"""Monthly vertical spread simulation on Reliance Industries.

Tests the user-suggested setups:

  1. Bull call spread  (buy ITM call, sell OTM call)
     - Net DEBIT. Profits if RIL goes up.
     - Max loss = debit. Max profit = strike_width - debit.

  2. Bear put spread   (buy ITM put,  sell OTM put)
     - Net DEBIT. Profits if RIL goes down.
     - Max loss = debit. Max profit = strike_width - debit.

(The user labelled (2) "bull put spread" but described the bear-put debit
combination, so we test what they actually described.)

Methodology
-----------
- Entry: first trading day of each calendar month, at close.
- Exit:  monthly expiry, settled to terminal payoff.
- Strikes: snap spot * (1 ± offset_pct) to the nearest Rs 10.
- Both legs priced via Black-Scholes off trailing 21-day realised vol.
- Optional `iv_skew_pct`: bumps the IV used for the OTM leg above the ATM
  IV to model the typical NSE stock-option skew (OTM puts trade richer).

Cycles where the close-to-close move exceeds 15% in any single day are
excluded as likely corporate actions.

Outputs
-------
  output/ril_spread_<variant>.csv     per-month PnL
  output/ril_spread_summary.csv       aggregate stats
  output/ril_spread_equity.png        cumulative equity curves
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fetch_reliance_history import load

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RV_WINDOW = 21
DAYS_PER_YEAR = 252
RISK_FREE = 0.06
STRIKE_TICK = 10.0           # Rs 10 spacing
SPLIT_FILTER_PCT = 0.15      # skip months with any single-day move > 15%
THURSDAY_ERA_END = pd.Timestamp("2025-08-26")


class Variant(NamedTuple):
    label: str
    direction: str           # "bull-call" or "bear-put"
    offset_pct: float        # how far each strike sits from spot, as fraction
    iv_skew_pct: float = 0.0 # extra IV applied to the OTM leg (e.g. 0.05 = 5%)


VARIANTS = [
    # Bull call: BUY ITM call, SELL OTM call. Pure directional bullish bet.
    # Equity call wing is roughly flat in IV, so no favourable skew to model.
    # The "skew_against" variant adds the more realistic case where the ITM
    # call you BUY trades at a vol premium (because by put-call parity the
    # ITM call IV equals the OTM put IV, which is rich on Indian equities).
    Variant("bull-call_0.5pct",          "bull-call", 0.005, 0.00),
    Variant("bull-call_1.0pct",          "bull-call", 0.010, 0.00),
    Variant("bull-call_1.0pct_skew_hurt", "bull-call", 0.010, -0.05),  # ITM-call you buy is dear

    # Bear put: BUY ITM put, SELL OTM put. Equity put skew = OTM puts trade
    # rich; selling OTM put against a less-rich ITM put captures that skew.
    Variant("bear-put_0.5pct",           "bear-put",  0.005, 0.00),
    Variant("bear-put_1.0pct",           "bear-put",  0.010, 0.00),
    Variant("bear-put_1.0pct_skew_help", "bear-put",  0.010, 0.05),    # OTM put you sell is rich
]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)


def realised_vol(close: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(DAYS_PER_YEAR)


def find_monthly_expiry_indices(df: pd.DataFrame) -> list[tuple[int, int]]:
    """For each calendar month, return (entry_pos, expiry_pos).

    entry_pos  = first trading day of the month
    expiry_pos = last trading day on or before the monthly expiry (last
                 Thursday pre-2025-08-28, last Tuesday after)
    """
    df = df.copy()
    df["weekday"] = df.index.weekday
    df["year_month"] = df.index.to_period("M")

    out: list[tuple[int, int]] = []
    for ym, group in df.groupby("year_month", sort=True):
        first_day = group.iloc[0]
        # Determine target expiry weekday for this month
        if group.index.max() <= THURSDAY_ERA_END:
            target_dow = 3  # Thursday
        else:
            target_dow = 1  # Tuesday
        # Last trading day <= target_dow within last week of month
        # Approach: find the LAST occurrence of weekday == target_dow; if
        # that day was a holiday, fall back to the previous trading day in
        # the same week.
        candidates = group[group["weekday"] == target_dow]
        if candidates.empty:
            # Fall back to last trading day of the month
            expiry = group.iloc[-1]
        else:
            expiry = candidates.iloc[-1]
        entry_pos = df.index.get_loc(first_day.name)
        expiry_pos = df.index.get_loc(expiry.name)
        if expiry_pos > entry_pos:
            out.append((entry_pos, expiry_pos))
    return out


def has_corporate_action(df: pd.DataFrame, start: int, end: int) -> bool:
    sub_close = df["close"].iloc[start:end + 1]
    rets = sub_close.pct_change().abs()
    return bool((rets > SPLIT_FILTER_PCT).any())


def simulate_variant(df: pd.DataFrame, variant: Variant,
                      cycles: list[tuple[int, int]]) -> pd.DataFrame:
    df = df.copy()
    df["rv21"] = realised_vol(df["close"])

    rows = []
    for entry_pos, expiry_pos in cycles:
        if entry_pos < RV_WINDOW:
            continue
        if has_corporate_action(df, entry_pos, expiry_pos):
            continue
        S0 = df["close"].iloc[entry_pos]
        S_T = df["close"].iloc[expiry_pos]
        sigma = df["rv21"].iloc[entry_pos]
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        n_days = expiry_pos - entry_pos
        T = n_days / DAYS_PER_YEAR

        if variant.direction == "bull-call":
            K_long = round(S0 * (1 - variant.offset_pct) / STRIKE_TICK) * STRIKE_TICK
            K_short = round(S0 * (1 + variant.offset_pct) / STRIKE_TICK) * STRIKE_TICK
            if K_short <= K_long:
                continue
            # iv_skew_pct < 0 means the ITM leg you BUY is dearer than ATM
            # (the put-call parity reality on Indian equities).
            sigma_long = sigma * (1 - variant.iv_skew_pct)
            sigma_short = sigma
            prem_long = bs_call(S0, K_long, T, RISK_FREE, sigma_long)
            prem_short = bs_call(S0, K_short, T, RISK_FREE, sigma_short)
            debit = prem_long - prem_short
            payoff = max(0.0, S_T - K_long) - max(0.0, S_T - K_short)
            width = K_short - K_long
        else:  # bear-put
            K_long = round(S0 * (1 + variant.offset_pct) / STRIKE_TICK) * STRIKE_TICK
            K_short = round(S0 * (1 - variant.offset_pct) / STRIKE_TICK) * STRIKE_TICK
            if K_long <= K_short:
                continue
            # iv_skew_pct > 0 means the OTM put you SELL trades richer than ATM.
            sigma_long = sigma
            sigma_short = sigma * (1 + variant.iv_skew_pct)
            prem_long = bs_put(S0, K_long, T, RISK_FREE, sigma_long)
            prem_short = bs_put(S0, K_short, T, RISK_FREE, sigma_short)
            debit = prem_long - prem_short
            payoff = max(0.0, K_long - S_T) - max(0.0, K_short - S_T)
            width = K_long - K_short

        if debit <= 0:
            continue
        pnl_pts = payoff - debit
        pnl_on_debit = pnl_pts / debit       # return on capital tied up
        pnl_on_underlying = pnl_pts / S0     # return as % of stock notional

        rows.append({
            "entry_date": df.index[entry_pos],
            "expiry_date": df.index[expiry_pos],
            "S0": S0,
            "S_T": S_T,
            "underlying_move_pct": (S_T / S0 - 1) * 100,
            "K_long": K_long,
            "K_short": K_short,
            "rv21": sigma,
            "T_years": T,
            "debit_pts": debit,
            "payoff_pts": payoff,
            "pnl_pts": pnl_pts,
            "max_profit_pts": width - debit,
            "pnl_on_debit_pct": pnl_on_debit * 100,
            "pnl_on_underlying_pct": pnl_on_underlying * 100,
        })
    return pd.DataFrame(rows).set_index("entry_date") if rows else pd.DataFrame()


def summarise(label: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"variant": label, "n": 0}
    pnl = df["pnl_on_debit_pct"]   # % return on the debit paid each cycle
    cycles_per_year = 12
    # Annualised return assuming a fixed-fraction (e.g. 5%) of capital per
    # cycle, so the monthly returns add roughly linearly. The "all-in"
    # cumprod() metric is misleading because a single -100% wipes it out
    # forever, even though no real trader sizes that way.
    annualised_return_pct = pnl.mean() * cycles_per_year
    return {
        "variant": label,
        "n": len(pnl),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "mean_return_per_cycle_pct": float(pnl.mean()),
        "median_return_per_cycle_pct": float(pnl.median()),
        "stdev_per_cycle_pct": float(pnl.std()),
        "worst_cycle_pct": float(pnl.min()),
        "best_cycle_pct": float(pnl.max()),
        "es5_per_cycle_pct": float(pnl.quantile(0.05)),
        "annualised_return_pct": float(annualised_return_pct),
        "annualised_sharpe": float(pnl.mean() / pnl.std() * math.sqrt(cycles_per_year)) if pnl.std() > 0 else 0.0,
        "mean_pnl_on_underlying_pct": float(df["pnl_on_underlying_pct"].mean()),
    }


def plot_equity(results: dict[str, pd.DataFrame]) -> Path:
    """Plot cumulative return assuming 5% of capital allocated per cycle.

    Avoids the all-in artifact where one -100% cycle zeroes the curve.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fraction = 0.05
    for label, df in results.items():
        if df.empty:
            continue
        # Effective per-cycle return on TOTAL capital = fraction * pnl_on_debit
        per_cycle = (df["pnl_on_debit_pct"] / 100) * fraction
        cum = (1 + per_cycle).cumprod()
        ax.plot(df.index, cum, label=label, lw=1.4)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_title("RIL monthly vertical spreads — 5% of capital allocated per cycle")
    ax.set_ylabel("Total-capital equity multiple")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "ril_spread_equity.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows  {df.index.min().date()} -> {df.index.max().date()}\n")

    cycles = find_monthly_expiry_indices(df)
    print(f"Identified {len(cycles)} monthly cycles\n")

    results: dict[str, pd.DataFrame] = {}
    summaries = []
    for v in VARIANTS:
        cycles_df = simulate_variant(df, v, cycles)
        results[v.label] = cycles_df
        summary = summarise(v.label, cycles_df)
        summaries.append(summary)
        if not cycles_df.empty:
            cycles_df.to_csv(OUT_DIR / f"ril_spread_{v.label}.csv")
        print(f"--- {v.label} ---")
        for k, val in summary.items():
            if isinstance(val, float):
                print(f"  {k:30s} {val:+.2f}")
            else:
                print(f"  {k:30s} {val}")
        print()

    pd.DataFrame(summaries).to_csv(OUT_DIR / "ril_spread_summary.csv", index=False)
    plot_equity(results)
    print("Realised RIL move stats over the cycles tested:")
    any_df = next((d for d in results.values() if not d.empty), None)
    if any_df is not None:
        m = any_df["underlying_move_pct"]
        print(f"  mean = {m.mean():+.2f}%   median = {m.median():+.2f}%")
        print(f"  pct positive months = {(m > 0).mean() * 100:.1f}%")
        print(f"  best month = {m.max():+.2f}%   worst = {m.min():+.2f}%")


if __name__ == "__main__":
    main()
