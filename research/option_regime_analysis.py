"""Expiry-aware short-strangle simulation on Nifty 50.

Replaces the Phase-1 Monday-to-Friday version (which didn't reflect how Indian
retail short-vol actually works — they sell late in the cycle to capture
accelerating theta).

Methodology
-----------
For each weekly expiry in the dataset:
  1. Find the trading day SELL_OFFSET trading days before expiry.
  2. SELL the ATM straddle at that day's close (strike = nearest 50).
  3. Find the trading day BUY_OFFSET trading days before expiry.
  4. BUY-BACK at that day's close.
  5. Both legs priced via Black-Scholes at theoretical fair value, using
     trailing 21-day realised vol as the IV input (no IV/RV premium baked
     in — see README for why this matters).

Two expiry eras are run separately:
  * Thursday era: 2008-01 to 2025-08-26  (DOW = 3)
  * Tuesday  era: 2025-08-28 onwards     (DOW = 1)

Within each era, we test two holding-period configurations:
  * "Theta harvest" : sell T-4, buy back T-1 (~3 trading days, Wed -> Mon
    for a Tuesday expiry; Fri -> Wed for a Thursday expiry).
  * "Full week"     : sell T-7, buy back T-1 (~6 trading days).

Outputs
-------
  output/strangle_v2_<era>_<config>.csv   per-cycle PnL
  output/strangle_v2_summary.csv          aggregate stats by era + config
  output/strangle_v2_equity.png           cumulative equity, all variants
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

from fetch_nifty_history import load

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RV_WINDOW = 21
DAYS_PER_YEAR = 252
RISK_FREE = 0.06

THURSDAY_ERA_END = pd.Timestamp("2025-08-26")
TUESDAY_ERA_START = pd.Timestamp("2025-08-28")


class EraConfig(NamedTuple):
    label: str
    expiry_dow: int   # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
    start: pd.Timestamp | None
    end: pd.Timestamp | None


ERAS = [
    EraConfig("thursday-era", 3, None, THURSDAY_ERA_END),
    EraConfig("tuesday-era", 1, TUESDAY_ERA_START, None),
]


class HoldConfig(NamedTuple):
    label: str
    sell_offset: int   # trading days BEFORE expiry to sell
    buy_offset: int    # trading days BEFORE expiry to buy back
    iv_premium: float = 1.0   # multiplier on RV when pricing the SELL leg


HOLDS = [
    HoldConfig("theta-harvest", 4, 1, 1.0),
    HoldConfig("full-week", 7, 1, 1.0),
    # Realistic VRP: Indian index ATM IV historically prices ~15% above
    # trailing RV. Selling at 1.15x RV and buying back at 1.0x RV captures
    # that premium and is the closest thing to what retail actually collects.
    HoldConfig("theta-harvest+15%VRP", 4, 1, 1.15),
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


def find_expiry_indices(df: pd.DataFrame, expiry_dow: int) -> list[int]:
    """Return positional indices of trading days that are weekly expiries.

    A trading day is an expiry if (a) its weekday == expiry_dow, OR (b) the
    weekly expiry was a holiday and the previous trading day inherits it. We
    keep it simple: take the LAST trading day in each week that is on or
    before the expiry weekday.
    """
    df = df.copy()
    df["weekday"] = df.index.weekday
    # Group by (year, iso_week)
    iso = df.index.isocalendar()
    df["year_week"] = iso.year.astype(int) * 100 + iso.week.astype(int)

    expiries: list[int] = []
    for _, group in df.groupby("year_week", sort=False):
        valid = group[group["weekday"] <= expiry_dow]
        if valid.empty:
            continue
        last = valid.iloc[-1]
        idx = df.index.get_loc(last.name)
        expiries.append(idx)
    return expiries


def simulate(df: pd.DataFrame, era: EraConfig, hold: HoldConfig) -> pd.DataFrame:
    sub = df.copy()
    if era.start is not None:
        sub = sub.loc[sub.index >= era.start]
    if era.end is not None:
        sub = sub.loc[sub.index <= era.end]
    if len(sub) < RV_WINDOW + hold.sell_offset + 5:
        return pd.DataFrame()

    sub["rv21"] = realised_vol(sub["close"])
    expiries = find_expiry_indices(sub, era.expiry_dow)

    rows = []
    n = len(sub)
    for exp_pos in expiries:
        sell_pos = exp_pos - hold.sell_offset
        buy_pos = exp_pos - hold.buy_offset
        if sell_pos < RV_WINDOW or buy_pos <= sell_pos or buy_pos >= n:
            continue
        sell_row = sub.iloc[sell_pos]
        buy_row = sub.iloc[buy_pos]
        sigma_sell = sell_row["rv21"]
        if not np.isfinite(sigma_sell) or sigma_sell <= 0:
            continue
        S_sell = sell_row["close"]
        S_buy = buy_row["close"]
        K = round(S_sell / 50) * 50

        T_sell = (hold.sell_offset) / DAYS_PER_YEAR
        T_buy = (hold.buy_offset) / DAYS_PER_YEAR
        # For buyback pricing, blend trailing RV at the buyback date so it
        # reflects how the market would re-price at that moment.
        sigma_buy = sub["rv21"].iloc[buy_pos]
        if not np.isfinite(sigma_buy) or sigma_buy <= 0:
            sigma_buy = sigma_sell

        sigma_sell_used = sigma_sell * hold.iv_premium
        prem_sell = bs_call(S_sell, K, T_sell, RISK_FREE, sigma_sell_used) + \
                    bs_put(S_sell, K, T_sell, RISK_FREE, sigma_sell_used)
        prem_buy = bs_call(S_buy, K, T_buy, RISK_FREE, sigma_buy) + \
                   bs_put(S_buy, K, T_buy, RISK_FREE, sigma_buy)
        pnl_pts = prem_sell - prem_buy
        pnl_pct = pnl_pts / S_sell
        rows.append({
            "sell_date": sell_row.name,
            "buy_date": buy_row.name,
            "expiry_date": sub.iloc[exp_pos].name,
            "S_sell": S_sell,
            "S_buy": S_buy,
            "strike": K,
            "rv21_sell": sigma_sell,
            "rv21_buy": sigma_buy,
            "prem_sell_pts": prem_sell,
            "prem_buy_pts": prem_buy,
            "pnl_pts": pnl_pts,
            "pnl_pct": pnl_pct,
            "abs_underlying_move_pct": (S_buy / S_sell - 1) * 100,
        })
    return pd.DataFrame(rows).set_index("sell_date") if rows else pd.DataFrame()


def summarise(label: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"variant": label, "n": 0}
    pct = df["pnl_pct"]
    cum = (1 + pct).prod()
    weeks_per_year = 52
    n = len(pct)
    cagr = cum ** (weeks_per_year / n) - 1 if n else 0.0
    return {
        "variant": label,
        "n": n,
        "win_rate_pct": float((pct > 0).mean() * 100),
        "mean_pnl_pct": float(pct.mean() * 100),
        "median_pnl_pct": float(pct.median() * 100),
        "stdev_pnl_pct": float(pct.std() * 100),
        "worst_pnl_pct": float(pct.min() * 100),
        "best_pnl_pct": float(pct.max() * 100),
        "es5_pct": float(pct.quantile(0.05) * 100),
        "cum_multiple": float(cum),
        "implied_cagr_pct": float(cagr * 100),
        "annualised_sharpe": float(pct.mean() / pct.std() * math.sqrt(weeks_per_year)) if pct.std() > 0 else 0.0,
    }


def plot_equity(results: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, df in results.items():
        if df.empty:
            continue
        cum = (1 + df["pnl_pct"]).cumprod()
        ax.plot(df.index, cum, label=label)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_title("Nifty short-strangle (fair-value priced) — cumulative equity by variant")
    ax.set_ylabel("Equity multiple")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "strangle_v2_equity.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows  {df.index.min().date()} -> {df.index.max().date()}\n")

    results: dict[str, pd.DataFrame] = {}
    summaries = []
    for era in ERAS:
        for hold in HOLDS:
            label = f"{era.label}/{hold.label}"
            cycles = simulate(df, era, hold)
            results[label] = cycles
            summary = summarise(label, cycles)
            summaries.append(summary)
            if not cycles.empty:
                cycles.to_csv(OUT_DIR / f"strangle_v2_{era.label}_{hold.label}.csv")
            print(f"--- {label} ---")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"  {k:24s} {v:+.3f}")
                else:
                    print(f"  {k:24s} {v}")
            print()

    pd.DataFrame(summaries).to_csv(OUT_DIR / "strangle_v2_summary.csv", index=False)
    plot_equity(results)


if __name__ == "__main__":
    main()
