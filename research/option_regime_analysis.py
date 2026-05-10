"""Naive weekly short-strangle simulation on Nifty 50.

Most retail "AI Nifty" platforms run short-vol strategies. We simulate the
simplest version: every Monday open, sell a 1-week ATM straddle at theoretical
fair value (Black-Scholes priced off trailing realised vol), held to Friday
close. Settlement = max(0, S_T - K) for the call and max(0, K - S_T) for the
put paid to the buyer; the seller keeps premium minus payout.

This is intentionally naive — no Greeks, no IV smile, no margin model — but it
characterises the realised drawdown distribution of short-vol on Nifty by
volatility regime, which is what an LLM agent would need to size against.

Outputs:
  output/short_strangle_weekly.csv   per-week PnL
  output/short_strangle_summary.csv  aggregate stats by vol regime
  output/short_strangle_equity.png   cumulative equity curve
  output/realised_vol.png            21-day realised vol over time
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fetch_nifty_history import load

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RV_WINDOW = 21      # days of trailing returns for realised vol
DAYS_PER_YEAR = 252
WEEK_DAYS = 5
RISK_FREE = 0.06    # ~Indian 10y, rough


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(0.0, S - K * math.exp(-r * T))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    call = bs_call(S, K, T, r, sigma)
    return call - S + K * math.exp(-r * T)


def realised_vol(close: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(DAYS_PER_YEAR)


def simulate_short_strangle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rv21"] = realised_vol(df["close"])
    df = df.dropna()
    weekday = df.index.weekday
    is_monday = (weekday == 0)
    df.loc[:, "is_monday"] = is_monday

    rows = []
    indices = df.index.tolist()
    for i, ts in enumerate(indices):
        if not is_monday[i]:
            continue
        if i + WEEK_DAYS - 1 >= len(indices):
            break
        # Use Monday OPEN as entry to capture the gap effect on the seller
        entry = df["open"].iloc[i]
        exit_ts = indices[min(i + WEEK_DAYS - 1, len(indices) - 1)]
        exit_px = df.loc[exit_ts, "close"]
        sigma = df["rv21"].iloc[i]
        if sigma <= 0 or not np.isfinite(sigma):
            continue
        T = WEEK_DAYS / DAYS_PER_YEAR
        K = round(entry / 50) * 50  # snap to nearest 50, like real Nifty strikes
        call_px = bs_call(entry, K, T, RISK_FREE, sigma)
        put_px = bs_put(entry, K, T, RISK_FREE, sigma)
        premium = call_px + put_px
        payout = max(0.0, exit_px - K) + max(0.0, K - exit_px)
        pnl_pts = premium - payout
        pnl_pct = pnl_pts / entry
        rows.append({
            "monday": ts,
            "friday": exit_ts,
            "entry": entry,
            "exit": exit_px,
            "strike": K,
            "rv21_in": sigma,
            "premium_pts": premium,
            "payout_pts": payout,
            "pnl_pts": pnl_pts,
            "pnl_pct": pnl_pct,
        })
    return pd.DataFrame(rows).set_index("monday")


def vol_regime(rv: float) -> str:
    if rv < 0.12:
        return "low (<12%)"
    if rv < 0.18:
        return "mid (12-18%)"
    if rv < 0.25:
        return "high (18-25%)"
    return "stress (>25%)"


def regime_summary(weeks: pd.DataFrame) -> pd.DataFrame:
    weeks = weeks.copy()
    weeks["regime"] = weeks["rv21_in"].apply(vol_regime)
    rows = []
    for regime, sub in weeks.groupby("regime"):
        n = len(sub)
        rows.append({
            "regime": regime,
            "n_weeks": n,
            "win_rate_pct": (sub["pnl_pct"] > 0).mean() * 100,
            "mean_pnl_pct": sub["pnl_pct"].mean() * 100,
            "median_pnl_pct": sub["pnl_pct"].median() * 100,
            "worst_pnl_pct": sub["pnl_pct"].min() * 100,
            "best_pnl_pct": sub["pnl_pct"].max() * 100,
            "stdev_pnl_pct": sub["pnl_pct"].std() * 100,
            "expected_shortfall_5pct_pct": sub["pnl_pct"].quantile(0.05) * 100,
        })
    out = pd.DataFrame(rows).sort_values("regime")
    return out


def plot_equity(weeks: pd.DataFrame) -> Path:
    cum = (1 + weeks["pnl_pct"]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weeks.index, cum, color="steelblue")
    ax.set_title("Naive weekly short-strangle on Nifty — cumulative equity (1 lot, no leverage)")
    ax.set_ylabel("Equity multiple")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "short_strangle_equity.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_rv(df: pd.DataFrame) -> Path:
    rv = realised_vol(df["close"]) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rv.index, rv, color="darkorange")
    for thr, label in [(12, "low"), (18, "mid"), (25, "high")]:
        ax.axhline(thr, color="gray", lw=0.6, ls="--")
    ax.set_title("Nifty 50 — 21-day annualised realised volatility")
    ax.set_ylabel("RV (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "realised_vol.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    df = load()
    weeks = simulate_short_strangle(df)
    summary = regime_summary(weeks)

    weeks.to_csv(OUT_DIR / "short_strangle_weekly.csv")
    summary.to_csv(OUT_DIR / "short_strangle_summary.csv", index=False)
    plot_equity(weeks)
    plot_rv(df)

    print(f"Simulated {len(weeks)} weekly short-strangles")
    print(f"  Period: {weeks.index.min().date()} -> {weeks.index.max().date()}")
    print()
    print("=== By volatility regime ===")
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))
    print()
    cum = (1 + weeks["pnl_pct"]).cumprod()
    print(f"Cumulative equity multiple (no leverage, no costs): {cum.iloc[-1]:.2f}x")
    cagr = cum.iloc[-1] ** (52 / len(weeks)) - 1
    print(f"Implied weekly-CAGR: {cagr * 100:.2f}%")
    weekly_sharpe = weeks["pnl_pct"].mean() / weeks["pnl_pct"].std() * math.sqrt(52)
    print(f"Annualised Sharpe (weekly returns): {weekly_sharpe:.2f}")
    worst = weeks["pnl_pct"].min() * 100
    print(f"Worst single-week PnL: {worst:.2f}%")
    print(f"5%-tail (expected shortfall) PnL: {weeks['pnl_pct'].quantile(0.05) * 100:.2f}%")


if __name__ == "__main__":
    main()
