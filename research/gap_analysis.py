"""Overnight gap behavior on Nifty 50.

Question: GIFT Nifty's premium/discount to prior NSE close is the dominant
pre-open signal in India. Empirically, what does the gap predict?

We measure for each session:
  gap     = open / prev_close - 1            (the realized GIFT-style signal)
  intra   = close / open - 1                 (intraday move after gap)
  whole   = close / prev_close - 1           (gap + intraday)
  follow  = sign(intra) == sign(gap)         (gap continuation)
  fill    = close crosses prev_close intraday (low <= prev_close <= high
            when gap up; high >= prev_close >= low when gap down)

Then bucket by absolute gap size and report follow-through and reversion stats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fetch_nifty_history import load

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKETS = [
    (0.000, 0.0025, "0.00-0.25%"),
    (0.0025, 0.005, "0.25-0.50%"),
    (0.005, 0.0075, "0.50-0.75%"),
    (0.0075, 0.010, "0.75-1.00%"),
    (0.010, 0.015, "1.00-1.50%"),
    (0.015, 0.025, "1.50-2.50%"),
    (0.025, 1.000, ">2.50%"),
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    prev_close = df["close"].shift(1)
    f["prev_close"] = prev_close
    f["open"] = df["open"]
    f["high"] = df["high"]
    f["low"] = df["low"]
    f["close"] = df["close"]
    f["gap"] = df["open"] / prev_close - 1
    f["intra"] = df["close"] / df["open"] - 1
    f["whole"] = df["close"] / prev_close - 1
    f["follow"] = np.sign(f["intra"]) == np.sign(f["gap"])
    gap_up = f["gap"] > 0
    gap_dn = f["gap"] < 0
    fill = ((gap_up & (df["low"] <= prev_close)) |
            (gap_dn & (df["high"] >= prev_close)))
    f["filled"] = fill
    return f.dropna()


def bucket_stats(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi, label in BUCKETS:
        mask = (f["gap"].abs() >= lo) & (f["gap"].abs() < hi)
        sub = f.loc[mask]
        n = len(sub)
        if n < 10:
            rows.append({"bucket": label, "n": n})
            continue
        # Directional: ride the gap (long if gap up, short if gap down)
        ride = np.sign(sub["gap"]) * sub["intra"]
        # Fade the gap (short if gap up, long if gap down)
        fade = -np.sign(sub["gap"]) * sub["intra"]
        rows.append({
            "bucket": label,
            "n": n,
            "follow_pct": sub["follow"].mean() * 100,
            "fill_pct": sub["filled"].mean() * 100,
            "mean_intra_bps": sub["intra"].mean() * 1e4,
            "mean_ride_bps": ride.mean() * 1e4,
            "mean_fade_bps": fade.mean() * 1e4,
            "ride_sharpe": ride.mean() / ride.std() * np.sqrt(252) if ride.std() > 0 else np.nan,
            "fade_sharpe": fade.mean() / fade.std() * np.sqrt(252) if fade.std() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def overall_stats(f: pd.DataFrame) -> dict:
    return {
        "rows": len(f),
        "date_min": str(f.index.min().date()),
        "date_max": str(f.index.max().date()),
        "mean_abs_gap_bps": float(f["gap"].abs().mean() * 1e4),
        "median_abs_gap_bps": float(f["gap"].abs().median() * 1e4),
        "p95_abs_gap_bps": float(f["gap"].abs().quantile(0.95) * 1e4),
        "p99_abs_gap_bps": float(f["gap"].abs().quantile(0.99) * 1e4),
        "pct_gap_up": float((f["gap"] > 0).mean() * 100),
        "pct_gap_dn": float((f["gap"] < 0).mean() * 100),
        "pct_follow_overall": float(f["follow"].mean() * 100),
        "pct_fill_overall": float(f["filled"].mean() * 100),
    }


def plot_gap_distribution(f: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    bps = f["gap"] * 1e4
    ax.hist(bps.clip(-300, 300), bins=80, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Nifty 50 overnight gap distribution (clipped to ±300 bps)")
    ax.set_xlabel("Open / prev close - 1  (bps)")
    ax.set_ylabel("Sessions")
    fig.tight_layout()
    out = OUT_DIR / "gap_distribution.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_bucket_edge(stats: pd.DataFrame) -> Path:
    s = stats.dropna(subset=["ride_sharpe"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(s))
    width = 0.35
    ax.bar(x - width/2, s["ride_sharpe"], width, label="Ride gap", color="steelblue")
    ax.bar(x + width/2, s["fade_sharpe"], width, label="Fade gap", color="indianred")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(s["bucket"], rotation=20)
    ax.set_ylabel("Annualised Sharpe (intraday-only PnL)")
    ax.set_title("Gap-direction strategy edge by gap-size bucket")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "gap_bucket_edge.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    df = load()
    f = build_features(df)
    overall = overall_stats(f)
    buckets = bucket_stats(f)

    f.to_csv(OUT_DIR / "gap_features.csv")
    buckets.to_csv(OUT_DIR / "gap_bucket_stats.csv", index=False)
    pd.Series(overall).to_csv(OUT_DIR / "gap_overall_stats.csv", header=["value"])

    plot_gap_distribution(f)
    plot_bucket_edge(buckets)

    print("=== Overall gap stats ===")
    for k, v in overall.items():
        print(f"  {k:25s} {v}")
    print()
    print("=== By gap-size bucket ===")
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(buckets.to_string(index=False))


if __name__ == "__main__":
    main()
