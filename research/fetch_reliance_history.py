"""Fetch Reliance Industries (RELIANCE.NS) daily OHLC and cache locally.

Uses a public GitHub mirror covering 2023-01 -> 2026-03. Yahoo Finance is
not reachable from this sandbox.

Note: prices in this dataset reflect the post-Oct-2024 1:1 split. Cycles
where the close-to-close move exceeds 15% in a single session are flagged
as likely corporate actions and excluded from spread simulations.

Cache: research/data/reliance_daily.csv
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "reliance_daily.csv"
META_PATH = DATA_DIR / "reliance_daily.meta.txt"

GITHUB_CSV = (
    "https://raw.githubusercontent.com/YahiyaV/"
    "Stock-Market-Prediction-with-Sentiment-Integration/"
    "2827d8f7754e383a1945976cadb809b08ea99cc1/"
    "data/RELIANCE.NS_2023-01-01_2026-03-04.csv"
)


def fetch() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(GITHUB_CSV, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    # Header is: Price,Close,High,Low,Open,Volume
    # Row 2:    Ticker,RELIANCE.NS,RELIANCE.NS...
    # Row 3:    Date,,,,,
    # Then data rows. Skip the two metadata rows.
    df = pd.read_csv(io.StringIO(raw), skiprows=[1, 2])
    df = df.rename(columns={"Price": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.to_csv(CACHE_PATH)
    META_PATH.write_text(
        f"source=github-fallback (YahiyaV)\n"
        f"rows={len(df)}\n"
        f"start={df.index.min().date()}\nend={df.index.max().date()}\n"
    )
    print(f"Saved {len(df):,} rows  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Cache: {CACHE_PATH}")
    return df


def load() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return fetch()
    return pd.read_csv(CACHE_PATH, index_col="date", parse_dates=["date"])


if __name__ == "__main__":
    df = fetch()
    print(df.tail())
