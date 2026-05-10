"""Fetch Nifty 50 daily OHLC and cache locally.

Primary source: yfinance (^NSEI). Falls back to a public GitHub-hosted CSV
(2008-01 through 2023-01) when yfinance is unreachable, which is common in
sandboxed environments.

Cache: research/data/nifty_daily.parquet
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "nifty_daily.csv"
META_PATH = DATA_DIR / "nifty_daily.meta.txt"

YF_TICKER = "^NSEI"
GITHUB_FALLBACK = (
    "https://raw.githubusercontent.com/manishkr1754/"
    "NIFTY50_Data_Analysis_NSETOOLS_NSEPY_Python/"
    "57873fe91ab740b0f15a804641c2840926357a1b/nifty50.csv"
)


def _from_yfinance() -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(YF_TICKER, period="max", interval="1d",
                         auto_adjust=False, progress=False)
    except Exception as e:
        print(f"yfinance failed: {e}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "date"
    return df


def _from_github_fallback() -> pd.DataFrame:
    print(f"Falling back to GitHub CSV: {GITHUB_FALLBACK}")
    req = urllib.request.Request(GITHUB_FALLBACK, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Trying yfinance ticker={YF_TICKER}...")
    df = _from_yfinance()
    source = "yfinance"
    if df is None or df.empty:
        df = _from_github_fallback()
        source = "github-fallback"
    df.to_csv(CACHE_PATH)
    META_PATH.write_text(
        f"source={source}\nrows={len(df)}\n"
        f"start={df.index.min().date()}\nend={df.index.max().date()}\n"
    )
    print(f"Source: {source}")
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
