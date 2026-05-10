"""Fetch Nifty 50 daily OHLC and cache locally.

Tries yfinance (^NSEI) first; falls back to two GitHub-hosted CSV mirrors
which between them cover 2008-01 through 2026-01. Sandboxed environments
typically block yfinance and NSE.

Cache: research/data/nifty_daily.csv
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

OLD_CSV = (
    "https://raw.githubusercontent.com/manishkr1754/"
    "NIFTY50_Data_Analysis_NSETOOLS_NSEPY_Python/"
    "57873fe91ab740b0f15a804641c2840926357a1b/nifty50.csv"
)
# Validated dataset covering 2021-02 -> 2026-01-31. Used for the Tuesday-expiry
# era (Aug 28 2025+) which the older CSV does not reach.
NEW_CSV = (
    "https://raw.githubusercontent.com/Bhavik-Sheth/"
    "Quant-Project-Info/6d5dfe940c76bc95ac66f15f83b8b2d565a448de/"
    "backend/data/validated/1d/%5ENSEI_1d_validated.csv"
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


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _load_old() -> pd.DataFrame:
    raw = _http_get(OLD_CSV)
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def _load_new() -> pd.DataFrame:
    raw = _http_get(NEW_CSV)
    df = pd.read_csv(io.StringIO(raw))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Strip the timezone+18:30 marker so dates align with the older CSV
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    # The 18:30 UTC stamp = end-of-day for previous Indian session, so subtract 1 day
    # to recover the actual trading session date in IST.
    df["date"] = df["date"] - pd.Timedelta(days=0)  # already correct after tz_convert
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def _from_github_fallback() -> pd.DataFrame:
    print(f"Loading old CSV: {OLD_CSV}")
    old = _load_old()
    print(f"  {len(old):,} rows  {old.index.min().date()} -> {old.index.max().date()}")
    print(f"Loading new CSV: {NEW_CSV}")
    new = _load_new()
    print(f"  {len(new):,} rows  {new.index.min().date()} -> {new.index.max().date()}")
    # Prefer the newer dataset where they overlap (it has cleaner OHLC).
    combined = pd.concat([old.loc[:new.index.min() - pd.Timedelta(days=1)], new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined.dropna()


def fetch() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Trying yfinance ticker={YF_TICKER}...")
    df = _from_yfinance()
    source = "yfinance"
    if df is None or df.empty:
        df = _from_github_fallback()
        source = "github-fallback-merged"
    df.to_csv(CACHE_PATH)
    META_PATH.write_text(
        f"source={source}\nrows={len(df)}\n"
        f"start={df.index.min().date()}\nend={df.index.max().date()}\n"
    )
    print(f"\nSource: {source}")
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
