"""Market data provider.

Two implementations:
  * HistoricalReplay  - reads the cached CSV produced by research/, walks
                        through it one trading session at a time. Used by
                        paper mode.
  * KiteLive          - placeholder for the real Kite Connect / Kite MCP
                        version. Not implemented in this paper-only
                        build; raises NotImplementedError so the live
                        code path is impossible to hit accidentally.

Both expose the same interface so strategy code doesn't know which one
it's talking to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Snapshot:
    """A single point-in-time view of the market the agent reacts to."""
    date: pd.Timestamp
    spot: float
    open: float
    high: float
    low: float
    close: float
    realised_vol_21d: float


class HistoricalReplay:
    """Walk through cached Nifty daily OHLC."""

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path, index_col="date", parse_dates=["date"])
        df = df.sort_index()
        # Trailing 21-day realised vol on log returns.
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["rv21"] = log_ret.rolling(21).std() * np.sqrt(252)
        self.df = df.dropna()
        self._cursor = 0

    @property
    def total_sessions(self) -> int:
        return len(self.df)

    def reset(self) -> None:
        self._cursor = 0

    def session_at(self, idx: int) -> Snapshot:
        row = self.df.iloc[idx]
        ts = self.df.index[idx]
        return Snapshot(
            date=ts,
            spot=float(row["close"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            realised_vol_21d=float(row["rv21"]),
        )

    def iterate(self):
        """Yield (idx, snapshot) for every session."""
        for i in range(len(self.df)):
            yield i, self.session_at(i)


class KiteLive:
    """Placeholder. Real implementation would call Kite Connect / Kite MCP."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Live Kite mode is not built in this paper-only milestone. "
            "Paper-trade for a few weeks first, then we'll wire up Kite."
        )
