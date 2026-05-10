"""VRP reader - the brain of the agent.

Given a market snapshot, decide whether to trade this week and at what
size. The whole agent's "intelligence" lives in this function and the
rules in config.py. The LLM doesn't pick direction; it doesn't pick
strikes; it doesn't make calls in the moment. This is just rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .market_data import Snapshot


@dataclass
class SizingDecision:
    trade: bool
    size_multiple: float       # 0.0, 0.5, or 1.0 of full size
    vrp: float
    reason: str


def assumed_vrp(rv: float) -> float:
    """Regime-dependent assumed VRP for paper mode.

    Matches the empirical Indian-index pattern: VRP is rich when markets
    are calm and collapses (or inverts) when vol spikes. The exact slope
    is a modelling assumption; live mode reads VRP directly from the
    Kite option chain.
    """
    if rv < 0.08:
        return 0.22         # complacency
    if rv < 0.14:
        return 0.15         # normal calm
    if rv < 0.20:
        return 0.10
    if rv < 0.26:
        return 0.05         # stress mode
    return -0.02            # crash regime - VRP inverts


def assumed_implied_vol(snap: Snapshot, cfg: Config) -> float:
    """Paper-mode IV = RV * (1 + assumed VRP for that RV regime)."""
    return snap.realised_vol_21d * (1.0 + assumed_vrp(snap.realised_vol_21d))


def decide(snap: Snapshot, cfg: Config) -> SizingDecision:
    rv = snap.realised_vol_21d
    if rv < cfg.min_rv_pct:
        return SizingDecision(
            trade=False, size_multiple=0.0, vrp=0.0,
            reason=f"skip: RV {rv:.1%} is below floor {cfg.min_rv_pct:.0%}; premium too thin",
        )
    if rv > cfg.max_rv_pct:
        return SizingDecision(
            trade=False, size_multiple=0.0, vrp=0.0,
            reason=f"skip: RV {rv:.1%} is above ceiling {cfg.max_rv_pct:.0%}; vol regime warning",
        )

    iv = assumed_implied_vol(snap, cfg)
    vrp = (iv / rv) - 1.0
    if vrp >= cfg.vrp_full_size_threshold:
        return SizingDecision(
            trade=True, size_multiple=1.0, vrp=vrp,
            reason=f"full size: VRP {vrp:.1%} rich",
        )
    if vrp >= cfg.vrp_half_size_threshold:
        return SizingDecision(
            trade=True, size_multiple=0.5, vrp=vrp,
            reason=f"half size: VRP {vrp:.1%} normal",
        )
    if vrp >= 0:
        return SizingDecision(
            trade=False, size_multiple=0.0, vrp=vrp,
            reason=f"skip: VRP {vrp:.1%} too thin",
        )
    return SizingDecision(
        trade=False, size_multiple=0.0, vrp=vrp,
        reason=f"PANIC SKIP: VRP {vrp:.1%} negative; market in stress mode",
    )
