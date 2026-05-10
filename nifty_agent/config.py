"""Configuration for the Nifty short-strangle paper-trading agent.

All numeric rules live here so a non-coder can edit them without touching
trading logic. Every value has a one-line plain-English explanation.

PAPER MODE ONLY. Live trading requires a separate live_config.py with
broker credentials and a deliberate code path swap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Capital and sizing ----------------------------------------------
    starting_capital_inr: float = 200_000.0
    # Max % of capital risked on a single weekly cycle's debit/margin.
    max_risk_per_cycle_pct: float = 0.05
    # If True, size each cycle off STARTING capital (no compounding).
    # Realistic for a paper test: position size doesn't snowball. The
    # honest backtest. Set False to model full Kelly-style compounding.
    fixed_sizing: bool = True
    # Hard cap on lots regardless of capital. Defends against the
    # paper-only compounding mirage AND against single-name liquidity
    # limits in any future live run.
    max_lots_per_cycle: int = 5
    # Lot size of Nifty options (NSE current spec: 75).
    lot_size: int = 75
    # Each option strike snapped to nearest multiple of this (Rs).
    strike_tick: float = 50.0

    # --- VRP rule table --------------------------------------------------
    # VRP = (implied vol / realised vol) - 1. Bigger = options are richer.
    # The agent uses this single number to decide whether and how big to
    # trade each week.
    vrp_full_size_threshold: float = 0.15   # VRP >= 15% -> full size
    vrp_half_size_threshold: float = 0.05   # 5% <= VRP < 15% -> half size
    # VRP < 5% -> skip the week
    # VRP < 0   -> skip and warn (panic regime)

    # --- Cycle timing (Tuesday expiry era, post 2025-08-28) -------------
    # Days before expiry to enter the position (sell strangle).
    # For Tuesday expiry: T-4 = previous Wednesday.
    # For Thursday expiry: T-4 = previous Friday.
    entry_days_before_expiry: int = 4
    # Days before expiry to plan exit (buy back).
    exit_days_before_expiry: int = 1

    # --- Risk tripwire ---------------------------------------------------
    # If the combined option value rises this multiple above what we
    # collected, force-close immediately. The "stop-loss" for short vol.
    stop_loss_multiple: float = 1.5
    # If the underlying moves this much in a single session, flag and
    # consider pre-emptive close. 2.5% intraday in Nifty is rare and
    # almost always means a regime shift.
    daily_move_panic_pct: float = 0.025

    # --- Realised vol window --------------------------------------------
    rv_window_days: int = 21

    # --- Skip rules ------------------------------------------------------
    # If trailing realised vol is below this, skip — too thin a premium.
    min_rv_pct: float = 0.06
    # If trailing realised vol is above this, skip — vol regime change.
    max_rv_pct: float = 0.35

    # --- Pricing assumptions (replay-only) ------------------------------
    # In paper mode we don't have a live option chain. We simulate today's
    # VRP as a function of trailing RV, matching the empirical pattern in
    # Indian index options:
    #   - calm markets (low RV)   -> rich VRP (~20%, complacency)
    #   - mid markets             -> normal VRP (~12%)
    #   - high vol                -> thin VRP (~6%)
    #   - stress vol              -> negative/inverted (skip regime)
    # Live mode replaces this with `kite.option_chain_iv(...)`.
    risk_free_rate: float = 0.06


DEFAULT = Config()
