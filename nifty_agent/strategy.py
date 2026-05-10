"""Weekly strangle strategy with stop-loss tripwire.

This is the meat of the agent. Each weekly cycle:

  1. Sunday  - read VRP, decide trade/skip/size for the upcoming week
  2. Wed     - enter short strangle if planned, set tripwire level
  3. daily   - mark-to-market, fire tripwire if breached
  4. Mon     - exit at planned target if still open

The tripwire is the single most important piece. Short-vol blows up when
sellers refuse to exit a losing position. Here it's enforced in code,
not left to discretion or to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import Config
from .market_data import HistoricalReplay, Snapshot
from .option_pricing import strangle_value
from .vrp_reader import decide, assumed_implied_vol

DAYS_PER_YEAR = 252

THURSDAY_ERA_END = pd.Timestamp("2025-08-26")


@dataclass
class OpenPosition:
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    target_exit_date: pd.Timestamp
    strike: float
    sigma_at_entry: float       # IV used to price the sell
    premium_collected: float    # per-share, in points
    contracts: int              # number of lots (1 lot = lot_size shares)
    stop_loss_level: float      # per-share strangle value; breach -> exit


@dataclass
class ClosedTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    expiry_date: pd.Timestamp
    strike: float
    contracts: int
    premium_collected: float    # per-share
    exit_value: float           # per-share
    pnl_per_share: float
    pnl_total_inr: float
    exit_reason: str
    vrp_at_entry: float
    sigma_at_entry: float


def _expiry_weekday(date: pd.Timestamp) -> int:
    """Tuesday after Aug 2025, Thursday before."""
    return 1 if date > THURSDAY_ERA_END else 3


def _find_expiry_indices(df: pd.DataFrame) -> list[tuple[int, int]]:
    """For each ISO week, return (entry_pos, expiry_pos) honouring era.

    entry_pos is the trading day cfg.entry_days_before_expiry sessions
    before the weekly expiry. Holiday-shifted to the previous trading day
    if needed.
    """
    df = df.copy()
    df["weekday"] = df.index.weekday
    iso = df.index.isocalendar()
    df["year_week"] = iso.year.astype(int) * 100 + iso.week.astype(int)
    out: list[tuple[int, int]] = []
    for _, group in df.groupby("year_week", sort=False):
        target_dow = _expiry_weekday(group.index.max())
        candidates = group[group["weekday"] <= target_dow]
        if candidates.empty:
            continue
        last = candidates.iloc[-1]
        expiry_pos = df.index.get_loc(last.name)
        out.append(expiry_pos)
    # Convert to (entry_pos, expiry_pos) pairs using the offset rule downstream.
    return list(set(out))


def _trading_day_offset(df: pd.DataFrame, expiry_pos: int, offset: int) -> int:
    """Position offset trading days before expiry_pos. Clamps at 0."""
    pos = expiry_pos - offset
    return max(pos, 0)


def run_paper_trader(replay: HistoricalReplay, cfg: Config) -> tuple[list[ClosedTrade], list[dict]]:
    """Walk historical sessions, simulate the full agent flow.

    Returns the closed trades list and a structured "diary" of every
    decision (skip, entry, daily monitor, exit) suitable for direct
    serialisation to JSON / CSV.
    """
    df = replay.df
    expiries = sorted(_find_expiry_indices(df))
    expiry_set = set(expiries)
    diary: list[dict] = []
    closed: list[ClosedTrade] = []
    capital = cfg.starting_capital_inr
    starting_capital = cfg.starting_capital_inr
    open_pos: Optional[OpenPosition] = None

    # For each expiry, plan an entry session offset days back, mark up
    # the entry calendar so the main loop can act on it.
    planned_entry_for: dict[int, int] = {}  # entry_pos -> expiry_pos
    for exp_pos in expiries:
        entry_pos = _trading_day_offset(df, exp_pos, cfg.entry_days_before_expiry)
        planned_entry_for[entry_pos] = exp_pos

    for i in range(len(df)):
        snap = replay.session_at(i)
        is_entry_day = i in planned_entry_for
        is_expiry_day = i in expiry_set

        # --- daily monitor of open position ----------------------------
        if open_pos is not None:
            t_remaining = max((open_pos.expiry_date - snap.date).days, 0) / DAYS_PER_YEAR
            mark = strangle_value(snap.spot, open_pos.strike, t_remaining,
                                   cfg.risk_free_rate, snap.realised_vol_21d or open_pos.sigma_at_entry)
            should_exit = False
            reason = ""
            if mark >= open_pos.stop_loss_level:
                should_exit = True
                reason = f"TRIPWIRE FIRED: mark {mark:.1f} >= stop {open_pos.stop_loss_level:.1f}"
            elif snap.date >= open_pos.target_exit_date:
                should_exit = True
                reason = "planned exit (T-1)"
            elif snap.date >= open_pos.expiry_date:
                should_exit = True
                reason = "held to expiry (target exit missed)"
            else:
                # Panic-move guard.
                day_move = abs(snap.close / snap.open - 1) if snap.open else 0
                if day_move > cfg.daily_move_panic_pct:
                    diary.append({
                        "date": str(snap.date.date()),
                        "event": "panic-move-flag",
                        "detail": f"intraday move {day_move:.2%} > {cfg.daily_move_panic_pct:.1%}; watching position",
                    })

            if should_exit:
                pnl_per_share = open_pos.premium_collected - mark
                pnl_total = pnl_per_share * cfg.lot_size * open_pos.contracts
                capital += pnl_total
                trade = ClosedTrade(
                    entry_date=open_pos.entry_date,
                    exit_date=snap.date,
                    expiry_date=open_pos.expiry_date,
                    strike=open_pos.strike,
                    contracts=open_pos.contracts,
                    premium_collected=open_pos.premium_collected,
                    exit_value=mark,
                    pnl_per_share=pnl_per_share,
                    pnl_total_inr=pnl_total,
                    exit_reason=reason,
                    vrp_at_entry=0.0,  # patched below
                    sigma_at_entry=open_pos.sigma_at_entry,
                )
                # Recover the entry vrp from the diary's most recent entry record.
                for d in reversed(diary):
                    if d.get("event") == "entry" and d.get("entry_date") == str(open_pos.entry_date.date()):
                        trade.vrp_at_entry = d.get("vrp", 0.0)
                        break
                closed.append(trade)
                diary.append({
                    "date": str(snap.date.date()),
                    "event": "exit",
                    "reason": reason,
                    "strike": open_pos.strike,
                    "premium_per_share": round(open_pos.premium_collected, 2),
                    "exit_per_share": round(mark, 2),
                    "pnl_per_share": round(pnl_per_share, 2),
                    "pnl_total_inr": round(pnl_total, 0),
                    "capital_after_inr": round(capital, 0),
                })
                open_pos = None

        # --- new entry --------------------------------------------------
        if is_entry_day and open_pos is None:
            exp_pos = planned_entry_for[i]
            decision = decide(snap, cfg)
            diary_entry = {
                "date": str(snap.date.date()),
                "event": "decision",
                "spot": round(snap.spot, 2),
                "rv21": round(snap.realised_vol_21d, 4),
                "iv_assumed": round(assumed_implied_vol(snap, cfg), 4),
                "vrp": round(decision.vrp, 4),
                "trade": decision.trade,
                "size_multiple": decision.size_multiple,
                "reason": decision.reason,
            }
            diary.append(diary_entry)
            if not decision.trade:
                continue
            t_to_expiry = (df.index[exp_pos] - snap.date).days / DAYS_PER_YEAR
            strike = round(snap.spot / cfg.strike_tick) * cfg.strike_tick
            sigma_sell = assumed_implied_vol(snap, cfg)
            premium = strangle_value(snap.spot, strike, t_to_expiry,
                                      cfg.risk_free_rate, sigma_sell)
            stop_level = premium * cfg.stop_loss_multiple
            # Size: budget = max_risk_per_cycle * capital * size_multiple.
            # Budget caps the WORST-CASE loss = (stop_level - premium) * lot_size * contracts.
            worst_case_per_lot = (stop_level - premium) * cfg.lot_size
            sizing_basis = starting_capital if cfg.fixed_sizing else capital
            risk_budget = sizing_basis * cfg.max_risk_per_cycle_pct * decision.size_multiple
            if worst_case_per_lot <= 0:
                continue
            contracts = max(int(risk_budget // worst_case_per_lot), 0)
            contracts = min(contracts, cfg.max_lots_per_cycle)
            if contracts == 0:
                diary.append({
                    "date": str(snap.date.date()),
                    "event": "skip",
                    "reason": f"risk budget Rs{risk_budget:,.0f} < one lot worst-case Rs{worst_case_per_lot:,.0f}",
                })
                continue
            target_exit_pos = _trading_day_offset(df, exp_pos, cfg.exit_days_before_expiry)
            target_exit_date = df.index[target_exit_pos]
            open_pos = OpenPosition(
                entry_date=snap.date,
                expiry_date=df.index[exp_pos],
                target_exit_date=target_exit_date,
                strike=strike,
                sigma_at_entry=sigma_sell,
                premium_collected=premium,
                contracts=contracts,
                stop_loss_level=stop_level,
            )
            diary.append({
                "date": str(snap.date.date()),
                "event": "entry",
                "entry_date": str(snap.date.date()),
                "expiry_date": str(df.index[exp_pos].date()),
                "target_exit_date": str(target_exit_date.date()),
                "spot": round(snap.spot, 2),
                "strike": strike,
                "contracts_lots": contracts,
                "premium_per_share": round(premium, 2),
                "stop_loss_per_share": round(stop_level, 2),
                "vrp": round(decision.vrp, 4),
                "size_multiple": decision.size_multiple,
                "premium_collected_inr": round(premium * cfg.lot_size * contracts, 0),
                "capital_at_entry_inr": round(capital, 0),
            })

    # If we ended with an open position (rare at end of data), settle it.
    if open_pos is not None:
        snap = replay.session_at(len(df) - 1)
        t_remaining = max((open_pos.expiry_date - snap.date).days, 0) / DAYS_PER_YEAR
        mark = strangle_value(snap.spot, open_pos.strike, t_remaining,
                               cfg.risk_free_rate, snap.realised_vol_21d or open_pos.sigma_at_entry)
        pnl_per_share = open_pos.premium_collected - mark
        pnl_total = pnl_per_share * cfg.lot_size * open_pos.contracts
        capital += pnl_total
        closed.append(ClosedTrade(
            entry_date=open_pos.entry_date,
            exit_date=snap.date,
            expiry_date=open_pos.expiry_date,
            strike=open_pos.strike,
            contracts=open_pos.contracts,
            premium_collected=open_pos.premium_collected,
            exit_value=mark,
            pnl_per_share=pnl_per_share,
            pnl_total_inr=pnl_total,
            exit_reason="end-of-data settlement",
            vrp_at_entry=0.0,
            sigma_at_entry=open_pos.sigma_at_entry,
        ))

    return closed, diary
