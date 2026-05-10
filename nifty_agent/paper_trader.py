"""Paper-trader runner.

End-to-end: load cached Nifty history, run the agent flow over every week,
write trade diary + summary + equity curve to nifty_agent/output/.

Usage:
    python3 -m nifty_agent.paper_trader

The runner is PAPER-ONLY by construction - it imports HistoricalReplay
and does not touch any broker. The KiteLive class exists only as a
placeholder that raises NotImplementedError.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT
from .market_data import HistoricalReplay
from .strategy import run_paper_trader

ROOT = Path(__file__).parent
RESEARCH_CSV = ROOT.parent / "research" / "data" / "nifty_daily.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def _equity_curve(starting: float, trades) -> pd.Series:
    if not trades:
        return pd.Series([starting])
    rows = [(t.exit_date, t.pnl_total_inr) for t in trades]
    s = pd.DataFrame(rows, columns=["date", "pnl"]).set_index("date").sort_index()
    s["capital"] = starting + s["pnl"].cumsum()
    return s["capital"]


def _summary(trades, starting: float) -> dict:
    if not trades:
        return {"n_trades": 0}
    pnl_series = pd.Series([t.pnl_total_inr for t in trades])
    win_rate = (pnl_series > 0).mean() * 100
    total_pnl = pnl_series.sum()
    end_capital = starting + total_pnl
    days = (trades[-1].exit_date - trades[0].entry_date).days or 1
    years = days / 365.25
    cagr = (end_capital / starting) ** (1 / years) - 1 if years > 0 else 0
    weekly = pnl_series / starting   # rough scale
    sharpe = (weekly.mean() / weekly.std() * math.sqrt(52)) if weekly.std() > 0 else 0
    tripwire_pct = sum(1 for t in trades if "TRIPWIRE" in t.exit_reason) / len(trades) * 100
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl_inr": round(total_pnl, 0),
        "starting_capital_inr": round(starting, 0),
        "ending_capital_inr": round(end_capital, 0),
        "return_pct": round((end_capital / starting - 1) * 100, 1),
        "years_simulated": round(years, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annualised_sharpe": round(sharpe, 2),
        "tripwire_firings_pct": round(tripwire_pct, 1),
        "worst_trade_inr": round(min(t.pnl_total_inr for t in trades), 0),
        "best_trade_inr": round(max(t.pnl_total_inr for t in trades), 0),
        "first_trade": str(trades[0].entry_date.date()),
        "last_trade": str(trades[-1].exit_date.date()),
    }


def _plot_equity(curve: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curve.index, curve.values, color="steelblue")
    ax.axhline(curve.iloc[0], color="black", lw=0.6)
    ax.set_title("Nifty short-strangle paper-trader - account equity")
    ax.set_ylabel("INR")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _human_diary(diary: list[dict]) -> str:
    """Produce a plain-English log of every decision."""
    lines = []
    for d in diary:
        date = d["date"]
        ev = d["event"]
        if ev == "decision":
            verdict = "TRADE" if d["trade"] else "SKIP"
            lines.append(
                f"{date}  [decision] {verdict:5}  spot={d['spot']}  "
                f"RV={d['rv21']:.1%}  IV={d['iv_assumed']:.1%}  "
                f"VRP={d['vrp']:+.1%}  size={d['size_multiple']}x  -- {d['reason']}"
            )
        elif ev == "entry":
            lines.append(
                f"{date}  [entry]    strike={d['strike']}  lots={d['contracts_lots']}  "
                f"premium=Rs{d['premium_per_share']:.1f}/sh  stop=Rs{d['stop_loss_per_share']:.1f}/sh  "
                f"collected Rs{d['premium_collected_inr']:,.0f}"
            )
        elif ev == "exit":
            lines.append(
                f"{date}  [exit]     strike={d['strike']}  "
                f"premium=Rs{d['premium_per_share']:.1f}  exit=Rs{d['exit_per_share']:.1f}  "
                f"PnL=Rs{d['pnl_total_inr']:,.0f}  capital=Rs{d['capital_after_inr']:,.0f}  -- {d['reason']}"
            )
        elif ev == "skip":
            lines.append(f"{date}  [skip]     {d['reason']}")
        elif ev == "panic-move-flag":
            lines.append(f"{date}  [flag]     {d['detail']}")
    return "\n".join(lines)


def main() -> None:
    if not RESEARCH_CSV.exists():
        raise SystemExit(
            f"Cache CSV missing: {RESEARCH_CSV}\n"
            f"Run `python3 research/fetch_nifty_history.py` first."
        )
    cfg = DEFAULT
    print(f"Paper-trader starting. Capital = Rs{cfg.starting_capital_inr:,.0f}")
    print(f"VRP rules: >={cfg.vrp_full_size_threshold:.0%} full, "
          f">={cfg.vrp_half_size_threshold:.0%} half, "
          f"<5% skip, <0 panic-skip")
    print(f"Cycle: enter T-{cfg.entry_days_before_expiry}, exit T-{cfg.exit_days_before_expiry}, "
          f"stop @ {cfg.stop_loss_multiple}x premium\n")

    replay = HistoricalReplay(RESEARCH_CSV)
    print(f"Loaded {replay.total_sessions:,} sessions  "
          f"{replay.df.index.min().date()} -> {replay.df.index.max().date()}\n")

    trades, diary = run_paper_trader(replay, cfg)
    summary = _summary(trades, cfg.starting_capital_inr)

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")

    # Persist artifacts
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "diary.json").write_text(json.dumps(diary, indent=2, default=str))
    (OUT / "diary.txt").write_text(_human_diary(diary))
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(OUT / "trades.csv", index=False)
    curve = _equity_curve(cfg.starting_capital_inr, trades)
    curve.to_csv(OUT / "equity_curve.csv")
    _plot_equity(curve, OUT / "equity_curve.png")

    print(f"\nArtifacts written to {OUT}/")
    print("  - summary.json     - one-page numeric summary")
    print("  - diary.txt        - plain-English decision log (every Wed)")
    print("  - diary.json       - same diary, machine-readable")
    print("  - trades.csv       - one row per closed trade")
    print("  - equity_curve.png - account value over time")


if __name__ == "__main__":
    main()
