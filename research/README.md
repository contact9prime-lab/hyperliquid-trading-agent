# Nifty / GIFT Nifty AI Trading Research

Phase 1 paper-research artifact. No live capital, no broker auth.

## Goal

Quantify two edges that are repeatedly cited in the AI-Nifty trading literature
and figure out which (if any) are tractable for an LLM-driven agent:

1. **Overnight-gap edge** — GIFT Nifty trades while NSE is closed, and its
   premium/discount to the prior NSE close is the dominant pre-open signal in
   India. We use the prior-close-to-open gap as the empirical realisation of
   that signal and ask: how often does the gap follow through vs mean-revert
   intraday? Is there a bucket size where the signal is exploitable?

2. **Option-selling regime** — Most retail "AI" Nifty platforms are short-vol
   strategies (short straddles/strangles, iron condors). We simulate a naive
   weekly short-strangle from close-to-close moves to characterise the
   drawdown distribution and tail risk by volatility regime.

## Why no broker yet

GIFT Nifty intraday ticks need a paid feed (NSE-IX, TradingView Pro, or a
prime broker). Phase 1 uses **yfinance** for Nifty 50 (`^NSEI`) daily OHLC,
which is free and sufficient to characterise gap and vol regimes. Phase 2
plugs in Kite Connect / Kite MCP for live option chain + IV term structure.

## Layout

```
research/
  fetch_nifty_history.py       # pulls ^NSEI daily, caches to data/
  gap_analysis.py              # overnight gap follow-through vs reversion
  option_regime_analysis.py    # realized vol regimes, short-strangle drawdowns
  data/                        # cached parquet/csv (gitignored)
  output/                      # CSV summaries + PNG plots
  REPORT.md                    # generated findings + architecture proposal
```

## Run

```bash
cd research
python3 fetch_nifty_history.py
python3 gap_analysis.py
python3 option_regime_analysis.py
```

Each script writes its outputs into `output/` and appends a section to
`REPORT.md`.
