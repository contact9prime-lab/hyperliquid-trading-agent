# Nifty Short-Strangle Paper-Trading Agent

A paper-trading version of the Nifty volatility-risk-premium (VRP) agent.
**No real money. No broker connection.** Replays 18 years of Nifty data
week by week, exactly as the live agent would behave.

## What it does, in plain English

Every weekly cycle:

1. **Wednesday morning** — reads today's Nifty spot, computes trailing
   realised vol, looks up the "would-be" implied vol from a regime model,
   and computes the VRP (volatility risk premium). Then a fixed rule
   table decides: full size, half size, or skip this week.
2. **Wednesday close** — if a trade is on, sells an at-the-money weekly
   strangle (call + put at the same strike). Records the entry premium
   and a stop-loss level at 1.5× the premium collected.
3. **Every session afterwards** — marks the strangle to model price.
   If it crosses the stop-loss, **closes the position immediately**.
4. **Monday close** — exits the position (the day before Tuesday
   expiry) if the stop hasn't fired.

Every decision is logged to `output/diary.txt` in plain English. Every
closed trade is appended to `output/trades.csv`. The account equity
over time goes to `output/equity_curve.png`.

## Run it

```bash
# Make sure the Nifty CSV cache exists first
python3 research/fetch_nifty_history.py

# Now run the paper trader
python3 -m nifty_agent.paper_trader
```

Output lands in `nifty_agent/output/`:

| File | What it contains |
|---|---|
| `summary.json` | One-page numeric summary |
| `diary.txt` | Plain-English log of every Wednesday decision and every entry/exit |
| `diary.json` | Same diary, machine-readable |
| `trades.csv` | One row per closed trade |
| `equity_curve.png` | Account value over time |

## The rule table (in `config.py`)

Edit `config.py` to change the rules. No need to touch trading logic.

| Setting | Default | What it means |
|---|---|---|
| `starting_capital_inr` | 200,000 | Your account size |
| `max_risk_per_cycle_pct` | 5% | Max % of capital risked per week |
| `fixed_sizing` | True | True = no compounding (honest backtest). False = Kelly-style. |
| `max_lots_per_cycle` | 5 | Hard cap on lots regardless of capital |
| `vrp_full_size_threshold` | 15% | VRP ≥ this → full size |
| `vrp_half_size_threshold` | 5% | VRP ≥ this → half size; below → skip |
| `entry_days_before_expiry` | 4 | T-4 = previous Wednesday for Tue expiry |
| `exit_days_before_expiry` | 1 | T-1 = day before expiry |
| `stop_loss_multiple` | 1.5× | Force-close if option price > 1.5× premium |
| `min_rv_pct` | 6% | Skip when market is too quiet (premium too thin) |
| `max_rv_pct` | 35% | Skip when market is in vol-blowup mode |

## What the latest run shows

Over 17.5 years (May 2008 → Nov 2025):

- 235 trades placed (the agent SKIPPED most weeks, deliberately)
- 69% win rate
- ₹200,000 → ₹606,000 (3.0× over 17.5 years, ~6.5% CAGR with no compounding)
- Worst single week: -₹18,264 (bounded by the tripwire)
- Best single week: +₹13,689
- Tripwire fired on 9% of trades — i.e., the agent autonomously cut
  losers nine times out of every hundred trades
- During the 2008 GFC: SKIPPED every week Oct–Dec
- During the 2020 COVID crash: SKIPPED every week from Mar 13 onwards

Caveats — paper trading is not real trading:

- IV is **simulated** in paper mode using a regime model
  (`vrp_reader.assumed_vrp`). Real IV varies more wildly.
- No slippage, no bid-ask spread, no commissions modelled. Real Sharpe
  will be materially lower.
- The Tuesday-expiry era (Aug 2025+) only has ~5 months of data here.
- This is a **single-strategy** test. Real trading should combine
  multiple uncorrelated signals.

## Architecture

```
nifty_agent/
  config.py           # rules and risk limits (edit me, not the code)
  market_data.py      # HistoricalReplay (paper) + KiteLive stub
  option_pricing.py   # Black-Scholes pricer
  vrp_reader.py       # decides full/half/skip from today's market
  strategy.py         # weekly cycle + stop-loss tripwire
  paper_trader.py     # end-to-end runner (paper mode only)
  output/             # generated artefacts
```

The intentional split: every numeric rule lives in `config.py`. The
"intelligence" is just a couple of if-statements in `vrp_reader.py`.
There's **no LLM in the decision loop** — this is a rules-based agent.
That's the design: the LLM's job (in a future version) is to read
broader context (event calendar, macro headlines, FII flows) and ADJUST
the rules between cycles, not pick trades in real time.

## Going from paper to live

When you're ready (and only after watching paper-mode for several weeks):

1. Sign up for Kite Connect (₹500/month) and complete OAuth.
2. Implement `KiteLive` in `market_data.py` — replace the stub with
   real calls to `kite.quote()` for Nifty spot and option chain.
3. Replace `vrp_reader.assumed_vrp` with a real IV reader that pulls
   ATM IV from the option chain.
4. Add a Kite order placer (mirror of the existing
   `src/trading/hyperliquid_api.py` pattern from the parent repo).
5. **Run paper and live in parallel for a month**; only switch off
   paper when they agree.

The strategy code (`strategy.py`) does not change. It already speaks
to `market_data` and `option_pricing` through interfaces — that's why
the live swap is mostly a data-source replacement.
