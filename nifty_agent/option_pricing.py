"""Black-Scholes pricing for the strangle paper-trader.

All option valuations in the agent flow through here so we can swap to a
real option-chain reader (Kite quote/oi) in live mode without changing
strategy logic.
"""

from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(spot: float, strike: float, t_years: float, r: float, sigma: float) -> float:
    if t_years <= 0 or sigma <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)


def bs_put(spot: float, strike: float, t_years: float, r: float, sigma: float) -> float:
    if t_years <= 0 or sigma <= 0:
        return max(0.0, strike - spot)
    return bs_call(spot, strike, t_years, r, sigma) - spot + strike * math.exp(-r * t_years)


def strangle_value(spot: float, strike: float, t_years: float, r: float, sigma: float) -> float:
    """Combined value of a call + put at the same strike."""
    return bs_call(spot, strike, t_years, r, sigma) + bs_put(spot, strike, t_years, r, sigma)
