"""Options pricing: Monte Carlo pricing over any simulator, payoff functions,
closed-form Black-Scholes benchmarks, and bump-and-revalue Greeks. Pure
NumPy/SciPy — no pandas, no MLflow.
"""
from __future__ import annotations

from src.pricing.black_scholes import bs_call, bs_greeks, bs_put
from src.pricing.greeks import GreeksCalculator, GreeksResult
from src.pricing.payoffs import (
    PAYOFF_REGISTRY,
    asian_call,
    asian_put,
    barrier_ko_call,
    barrier_ko_put,
    european_call,
    european_put,
    lookback_call,
    lookback_put,
)
from src.pricing.pricer import MCPricer, PricingResult

__all__ = [
    "MCPricer",
    "PricingResult",
    "GreeksCalculator",
    "GreeksResult",
    "PAYOFF_REGISTRY",
    "european_call",
    "european_put",
    "asian_call",
    "asian_put",
    "barrier_ko_call",
    "barrier_ko_put",
    "lookback_call",
    "lookback_put",
    "bs_call",
    "bs_put",
    "bs_greeks",
]
