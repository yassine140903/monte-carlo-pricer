"""Closed-form Black-Scholes prices and Greeks.

Standalone pure functions — nothing here touches the simulators. They serve
two purposes: a benchmark the Monte Carlo engine is checked against, and the
analytic Greeks the bump-and-revalue calculator is validated against.

Units follow the usual market conventions: vega is per 1.0 of volatility (so
divide by 100 for "per vol point"), theta is per year (divide by 252 for per
trading day), and rho is per 1.0 of rate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _validate(sigma: float, T: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """European call. At T <= 0 the option is expiring, so it is worth its
    intrinsic value.
    """
    _validate(sigma, T)
    if T == 0:
        return float(max(S - K, 0.0))

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    _validate(sigma, T)
    if T == 0:
        return float(max(K - S, 0.0))

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> dict:
    """Analytic delta, gamma, vega, theta and rho.

    ``option_type`` is "call" or "put"; gamma and vega are identical for both.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    _validate(sigma, T)

    if T == 0:
        # At expiry the value is a kink in S: delta is a step, everything else
        # has collapsed. Report the one-sided limits rather than dividing by 0.
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    discount_q = np.exp(-q * T)
    discount_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = discount_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * discount_q * pdf_d1 * sqrt_T
    time_decay = -S * discount_q * pdf_d1 * sigma / (2 * sqrt_T)

    if option_type == "call":
        delta = discount_q * norm.cdf(d1)
        theta = time_decay + q * S * discount_q * norm.cdf(d1) - r * K * discount_r * norm.cdf(d2)
        rho = K * T * discount_r * norm.cdf(d2)
    else:
        delta = -discount_q * norm.cdf(-d1)
        theta = (
            time_decay - q * S * discount_q * norm.cdf(-d1) + r * K * discount_r * norm.cdf(-d2)
        )
        rho = -K * T * discount_r * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
