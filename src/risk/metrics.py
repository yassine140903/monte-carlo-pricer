"""Risk metrics over simulated paths.

Every function here is pure: NumPy arrays and scalars in, scalars or a
Pydantic bundle out. Nothing imports a simulator and nothing logs — where the
paths came from, and what is recorded about them, is the caller's business.

Sign convention: VaR and CVaR are returned as raw P&L quantiles, so a loss is
*negative*. Flipping to the positive "value at risk" figure a report usually
quotes is left to the presentation layer; keeping the sign as the arithmetic
produces it means the comparisons below (CVaR <= VaR, stressed <= baseline)
read the same way as the maths.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]


class RiskMetrics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    var: dict[float, float]
    cvar: dict[float, float]
    max_drawdown_mean: float
    max_drawdown_95th: float
    sharpe_ratio: float
    prob_of_loss: float
    n_simulations: int


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence}")


def compute_var(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk: the ``1 - confidence`` quantile of the P&L distribution.

    At 95% confidence this is the 5th percentile — the loss that only the
    worst 5% of simulations exceed.
    """
    _validate_confidence(confidence)
    return float(np.percentile(np.asarray(pnl, dtype=float), 100 * (1 - confidence)))


def compute_cvar(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR: the mean P&L of the paths at or below the VaR level.

    Where VaR only marks the edge of the tail, CVaR averages over it, so it
    sees how bad the bad cases actually get. It is always <= VaR.
    """
    pnl = np.asarray(pnl, dtype=float)
    var = compute_var(pnl, confidence)
    return float(np.mean(pnl[pnl <= var]))


def compute_max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Peak-to-trough drawdown of every path, as a fraction in [0, 1].

    Args:
        paths: Shape (n_simulations, n_steps+1).

    Returns:
        Shape (n_simulations,). A path that never falls below its running
        maximum scores 0.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must be a 2-D (n_simulations, n_steps+1) array")

    running_max = np.maximum.accumulate(paths, axis=1)
    drawdown = (running_max - paths) / running_max
    return np.max(drawdown, axis=1)


def compute_sharpe(pnl: np.ndarray, S0: float, r: float, T: float) -> float:
    """Sharpe ratio of the simulated returns over the horizon ``T``.

    The excess is measured against the risk-free return actually earned over
    the period, exp(r*T) - 1, not against r itself. The denominator is the
    total return volatility rather than the excess volatility, which is the
    standard convention and identical here anyway — the risk-free leg is a
    constant, so subtracting it cannot change the spread.
    """
    pnl = np.asarray(pnl, dtype=float)
    returns = pnl / S0
    rf_return = np.exp(r * T) - 1
    excess = returns - rf_return
    return float(np.mean(excess) / np.std(returns, ddof=1))


def compute_prob_of_loss(terminal_values: np.ndarray, S0: float) -> float:
    """Fraction of paths that finish below where they started."""
    return float(np.mean(np.asarray(terminal_values, dtype=float) < S0))


def compute_risk_metrics(
    paths: np.ndarray,
    S0: float,
    r: float,
    T: float,
    confidence_levels: list[float] | None = None,
) -> RiskMetrics:
    """Every metric in this module over one path array, as one bundle.

    Args:
        paths: Shape (n_simulations, n_steps+1) — prices, or portfolio values.
        S0: The starting level the P&L is measured from.
        r: Risk-free rate, for the Sharpe benchmark.
        T: Horizon in years.
        confidence_levels: VaR/CVaR levels; defaults to 95% and 99%.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must be a 2-D (n_simulations, n_steps+1) array")

    levels = (
        DEFAULT_CONFIDENCE_LEVELS if confidence_levels is None else confidence_levels
    )

    terminal = paths[:, -1]
    pnl = terminal - S0
    drawdowns = compute_max_drawdown(paths)

    return RiskMetrics(
        var={level: compute_var(pnl, level) for level in levels},
        cvar={level: compute_cvar(pnl, level) for level in levels},
        max_drawdown_mean=float(np.mean(drawdowns)),
        max_drawdown_95th=float(np.percentile(drawdowns, 95)),
        sharpe_ratio=compute_sharpe(pnl, S0, r, T),
        prob_of_loss=compute_prob_of_loss(terminal, S0),
        n_simulations=paths.shape[0],
    )
