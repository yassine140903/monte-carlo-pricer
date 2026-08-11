"""Pure-NumPy helpers shared by the simulators: step-count derivation, RNG
construction, and summary statistics over simulated paths. No pandas anywhere
in src/simulation/.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def resolve_n_steps(T: float, dt: float) -> int:
    """Number of time steps for a horizon ``T`` discretized at ``dt``.

    Rounded rather than truncated so that e.g. T=1, dt=1/252 gives exactly 252
    steps instead of 251 from floating-point error.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    n_steps = int(round(T / dt))
    if n_steps < 1:
        raise ValueError(f"T={T} and dt={dt} give fewer than one time step")
    return n_steps


def make_rng(seed: int | None = None) -> np.random.Generator:
    """An isolated Generator for a single simulate() call.

    Every simulation builds its own Generator from the caller's seed; the
    global np.random state is never touched, so concurrent simulations cannot
    perturb each other's draws.
    """
    return np.random.default_rng(seed)


def terminal_values(paths: np.ndarray) -> np.ndarray:
    """Terminal price of every path — the last column of ``paths``."""
    return np.asarray(paths)[:, -1]


def path_statistics(paths: np.ndarray) -> dict:
    """Summary statistics of a (n_simulations, n_steps+1) path array.

    Terminal-value moments and quantiles, plus the standard error of the mean
    terminal value — the quantity variance reduction is meant to shrink.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must be a 2-D (n_simulations, n_steps+1) array")

    terminal = terminal_values(paths)
    n_simulations = paths.shape[0]
    terminal_std = float(np.std(terminal, ddof=1)) if n_simulations > 1 else 0.0

    return {
        "n_simulations": n_simulations,
        "n_steps": paths.shape[1] - 1,
        "terminal_mean": float(np.mean(terminal)),
        "terminal_std": terminal_std,
        "terminal_stderr": terminal_std / np.sqrt(n_simulations),
        "terminal_min": float(np.min(terminal)),
        "terminal_max": float(np.max(terminal)),
        "terminal_p05": float(np.quantile(terminal, 0.05)),
        "terminal_median": float(np.quantile(terminal, 0.50)),
        "terminal_p95": float(np.quantile(terminal, 0.95)),
        "max_path_value": float(np.max(paths)),
        "min_path_value": float(np.min(paths)),
    }
