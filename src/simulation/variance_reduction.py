"""Variance reduction for the normal draws that feed the simulators.

Every simulator asks for its standard normals through :func:`generate_samples`
rather than calling ``rng.standard_normal`` directly, so a single
``variance_reduction`` argument switches the sampling scheme without touching
any of the path-evolution code.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

VALID_METHODS = (None, "antithetic", "stratified")


def generate_samples(
    rng: np.random.Generator,
    shape: tuple[int, int],
    method: str | None = None,
) -> np.ndarray:
    """Standard normal draws of ``shape`` = (n_simulations, n_columns).

    ``method`` is one of:

    - ``None`` — plain i.i.d. draws.
    - ``"antithetic"`` — the first half of the simulations is drawn, the second
      half is its negation. Each pair straddles the mean, so path functionals
      that are monotone in the driving noise get a negatively correlated
      partner and the estimator variance drops. With an odd number of
      simulations the final row is left unpaired.
    - ``"stratified"`` — each column is stratified into ``n_simulations`` equal
      probability bins, one draw per bin, which forces the marginal of every
      step to cover its distribution evenly. Strata are shuffled independently
      per column so the columns stay independent of each other.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown variance_reduction {method!r}; expected one of {VALID_METHODS}"
        )

    n_rows, n_cols = shape
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"shape must be positive in both dimensions, got {shape}")

    if method is None:
        return rng.standard_normal(shape)

    if method == "antithetic":
        n_base = (n_rows + 1) // 2
        base = rng.standard_normal((n_base, n_cols))
        return np.concatenate([base, -base], axis=0)[:n_rows]

    return _stratified_normals(rng, n_rows, n_cols)


def _stratified_normals(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray:
    """One uniform per equal-probability stratum per column, mapped through
    the inverse normal CDF.
    """
    strata = np.tile(np.arange(n_rows)[:, None], (1, n_cols))
    # Independent shuffle per column: without it every simulation would sit in
    # the same stratum at every step, which correlates the columns.
    strata = rng.permuted(strata, axis=0)

    uniforms = (strata + rng.random((n_rows, n_cols))) / n_rows
    return norm.ppf(uniforms)
