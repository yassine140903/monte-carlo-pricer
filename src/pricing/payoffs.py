"""Option payoff functions and the registry the pricer looks them up in.

Every payoff takes the simulated paths — shape (n_simulations, n_steps+1),
column 0 being S0 — as its first argument and returns the undiscounted payoff
per path, shape (n_simulations,). Discounting is the pricer's job.

Path-dependent payoffs read ``paths[:, 1:]``: S0 is a known constant, not a
sampled observation, so including it would let the option's own starting
point knock a barrier out or drag a lookback extremum.

Barriers and lookbacks are monitored *discretely*, at the step dates only. A
continuously monitored contract is worth less (knock-out) or more (lookback),
since the extremum between step dates goes unseen; shrinking ``dt`` narrows
that gap.
"""
from __future__ import annotations

import numpy as np


def _terminal(paths: np.ndarray) -> np.ndarray:
    return paths[:, -1]


def _monitored(paths: np.ndarray) -> np.ndarray:
    """The observation window: every step date after t=0."""
    return paths[:, 1:]


def european_call(paths: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(_terminal(paths) - K, 0.0)


def european_put(paths: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - _terminal(paths), 0.0)


def asian_call(paths: np.ndarray, K: float) -> np.ndarray:
    """Arithmetic-average-price call on the post-t0 step dates."""
    return np.maximum(_monitored(paths).mean(axis=1) - K, 0.0)


def asian_put(paths: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - _monitored(paths).mean(axis=1), 0.0)


def barrier_ko_call(paths: np.ndarray, K: float, *, barrier: float) -> np.ndarray:
    """Up-and-out call: a European call that dies if the path ever touches
    ``barrier`` from below.
    """
    knocked_out = np.any(_monitored(paths) >= barrier, axis=1)
    return np.where(knocked_out, 0.0, european_call(paths, K))


def barrier_ko_put(paths: np.ndarray, K: float, *, barrier: float) -> np.ndarray:
    """Down-and-out put: a European put that dies if the path ever touches
    ``barrier`` from above.
    """
    knocked_out = np.any(_monitored(paths) <= barrier, axis=1)
    return np.where(knocked_out, 0.0, european_put(paths, K))


def lookback_call(paths: np.ndarray) -> np.ndarray:
    """Floating-strike lookback call: buy at the realized minimum. No strike."""
    return _terminal(paths) - _monitored(paths).min(axis=1)


def lookback_put(paths: np.ndarray) -> np.ndarray:
    """Floating-strike lookback put: sell at the realized maximum. No strike."""
    return _monitored(paths).max(axis=1) - _terminal(paths)


PAYOFF_REGISTRY = {
    "european_call": european_call,
    "european_put": european_put,
    "asian_call": asian_call,
    "asian_put": asian_put,
    "barrier_ko_call": barrier_ko_call,
    "barrier_ko_put": barrier_ko_put,
    "lookback_call": lookback_call,
    "lookback_put": lookback_put,
}
