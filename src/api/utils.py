"""Helpers shared by the routers.

Three kinds of thing live here, none of them quantitative:

- turning engine output into something JSON-serializable and small enough to
  send — thinning path arrays, binning payoffs, re-keying dicts;
- reading a price history out of storage, which several endpoints need in the
  same shape;
- MLflow logging, deliberately best-effort: an unreachable tracking server
  degrades a response to ``mlflow_run_id=None`` rather than failing the
  request the user actually asked for.

The path budget is also enforced from here. It is an API policy — how much
compute one HTTP request may claim — not a modelling constraint, which is why
it gets its own exception type rather than reusing ValueError.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy import Engine

from src.config import settings
from src.data import storage
from src.data.models import StockData

logger = logging.getLogger(__name__)

MAX_SIMULATIONS = 100_000

DEFAULT_TARGET_POINTS = 50
DEFAULT_HISTOGRAM_BINS = 50


class SimulationTooLargeError(Exception):
    """Raised when a request asks for more paths than the API will run.

    Mapped to HTTP 413 by the handler in main.py — the request is well formed,
    it is simply too big, which is a different answer from "invalid".
    """

    def __init__(self, n_simulations: int, limit: int = MAX_SIMULATIONS) -> None:
        self.n_simulations = n_simulations
        self.limit = limit
        super().__init__(
            f"n_simulations={n_simulations} exceeds the API limit of {limit}"
        )


def enforce_simulation_budget(n_simulations: int, limit: int = MAX_SIMULATIONS) -> None:
    if n_simulations > limit:
        raise SimulationTooLargeError(n_simulations, limit)


MIN_CALIBRATION_OBSERVATIONS = 30


def load_recent_closes(
    ticker: str,
    lookback_days: int,
    engine: Engine,
    minimum: int = MIN_CALIBRATION_OBSERVATIONS,
) -> tuple[list[StockData], np.ndarray]:
    """The last ``lookback_days`` stored bars for ``ticker``, and their closes.

    ``lookback_days`` counts *stored trading days*, not calendar days, so the
    window is taken by slicing the tail of the series rather than by date
    arithmetic — no assumption about holidays is needed.

    Raises:
        KeyError: the ticker has no stored data at all.
        ValueError: it has some, but too little to fit anything to.
    """
    bars = storage.load_ohlcv(ticker, engine=engine)
    if not bars:
        raise KeyError(f"No stored data for ticker {ticker!r}")

    bars = bars[-lookback_days:]
    if len(bars) < minimum:
        raise ValueError(
            f"Ticker {ticker!r} has only {len(bars)} bars in the requested "
            f"lookback of {lookback_days}; at least {minimum} are needed"
        )

    return bars, np.array([bar.close for bar in bars], dtype=float)


def _downsample_indices(length: int, target_points: int) -> list[int]:
    """Indices of an evenly strided subsample that keeps both endpoints.

    Returned rather than applied so several arrays sharing a time axis — the
    percentile bands, the sample paths and the time axis itself — can be
    thinned to exactly the same dates.
    """
    if target_points < 2 or length <= target_points:
        return list(range(length))

    stride = max(length // target_points, 1)
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def downsample(array: np.ndarray, target_points: int = DEFAULT_TARGET_POINTS) -> np.ndarray:
    """Thin ``array`` along its time axis to roughly ``target_points`` samples.

    Takes every Nth element with N = length // target_points, and always keeps
    the first and last point so the horizon endpoints survive. A 1-D array is
    thinned directly; a 2-D (n_paths, n_steps+1) array is thinned column-wise,
    since time runs along the last axis in both cases.

    The result is *at most* target_points + 1 long, not exactly target_points —
    an exact count would need interpolation, which would invent prices that no
    simulated path ever visited.
    """
    arr = np.asarray(array)
    if arr.ndim not in (1, 2):
        raise ValueError("downsample expects a 1-D or 2-D array")

    axis = arr.ndim - 1
    indices = _downsample_indices(arr.shape[axis], target_points)
    return np.take(arr, indices, axis=axis)


def compute_histogram(
    values: np.ndarray, bins: int = DEFAULT_HISTOGRAM_BINS
) -> dict[str, list[float]]:
    """Bin ``values`` into plain Python lists.

    ``bin_edges`` has one more entry than ``counts`` — the standard NumPy
    convention, kept so the client can draw the bars without guessing widths.
    """
    counts, bin_edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return {
        "bin_edges": [float(edge) for edge in bin_edges],
        "counts": [int(count) for count in counts],
    }


def terminal_summary(paths: np.ndarray) -> dict[str, float]:
    """Mean, median, std, min and max of the terminal values of ``paths``."""
    terminal = np.asarray(paths, dtype=float)[:, -1]
    return {
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "std": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
        "min": float(np.min(terminal)),
        "max": float(np.max(terminal)),
    }


PERCENTILE_BANDS = {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}


def percentile_bands(
    paths: np.ndarray, target_points: int = DEFAULT_TARGET_POINTS
) -> dict[str, list[float]]:
    """Cross-sectional percentiles of ``paths`` at each step, then downsampled.

    Percentiles are taken across simulations *before* thinning, so every band
    reflects all the paths even though only ~``target_points`` dates are
    reported.
    """
    paths = np.asarray(paths, dtype=float)
    return {
        label: [float(v) for v in downsample(np.percentile(paths, q, axis=0), target_points)]
        for label, q in PERCENTILE_BANDS.items()
    }


def confidence_key(level: float) -> str:
    """The response key for a confidence level: 0.95 -> "95", 0.995 -> "99.5"."""
    return f"{level * 100:g}"


def log_to_mlflow(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> str | None:
    """Log one run and return its id, or None if MLflow is unavailable.

    Non-finite metrics are dropped rather than sent — MLflow rejects NaN and
    infinity, and one bad metric would otherwise cost the whole run.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and np.isfinite(v)
                }
            )
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 - logging must never break a request
        logger.warning("MLflow logging to %r failed: %s", experiment_name, exc)
        return None
