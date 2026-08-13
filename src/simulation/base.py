"""Abstract base class for path simulators.

Simulators are stateless, the mirror image of the calibrators in
src/calibration/: ``simulate()`` returns the path array and stores nothing on
``self``, so one instance can be reused across tickers, horizons and threads.
Randomness is likewise per-call — each ``simulate()`` builds its own Generator
from the caller's seed and never touches the global np.random state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class SimulatorBase(ABC):
    @abstractmethod
    def simulate(
        self,
        S0: float,
        params: BaseModel,
        T: float,
        dt: float,
        n_simulations: int,
        seed: int | None = None,
        variance_reduction: str | None = None,
        z_extern: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate price paths under the model's dynamics.

        Args:
            S0: Spot price at t=0.
            params: The calibrated params for this process (GBMParams,
                JumpDiffusionParams or HestonParams).
            T: Horizon in years.
            dt: Time step in years (1/252 for daily steps).
            n_simulations: Number of paths.
            seed: Seed for this call's Generator; None draws from OS entropy.
            variance_reduction: "antithetic", "stratified" or None.
            z_extern: Externally supplied standard normals of shape
                (n_simulations, n_steps) driving the price process, in place
                of the ones this call would have drawn itself. This is how
                src/risk/portfolio.py injects Cholesky-correlated normals to
                simulate several assets under one correlation structure.
                Mutually exclusive with ``variance_reduction``, which is the
                other way of controlling those same draws.

        Subclasses also accept a ``mu`` keyword that overrides the drift
        carried in ``params`` (HestonParams has none, so there it is simply
        the drift, defaulting to 0). Pricing code passes r - q to switch from
        the calibrated real-world measure to the risk-neutral one. ``mu`` and
        ``z_extern`` are independent and may be combined.

        Returns:
            Array of shape (n_simulations, n_steps+1) where
            n_steps = round(T / dt). Column 0 is S0 for every path.
        """

    @staticmethod
    def _validate_inputs(S0: float, n_simulations: int) -> None:
        if S0 <= 0:
            raise ValueError("S0 must be positive")
        if n_simulations < 1:
            raise ValueError("n_simulations must be at least 1")

    @staticmethod
    def _validate_z_extern(
        z_extern: np.ndarray | None,
        variance_reduction: str | None,
        n_simulations: int,
        n_steps: int,
    ) -> None:
        """Reject externally supplied normals that clash with the call.

        Passing both ``z_extern`` and ``variance_reduction`` is a programming
        error rather than a preference to resolve: each is a complete recipe
        for the driving normals, so honouring one would silently discard the
        other.
        """
        if z_extern is None:
            return

        if variance_reduction is not None:
            raise ValueError(
                "Cannot use z_extern with variance_reduction — they both control "
                "the random normals"
            )

        expected = (n_simulations, n_steps)
        if z_extern.shape != expected:
            raise ValueError(
                f"z_extern must have shape {expected}, got {z_extern.shape}"
            )

    @staticmethod
    def _paths_from_log_increments(S0: float, log_increments: np.ndarray) -> np.ndarray:
        """Prepend the S0 column and exponentiate the cumulated log increments.

        Building paths in log space keeps prices strictly positive regardless
        of step size, which an Euler discretization of dS itself would not.
        """
        n_simulations = log_increments.shape[0]
        log_paths = np.concatenate(
            [np.zeros((n_simulations, 1)), np.cumsum(log_increments, axis=1)], axis=1
        )
        return S0 * np.exp(log_paths)
