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

        Subclasses also accept a ``mu`` keyword that overrides the drift
        carried in ``params`` (HestonParams has none, so there it is simply
        the drift, defaulting to 0). Pricing code passes r - q to switch from
        the calibrated real-world measure to the risk-neutral one.

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
