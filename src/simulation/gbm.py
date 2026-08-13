"""Geometric Brownian Motion path simulation.

Uses the exact solution of the GBM SDE, not an Euler discretization: the log
increment over a step is normal with mean (mu - sigma**2/2)*dt and variance
sigma**2*dt, so the scheme is exact at the step dates for any dt.
"""
from __future__ import annotations

import numpy as np

from src.calibration.gbm import GBMParams
from src.simulation.base import SimulatorBase
from src.simulation.utils import make_rng, resolve_n_steps
from src.simulation.variance_reduction import generate_samples


class GBMSimulator(SimulatorBase):
    def simulate(
        self,
        S0: float,
        params: GBMParams,
        T: float,
        dt: float,
        n_simulations: int,
        seed: int | None = None,
        variance_reduction: str | None = None,
        mu: float | None = None,
        z_extern: np.ndarray | None = None,
    ) -> np.ndarray:
        """``mu`` overrides ``params.mu`` when given — pricing passes r - q to
        move from the calibrated real-world drift to the risk-neutral one.

        ``z_extern`` supplies the standard normals instead of drawing them,
        which leaves ``seed`` with nothing to do: the Brownian shocks are the
        whole of GBM's randomness.
        """
        self._validate_inputs(S0, n_simulations)
        n_steps = resolve_n_steps(T, dt)
        self._validate_z_extern(z_extern, variance_reduction, n_simulations, n_steps)

        if z_extern is not None:
            z = z_extern
        else:
            z = generate_samples(
                make_rng(seed), (n_simulations, n_steps), variance_reduction
            )

        effective_mu = mu if mu is not None else params.mu
        drift = (effective_mu - 0.5 * params.sigma**2) * dt
        diffusion = params.sigma * np.sqrt(dt) * z

        return self._paths_from_log_increments(S0, drift + diffusion)
