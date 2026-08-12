"""Merton jump-diffusion path simulation.

The diffusion part is the exact GBM solution; on top of it each step draws a
Poisson jump count and, where at least one jump landed, adds a single
log-normal jump (the single-jump approximation). That approximation is only
accurate when at most one jump per step is likely, so ``lambda_j * dt`` is
validated against MAX_LAMBDA_DT.

The drift is compensated: subtracting ``lambda_j * k`` cancels the expected
contribution of the jumps, so E[S_T] = S0 * exp(mu * T) exactly as under GBM
and ``mu`` keeps its meaning as the total expected return.
"""
from __future__ import annotations

import numpy as np

from src.calibration.jump_diffusion import JumpDiffusionParams
from src.simulation.base import SimulatorBase
from src.simulation.utils import make_rng, resolve_n_steps
from src.simulation.variance_reduction import generate_samples

MAX_LAMBDA_DT = 0.1


class JumpDiffusionSimulator(SimulatorBase):
    def simulate(
        self,
        S0: float,
        params: JumpDiffusionParams,
        T: float,
        dt: float,
        n_simulations: int,
        seed: int | None = None,
        variance_reduction: str | None = None,
        mu: float | None = None,
    ) -> np.ndarray:
        """``mu`` overrides ``params.mu`` when given — pricing passes r - q,
        which the compensator below turns into the risk-neutral drift
        r - q - lambda_j * k for the diffusion part.
        """
        self._validate_inputs(S0, n_simulations)
        n_steps = resolve_n_steps(T, dt)
        self._validate_jump_intensity(params.lambda_j, dt)
        rng = make_rng(seed)

        shape = (n_simulations, n_steps)
        z = generate_samples(rng, shape, variance_reduction)

        # Expected proportional jump size E[e^J - 1] for J ~ N(mu_j, sigma_j**2).
        k = np.expm1(params.mu_j + 0.5 * params.sigma_j**2)

        effective_mu = mu if mu is not None else params.mu
        drift = (effective_mu - params.lambda_j * k - 0.5 * params.sigma**2) * dt
        diffusion = params.sigma * np.sqrt(dt) * z

        # Variance reduction deliberately does not extend to the jump draws:
        # antithetic pairing of a Poisson count is not well defined, and the
        # jump component is a small share of the total variance by construction.
        jump_occurred = rng.poisson(params.lambda_j * dt, size=shape) > 0
        jump_sizes = params.mu_j + params.sigma_j * rng.standard_normal(shape)
        jumps = np.where(jump_occurred, jump_sizes, 0.0)

        return self._paths_from_log_increments(S0, drift + diffusion + jumps)

    @staticmethod
    def _validate_jump_intensity(lambda_j: float, dt: float) -> None:
        if lambda_j < 0:
            raise ValueError("lambda_j must be non-negative")
        if lambda_j * dt >= MAX_LAMBDA_DT:
            raise ValueError(
                f"lambda_j * dt = {lambda_j * dt:.4f} exceeds {MAX_LAMBDA_DT}; the "
                "single-jump-per-step approximation is not valid at this step size. "
                "Use a smaller dt."
            )
