"""Merton jump-diffusion calibration.

Initial guesses come from a simple threshold rule (returns more than 3
diffusion-sigmas from the GBM drift are flagged as jumps); those guesses are
then refined by maximizing the exact mixture log-likelihood.

The likelihood is written in log-return space, so the drift it fits is that of
the diffusion component alone: alpha = mu - lambda_j * k - sigma**2 / 2, where
k = E[e^J - 1] is the expected proportional jump size. The reported ``mu`` adds
both terms back, so across this package ``mu`` always means the expected return
of the price process — the same convention GBMCalibrator reports and the
simulators in src/simulation/ expect.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.stats import norm

from src.calibration.base import CalibratorBase
from src.calibration.gbm import GBMCalibrator, GBMParams
from src.calibration.utils import log_returns


class JumpDiffusionParams(BaseModel):
    """Merton params. ``mu`` is the price-level drift (expected return of S),
    matching GBMParams — not the log-return drift.

    ``model_type`` is the discriminator tag described on GBMParams.
    """

    model_type: Literal["jump_diffusion"] = "jump_diffusion"
    mu: float
    sigma: float
    lambda_j: float
    mu_j: float
    sigma_j: float


def _neg_log_likelihood(x: np.ndarray, returns: np.ndarray, dt: float) -> float:
    """Negative log-likelihood of the jump/no-jump mixture over log returns.

    The first element of ``x`` is the *log-space* drift — log returns have mean
    mu_log * dt, not mu * dt. calibrate() converts it before reporting.
    """
    mu_log, sigma, lambda_j, mu_j, sigma_j = x
    sigma = max(sigma, 1e-8)
    sigma_j = max(sigma_j, 1e-8)
    lambda_j = max(lambda_j, 0.0)

    p_jump = min(lambda_j * dt, 1.0)

    diffusion_pdf = norm.pdf(returns, loc=mu_log * dt, scale=sigma * np.sqrt(dt))
    jump_pdf = norm.pdf(
        returns, loc=mu_log * dt + mu_j, scale=np.sqrt(sigma**2 * dt + sigma_j**2)
    )

    mixture_pdf = (1 - p_jump) * diffusion_pdf + p_jump * jump_pdf
    mixture_pdf = np.clip(mixture_pdf, 1e-300, None)
    return -float(np.sum(np.log(mixture_pdf)))


class JumpDiffusionCalibrator(CalibratorBase):
    def calibrate(
        self,
        prices: np.ndarray,
        dt: float = 1 / 252,
        gbm_params: GBMParams | None = None,
    ) -> JumpDiffusionParams:
        returns = log_returns(prices)

        if gbm_params is None:
            gbm_params = GBMCalibrator().calibrate(prices, dt)
        mu_gbm, sigma_gbm = gbm_params.mu, gbm_params.sigma

        threshold = 3 * sigma_gbm * np.sqrt(dt)
        # centre on the log-return mean, since ``returns`` are log returns
        deviations = returns - (mu_gbm - 0.5 * sigma_gbm**2) * dt
        jump_returns = returns[np.abs(deviations) > threshold]

        n_years = len(returns) * dt
        if jump_returns.size > 0:
            lambda_j0 = jump_returns.size / n_years
            mu_j0 = float(np.mean(jump_returns))
            sigma_j0 = float(np.std(jump_returns, ddof=1)) if jump_returns.size > 1 else 0.01
        else:
            lambda_j0, mu_j0, sigma_j0 = 0.0, 0.0, 0.01

        # x0[0] seeds the log-space drift the likelihood fits, so strip the
        # Ito correction GBMCalibrator added to mu_gbm.
        x0 = [mu_gbm - 0.5 * sigma_gbm**2, sigma_gbm, lambda_j0, mu_j0, max(sigma_j0, 1e-4)]
        bounds = [
            (None, None),  # mu_log
            (1e-6, None),  # sigma
            (0.0, None),  # lambda_j
            (None, None),  # mu_j
            (1e-6, None),  # sigma_j
        ]

        result = minimize(
            _neg_log_likelihood,
            x0=x0,
            args=(returns, dt),
            method="L-BFGS-B",
            bounds=bounds,
        )

        mu_log, sigma, lambda_j, mu_j, sigma_j = result.x
        # lambda_j converging to ~0 is a valid outcome (no detectable jumps).

        # The likelihood fits the drift of the *diffusion component* in log
        # space, alpha = mu - lambda_j * k - sigma**2 / 2, so both terms come
        # back to reach the price-level expected return that GBMCalibrator
        # reports and JumpDiffusionSimulator expects. Dropping the Ito term
        # understates mu by sigma**2/2; dropping the compensator understates
        # it by lambda_j * k, which is the larger error on jumpy data.
        k = np.expm1(mu_j + 0.5 * sigma_j**2)
        mu = mu_log + 0.5 * sigma**2 + lambda_j * k

        self.params = JumpDiffusionParams(
            mu=mu, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
        )
        self.residuals = returns - np.mean(returns)
        self._is_calibrated = True
        return self.params
