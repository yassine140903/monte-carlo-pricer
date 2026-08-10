"""Merton jump-diffusion calibration.

Initial guesses come from a simple threshold rule (returns more than 3
diffusion-sigmas from the GBM drift are flagged as jumps); those guesses are
then refined by maximizing the exact mixture log-likelihood.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.stats import norm

from src.calibration.base import CalibratorBase
from src.calibration.gbm import GBMCalibrator, GBMParams
from src.calibration.utils import log_returns


class JumpDiffusionParams(BaseModel):
    mu: float
    sigma: float
    lambda_j: float
    mu_j: float
    sigma_j: float


def _neg_log_likelihood(x: np.ndarray, returns: np.ndarray, dt: float) -> float:
    mu, sigma, lambda_j, mu_j, sigma_j = x
    sigma = max(sigma, 1e-8)
    sigma_j = max(sigma_j, 1e-8)
    lambda_j = max(lambda_j, 0.0)

    p_jump = min(lambda_j * dt, 1.0)

    diffusion_pdf = norm.pdf(returns, loc=mu * dt, scale=sigma * np.sqrt(dt))
    jump_pdf = norm.pdf(
        returns, loc=mu * dt + mu_j, scale=np.sqrt(sigma**2 * dt + sigma_j**2)
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
        deviations = returns - mu_gbm * dt
        jump_returns = returns[np.abs(deviations) > threshold]

        n_years = len(returns) * dt
        if jump_returns.size > 0:
            lambda_j0 = jump_returns.size / n_years
            mu_j0 = float(np.mean(jump_returns))
            sigma_j0 = float(np.std(jump_returns, ddof=1)) if jump_returns.size > 1 else 0.01
        else:
            lambda_j0, mu_j0, sigma_j0 = 0.0, 0.0, 0.01

        x0 = [mu_gbm, sigma_gbm, lambda_j0, mu_j0, max(sigma_j0, 1e-4)]
        bounds = [
            (None, None),  # mu
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

        mu, sigma, lambda_j, mu_j, sigma_j = result.x
        # lambda_j converging to ~0 is a valid outcome (no detectable jumps).

        self.params = JumpDiffusionParams(
            mu=mu, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j
        )
        self.residuals = returns - np.mean(returns)
        self._is_calibrated = True
        return self.params
