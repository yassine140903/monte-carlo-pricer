"""Geometric Brownian Motion calibration from log returns."""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from src.calibration.base import CalibratorBase
from src.calibration.utils import annualize_mean, annualize_volatility, log_returns


class GBMParams(BaseModel):
    """``model_type`` is a constant tag, not a fitted parameter: it lets the API
    layer treat the three params models as a discriminated union. It defaults,
    so nothing that constructed these params before needs to know about it.
    """

    model_type: Literal["gbm"] = "gbm"
    mu: float
    sigma: float


class GBMCalibrator(CalibratorBase):
    def calibrate(self, prices: np.ndarray, dt: float = 1 / 252) -> GBMParams:
        returns = log_returns(prices)

        sigma = annualize_volatility(float(np.std(returns, ddof=1)), dt)
        # Log returns have mean (mu - 0.5 * sigma**2) * dt, so add back the
        # Ito correction to recover the drift mu of the underlying GBM.
        mu = annualize_mean(float(np.mean(returns)), dt) + 0.5 * sigma**2

        self.params = GBMParams(mu=mu, sigma=sigma)
        self.residuals = returns - np.mean(returns)
        self._is_calibrated = True
        return self.params
