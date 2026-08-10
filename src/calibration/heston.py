"""Heston stochastic-volatility calibration by fitting the realized
volatility term structure against the model's expected-variance curve.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize

from src.calibration.base import CalibratorBase
from src.calibration.utils import log_returns, rolling_volatility

_HORIZONS_DAYS = (5, 21, 63, 126, 252)


class HestonParams(BaseModel):
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    feller_satisfied: bool


def feller_condition_satisfied(kappa: float, theta: float, xi: float) -> bool:
    """2 * kappa * theta > xi**2 — variance process stays strictly positive."""
    return bool(2 * kappa * theta > xi**2)


def _model_vols(params: np.ndarray, horizons_days: list[int], dt: float) -> np.ndarray:
    kappa, theta, xi, rho, v0 = params
    t = np.asarray(horizons_days, dtype=float) * dt
    variance = theta + (v0 - theta) * np.exp(-kappa * t)
    variance = np.clip(variance, 1e-12, None)
    return np.sqrt(variance)


def _loss(
    params: np.ndarray, horizons_days: list[int], dt: float, realized_vols: np.ndarray
) -> float:
    model_vols = _model_vols(params, horizons_days, dt)
    return float(np.sum((realized_vols - model_vols) ** 2))


class HestonCalibrator(CalibratorBase):
    def calibrate(self, prices: np.ndarray, dt: float = 1 / 252) -> HestonParams:
        returns = log_returns(prices)

        horizons_days = [h for h in _HORIZONS_DAYS if h <= len(returns)]
        if not horizons_days:
            raise ValueError(
                "Not enough return observations to build a realized volatility term structure"
            )

        realized_vols = np.array(
            [rolling_volatility(returns, h)[-1] for h in horizons_days]
        )

        v0_guess = float(realized_vols[0] ** 2)
        theta_guess = float(realized_vols[-1] ** 2)
        x0 = np.array([2.0, theta_guess, 0.3, -0.5, v0_guess])

        bounds = [
            (1e-4, 20.0),  # kappa
            (1e-6, 5.0),  # theta
            (1e-4, 5.0),  # xi
            (-0.999, 0.999),  # rho
            (1e-6, 5.0),  # v0
        ]

        unconstrained = minimize(
            _loss,
            x0=x0,
            args=(horizons_days, dt, realized_vols),
            method="L-BFGS-B",
            bounds=bounds,
        )

        constrained = minimize(
            _loss,
            x0=x0,
            args=(horizons_days, dt, realized_vols),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x: 2 * x[0] * x[1] - x[2] ** 2,
                }
            ],
        )

        unconstrained_loss = _loss(unconstrained.x, horizons_days, dt, realized_vols)
        constrained_loss = _loss(constrained.x, horizons_days, dt, realized_vols)

        if constrained_loss > 2 * unconstrained_loss:
            kappa, theta, xi, rho, v0 = unconstrained.x
            feller_satisfied = False
        else:
            kappa, theta, xi, rho, v0 = constrained.x
            feller_satisfied = feller_condition_satisfied(kappa, theta, xi)

        self.params = HestonParams(
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            feller_satisfied=feller_satisfied,
        )
        model_vols = _model_vols(np.array([kappa, theta, xi, rho, v0]), horizons_days, dt)
        self.residuals = realized_vols - model_vols
        self._is_calibrated = True
        return self.params
