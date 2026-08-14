"""Abstract base class for stochastic-process calibrators.

Calibrators are stateful: ``calibrate()`` stores the fitted params and the
residuals used to derive them on ``self``, so ``goodness_of_fit()`` can be
computed afterwards without re-running the fit. Calibrators never log to
MLflow themselves — that is the caller's responsibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel
from scipy import stats


class CalibratorBase(ABC):
    def __init__(self) -> None:
        self.params: BaseModel | None = None
        self.residuals: np.ndarray | None = None
        self._is_calibrated: bool = False

    @abstractmethod
    def calibrate(self, prices: np.ndarray, dt: float = 1 / 252) -> BaseModel:
        """Fit the model to ``prices`` and return the resulting params.

        Subclasses must store the params and residuals on ``self`` and set
        ``self._is_calibrated = True`` before returning.
        """

    def goodness_of_fit(self) -> dict:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() first")

        residuals = self.residuals
        mean = np.mean(residuals)
        std = np.std(residuals, ddof=1)
        log_likelihood = float(np.sum(stats.norm.logpdf(residuals, loc=mean, scale=std)))

        # ``model_type`` is a constant discriminator tag the API layer relies
        # on, not something the fit estimated, so it must not add to the AIC/BIC
        # parameter penalty.
        k = sum(1 for name in type(self.params).model_fields if name != "model_type")
        n = len(residuals)

        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        jb_stat, jb_pvalue = stats.jarque_bera(residuals)

        return {
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
        }
