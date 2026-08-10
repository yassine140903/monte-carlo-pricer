"""Tests for src/calibration/. No DB or network access needed — everything
here runs against synthetic in-memory price paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.base import CalibratorBase
from src.calibration.gbm import GBMCalibrator, GBMParams
from src.calibration.heston import HestonCalibrator, feller_condition_satisfied
from src.calibration.jump_diffusion import JumpDiffusionCalibrator
from src.calibration.utils import (
    TRADING_DAYS_PER_YEAR,
    annualize_mean,
    annualize_volatility,
    log_returns,
    rolling_volatility,
)

DT = 1 / 252


def _simulate_gbm_prices(mu: float, sigma: float, n_days: int, seed: int, s0: float = 100.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_days)
    log_rets = (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * z
    prices = s0 * np.exp(np.cumsum(log_rets))
    return np.insert(prices, 0, s0)


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------


class TestUtils:
    def test_log_returns_known_prices(self):
        prices = np.array([100.0, 105.0, 110.25])
        returns = log_returns(prices)
        expected = np.array([np.log(105.0 / 100.0), np.log(110.25 / 105.0)])
        np.testing.assert_allclose(returns, expected)

    def test_log_returns_drops_nan(self):
        prices = np.array([100.0, 105.0, np.nan, 110.0, 115.0])
        returns = log_returns(prices)
        # raw returns: [ln(105/100), nan, nan, ln(115/110)] -> nans dropped
        assert len(returns) == 2
        assert not np.any(np.isnan(returns))
        np.testing.assert_allclose(returns[0], np.log(105.0 / 100.0))
        np.testing.assert_allclose(returns[1], np.log(115.0 / 110.0))

    def test_annualize_mean(self):
        assert annualize_mean(0.0004, dt=1 / 252) == pytest.approx(0.0004 * 252)

    def test_annualize_volatility(self):
        assert annualize_volatility(0.01, dt=1 / 252) == pytest.approx(0.01 * np.sqrt(252))

    def test_trading_days_constant(self):
        assert TRADING_DAYS_PER_YEAR == 252

    def test_rolling_volatility_output_length(self):
        returns = np.arange(1, 21, dtype=float) / 1000
        window = 5
        vols = rolling_volatility(returns, window)
        assert len(vols) == len(returns) - window + 1

    def test_rolling_volatility_matches_manual_std(self):
        rng = np.random.default_rng(0)
        returns = rng.standard_normal(30) * 0.01
        window = 10
        vols = rolling_volatility(returns, window)

        first_window_std = np.std(returns[:window], ddof=1)
        expected_first = annualize_volatility(first_window_std)
        np.testing.assert_allclose(vols[0], expected_first)

    def test_rolling_volatility_window_too_large(self):
        returns = np.ones(5)
        assert len(rolling_volatility(returns, 10)) == 0


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------


class TestCalibratorBase:
    def test_goodness_of_fit_before_calibrate_raises(self):
        cal = GBMCalibrator()
        with pytest.raises(RuntimeError, match="Call calibrate\\(\\) first"):
            cal.goodness_of_fit()

    def test_goodness_of_fit_after_calibrate(self):
        prices = _simulate_gbm_prices(mu=0.1, sigma=0.25, n_days=252 * 5, seed=1)
        cal = GBMCalibrator()
        cal.calibrate(prices, DT)
        gof = cal.goodness_of_fit()
        assert set(gof) == {
            "log_likelihood",
            "aic",
            "bic",
            "jarque_bera_stat",
            "jarque_bera_pvalue",
        }
        assert np.isfinite(gof["log_likelihood"])
        assert np.isfinite(gof["aic"])
        assert np.isfinite(gof["bic"])

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            CalibratorBase()


# --------------------------------------------------------------------------
# GBM
# --------------------------------------------------------------------------


class TestGBMCalibrator:
    def test_recovers_known_parameters(self):
        mu_true, sigma_true = 0.10, 0.25
        prices = _simulate_gbm_prices(mu_true, sigma_true, n_days=252 * 40, seed=3)

        cal = GBMCalibrator()
        params = cal.calibrate(prices, DT)

        assert isinstance(params, GBMParams)
        assert params.mu == pytest.approx(mu_true, rel=0.15)
        assert params.sigma == pytest.approx(sigma_true, rel=0.15)

    def test_stores_state_on_self(self):
        prices = _simulate_gbm_prices(0.1, 0.25, n_days=252 * 2, seed=2)
        cal = GBMCalibrator()
        assert cal.params is None
        assert cal.residuals is None
        assert cal._is_calibrated is False

        cal.calibrate(prices, DT)
        assert cal.params is not None
        assert cal.residuals is not None
        assert cal._is_calibrated is True
        assert cal.residuals.shape == (len(prices) - 1,)


# --------------------------------------------------------------------------
# Jump diffusion
# --------------------------------------------------------------------------


class TestJumpDiffusionCalibrator:
    def test_detects_injected_jumps(self):
        mu_true, sigma_true = 0.08, 0.20
        n_days = 252 * 5
        rng = np.random.default_rng(7)
        z = rng.standard_normal(n_days)
        log_rets = (mu_true - 0.5 * sigma_true**2) * DT + sigma_true * np.sqrt(DT) * z

        n_jumps = 5
        jump_idx = rng.choice(n_days, size=n_jumps, replace=False)
        jump_sizes = rng.choice([-1, 1], size=n_jumps) * rng.uniform(0.08, 0.15, size=n_jumps)
        log_rets[jump_idx] += jump_sizes

        prices = 100.0 * np.exp(np.cumsum(log_rets))
        prices = np.insert(prices, 0, 100.0)

        cal = JumpDiffusionCalibrator()
        params = cal.calibrate(prices, DT)

        assert params.lambda_j > 0
        # roughly the right order of magnitude: 5 jumps over 5 years -> ~1/year
        assert 0.1 < params.lambda_j < 10
        assert params.sigma_j > 0

    def test_no_jumps_still_converges(self):
        prices = _simulate_gbm_prices(0.05, 0.2, n_days=252 * 5, seed=11)
        cal = JumpDiffusionCalibrator()
        params = cal.calibrate(prices, DT)
        assert params.lambda_j >= 0

    def test_stores_state_on_self(self):
        prices = _simulate_gbm_prices(0.05, 0.2, n_days=252 * 3, seed=12)
        cal = JumpDiffusionCalibrator()
        cal.calibrate(prices, DT)
        assert cal._is_calibrated is True
        assert cal.params is not None
        assert cal.residuals is not None


# --------------------------------------------------------------------------
# Heston
# --------------------------------------------------------------------------


class TestHestonCalibrator:
    def test_feller_condition_satisfied(self):
        assert feller_condition_satisfied(kappa=2.0, theta=0.04, xi=0.1) is True

    def test_feller_condition_violated(self):
        assert feller_condition_satisfied(kappa=0.5, theta=0.01, xi=1.0) is False

    def test_calibrate_sets_feller_flag_consistently(self):
        rng = np.random.default_rng(1)
        n_days = 252 * 3
        sigma_path = np.concatenate(
            [np.full(n_days // 2, 0.15), np.full(n_days - n_days // 2, 0.45)]
        )
        z = rng.standard_normal(n_days)
        log_rets = -0.5 * sigma_path**2 * DT + sigma_path * np.sqrt(DT) * z
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        prices = np.insert(prices, 0, 100.0)

        cal = HestonCalibrator()
        params = cal.calibrate(prices, DT)

        # the flag must agree with the actual Feller math whenever it claims
        # to be satisfied (the fallback path is allowed to be conservative
        # and report False even when the unconstrained fit happens to pass).
        actual = feller_condition_satisfied(params.kappa, params.theta, params.xi)
        if params.feller_satisfied:
            assert actual is True

    def test_stores_state_on_self(self):
        prices = _simulate_gbm_prices(0.05, 0.2, n_days=252 * 3, seed=13)
        cal = HestonCalibrator()
        cal.calibrate(prices, DT)
        assert cal._is_calibrated is True
        assert cal.params is not None
        assert cal.residuals is not None
