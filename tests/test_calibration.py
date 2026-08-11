"""Tests for src/calibration/. No DB or network access needed — everything
here runs against synthetic in-memory price paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.base import CalibratorBase
from src.calibration.gbm import GBMCalibrator, GBMParams
from src.calibration.heston import HestonCalibrator, feller_condition_satisfied
from src.calibration.jump_diffusion import JumpDiffusionCalibrator, JumpDiffusionParams
from src.calibration.utils import (
    TRADING_DAYS_PER_YEAR,
    annualize_mean,
    annualize_volatility,
    log_returns,
    rolling_volatility,
)
from src.simulation.jump_diffusion import JumpDiffusionSimulator

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

    def test_recovers_gbm_drift_when_no_jumps(self):
        """Both calibrators report mu in the price-level convention, so on
        jump-free data they must agree. Fitting the log-return drift and
        reporting it as mu would leave JD low by sigma**2/2 (~2%/yr here).
        """
        mu_true, sigma_true = 0.10, 0.20
        prices = _simulate_gbm_prices(mu_true, sigma_true, n_days=252 * 20, seed=21)

        gbm_params = GBMCalibrator().calibrate(prices, DT)
        jd_params = JumpDiffusionCalibrator().calibrate(prices, DT)

        assert jd_params.mu == pytest.approx(gbm_params.mu, rel=0.05)
        assert jd_params.mu == pytest.approx(mu_true, rel=0.25)
        # the gap must be far smaller than the correction itself
        assert abs(jd_params.mu - gbm_params.mu) < 0.25 * (0.5 * sigma_true**2)

    def test_mu_is_price_level_drift_not_log_drift(self):
        """Pins the convention directly against the sample.

        The likelihood fits the diffusion drift alpha in log space, and the
        mixture splits the observed log-return mean between that drift and the
        jumps: mean(returns)/dt == alpha + lambda_j * mu_j. Recovering alpha
        from the reported mu means undoing both the Ito correction and the
        jump compensator.
        """
        prices = _simulate_gbm_prices(0.10, 0.20, n_days=252 * 20, seed=22)
        params = JumpDiffusionCalibrator().calibrate(prices, DT)

        k = np.expm1(params.mu_j + 0.5 * params.sigma_j**2)
        alpha = params.mu - 0.5 * params.sigma**2 - params.lambda_j * k
        mu_log_sample = float(np.mean(log_returns(prices))) / DT

        assert alpha + params.lambda_j * params.mu_j == pytest.approx(
            mu_log_sample, abs=1e-3
        )

        # independent of the algebra above: mu is a price-level drift, so it
        # should track the mean *simple* return, not the mean log return
        simple_returns = np.diff(prices) / prices[:-1]
        assert params.mu == pytest.approx(float(np.mean(simple_returns)) / DT, abs=0.01)
        assert abs(params.mu - mu_log_sample) > 0.5 * (0.5 * params.sigma**2)

    def test_round_trip_recovers_mu_on_jumpy_data(self):
        """Simulate with known params, calibrate back, compare mu.

        Jump activity is deliberately heavy (lambda_j=2/yr, mu_j=-5%) so the
        compensator lambda_j*k is about -0.095 — roughly five times the Ito
        term. Dropping either correction moves the recovered mu far outside
        the tolerance below.

        A single 20y path pins the drift only to about sigma/sqrt(T) = 4.5%,
        so 20 independent paths are calibrated and averaged.
        """
        true_params = JumpDiffusionParams(
            mu=0.10, sigma=0.20, lambda_j=2.0, mu_j=-0.05, sigma_j=0.05
        )
        n_paths, years = 20, 20.0

        paths = JumpDiffusionSimulator().simulate(
            100.0, true_params, T=years, dt=DT, n_simulations=n_paths, seed=2024
        )
        recovered = [JumpDiffusionCalibrator().calibrate(path, DT) for path in paths]

        mus = np.array([p.mu for p in recovered])
        stderr = mus.std(ddof=1) / np.sqrt(n_paths)
        assert mus.mean() == pytest.approx(true_params.mu, abs=max(4 * stderr, 0.02))

        # the other params must come back too, or the mu agreement above could
        # be a cancellation of two offsetting errors
        assert np.mean([p.sigma for p in recovered]) == pytest.approx(
            true_params.sigma, rel=0.05
        )
        assert np.mean([p.lambda_j for p in recovered]) == pytest.approx(
            true_params.lambda_j, rel=0.25
        )
        assert np.mean([p.mu_j for p in recovered]) == pytest.approx(
            true_params.mu_j, rel=0.30
        )

    def test_round_trip_fails_without_the_jump_compensator(self):
        """The guard the previous test needs to be meaningful: strip the
        lambda_j*k term back off and mu must land visibly wrong, so the round
        trip cannot pass by accident under the old convention.
        """
        true_params = JumpDiffusionParams(
            mu=0.10, sigma=0.20, lambda_j=2.0, mu_j=-0.05, sigma_j=0.05
        )
        paths = JumpDiffusionSimulator().simulate(
            100.0, true_params, T=20.0, dt=DT, n_simulations=20, seed=2024
        )
        recovered = [JumpDiffusionCalibrator().calibrate(path, DT) for path in paths]

        without_compensator = np.array(
            [p.mu - p.lambda_j * np.expm1(p.mu_j + 0.5 * p.sigma_j**2) for p in recovered]
        )
        assert abs(without_compensator.mean() - true_params.mu) > 0.05

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
