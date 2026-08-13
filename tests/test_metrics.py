"""Tests for src/risk/metrics.py.

The metrics are pure functions, so most tests here are exact: hand-built
arrays with a drawdown or a quantile that can be worked out by hand. The two
statistical tests pin a seed and check against a closed form rather than a
recorded number.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.calibration.gbm import GBMParams
from src.risk.metrics import (
    RiskMetrics,
    compute_cvar,
    compute_max_drawdown,
    compute_prob_of_loss,
    compute_risk_metrics,
    compute_sharpe,
    compute_var,
)
from src.simulation.gbm import GBMSimulator

DT = 1 / 252
KNOWN_PNL = np.array([-10.0, -5.0, -2.0, 0.0, 1.0, 3.0, 5.0, 8.0, 10.0, 15.0])


def gbm_paths(mu: float = 0.08, sigma: float = 0.20, n_simulations: int = 20_000):
    return GBMSimulator().simulate(
        100.0, GBMParams(mu=mu, sigma=sigma), 1.0, DT, n_simulations, seed=42
    )


# --------------------------------------------------------------------------
# VaR / CVaR
# --------------------------------------------------------------------------


class TestVaR:
    def test_matches_the_corresponding_percentile(self):
        assert compute_var(KNOWN_PNL, confidence=0.90) == pytest.approx(
            np.percentile(KNOWN_PNL, 10)
        )

    def test_default_confidence_is_the_fifth_percentile(self):
        assert compute_var(KNOWN_PNL) == pytest.approx(np.percentile(KNOWN_PNL, 5))

    def test_higher_confidence_means_deeper_into_the_tail(self):
        pnl = np.random.default_rng(0).normal(0, 10, 10_000)
        assert compute_var(pnl, 0.99) < compute_var(pnl, 0.95) < compute_var(pnl, 0.90)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_confidence_outside_the_open_unit_interval(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            compute_var(KNOWN_PNL, confidence)


class TestCVaR:
    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_never_exceeds_var(self, confidence):
        pnl = np.random.default_rng(42).normal(0, 10, 10_000)

        assert compute_cvar(pnl, confidence) <= compute_var(pnl, confidence)

    def test_matches_the_closed_form_for_a_normal_distribution(self):
        """For N(0,1), CVaR at level a is -phi(Phi^-1(1-a)) / (1-a) ~ -2.0627."""
        confidence = 0.95
        tail = 1 - confidence
        analytical = -norm.pdf(norm.ppf(tail)) / tail

        samples = np.random.default_rng(7).standard_normal(1_000_000)

        assert compute_cvar(samples, confidence) == pytest.approx(analytical, abs=0.05)
        assert analytical == pytest.approx(-2.0627, abs=1e-3)

    def test_averages_the_tail_rather_than_marking_its_edge(self):
        # The worst two of ten sit at or below the 10th percentile.
        var = compute_var(KNOWN_PNL, 0.90)
        expected = np.mean(KNOWN_PNL[KNOWN_PNL <= var])

        assert compute_cvar(KNOWN_PNL, 0.90) == pytest.approx(expected)

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_rejects_confidence_outside_the_open_unit_interval(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            compute_cvar(KNOWN_PNL, confidence)


# --------------------------------------------------------------------------
# Drawdown
# --------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_monotonically_increasing_path_never_draws_down(self):
        paths = np.array([[100.0, 110.0, 120.0, 130.0]])

        assert compute_max_drawdown(paths)[0] == pytest.approx(0.0, abs=1e-12)

    def test_known_peak_to_trough(self):
        # Peak 120, trough 80 -> (120 - 80) / 120 = 1/3. The later recovery to
        # 110 does not reduce a drawdown already taken.
        paths = np.array([[100.0, 120.0, 80.0, 110.0]])

        assert compute_max_drawdown(paths)[0] == pytest.approx(1 / 3, abs=1e-10)

    def test_measured_from_the_running_peak_not_the_start(self):
        # Rises to 200, falls to 100: the drawdown is 50%, not 0% just because
        # the path finished where it began.
        paths = np.array([[100.0, 200.0, 100.0]])

        assert compute_max_drawdown(paths)[0] == pytest.approx(0.5)

    def test_returns_one_value_per_path(self):
        paths = gbm_paths(n_simulations=64)
        drawdowns = compute_max_drawdown(paths)

        assert drawdowns.shape == (64,)

    def test_always_a_fraction_between_zero_and_one(self):
        drawdowns = compute_max_drawdown(gbm_paths(n_simulations=2_000))

        assert np.all(drawdowns >= 0.0)
        assert np.all(drawdowns <= 1.0)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_max_drawdown(np.array([100.0, 120.0, 80.0]))


# --------------------------------------------------------------------------
# Sharpe and probability of loss
# --------------------------------------------------------------------------


class TestSharpe:
    def test_positive_for_strong_drift_and_low_vol(self):
        paths = gbm_paths(mu=0.30, sigma=0.10)
        pnl = paths[:, -1] - 100.0

        assert compute_sharpe(pnl, S0=100.0, r=0.02, T=1.0) > 0

    def test_negative_when_the_drift_lags_the_risk_free_rate(self):
        paths = gbm_paths(mu=-0.10, sigma=0.10)
        pnl = paths[:, -1] - 100.0

        assert compute_sharpe(pnl, S0=100.0, r=0.05, T=1.0) < 0

    def test_falls_as_volatility_rises_for_the_same_drift(self):
        calm = gbm_paths(mu=0.20, sigma=0.10)[:, -1] - 100.0
        wild = gbm_paths(mu=0.20, sigma=0.40)[:, -1] - 100.0

        calm_sharpe = compute_sharpe(calm, S0=100.0, r=0.02, T=1.0)
        wild_sharpe = compute_sharpe(wild, S0=100.0, r=0.02, T=1.0)

        assert calm_sharpe > wild_sharpe

    def test_benchmark_is_the_compounded_rate_over_the_horizon(self):
        """A portfolio returning exactly exp(r*T) - 1 on every path has zero
        excess; a fixed spread above it scales as that spread."""
        S0, r, T = 100.0, 0.05, 2.0
        risk_free_pnl = np.full(1_000, S0 * (np.exp(r * T) - 1))
        pnl = risk_free_pnl + np.linspace(-10.0, 10.0, 1_000)

        excess = np.mean(pnl / S0) - (np.exp(r * T) - 1)
        expected = excess / np.std(pnl / S0, ddof=1)

        assert compute_sharpe(pnl, S0, r, T) == pytest.approx(expected)


class TestProbOfLoss:
    def test_half_the_paths_below_start(self):
        terminal = np.array([90.0, 95.0, 105.0, 110.0])

        assert compute_prob_of_loss(terminal, S0=100.0) == pytest.approx(0.5)

    def test_all_paths_above_start(self):
        terminal = np.array([101.0, 105.0, 110.0])

        assert compute_prob_of_loss(terminal, S0=100.0) == 0.0

    def test_all_paths_below_start(self):
        terminal = np.array([99.0, 95.0, 90.0])

        assert compute_prob_of_loss(terminal, S0=100.0) == 1.0

    def test_a_path_exactly_at_the_start_is_not_a_loss(self):
        assert compute_prob_of_loss(np.array([100.0, 100.0]), S0=100.0) == 0.0


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------


class TestComputeRiskMetrics:
    @pytest.fixture(scope="class")
    def metrics(self):
        return compute_risk_metrics(gbm_paths(), S0=100.0, r=0.02, T=1.0)

    def test_returns_the_bundle(self, metrics):
        assert isinstance(metrics, RiskMetrics)

    def test_default_confidence_levels(self, metrics):
        assert set(metrics.var) == {0.95, 0.99}
        assert set(metrics.cvar) == {0.95, 0.99}

    def test_cvar_never_exceeds_var_at_any_level(self, metrics):
        for level in metrics.var:
            assert metrics.cvar[level] <= metrics.var[level]

    def test_higher_confidence_is_the_worse_number(self, metrics):
        assert metrics.var[0.99] < metrics.var[0.95]
        assert metrics.cvar[0.99] < metrics.cvar[0.95]

    def test_probabilities_and_drawdowns_are_fractions(self, metrics):
        assert 0.0 <= metrics.prob_of_loss <= 1.0
        assert 0.0 <= metrics.max_drawdown_mean <= 1.0
        assert 0.0 <= metrics.max_drawdown_95th <= 1.0

    def test_the_95th_percentile_drawdown_is_the_worse_one(self, metrics):
        assert metrics.max_drawdown_95th >= metrics.max_drawdown_mean

    def test_reports_the_simulation_count(self, metrics):
        assert metrics.n_simulations == 20_000

    def test_custom_confidence_levels(self):
        metrics = compute_risk_metrics(
            gbm_paths(n_simulations=2_000),
            S0=100.0,
            r=0.02,
            T=1.0,
            confidence_levels=[0.80, 0.90, 0.999],
        )

        assert set(metrics.var) == {0.80, 0.90, 0.999}

    def test_agrees_with_the_individual_functions(self):
        paths = gbm_paths(n_simulations=2_000)
        metrics = compute_risk_metrics(paths, S0=100.0, r=0.02, T=1.0)
        pnl = paths[:, -1] - 100.0

        assert metrics.var[0.95] == pytest.approx(compute_var(pnl, 0.95))
        assert metrics.cvar[0.95] == pytest.approx(compute_cvar(pnl, 0.95))
        assert metrics.sharpe_ratio == pytest.approx(
            compute_sharpe(pnl, 100.0, 0.02, 1.0)
        )
        assert metrics.prob_of_loss == pytest.approx(
            compute_prob_of_loss(paths[:, -1], 100.0)
        )

    def test_more_volatile_paths_produce_worse_risk(self):
        calm = compute_risk_metrics(gbm_paths(sigma=0.10), S0=100.0, r=0.02, T=1.0)
        wild = compute_risk_metrics(gbm_paths(sigma=0.50), S0=100.0, r=0.02, T=1.0)

        assert wild.cvar[0.95] < calm.cvar[0.95]
        assert wild.max_drawdown_mean > calm.max_drawdown_mean
        assert wild.prob_of_loss > calm.prob_of_loss

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_risk_metrics(np.array([100.0, 110.0]), S0=100.0, r=0.02, T=1.0)
