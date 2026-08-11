"""Tests for src/simulation/. No DB or network access needed — everything
here runs against synthetic in-memory paths.

Several tests are statistical. They all pin a seed, and their tolerances are
stated in standard errors rather than absolute numbers so they stay meaningful
rather than merely passing.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.calibration.gbm import GBMParams
from src.calibration.heston import HestonParams
from src.calibration.jump_diffusion import JumpDiffusionParams
from src.simulation.base import SimulatorBase
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import PSI_CRIT, HestonSimulator
from src.simulation.jump_diffusion import MAX_LAMBDA_DT, JumpDiffusionSimulator
from src.simulation.utils import path_statistics, resolve_n_steps, terminal_values
from src.simulation.variance_reduction import generate_samples

DT = 1 / 252

GBM_PARAMS = GBMParams(mu=0.08, sigma=0.20)
JD_PARAMS = JumpDiffusionParams(mu=0.08, sigma=0.20, lambda_j=1.0, mu_j=-0.03, sigma_j=0.08)
HESTON_PARAMS = HestonParams(
    kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04, feller_satisfied=True
)
# 2*kappa*theta = 0.08 < xi**2 = 1.0 — the variance process hits zero often,
# which is exactly where a naive Euler scheme would go negative.
HESTON_PARAMS_FELLER_VIOLATED = HestonParams(
    kappa=1.0, theta=0.04, xi=1.0, rho=-0.7, v0=0.04, feller_satisfied=False
)


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------


class TestUtils:
    def test_resolve_n_steps_daily(self):
        assert resolve_n_steps(T=1.0, dt=DT) == 252

    def test_resolve_n_steps_rounds_not_truncates(self):
        # 0.5 / (1/252) = 125.99999999999999 in floating point
        assert resolve_n_steps(T=0.5, dt=DT) == 126

    @pytest.mark.parametrize("T, dt", [(0.0, DT), (-1.0, DT), (1.0, 0.0), (1.0, -DT)])
    def test_resolve_n_steps_rejects_non_positive(self, T, dt):
        with pytest.raises(ValueError):
            resolve_n_steps(T, dt)

    def test_resolve_n_steps_rejects_sub_step_horizon(self):
        with pytest.raises(ValueError, match="fewer than one time step"):
            resolve_n_steps(T=0.1, dt=1.0)

    def test_terminal_values_is_last_column(self):
        paths = np.arange(12, dtype=float).reshape(3, 4)
        np.testing.assert_array_equal(terminal_values(paths), [3.0, 7.0, 11.0])

    def test_path_statistics(self):
        paths = GBMSimulator().simulate(100.0, GBM_PARAMS, 1.0, 1 / 12, 5_000, seed=1)
        stats_dict = path_statistics(paths)

        assert stats_dict["n_simulations"] == 5_000
        assert stats_dict["n_steps"] == 12
        assert stats_dict["terminal_p05"] < stats_dict["terminal_median"]
        assert stats_dict["terminal_median"] < stats_dict["terminal_p95"]
        assert stats_dict["min_path_value"] > 0
        expected_stderr = stats_dict["terminal_std"] / np.sqrt(5_000)
        assert stats_dict["terminal_stderr"] == pytest.approx(expected_stderr)

    def test_path_statistics_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            path_statistics(np.arange(5, dtype=float))


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------


class TestSimulatorBase:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            SimulatorBase()

    @pytest.mark.parametrize(
        "simulator, params",
        [
            (GBMSimulator(), GBM_PARAMS),
            (JumpDiffusionSimulator(), JD_PARAMS),
            (HestonSimulator(), HESTON_PARAMS),
        ],
    )
    def test_shape_and_first_column(self, simulator, params):
        paths = simulator.simulate(100.0, params, T=1.0, dt=DT, n_simulations=50, seed=0)
        assert paths.shape == (50, 253)
        np.testing.assert_array_equal(paths[:, 0], 100.0)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)

    @pytest.mark.parametrize(
        "simulator, params",
        [
            (GBMSimulator(), GBM_PARAMS),
            (JumpDiffusionSimulator(), JD_PARAMS),
            (HestonSimulator(), HESTON_PARAMS),
        ],
    )
    def test_stores_nothing_on_self(self, simulator, params):
        before = set(vars(simulator))
        simulator.simulate(100.0, params, T=1.0, dt=DT, n_simulations=10, seed=0)
        assert set(vars(simulator)) == before

    @pytest.mark.parametrize(
        "simulator", [GBMSimulator(), JumpDiffusionSimulator(), HestonSimulator()]
    )
    def test_rejects_bad_inputs(self, simulator):
        params = {
            GBMSimulator: GBM_PARAMS,
            JumpDiffusionSimulator: JD_PARAMS,
            HestonSimulator: HESTON_PARAMS,
        }[type(simulator)]
        with pytest.raises(ValueError, match="S0 must be positive"):
            simulator.simulate(0.0, params, T=1.0, dt=DT, n_simulations=10)
        with pytest.raises(ValueError, match="n_simulations must be at least 1"):
            simulator.simulate(100.0, params, T=1.0, dt=DT, n_simulations=0)

    def test_does_not_touch_global_random_state(self):
        """Even an unseeded call must leave np.random alone, so a simulation
        can never perturb another caller's draws.
        """
        np.random.seed(1234)
        before = np.random.get_state()[1].copy()
        GBMSimulator().simulate(100.0, GBM_PARAMS, 1.0, DT, 100, seed=None)
        np.testing.assert_array_equal(np.random.get_state()[1], before)


# --------------------------------------------------------------------------
# variance reduction primitives
# --------------------------------------------------------------------------


class TestGenerateSamples:
    @pytest.mark.parametrize("method", [None, "antithetic", "stratified"])
    def test_shape(self, method):
        rng = np.random.default_rng(0)
        assert generate_samples(rng, (17, 5), method).shape == (17, 5)

    def test_unknown_method_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unknown variance_reduction"):
            generate_samples(rng, (10, 2), "control_variate")

    def test_antithetic_pairs_are_negations(self):
        rng = np.random.default_rng(0)
        samples = generate_samples(rng, (8, 3), "antithetic")
        np.testing.assert_allclose(samples[:4], -samples[4:])
        assert np.allclose(samples.sum(axis=0), 0.0)

    def test_antithetic_odd_count_leaves_last_row_unpaired(self):
        rng = np.random.default_rng(0)
        samples = generate_samples(rng, (7, 3), "antithetic")
        assert samples.shape == (7, 3)
        np.testing.assert_allclose(samples[:3], -samples[4:7])

    def test_stratified_covers_every_stratum_once(self):
        n_rows = 500
        rng = np.random.default_rng(0)
        samples = generate_samples(rng, (n_rows, 4), "stratified")

        # each column must place exactly one draw in each equal-probability bin
        for column in samples.T:
            bins = np.floor(stats.norm.cdf(column) * n_rows).astype(int)
            assert len(np.unique(bins)) == n_rows

    def test_stratified_columns_are_not_correlated(self):
        rng = np.random.default_rng(0)
        samples = generate_samples(rng, (2_000, 2), "stratified")
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert abs(corr) < 0.06


# --------------------------------------------------------------------------
# GBM
# --------------------------------------------------------------------------


class TestGBMSimulator:
    def test_mean_path_converges_to_analytic(self):
        S0, T, dt, n_sims = 100.0, 1.0, 1 / 12, 100_000
        paths = GBMSimulator().simulate(S0, GBM_PARAMS, T, dt, n_sims, seed=42)

        times = np.arange(paths.shape[1]) * dt
        expected = S0 * np.exp(GBM_PARAMS.mu * times)
        realized = paths.mean(axis=0)
        stderr = paths.std(axis=0, ddof=1) / np.sqrt(n_sims)

        # column 0 is exactly S0 (zero stderr); check the rest within 4 se
        assert realized[0] == S0
        z_scores = np.abs(realized[1:] - expected[1:]) / stderr[1:]
        assert z_scores.max() < 4.0

    def test_terminal_distribution_is_lognormal(self):
        S0, T = 100.0, 1.0
        paths = GBMSimulator().simulate(S0, GBM_PARAMS, T, DT, 20_000, seed=7)

        log_returns = np.log(terminal_values(paths) / S0)
        loc = (GBM_PARAMS.mu - 0.5 * GBM_PARAMS.sigma**2) * T
        scale = GBM_PARAMS.sigma * np.sqrt(T)

        _, p_value = stats.kstest(log_returns, "norm", args=(loc, scale))
        assert p_value > 0.01

    def test_terminal_variance_matches_analytic(self):
        S0, T = 100.0, 1.0
        paths = GBMSimulator().simulate(S0, GBM_PARAMS, T, DT, 100_000, seed=3)

        variance = np.var(terminal_values(paths), ddof=1)
        expected = (
            S0**2
            * np.exp(2 * GBM_PARAMS.mu * T)
            * (np.exp(GBM_PARAMS.sigma**2 * T) - 1)
        )
        assert variance == pytest.approx(expected, rel=0.05)

    def test_zero_volatility_is_deterministic(self):
        params = GBMParams(mu=0.10, sigma=0.0)
        paths = GBMSimulator().simulate(100.0, params, T=2.0, dt=DT, n_simulations=5, seed=0)
        expected = 100.0 * np.exp(params.mu * 2.0)
        np.testing.assert_allclose(terminal_values(paths), expected)


# --------------------------------------------------------------------------
# Jump diffusion
# --------------------------------------------------------------------------


class TestJumpDiffusionSimulator:
    def test_zero_intensity_matches_gbm(self):
        params = JumpDiffusionParams(
            mu=GBM_PARAMS.mu, sigma=GBM_PARAMS.sigma, lambda_j=0.0, mu_j=-0.05, sigma_j=0.1
        )
        jd_paths = JumpDiffusionSimulator().simulate(100.0, params, 1.0, DT, 1_000, seed=99)
        gbm_paths = GBMSimulator().simulate(100.0, GBM_PARAMS, 1.0, DT, 1_000, seed=99)
        np.testing.assert_allclose(jd_paths, gbm_paths, rtol=1e-12)

    def test_rejects_intensity_too_high_for_step_size(self):
        params = JumpDiffusionParams(
            mu=0.05, sigma=0.2, lambda_j=1.0, mu_j=0.0, sigma_j=0.05
        )
        # lambda_j * dt = 0.25 > MAX_LAMBDA_DT
        with pytest.raises(ValueError, match="single-jump-per-step approximation"):
            JumpDiffusionSimulator().simulate(100.0, params, T=1.0, dt=0.25, n_simulations=10)

        # just inside the threshold is fine
        dt_ok = 0.99 * MAX_LAMBDA_DT / params.lambda_j
        paths = JumpDiffusionSimulator().simulate(
            100.0, params, T=1.0, dt=dt_ok, n_simulations=10, seed=0
        )
        assert np.all(paths > 0)

    def test_rejects_negative_intensity(self):
        params = JumpDiffusionParams(
            mu=0.05, sigma=0.2, lambda_j=-1.0, mu_j=0.0, sigma_j=0.05
        )
        with pytest.raises(ValueError, match="lambda_j must be non-negative"):
            JumpDiffusionSimulator().simulate(100.0, params, T=1.0, dt=DT, n_simulations=10)

    def test_compensated_drift_preserves_expected_terminal_value(self):
        S0, T, n_sims = 100.0, 1.0, 200_000
        params = JumpDiffusionParams(
            mu=0.08, sigma=0.15, lambda_j=2.0, mu_j=-0.05, sigma_j=0.10
        )
        paths = JumpDiffusionSimulator().simulate(S0, params, T, DT, n_sims, seed=5)

        terminal = terminal_values(paths)
        stderr = np.std(terminal, ddof=1) / np.sqrt(n_sims)
        expected = S0 * np.exp(params.mu * T)
        assert abs(terminal.mean() - expected) < 4 * stderr

    def test_jumps_fatten_the_tails_relative_to_gbm(self):
        S0, T, n_sims = 100.0, 1.0, 50_000
        params = JumpDiffusionParams(
            mu=0.08, sigma=0.15, lambda_j=3.0, mu_j=-0.06, sigma_j=0.12
        )
        gbm_only = GBMParams(mu=params.mu, sigma=params.sigma)

        jd_returns = np.log(
            terminal_values(
                JumpDiffusionSimulator().simulate(S0, params, T, DT, n_sims, seed=8)
            )
            / S0
        )
        gbm_returns = np.log(
            terminal_values(GBMSimulator().simulate(S0, gbm_only, T, DT, n_sims, seed=8)) / S0
        )

        assert stats.kurtosis(jd_returns) > stats.kurtosis(gbm_returns) + 0.2
        assert jd_returns.std(ddof=1) > gbm_returns.std(ddof=1)


# --------------------------------------------------------------------------
# Heston
# --------------------------------------------------------------------------


class TestHestonSimulator:
    @pytest.mark.parametrize(
        "params",
        [HESTON_PARAMS, HESTON_PARAMS_FELLER_VIOLATED],
        ids=["feller_ok", "feller_violated"],
    )
    def test_variance_never_negative(self, params):
        _, variance = HestonSimulator().simulate_with_variance(
            100.0, params, T=2.0, dt=DT, n_simulations=2_000, seed=11
        )
        assert variance.shape == (2_000, 505)
        assert np.all(variance >= 0.0)
        assert np.all(np.isfinite(variance))

    def test_variance_starts_at_v0(self):
        _, variance = HestonSimulator().simulate_with_variance(
            100.0, HESTON_PARAMS, T=1.0, dt=DT, n_simulations=100, seed=1
        )
        np.testing.assert_array_equal(variance[:, 0], HESTON_PARAMS.v0)

    def test_feller_violated_variance_actually_hits_zero(self):
        """Guards the exponential branch: if this never fired, the
        non-negativity test above would be passing vacuously.
        """
        _, variance = HestonSimulator().simulate_with_variance(
            100.0, HESTON_PARAMS_FELLER_VIOLATED, T=2.0, dt=DT, n_simulations=1_000, seed=11
        )
        assert np.any(variance == 0.0)

    def test_variance_mean_reverts_to_theta(self):
        params = HestonParams(
            kappa=3.0, theta=0.09, xi=0.4, rho=-0.5, v0=0.01, feller_satisfied=True
        )
        _, variance = HestonSimulator().simulate_with_variance(
            100.0, params, T=5.0, dt=DT, n_simulations=20_000, seed=2
        )
        # E[v_t] = theta + (v0 - theta) e^{-kappa t}; after 5y at kappa=3 the
        # exponential term is ~3e-7, so the mean should sit on theta.
        assert variance[:, -1].mean() == pytest.approx(params.theta, rel=0.05)
        assert variance[:, 0].mean() == pytest.approx(params.v0)

    def test_expected_terminal_price_matches_drift(self):
        S0, T, mu, n_sims = 100.0, 1.0, 0.03, 100_000
        paths = HestonSimulator().simulate(
            S0, HESTON_PARAMS, T, DT, n_sims, seed=4, mu=mu
        )
        terminal = terminal_values(paths)
        stderr = np.std(terminal, ddof=1) / np.sqrt(n_sims)
        # the QE log-price step is a discretization, not exact, so allow a
        # small bias on top of the sampling error
        assert abs(terminal.mean() - S0 * np.exp(mu * T)) < 4 * stderr + 0.02 * S0

    def test_degenerates_to_black_scholes_as_vol_of_vol_vanishes(self):
        """With xi -> 0 the variance is pinned at v0 = theta, so Heston must
        reprice a European call exactly like Black-Scholes. This is the
        sharpest available check on the K0..K4 log-price discretization.
        """
        S0, T, mu, sigma, n_sims = 100.0, 1.0, 0.03, 0.2, 50_000
        params = HestonParams(
            kappa=2.0, theta=sigma**2, xi=1e-8, rho=0.0, v0=sigma**2, feller_satisfied=True
        )
        terminal = terminal_values(
            HestonSimulator().simulate(S0, params, T, 1 / 52, n_sims, seed=0, mu=mu)
        )

        forward = S0 * np.exp(mu * T)
        d1 = (np.log(forward / 100.0) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        analytic = forward * stats.norm.cdf(d1) - 100.0 * stats.norm.cdf(d2)

        payoffs = np.maximum(terminal - 100.0, 0.0)
        stderr = payoffs.std(ddof=1) / np.sqrt(n_sims)
        assert abs(payoffs.mean() - analytic) < 4 * stderr

    def test_negative_correlation_skews_returns_left(self):
        S0, T, n_sims = 100.0, 1.0, 40_000
        down = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.8, v0=0.04, feller_satisfied=False)
        up = down.model_copy(update={"rho": 0.8})

        skew_down = stats.skew(
            np.log(terminal_values(HestonSimulator().simulate(S0, down, T, DT, n_sims, seed=6)))
        )
        skew_up = stats.skew(
            np.log(terminal_values(HestonSimulator().simulate(S0, up, T, DT, n_sims, seed=6)))
        )
        assert skew_down < skew_up

    def test_psi_crit_is_andersens_recommendation(self):
        assert PSI_CRIT == 1.5

    @pytest.mark.parametrize(
        "update, match",
        [
            ({"kappa": 0.0}, "kappa must be positive"),
            ({"theta": -0.01}, "theta and v0 must be non-negative"),
            ({"v0": -0.01}, "theta and v0 must be non-negative"),
            ({"xi": 0.0}, "xi must be positive"),
            ({"rho": 1.5}, r"rho must lie in \[-1, 1\]"),
        ],
    )
    def test_rejects_invalid_params(self, update, match):
        params = HESTON_PARAMS.model_copy(update=update)
        with pytest.raises(ValueError, match=match):
            HestonSimulator().simulate(100.0, params, T=1.0, dt=DT, n_simulations=10)

    def test_simulate_matches_simulate_with_variance(self):
        prices_only = HestonSimulator().simulate(
            100.0, HESTON_PARAMS, 1.0, DT, 100, seed=17
        )
        prices, _ = HestonSimulator().simulate_with_variance(
            100.0, HESTON_PARAMS, 1.0, DT, 100, seed=17
        )
        np.testing.assert_array_equal(prices_only, prices)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


class TestReproducibility:
    SIMULATORS = [
        (GBMSimulator(), GBM_PARAMS),
        (JumpDiffusionSimulator(), JD_PARAMS),
        (HestonSimulator(), HESTON_PARAMS),
    ]

    @pytest.mark.parametrize("simulator, params", SIMULATORS)
    @pytest.mark.parametrize("method", [None, "antithetic", "stratified"])
    def test_same_seed_gives_identical_arrays(self, simulator, params, method):
        kwargs = dict(T=1.0, dt=DT, n_simulations=200, seed=123, variance_reduction=method)
        first = simulator.simulate(100.0, params, **kwargs)
        second = simulator.simulate(100.0, params, **kwargs)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("simulator, params", SIMULATORS)
    def test_different_seeds_give_different_arrays(self, simulator, params):
        first = simulator.simulate(100.0, params, 1.0, DT, 200, seed=1)
        second = simulator.simulate(100.0, params, 1.0, DT, 200, seed=2)
        assert not np.allclose(first, second)

    @pytest.mark.parametrize("simulator, params", SIMULATORS)
    def test_a_fresh_instance_reproduces_the_same_seed(self, simulator, params):
        """Statelessness in practice: a reused instance and a brand new one of
        the same type must agree, so no draw state leaks between calls.
        """
        reused = simulator.simulate(100.0, params, 1.0, DT, 100, seed=55)
        fresh = type(simulator)().simulate(100.0, params, 1.0, DT, 100, seed=55)
        np.testing.assert_array_equal(reused, fresh)


# --------------------------------------------------------------------------
# Variance reduction, end to end
# --------------------------------------------------------------------------


def _mean_terminal_estimates(method: str | None, n_runs: int, n_sims: int) -> np.ndarray:
    """The Monte Carlo estimator of E[S_T], recomputed over independent runs."""
    simulator = GBMSimulator()
    return np.array(
        [
            terminal_values(
                simulator.simulate(
                    100.0, GBM_PARAMS, 1.0, 1 / 12, n_sims, seed=seed, variance_reduction=method
                )
            ).mean()
            for seed in range(n_runs)
        ]
    )


class TestVarianceReductionEffect:
    def test_antithetic_reduces_estimator_variance(self):
        n_runs, n_sims = 60, 500
        naive = _mean_terminal_estimates(None, n_runs, n_sims)
        antithetic = _mean_terminal_estimates("antithetic", n_runs, n_sims)

        naive_var = np.var(naive, ddof=1)
        antithetic_var = np.var(antithetic, ddof=1)
        assert antithetic_var < 0.7 * naive_var

        # both estimators must still be centred on the same analytic value
        expected = 100.0 * np.exp(GBM_PARAMS.mu)
        assert antithetic.mean() == pytest.approx(expected, rel=0.01)

    def test_stratified_improves_distributional_coverage(self):
        """One step, so the terminal marginal is exactly what gets stratified.

        Compares the KS distance between the simulated terminal log return and
        its analytic normal law, averaged over independent runs.
        """
        n_runs, n_sims = 20, 500
        loc = (GBM_PARAMS.mu - 0.5 * GBM_PARAMS.sigma**2) * 1.0
        scale = GBM_PARAMS.sigma

        def mean_ks(method: str | None) -> float:
            simulator = GBMSimulator()
            distances = []
            for seed in range(n_runs):
                paths = simulator.simulate(
                    100.0, GBM_PARAMS, 1.0, 1.0, n_sims, seed=seed, variance_reduction=method
                )
                log_returns = np.log(terminal_values(paths) / 100.0)
                distances.append(stats.kstest(log_returns, "norm", args=(loc, scale)).statistic)
            return float(np.mean(distances))

        assert mean_ks("stratified") < 0.5 * mean_ks(None)

    @pytest.mark.parametrize("method", ["antithetic", "stratified"])
    def test_variance_reduction_leaves_the_mean_unbiased(self, method):
        n_sims = 100_000
        paths = GBMSimulator().simulate(
            100.0, GBM_PARAMS, 1.0, 1 / 12, n_sims, seed=21, variance_reduction=method
        )
        terminal = terminal_values(paths)
        stderr = np.std(terminal, ddof=1) / np.sqrt(n_sims)
        assert abs(terminal.mean() - 100.0 * np.exp(GBM_PARAMS.mu)) < 4 * stderr
