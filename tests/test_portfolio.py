"""Tests for src/risk/portfolio.py.

The central claim is that the correlation asked for is the correlation you
get, so most tests here recover it from the simulated terminal returns and
compare against the input. Those tests pin a seed and use enough paths that
the sampling error is well inside the tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.gbm import GBMParams
from src.calibration.heston import HestonParams
from src.calibration.jump_diffusion import JumpDiffusionParams
from src.risk.metrics import RiskMetrics, compute_risk_metrics
from src.risk.portfolio import (
    PortfolioAsset,
    PortfolioResult,
    estimate_correlation_matrix,
    simulate_portfolio,
)
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.jump_diffusion import JumpDiffusionSimulator

DT = 1 / 252
T = 1.0
GBM_PARAMS = GBMParams(mu=0.08, sigma=0.20)


def corr_matrix(rho: float) -> np.ndarray:
    return np.array([[1.0, rho], [rho, 1.0]])


def two_gbm_assets(
    weights: tuple[float, float] = (0.5, 0.5),
    spots: tuple[float, float] = (100.0, 50.0),
) -> list[PortfolioAsset]:
    return [
        PortfolioAsset(
            ticker=ticker,
            weight=weight,
            S0=spot,
            simulator=GBMSimulator(),
            params=GBM_PARAMS,
        )
        for ticker, weight, spot in zip(("AAA", "BBB"), weights, spots)
    ]


def terminal_log_returns(result: PortfolioResult, ticker: str, S0: float) -> np.ndarray:
    return np.log(result.asset_paths[ticker][:, -1] / S0)


def realized_correlation(result: PortfolioResult, assets: list[PortfolioAsset]) -> float:
    first, second = (
        terminal_log_returns(result, asset.ticker, asset.S0) for asset in assets
    )
    return float(np.corrcoef(first, second)[0, 1])


# --------------------------------------------------------------------------
# Correlation estimation
# --------------------------------------------------------------------------


class TestEstimateCorrelationMatrix:
    def test_recovers_a_known_correlation(self):
        rho = 0.6
        rng = np.random.default_rng(0)
        independent = rng.standard_normal((2, 100_000))
        correlated = np.linalg.cholesky(corr_matrix(rho)) @ independent

        estimated, tickers = estimate_correlation_matrix(
            {"AAA": correlated[0], "BBB": correlated[1]}
        )

        assert estimated[0, 1] == pytest.approx(rho, abs=0.01)
        assert tickers == ["AAA", "BBB"]

    def test_orders_tickers_alphabetically_regardless_of_insertion_order(self):
        rng = np.random.default_rng(1)
        series = {name: rng.standard_normal(500) for name in ("ZZZ", "AAA", "MMM")}

        matrix, tickers = estimate_correlation_matrix(series)

        assert tickers == ["AAA", "MMM", "ZZZ"]
        assert matrix.shape == (3, 3)

    def test_ordering_is_stable_across_insertion_orders(self):
        rng = np.random.default_rng(2)
        a, b = rng.standard_normal(500), rng.standard_normal(500)

        forward, forward_tickers = estimate_correlation_matrix({"AAA": a, "BBB": b})
        reversed_, reversed_tickers = estimate_correlation_matrix({"BBB": b, "AAA": a})

        assert forward_tickers == reversed_tickers
        assert np.allclose(forward, reversed_)

    def test_diagonal_is_unit_and_matrix_is_symmetric(self):
        rng = np.random.default_rng(3)
        series = {name: rng.standard_normal(500) for name in ("AAA", "BBB", "CCC")}

        matrix, _ = estimate_correlation_matrix(series)

        assert np.allclose(np.diag(matrix), 1.0)
        assert np.allclose(matrix, matrix.T)

    def test_rejects_series_of_different_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            estimate_correlation_matrix(
                {"AAA": np.zeros(100), "BBB": np.zeros(99)}
            )

    def test_rejects_fewer_than_two_series(self):
        with pytest.raises(ValueError, match="at least two"):
            estimate_correlation_matrix({"AAA": np.zeros(100)})


# --------------------------------------------------------------------------
# Correlation is preserved through simulation
# --------------------------------------------------------------------------


class TestCorrelationPreservation:
    @pytest.mark.parametrize("rho", [0.7, -0.5, 0.3])
    def test_simulated_returns_carry_the_requested_correlation(self, rho):
        assets = two_gbm_assets()
        result = simulate_portfolio(
            assets, corr_matrix(rho), T, DT, n_simulations=50_000, seed=42
        )

        assert realized_correlation(result, assets) == pytest.approx(rho, abs=0.05)

    def test_identity_correlation_leaves_the_assets_independent(self):
        assets = two_gbm_assets()
        result = simulate_portfolio(
            assets, np.eye(2), T, DT, n_simulations=50_000, seed=42
        )

        assert realized_correlation(result, assets) == pytest.approx(0.0, abs=0.02)

    def test_perfect_correlation_makes_identical_assets_move_together(self):
        assets = two_gbm_assets(spots=(100.0, 100.0))
        result = simulate_portfolio(
            assets, corr_matrix(1.0), T, DT, n_simulations=10_000, seed=42
        )

        assert realized_correlation(result, assets) == pytest.approx(1.0, abs=1e-6)
        # Not bitwise identical: rho = 1 is singular, so the Cholesky ridge
        # leaks sqrt(1e-10) = 1e-5 of an independent shock into the second
        # asset per step, worth ~1e-6 relative by the terminal date.
        assert np.allclose(
            result.asset_paths["AAA"], result.asset_paths["BBB"], rtol=1e-4
        )

    def test_three_assets_carry_a_full_correlation_structure(self):
        target = np.array(
            [[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]]
        )
        assets = [
            PortfolioAsset(
                ticker=ticker,
                weight=1 / 3,
                S0=100.0,
                simulator=GBMSimulator(),
                params=GBM_PARAMS,
            )
            for ticker in ("AAA", "BBB", "CCC")
        ]

        result = simulate_portfolio(
            assets, target, T, DT, n_simulations=50_000, seed=42
        )
        returns = np.vstack(
            [terminal_log_returns(result, asset.ticker, asset.S0) for asset in assets]
        )

        assert np.allclose(np.corrcoef(returns), target, atol=0.05)


# --------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------


class TestPortfolioBlending:
    def test_identical_perfectly_correlated_assets_reproduce_the_single_asset(self):
        """0.5X + 0.5X = X. With the same params, same spot and rho = 1 the
        portfolio return path must equal either asset's return path — up to
        the Cholesky ridge that keeps the singular matrix factorable.
        """
        assets = two_gbm_assets(spots=(100.0, 100.0))
        result = simulate_portfolio(
            assets, corr_matrix(1.0), T, DT, n_simulations=5_000, seed=42
        )

        single = result.asset_paths["AAA"] / 100.0

        assert np.allclose(result.portfolio_paths, single, rtol=1e-4)

    def test_every_path_starts_at_the_initial_value(self):
        assets = two_gbm_assets()
        result = simulate_portfolio(
            assets, corr_matrix(0.4), T, DT, n_simulations=1_000, seed=1,
            initial_value=1_000_000.0,
        )

        assert np.allclose(result.portfolio_paths[:, 0], 1_000_000.0)

    def test_initial_value_scales_the_paths_linearly(self):
        assets = two_gbm_assets()
        kwargs = dict(T=T, dt=DT, n_simulations=1_000, seed=1)

        unit = simulate_portfolio(assets, corr_matrix(0.4), **kwargs)
        scaled = simulate_portfolio(
            assets, corr_matrix(0.4), initial_value=500.0, **kwargs
        )

        assert np.allclose(scaled.portfolio_paths, 500.0 * unit.portfolio_paths)

    def test_weights_shift_the_blend_toward_the_heavier_asset(self):
        """Two assets differing only in volatility: loading the calmer one
        must reduce the spread of portfolio outcomes."""
        calm = PortfolioAsset(
            ticker="CALM", weight=0.9, S0=100.0,
            simulator=GBMSimulator(), params=GBMParams(mu=0.08, sigma=0.05),
        )
        wild = PortfolioAsset(
            ticker="WILD", weight=0.1, S0=100.0,
            simulator=GBMSimulator(), params=GBMParams(mu=0.08, sigma=0.50),
        )
        kwargs = dict(T=T, dt=DT, n_simulations=20_000, seed=42)

        calm_heavy = simulate_portfolio([calm, wild], corr_matrix(0.3), **kwargs)
        calm.weight, wild.weight = 0.1, 0.9
        wild_heavy = simulate_portfolio([calm, wild], corr_matrix(0.3), **kwargs)

        assert np.std(calm_heavy.portfolio_paths[:, -1]) < np.std(
            wild_heavy.portfolio_paths[:, -1]
        )

    def test_diversification_reduces_variance_versus_perfect_correlation(self):
        assets = two_gbm_assets(spots=(100.0, 100.0))
        kwargs = dict(T=T, dt=DT, n_simulations=20_000, seed=42)

        diversified = simulate_portfolio(assets, np.eye(2), **kwargs)
        concentrated = simulate_portfolio(assets, corr_matrix(1.0), **kwargs)

        assert np.std(diversified.portfolio_paths[:, -1]) < np.std(
            concentrated.portfolio_paths[:, -1]
        )

    def test_result_shapes_and_metadata(self):
        assets = two_gbm_assets()
        result = simulate_portfolio(
            assets, corr_matrix(0.4), T, DT, n_simulations=100, seed=1
        )

        assert result.portfolio_paths.shape == (100, 253)
        assert set(result.asset_paths) == {"AAA", "BBB"}
        assert all(paths.shape == (100, 253) for paths in result.asset_paths.values())
        assert result.weights == {"AAA": 0.5, "BBB": 0.5}
        assert result.tickers == ["AAA", "BBB"]
        assert np.allclose(result.correlation_matrix, corr_matrix(0.4))

    def test_asset_paths_start_at_their_own_spots(self):
        assets = two_gbm_assets(spots=(100.0, 50.0))
        result = simulate_portfolio(
            assets, corr_matrix(0.4), T, DT, n_simulations=100, seed=1
        )

        assert np.allclose(result.asset_paths["AAA"][:, 0], 100.0)
        assert np.allclose(result.asset_paths["BBB"][:, 0], 50.0)


class TestReproducibility:
    def test_same_seed_reproduces_the_portfolio(self):
        assets = two_gbm_assets()
        kwargs = dict(T=T, dt=DT, n_simulations=1_000, seed=42)

        first = simulate_portfolio(assets, corr_matrix(0.5), **kwargs)
        second = simulate_portfolio(assets, corr_matrix(0.5), **kwargs)

        assert np.array_equal(first.portfolio_paths, second.portfolio_paths)

    def test_different_seeds_give_different_portfolios(self):
        assets = two_gbm_assets()
        kwargs = dict(T=T, dt=DT, n_simulations=1_000)

        first = simulate_portfolio(assets, corr_matrix(0.5), seed=1, **kwargs)
        second = simulate_portfolio(assets, corr_matrix(0.5), seed=2, **kwargs)

        assert not np.allclose(first.portfolio_paths, second.portfolio_paths)


class TestMixedModels:
    """Correlation is imposed from outside the simulators, so the models in a
    portfolio need not match."""

    @pytest.fixture()
    def mixed_assets(self):
        return [
            PortfolioAsset(
                ticker="GBM", weight=1 / 3, S0=100.0,
                simulator=GBMSimulator(), params=GBM_PARAMS,
            ),
            PortfolioAsset(
                ticker="JD", weight=1 / 3, S0=80.0,
                simulator=JumpDiffusionSimulator(),
                params=JumpDiffusionParams(
                    mu=0.08, sigma=0.20, lambda_j=1.0, mu_j=-0.03, sigma_j=0.08
                ),
            ),
            PortfolioAsset(
                ticker="HESTON", weight=1 / 3, S0=120.0,
                simulator=HestonSimulator(),
                params=HestonParams(
                    kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04,
                    feller_satisfied=True,
                ),
            ),
        ]

    def test_simulates_a_portfolio_of_three_different_models(self, mixed_assets):
        target = np.array(
            [[1.0, 0.5, 0.4], [0.5, 1.0, 0.3], [0.4, 0.3, 1.0]]
        )

        result = simulate_portfolio(
            mixed_assets, target, T, DT, n_simulations=5_000, seed=42
        )

        assert result.portfolio_paths.shape == (5_000, 253)
        assert np.all(result.portfolio_paths > 0)
        assert set(result.asset_paths) == {"GBM", "JD", "HESTON"}

    def test_mixed_model_portfolio_is_reproducible(self, mixed_assets):
        """Per-asset seeds are derived from the master seed, so the jump and
        variance draws repeat too — not just the correlated normals."""
        target = np.array(
            [[1.0, 0.5, 0.4], [0.5, 1.0, 0.3], [0.4, 0.3, 1.0]]
        )
        kwargs = dict(T=T, dt=DT, n_simulations=1_000, seed=7)

        first = simulate_portfolio(mixed_assets, target, **kwargs)
        second = simulate_portfolio(mixed_assets, target, **kwargs)

        assert np.array_equal(first.portfolio_paths, second.portfolio_paths)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class TestValidation:
    def test_rejects_fewer_than_two_assets(self):
        assets = two_gbm_assets()[:1]
        assets[0].weight = 1.0

        with pytest.raises(ValueError, match="at least two assets"):
            simulate_portfolio(assets, np.eye(1), T, DT, 100)

    @pytest.mark.parametrize("weights", [(0.25, 0.25), (0.8, 0.8), (0.0, 0.0)])
    def test_rejects_weights_that_do_not_sum_to_one(self, weights):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            simulate_portfolio(
                two_gbm_assets(weights=weights), corr_matrix(0.5), T, DT, 100
            )

    def test_accepts_weights_within_floating_point_tolerance(self):
        third = 1 / 3
        assets = [
            PortfolioAsset(
                ticker=ticker, weight=third, S0=100.0,
                simulator=GBMSimulator(), params=GBM_PARAMS,
            )
            for ticker in ("AAA", "BBB", "CCC")
        ]

        result = simulate_portfolio(assets, np.eye(3), T, DT, 100, seed=1)

        assert result.portfolio_paths.shape == (100, 253)

    def test_rejects_a_matrix_sized_for_a_different_number_of_assets(self):
        assets = [
            PortfolioAsset(
                ticker=ticker, weight=1 / 3, S0=100.0,
                simulator=GBMSimulator(), params=GBM_PARAMS,
            )
            for ticker in ("AAA", "BBB", "CCC")
        ]

        with pytest.raises(ValueError, match="must have shape"):
            simulate_portfolio(assets, np.eye(2), T, DT, 100)

    def test_rejects_an_asymmetric_matrix(self):
        asymmetric = np.array([[1.0, 0.5], [0.2, 1.0]])

        with pytest.raises(ValueError, match="not symmetric"):
            simulate_portfolio(two_gbm_assets(), asymmetric, T, DT, 100)

    def test_rejects_a_non_unit_diagonal(self):
        bad_diagonal = np.array([[0.9, 0.5], [0.5, 0.9]])

        with pytest.raises(ValueError, match="diagonal"):
            simulate_portfolio(two_gbm_assets(), bad_diagonal, T, DT, 100)

    def test_rejects_a_non_positive_semi_definite_matrix(self):
        # |rho| > 1 is not a possible correlation: the matrix has a negative
        # eigenvalue, so no Cholesky factor exists.
        with pytest.raises(ValueError, match="positive semi-definite"):
            simulate_portfolio(two_gbm_assets(), corr_matrix(1.5), T, DT, 100)

    def test_rejects_an_impossible_three_way_correlation(self):
        # Pairwise plausible, jointly impossible: A and B both track C
        # strongly, so they cannot be strongly negatively correlated.
        impossible = np.array(
            [[1.0, -0.9, 0.9], [-0.9, 1.0, 0.9], [0.9, 0.9, 1.0]]
        )
        assets = [
            PortfolioAsset(
                ticker=ticker, weight=1 / 3, S0=100.0,
                simulator=GBMSimulator(), params=GBM_PARAMS,
            )
            for ticker in ("AAA", "BBB", "CCC")
        ]

        with pytest.raises(ValueError, match="positive semi-definite"):
            simulate_portfolio(assets, impossible, T, DT, 100)

    def test_singular_but_valid_matrix_survives_via_the_ridge(self):
        """rho = 1 is positive semi-definite but not positive definite, so a
        bare Cholesky fails; the ridge fallback must carry it through."""
        result = simulate_portfolio(
            two_gbm_assets(), corr_matrix(1.0), T, DT, 1_000, seed=1
        )

        assert result.portfolio_paths.shape == (1_000, 253)
        assert np.all(np.isfinite(result.portfolio_paths))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


class TestPortfolioRiskMetrics:
    def test_portfolio_paths_feed_the_metrics_unchanged(self):
        assets = two_gbm_assets()
        result = simulate_portfolio(
            assets, corr_matrix(0.6), T, DT, n_simulations=20_000, seed=42,
            initial_value=1_000_000.0,
        )

        metrics = compute_risk_metrics(
            result.portfolio_paths, S0=1_000_000.0, r=0.02, T=T
        )

        assert isinstance(metrics, RiskMetrics)
        assert metrics.n_simulations == 20_000
        assert metrics.cvar[0.95] <= metrics.var[0.95] < 0
        assert 0.0 < metrics.prob_of_loss < 1.0
        assert 0.0 <= metrics.max_drawdown_mean <= 1.0

    def test_a_diversified_portfolio_carries_less_tail_risk(self):
        assets = two_gbm_assets(spots=(100.0, 100.0))
        kwargs = dict(T=T, dt=DT, n_simulations=20_000, seed=42)

        diversified = simulate_portfolio(assets, np.eye(2), **kwargs)
        concentrated = simulate_portfolio(assets, corr_matrix(1.0), **kwargs)

        diversified_cvar = compute_risk_metrics(
            diversified.portfolio_paths, S0=1.0, r=0.02, T=T
        ).cvar[0.95]
        concentrated_cvar = compute_risk_metrics(
            concentrated.portfolio_paths, S0=1.0, r=0.02, T=T
        ).cvar[0.95]

        assert diversified_cvar > concentrated_cvar
