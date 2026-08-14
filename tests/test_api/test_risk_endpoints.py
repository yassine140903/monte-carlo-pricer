"""POST /risk/metrics, POST /risk/portfolio and GET /risk/scenarios."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_api.conftest import GBM_PARAMS, HESTON_PARAMS, JUMP_PARAMS

METRIC_KEYS = {
    "var",
    "cvar",
    "max_drawdown",
    "sharpe_ratio",
    "probability_of_loss",
    "n_simulations",
}


def metrics_body(**overrides) -> dict:
    body = {
        "S0": 100.0,
        "model_params": GBM_PARAMS,
        "T": 1.0,
        "n_simulations": 2000,
        "r": 0.03,
        "seed": 21,
    }
    body.update(overrides)
    return body


def portfolio_body(**overrides) -> dict:
    body = {
        "assets": [
            {"ticker": "AAPL", "weight": 0.5, "S0": 180.0, "model_params": GBM_PARAMS},
            {"ticker": "MSFT", "weight": 0.3, "S0": 400.0, "model_params": JUMP_PARAMS},
            {"ticker": "GOOGL", "weight": 0.2, "S0": 140.0, "model_params": HESTON_PARAMS},
        ],
        "T": 1.0,
        "n_simulations": 1000,
        "lookback_days": 500,
        "r": 0.03,
        "seed": 31,
    }
    body.update(overrides)
    return body


@pytest.fixture()
def single_asset_metrics(client) -> dict:
    response = client.post("/risk/metrics", json=metrics_body())
    assert response.status_code == 200
    return response.json()


class TestSingleAssetMetrics:
    def test_has_every_documented_field(self, single_asset_metrics):
        assert set(single_asset_metrics) == METRIC_KEYS

    def test_var_and_cvar_are_keyed_by_confidence_percentage(self, single_asset_metrics):
        assert set(single_asset_metrics["var"]) == {"95", "99"}
        assert set(single_asset_metrics["cvar"]) == {"95", "99"}

    def test_losses_keep_their_negative_sign(self, single_asset_metrics):
        """The engine's convention is carried through untouched: a loss is
        negative, and CVaR is at least as bad as VaR."""
        var, cvar = single_asset_metrics["var"], single_asset_metrics["cvar"]

        assert var["95"] < 0
        assert cvar["95"] <= var["95"]
        assert var["99"] <= var["95"]

    def test_max_drawdown_reports_a_mean_and_a_tail(self, single_asset_metrics):
        drawdown = single_asset_metrics["max_drawdown"]

        assert set(drawdown) == {"mean", "percentile_95"}
        assert 0 <= drawdown["mean"] <= drawdown["percentile_95"] <= 1

    def test_probability_of_loss_is_a_probability(self, single_asset_metrics):
        assert 0.0 <= single_asset_metrics["probability_of_loss"] <= 1.0

    def test_custom_confidence_levels_are_honoured(self, client):
        body = client.post(
            "/risk/metrics", json=metrics_body(confidence_levels=[0.9, 0.975])
        ).json()

        assert set(body["var"]) == {"90", "97.5"}

    def test_runs_under_the_physical_measure(self, client):
        """A strongly positive calibrated drift must lower the loss
        probability; overriding it with r - q would erase the difference."""
        drifty = client.post(
            "/risk/metrics",
            json=metrics_body(model_params=dict(GBM_PARAMS, mu=0.40)),
        ).json()
        flat = client.post(
            "/risk/metrics",
            json=metrics_body(model_params=dict(GBM_PARAMS, mu=0.0)),
        ).json()

        assert drifty["probability_of_loss"] < flat["probability_of_loss"]

    @pytest.mark.parametrize(
        "params", [GBM_PARAMS, JUMP_PARAMS, HESTON_PARAMS], ids=["gbm", "jump", "heston"]
    )
    def test_every_model_family_is_supported(self, client, params):
        response = client.post("/risk/metrics", json=metrics_body(model_params=params))

        assert response.status_code == 200

    def test_a_preset_scenario_widens_the_loss_tail(self, client):
        """Same seed either side, so the gap is the shock and not noise."""
        baseline = client.post("/risk/metrics", json=metrics_body()).json()
        stressed = client.post(
            "/risk/metrics", json=metrics_body(scenario="vol_spike")
        ).json()

        assert stressed["var"]["95"] < baseline["var"]["95"]
        assert stressed["cvar"]["99"] < baseline["cvar"]["99"]

    def test_a_custom_scenario_is_applied(self, client):
        baseline = client.post("/risk/metrics", json=metrics_body()).json()
        stressed = client.post(
            "/risk/metrics",
            json=metrics_body(
                custom_scenario={
                    "name": "Vol Doubling",
                    "description": "",
                    "multipliers": {"sigma": 2.0},
                }
            ),
        ).json()

        assert stressed["max_drawdown"]["mean"] > baseline["max_drawdown"]["mean"]

    def test_a_scenario_variant_is_picked_per_model_family(self, client):
        """vol_spike names sigma for GBM and v0/theta/xi for Heston; the
        request carries neither, the preset lookup resolves it."""
        response = client.post(
            "/risk/metrics",
            json=metrics_body(model_params=HESTON_PARAMS, scenario="vol_spike"),
        )

        assert response.status_code == 200

    def test_an_unknown_scenario_is_404(self, client):
        response = client.post(
            "/risk/metrics", json=metrics_body(scenario="alien_invasion")
        )

        assert response.status_code == 404

    def test_both_scenario_kinds_at_once_is_422(self, client):
        response = client.post(
            "/risk/metrics",
            json=metrics_body(
                scenario="vol_spike",
                custom_scenario={"name": "x", "description": "", "multipliers": {"sigma": 2.0}},
            ),
        )

        assert response.status_code == 422

    def test_a_confidence_level_outside_the_unit_interval_is_422(self, client):
        response = client.post("/risk/metrics", json=metrics_body(confidence_levels=[1.5]))

        assert response.status_code == 422

    def test_over_the_path_budget_is_413(self, client):
        response = client.post("/risk/metrics", json=metrics_body(n_simulations=100_001))

        assert response.status_code == 413


@pytest.fixture()
def portfolio(client) -> dict:
    response = client.post("/risk/portfolio", json=portfolio_body())
    assert response.status_code == 200
    return response.json()


class TestPortfolioRisk:
    def test_has_every_documented_field(self, portfolio):
        assert set(portfolio) == {
            "risk_metrics",
            "correlation_matrix",
            "tickers",
            "scenario_applied",
        }
        assert set(portfolio["risk_metrics"]) == METRIC_KEYS

    def test_tickers_and_matrix_share_one_ordering(self, portfolio):
        """estimate_correlation_matrix sorts alphabetically, and the assets are
        reordered to match — simulate_portfolio cannot detect a mismatch."""
        assert portfolio["tickers"] == ["AAPL", "GOOGL", "MSFT"]
        assert len(portfolio["correlation_matrix"]) == 3

    def test_the_correlation_matrix_is_a_correlation_matrix(self, portfolio):
        matrix = np.array(portfolio["correlation_matrix"])

        assert np.allclose(np.diag(matrix), 1.0)
        assert np.allclose(matrix, matrix.T)
        assert np.all(np.linalg.eigvalsh(matrix) > -1e-8)

    def test_the_estimated_correlations_are_positive(self, portfolio):
        """The fixture series share a common factor, so they must correlate."""
        matrix = np.array(portfolio["correlation_matrix"])
        off_diagonal = matrix[~np.eye(3, dtype=bool)]

        assert np.all(off_diagonal > 0.1)

    def test_two_assets_are_enough(self, client):
        response = client.post(
            "/risk/portfolio",
            json=portfolio_body(
                assets=[
                    {"ticker": "AAPL", "weight": 0.6, "S0": 180.0, "model_params": GBM_PARAMS},
                    {"ticker": "MSFT", "weight": 0.4, "S0": 400.0, "model_params": GBM_PARAMS},
                ]
            ),
        )

        assert response.status_code == 200
        assert response.json()["tickers"] == ["AAPL", "MSFT"]

    def test_no_scenario_applied_by_default(self, portfolio):
        assert portfolio["scenario_applied"] is None

    def test_a_preset_scenario_worsens_the_tail(self, client):
        baseline = client.post("/risk/portfolio", json=portfolio_body()).json()
        stressed = client.post(
            "/risk/portfolio", json=portfolio_body(scenario="vol_spike")
        ).json()

        assert stressed["scenario_applied"] == "vol_spike"
        assert stressed["risk_metrics"]["var"]["95"] < baseline["risk_metrics"]["var"]["95"]

    def test_a_custom_scenario_is_applied_by_name(self, client):
        response = client.post(
            "/risk/portfolio",
            json=portfolio_body(
                assets=[
                    {"ticker": "AAPL", "weight": 0.5, "S0": 180.0, "model_params": GBM_PARAMS},
                    {"ticker": "MSFT", "weight": 0.5, "S0": 400.0, "model_params": GBM_PARAMS},
                ],
                custom_scenario={
                    "name": "Mild Squeeze",
                    "description": "vol up 50%",
                    "multipliers": {"sigma": 1.5},
                },
            ),
        )

        assert response.status_code == 200
        assert response.json()["scenario_applied"] == "Mild Squeeze"

    def test_a_custom_scenario_naming_an_absent_parameter_is_422(self, client):
        response = client.post(
            "/risk/portfolio",
            json=portfolio_body(
                custom_scenario={
                    "name": "Nope",
                    "description": "",
                    "multipliers": {"nonsense": 2.0},
                }
            ),
        )

        assert response.status_code == 422

    def test_both_scenario_kinds_at_once_is_422(self, client):
        response = client.post(
            "/risk/portfolio",
            json=portfolio_body(
                scenario="vol_spike",
                custom_scenario={"name": "x", "description": "", "multipliers": {"sigma": 2.0}},
            ),
        )

        assert response.status_code == 422

    def test_an_unknown_scenario_is_404(self, client):
        response = client.post("/risk/portfolio", json=portfolio_body(scenario="alien_invasion"))

        assert response.status_code == 404

    def test_weights_that_do_not_sum_to_one_are_422(self, client):
        assets = portfolio_body()["assets"]
        assets[0]["weight"] = 0.9

        response = client.post("/risk/portfolio", json=portfolio_body(assets=assets))

        assert response.status_code == 422

    def test_a_single_asset_is_rejected_before_it_reaches_the_engine(self, client):
        response = client.post(
            "/risk/portfolio",
            json=portfolio_body(
                assets=[
                    {"ticker": "AAPL", "weight": 1.0, "S0": 180.0, "model_params": GBM_PARAMS}
                ]
            ),
        )

        assert response.status_code == 422

    def test_an_unknown_ticker_is_404(self, client):
        assets = portfolio_body()["assets"]
        assets[0]["ticker"] = "NOSUCH"

        response = client.post("/risk/portfolio", json=portfolio_body(assets=assets))

        assert response.status_code == 404


class TestScenarioListing:
    def test_lists_every_preset_family(self, client):
        response = client.get("/risk/scenarios")

        assert response.status_code == 200
        keys = [family["key"] for family in response.json()["scenarios"]]
        assert set(keys) == {"vol_spike", "market_crash"}

    def test_each_family_names_the_models_it_covers(self, client):
        families = client.get("/risk/scenarios").json()["scenarios"]

        for family in families:
            assert family["model_types"] == ["gbm", "heston", "jump_diffusion"]
            assert len(family["variants"]) == 3
            assert all(variant["description"] for variant in family["variants"])
