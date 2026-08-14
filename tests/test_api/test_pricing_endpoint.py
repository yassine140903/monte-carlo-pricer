"""POST /price-option."""
from __future__ import annotations

import pytest

from tests.test_api.conftest import GBM_PARAMS, HESTON_PARAMS, JUMP_PARAMS

GREEK_KEYS = {"delta", "gamma", "vega", "theta", "rho"}


def price_body(**overrides) -> dict:
    body = {
        "option_type": "european_call",
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.03,
        "model_params": GBM_PARAMS,
        "n_simulations": 1000,
        "seed": 5,
    }
    body.update(overrides)
    return body


@pytest.fixture()
def european_call(client) -> dict:
    response = client.post("/price-option", json=price_body())
    assert response.status_code == 200
    return response.json()


class TestEuropeanCall:
    def test_has_every_documented_field(self, european_call):
        assert set(european_call) == {
            "price",
            "std_error",
            "confidence_interval_95",
            "greeks",
            "bs_benchmark",
            "payoff_histogram",
            "n_simulations",
            "computation_time_ms",
            "mlflow_run_id",
        }

    def test_price_is_positive_and_bracketed_by_its_interval(self, european_call):
        low, high = european_call["confidence_interval_95"]

        assert european_call["price"] > 0
        assert low < european_call["price"] < high
        assert european_call["std_error"] > 0

    def test_all_five_greeks_are_present(self, european_call):
        greeks = european_call["greeks"]

        assert set(greeks) == GREEK_KEYS
        # An at-the-money call: delta near 0.5, positive gamma and vega, decay.
        assert 0.3 < greeks["delta"] < 0.8
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0
        assert greeks["theta"] < 0


class TestBlackScholesBenchmark:
    def test_included_for_a_gbm_european_call(self, european_call):
        benchmark = european_call["bs_benchmark"]

        assert benchmark is not None
        assert set(benchmark) == {"price", "greeks", "relative_error"}
        assert set(benchmark["greeks"]) == GREEK_KEYS

    def test_the_monte_carlo_price_tracks_it(self, client):
        body = client.post("/price-option", json=price_body(n_simulations=20_000)).json()

        assert body["price"] == pytest.approx(body["bs_benchmark"]["price"], rel=0.05)

    def test_included_for_a_gbm_european_put(self, client):
        body = client.post(
            "/price-option", json=price_body(option_type="european_put")
        ).json()

        assert body["bs_benchmark"] is not None
        assert body["greeks"]["delta"] < 0

    @pytest.mark.parametrize(
        "option_type", ["asian_call", "barrier_ko_call", "lookback_call"]
    )
    def test_omitted_for_exotics(self, client, option_type):
        """Black-Scholes prices vanillas; there is no closed form to quote here."""
        extra = {"barrier": 130.0} if option_type.startswith("barrier") else {}
        if option_type.startswith("lookback"):
            extra["K"] = None

        body = client.post(
            "/price-option", json=price_body(option_type=option_type, **extra)
        ).json()

        assert body["bs_benchmark"] is None

    @pytest.mark.parametrize(
        "params", [JUMP_PARAMS, HESTON_PARAMS], ids=["jump", "heston"]
    )
    def test_omitted_for_non_gbm_models(self, client, params):
        """A jump-diffusion sigma looks Black-Scholes-shaped but is not: the
        closed form cannot see the jumps, so quoting it would mislead."""
        body = client.post("/price-option", json=price_body(model_params=params)).json()

        assert body["bs_benchmark"] is None


class TestPayoffHistogram:
    def test_edges_are_one_longer_than_counts(self, european_call):
        histogram = european_call["payoff_histogram"]

        assert set(histogram) == {"bin_edges", "counts"}
        assert len(histogram["bin_edges"]) == 51
        assert len(histogram["counts"]) == 50

    def test_counts_cover_every_simulated_path(self, european_call):
        assert sum(european_call["payoff_histogram"]["counts"]) == 1000

    def test_edges_are_increasing_and_non_negative(self, european_call):
        edges = european_call["payoff_histogram"]["bin_edges"]

        assert edges == sorted(edges)
        assert edges[0] >= 0  # a call payoff is floored at zero


class TestExoticsAndValidation:
    def test_barrier_option_prices_below_the_vanilla(self, client):
        vanilla = client.post("/price-option", json=price_body()).json()["price"]
        knocked = client.post(
            "/price-option",
            json=price_body(option_type="barrier_ko_call", barrier=115.0),
        ).json()["price"]

        assert 0 <= knocked < vanilla

    def test_lookback_needs_no_strike(self, client):
        response = client.post(
            "/price-option", json=price_body(option_type="lookback_call", K=None)
        )

        assert response.status_code == 200
        assert response.json()["price"] > 0

    def test_unknown_option_type_is_422(self, client):
        response = client.post("/price-option", json=price_body(option_type="rainbow"))

        assert response.status_code == 422

    def test_a_barrier_option_without_a_barrier_is_422(self, client):
        response = client.post(
            "/price-option", json=price_body(option_type="barrier_ko_call")
        )

        assert response.status_code == 422

    def test_over_the_path_budget_is_413(self, client):
        response = client.post("/price-option", json=price_body(n_simulations=100_001))

        assert response.status_code == 413


class TestMLflowLogging:
    def test_price_and_greeks_are_logged(self, client, local_mlflow):
        import mlflow

        run_id = client.post("/price-option", json=price_body()).json()["mlflow_run_id"]
        assert run_id is not None

        mlflow.set_tracking_uri(local_mlflow)
        run = mlflow.get_run(run_id)

        assert run.data.params["option_type"] == "european_call"
        assert run.data.params["model_type"] == "gbm"
        assert {"price", "std_error"} | GREEK_KEYS <= set(run.data.metrics)
        assert (
            mlflow.get_experiment(run.info.experiment_id).name
            == "pricing/european_call"
        )
