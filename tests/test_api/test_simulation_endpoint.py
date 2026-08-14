"""POST /simulate."""
from __future__ import annotations

import pytest

from src.api.utils import DEFAULT_TARGET_POINTS
from tests.test_api.conftest import GBM_PARAMS, HESTON_PARAMS, JUMP_PARAMS

BAND_KEYS = {"p5", "p25", "p50", "p75", "p95"}
SUMMARY_KEYS = {"mean", "median", "std", "min", "max"}


def simulate_body(**overrides) -> dict:
    body = {
        "S0": 100.0,
        "model_params": GBM_PARAMS,
        "T": 1.0,
        "n_simulations": 1000,
        "seed": 11,
    }
    body.update(overrides)
    return body


@pytest.fixture()
def gbm_simulation(client) -> dict:
    response = client.post("/simulate", json=simulate_body())
    assert response.status_code == 200
    return response.json()


class TestResponseShape:
    def test_has_every_documented_field(self, gbm_simulation):
        assert set(gbm_simulation) == {
            "summary",
            "percentile_bands",
            "sample_paths",
            "time_axis",
            "n_simulations",
            "computation_time_ms",
        }

    def test_summary_describes_the_terminal_distribution(self, gbm_simulation):
        summary = gbm_simulation["summary"]

        assert set(summary) == SUMMARY_KEYS
        assert summary["min"] <= summary["median"] <= summary["max"]
        assert summary["std"] > 0

    def test_percentile_bands_are_ordered(self, gbm_simulation):
        bands = gbm_simulation["percentile_bands"]

        assert set(bands) == BAND_KEYS
        for step in range(len(bands["p5"])):
            values = [bands[key][step] for key in ("p5", "p25", "p50", "p75", "p95")]
            assert values == sorted(values)

    def test_reports_the_requested_path_count(self, gbm_simulation):
        assert gbm_simulation["n_simulations"] == 1000

    def test_computation_time_is_recorded(self, gbm_simulation):
        assert gbm_simulation["computation_time_ms"] > 0


class TestDownsampling:
    def test_series_are_thinned_to_about_fifty_points(self, gbm_simulation):
        """253 daily steps in, ~50 out — exactly 50 would need interpolation."""
        length = len(gbm_simulation["time_axis"])

        assert DEFAULT_TARGET_POINTS <= length <= DEFAULT_TARGET_POINTS + 2

    def test_bands_paths_and_axis_share_one_time_grid(self, gbm_simulation):
        length = len(gbm_simulation["time_axis"])

        assert all(len(band) == length for band in gbm_simulation["percentile_bands"].values())
        assert all(len(path) == length for path in gbm_simulation["sample_paths"])

    def test_the_axis_keeps_both_endpoints(self, gbm_simulation):
        axis = gbm_simulation["time_axis"]

        assert axis[0] == pytest.approx(0.0)
        assert axis[-1] == pytest.approx(1.0, abs=1 / 252)

    def test_five_sample_paths_are_returned_starting_at_S0(self, gbm_simulation):
        paths = gbm_simulation["sample_paths"]

        assert len(paths) == 5
        assert all(path[0] == pytest.approx(100.0) for path in paths)


class TestModels:
    @pytest.mark.parametrize(
        "params", [GBM_PARAMS, JUMP_PARAMS, HESTON_PARAMS], ids=["gbm", "jump", "heston"]
    )
    def test_every_model_family_simulates(self, client, params):
        response = client.post("/simulate", json=simulate_body(model_params=params))

        assert response.status_code == 200
        assert response.json()["summary"]["mean"] > 0

    @pytest.mark.parametrize("method", ["antithetic", "stratified"])
    def test_variance_reduction_is_accepted(self, client, method):
        response = client.post(
            "/simulate", json=simulate_body(variance_reduction=method)
        )

        assert response.status_code == 200

    def test_seeding_makes_the_response_reproducible(self, client):
        first = client.post("/simulate", json=simulate_body(seed=99)).json()
        second = client.post("/simulate", json=simulate_body(seed=99)).json()

        assert first["summary"] == second["summary"]
        assert first["sample_paths"] == second["sample_paths"]

    def test_the_physical_drift_is_used_not_a_risk_neutral_one(self, client):
        """mu=0.30 must show up in the terminal mean; pricing's r-q override
        does not apply to this endpoint."""
        drifty = dict(GBM_PARAMS, mu=0.30, sigma=0.05)

        mean = client.post(
            "/simulate",
            json=simulate_body(model_params=drifty, n_simulations=4000),
        ).json()["summary"]["mean"]

        assert mean == pytest.approx(100 * 2.718281828**0.30, rel=0.03)


class TestLimits:
    def test_over_the_path_budget_is_413(self, client):
        response = client.post("/simulate", json=simulate_body(n_simulations=100_001))

        assert response.status_code == 413
        body = response.json()
        assert body["error"] == "simulation_too_large"
        assert body["context"]["limit"] == 100_000

    def test_exactly_at_the_budget_is_allowed(self, client):
        """The limit is inclusive — 100k is a legal request, 100k+1 is not."""
        response = client.post(
            "/simulate", json=simulate_body(n_simulations=100_000, T=0.02)
        )

        assert response.status_code == 200
