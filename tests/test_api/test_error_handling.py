"""How domain errors become HTTP statuses.

The mapping is the one thing the API layer genuinely owns, so it is tested
directly rather than only as a side effect of the endpoint tests.
"""
from __future__ import annotations

import pytest

from src.api.dependencies import get_calibrator, get_simulator
from tests.test_api.conftest import GBM_PARAMS


def error_body(response) -> dict:
    body = response.json()
    assert set(body) == {"error", "detail", "context"}
    return body


class TestHealth:
    def test_health_reports_healthy(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestInvalidModelType:
    def test_an_unknown_discriminator_tag_is_422(self, client):
        response = client.post(
            "/simulate",
            json={
                "S0": 100.0,
                "model_params": {"model_type": "quantum", "mu": 0.1, "sigma": 0.2},
                "T": 1.0,
                "n_simulations": 100,
            },
        )

        assert response.status_code == 422
        assert error_body(response)["error"] == "validation_error"

    def test_params_that_do_not_match_their_tag_are_422(self, client):
        """Tagged gbm but carrying Heston fields — validated against GBM alone,
        which is the point of discriminating on model_type."""
        response = client.post(
            "/simulate",
            json={
                "S0": 100.0,
                "model_params": {"model_type": "gbm", "kappa": 2.0, "theta": 0.04},
                "T": 1.0,
                "n_simulations": 100,
            },
        )

        assert response.status_code == 422

    def test_an_unknown_calibration_model_type_is_422(self, client):
        response = client.post(
            "/calibrate", json={"ticker": "AAPL", "model_type": "quantum"}
        )

        assert response.status_code == 422

    def test_the_factories_reject_unknown_types_with_ValueError(self):
        """The ValueError the 422 handler keys on, at its source."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_simulator("quantum")
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_calibrator("quantum")


class TestUnknownTicker:
    @pytest.mark.parametrize("path", ["/data/NOSUCH"])
    def test_data_lookup_is_404(self, client, path):
        response = client.get(path)

        assert response.status_code == 404
        body = error_body(response)
        assert body["error"] == "not_found"
        # The KeyError message survives, without KeyError's own quoting.
        assert body["detail"].startswith("No stored data")

    def test_calibration_is_404(self, client):
        response = client.post(
            "/calibrate", json={"ticker": "NOSUCH", "model_type": "gbm"}
        )

        assert response.status_code == 404

    def test_portfolio_is_404(self, client):
        response = client.post(
            "/risk/portfolio",
            json={
                "assets": [
                    {"ticker": "AAPL", "weight": 0.5, "S0": 180.0, "model_params": GBM_PARAMS},
                    {"ticker": "NOSUCH", "weight": 0.5, "S0": 100.0, "model_params": GBM_PARAMS},
                ],
                "T": 1.0,
                "n_simulations": 500,
            },
        )

        assert response.status_code == 404


class TestOversizedSimulation:
    @pytest.mark.parametrize(
        ("path", "body"),
        [
            (
                "/simulate",
                {"S0": 100.0, "model_params": GBM_PARAMS, "T": 1.0, "n_simulations": 100_001},
            ),
            (
                "/risk/metrics",
                {"S0": 100.0, "model_params": GBM_PARAMS, "T": 1.0, "n_simulations": 250_000},
            ),
            (
                "/price-option",
                {
                    "option_type": "european_call",
                    "S0": 100.0,
                    "K": 100.0,
                    "T": 1.0,
                    "r": 0.03,
                    "model_params": GBM_PARAMS,
                    "n_simulations": 500_000,
                },
            ),
        ],
    )
    def test_is_413_everywhere_paths_are_simulated(self, client, path, body):
        response = client.post(path, json=body)

        assert response.status_code == 413
        payload = error_body(response)
        assert payload["error"] == "simulation_too_large"
        assert payload["context"] == {
            "n_simulations": body["n_simulations"],
            "limit": 100_000,
        }

    def test_the_check_runs_before_any_paths_are_drawn(self, client):
        """A 413 must be cheap — it is refusing the work, not doing it first."""
        import time

        started = time.perf_counter()
        client.post(
            "/simulate",
            json={
                "S0": 100.0,
                "model_params": GBM_PARAMS,
                "T": 30.0,
                "n_simulations": 10_000_000,
            },
        )

        assert time.perf_counter() - started < 2.0


class TestDomainValueErrors:
    def test_a_horizon_shorter_than_one_step_is_422(self, client):
        """resolve_n_steps raises ValueError; the handler turns it into 422."""
        response = client.post(
            "/simulate",
            json={
                "S0": 100.0,
                "model_params": GBM_PARAMS,
                "T": 0.0001,
                "dt": 1.0,
                "n_simulations": 100,
            },
        )

        assert response.status_code == 422
        assert error_body(response)["error"] == "invalid_request"

    def test_a_jump_intensity_too_high_for_the_step_size_is_422(self, client):
        """The single-jump-per-step approximation guard in the simulator."""
        response = client.post(
            "/simulate",
            json={
                "S0": 100.0,
                "model_params": {
                    "model_type": "jump_diffusion",
                    "mu": 0.05,
                    "sigma": 0.2,
                    "lambda_j": 100.0,
                    "mu_j": -0.02,
                    "sigma_j": 0.05,
                },
                "T": 1.0,
                "n_simulations": 100,
            },
        )

        assert response.status_code == 422
        assert "single-jump-per-step" in error_body(response)["detail"]

    def test_a_negative_spot_is_rejected_by_the_schema(self, client):
        response = client.post(
            "/simulate",
            json={
                "S0": -100.0,
                "model_params": GBM_PARAMS,
                "T": 1.0,
                "n_simulations": 100,
            },
        )

        assert response.status_code == 422


class TestNotFoundRouting:
    def test_an_unrouted_path_still_returns_the_error_shape(self, client):
        response = client.get("/does-not-exist")

        assert response.status_code == 404
        assert error_body(response)["error"] == "not_found"
