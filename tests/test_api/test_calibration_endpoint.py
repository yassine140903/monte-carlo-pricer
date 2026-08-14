"""POST /calibrate."""
from __future__ import annotations

import pytest

FIT_METRIC_KEYS = {
    "log_likelihood",
    "aic",
    "bic",
    "jarque_bera_stat",
    "jarque_bera_pvalue",
}

EXPECTED_PARAM_KEYS = {
    "gbm": {"model_type", "mu", "sigma"},
    "jump_diffusion": {
        "model_type",
        "mu",
        "sigma",
        "lambda_j",
        "mu_j",
        "sigma_j",
    },
    "heston": {
        "model_type",
        "kappa",
        "theta",
        "xi",
        "rho",
        "v0",
        "feller_satisfied",
    },
}


@pytest.mark.parametrize("model_type", ["gbm", "jump_diffusion", "heston"])
class TestCalibrateEachModel:
    def test_returns_the_params_for_that_model(self, client, model_type):
        response = client.post(
            "/calibrate",
            json={"ticker": "AAPL", "model_type": model_type, "lookback_days": 500},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == model_type
        assert body["params"]["model_type"] == model_type
        assert set(body["params"]) == EXPECTED_PARAM_KEYS[model_type]

    def test_reports_goodness_of_fit(self, client, model_type):
        body = client.post(
            "/calibrate",
            json={"ticker": "AAPL", "model_type": model_type, "lookback_days": 500},
        ).json()

        assert set(body["fit_metrics"]) == FIT_METRIC_KEYS
        assert all(isinstance(v, float) for v in body["fit_metrics"].values())


class TestCalibrateBehaviour:
    def test_fitted_gbm_params_are_plausible(self, client):
        params = client.post(
            "/calibrate", json={"ticker": "AAPL", "model_type": "gbm"}
        ).json()["params"]

        # The fixture data is built from ~1.2% daily moves, i.e. ~19% annualized.
        assert params["sigma"] == pytest.approx(0.19, abs=0.05)

    def test_lookback_days_narrows_the_sample(self, client):
        short = client.post(
            "/calibrate",
            json={"ticker": "AAPL", "model_type": "gbm", "lookback_days": 100},
        ).json()
        long = client.post(
            "/calibrate",
            json={"ticker": "AAPL", "model_type": "gbm", "lookback_days": 700},
        ).json()

        assert short["params"] != long["params"]

    def test_window_days_narrows_a_gbm_fit_further(self, client):
        body = {"ticker": "AAPL", "model_type": "gbm", "lookback_days": 500}
        full = client.post("/calibrate", json=body).json()
        windowed = client.post("/calibrate", json={**body, "window_days": 60}).json()

        assert windowed["params"]["sigma"] != full["params"]["sigma"]

    def test_ticker_is_case_insensitive(self, client):
        assert client.post(
            "/calibrate", json={"ticker": "aapl", "model_type": "gbm"}
        ).status_code == 200

    def test_unknown_ticker_is_404(self, client):
        response = client.post(
            "/calibrate", json={"ticker": "NOSUCH", "model_type": "gbm"}
        )

        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    def test_unknown_model_type_is_422(self, client):
        response = client.post(
            "/calibrate", json={"ticker": "AAPL", "model_type": "stochastic_vibes"}
        )

        assert response.status_code == 422

    def test_too_short_a_lookback_is_422(self, client):
        response = client.post(
            "/calibrate",
            json={"ticker": "AAPL", "model_type": "gbm", "lookback_days": 5},
        )

        assert response.status_code == 422
        assert "at least" in response.json()["detail"]


class TestMLflowLogging:
    def test_a_run_is_created_and_retrievable(self, client, local_mlflow):
        import mlflow

        run_id = client.post(
            "/calibrate",
            json={"ticker": "MSFT", "model_type": "gbm", "lookback_days": 400},
        ).json()["mlflow_run_id"]
        assert run_id is not None

        mlflow.set_tracking_uri(local_mlflow)
        run = mlflow.get_run(run_id)

        assert run.data.params["ticker"] == "MSFT"
        assert run.data.params["model_type"] == "gbm"
        assert run.data.params["lookback_days"] == "400"
        assert FIT_METRIC_KEYS <= set(run.data.metrics)

    def test_the_run_lands_in_a_per_ticker_experiment(self, client, local_mlflow):
        import mlflow

        run_id = client.post(
            "/calibrate", json={"ticker": "GOOGL", "model_type": "gbm"}
        ).json()["mlflow_run_id"]

        mlflow.set_tracking_uri(local_mlflow)
        experiment_id = mlflow.get_run(run_id).info.experiment_id

        assert mlflow.get_experiment(experiment_id).name == "calibration/GOOGL"

    def test_an_unusable_tracking_uri_does_not_fail_the_request(
        self, client, monkeypatch
    ):
        """MLflow is a companion, not a dependency: the calibration still ships.

        A bad *scheme* rather than a dead host, so the failure is immediate —
        pointing at a closed port would make the test wait out MLflow's HTTP
        retry schedule to prove the same thing.
        """
        from src.api import utils

        monkeypatch.setattr(utils.settings, "MLFLOW_TRACKING_URI", "nonsense://nowhere")

        response = client.post(
            "/calibrate", json={"ticker": "AAPL", "model_type": "gbm"}
        )

        assert response.status_code == 200
        assert response.json()["mlflow_run_id"] is None
