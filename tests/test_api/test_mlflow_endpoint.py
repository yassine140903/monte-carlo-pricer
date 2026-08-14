"""GET /mlflow/experiments."""
from __future__ import annotations


class TestExperimentListing:
    def test_lists_experiments_logged_by_the_other_endpoints(self, client, local_mlflow):
        client.post(
            "/calibrate",
            json={"ticker": "AAPL", "model_type": "gbm", "lookback_days": 300},
        )

        response = client.get("/mlflow/experiments")

        assert response.status_code == 200
        experiments = response.json()["experiments"]
        by_name = {entry["name"]: entry for entry in experiments}
        assert "calibration/AAPL" in by_name
        assert by_name["calibration/AAPL"]["run_count"] >= 1

    def test_an_unusable_tracking_uri_yields_an_empty_list_not_an_error(
        self, client, monkeypatch
    ):
        """MLflow is optional infrastructure — a dashboard should not go down
        because nobody started the tracking server."""
        from src.api.routers import mlflow_runs

        monkeypatch.setattr(
            mlflow_runs.settings, "MLFLOW_TRACKING_URI", "nonsense://nowhere"
        )

        response = client.get("/mlflow/experiments")

        assert response.status_code == 200
        assert response.json() == {"experiments": []}
