"""MLflow experiment listing.

Deliberately shallow: enough for a client to show what has been logged and
link out. Browsing runs, comparing them and reading artifacts is what the
MLflow UI already does well, and reimplementing it behind this API would mean
maintaining a second, worse version of it.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas import ExperimentListResponse, ExperimentSummary
from src.config import settings

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

logger = logging.getLogger(__name__)


def _list_experiments() -> list[ExperimentSummary]:
    import mlflow

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    summaries = []
    for experiment in mlflow.search_experiments():
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], output_format="list"
        )
        summaries.append(
            ExperimentSummary(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                run_count=len(runs),
            )
        )
    return summaries


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments() -> ExperimentListResponse:
    """Experiments on the configured tracking server.

    An unreachable tracking server yields an empty list rather than an error:
    MLflow is an optional companion to this service, and a dashboard should
    not go down because nobody started it.
    """
    try:
        experiments = await asyncio.to_thread(_list_experiments)
    except Exception as exc:  # noqa: BLE001 - MLflow is optional infrastructure
        logger.warning("Could not reach MLflow at %s: %s", settings.MLFLOW_TRACKING_URI, exc)
        experiments = []

    return ExperimentListResponse(experiments=experiments)
