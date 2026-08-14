"""Model calibration endpoint.

Loads a stored price history, hands it to the calibrator for the requested
model family, and reports the fitted params alongside the goodness-of-fit
diagnostics. The fit itself runs in a worker thread — it is a SciPy optimizer
for two of the three models, and blocking the event loop on it would stall
every other request in the process.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy import Engine

from src.api.dependencies import get_calibrator, get_engine
from src.api.schemas import CalibrateRequest, CalibrateResponse
from src.api.utils import load_recent_closes, log_to_mlflow

router = APIRouter(tags=["calibration"])


@router.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(
    request: CalibrateRequest, engine: Engine = Depends(get_engine)
) -> CalibrateResponse:
    """Fit ``request.model_type`` to the last ``lookback_days`` of closes.

    ``window_days``, when given, narrows the GBM fit further to the most
    recent ``window_days`` observations — a shorter estimation window tracks
    the current volatility regime more closely at the cost of a noisier
    estimate. It is ignored for the other models, whose calibrators fit a term
    structure or a likelihood over the whole sample and have no equivalent.
    """
    _, closes = load_recent_closes(request.ticker, request.lookback_days, engine)

    if request.model_type == "gbm" and request.window_days is not None:
        closes = closes[-request.window_days :]

    calibrator = get_calibrator(request.model_type)
    params = await asyncio.to_thread(calibrator.calibrate, closes)
    fit_metrics = calibrator.goodness_of_fit()

    run_id = log_to_mlflow(
        experiment_name=f"calibration/{request.ticker}",
        params={
            "ticker": request.ticker,
            "model_type": request.model_type,
            "lookback_days": request.lookback_days,
            "window_days": request.window_days,
            "n_observations": len(closes),
        },
        metrics=fit_metrics,
    )

    return CalibrateResponse(
        model_type=request.model_type,
        params=params,
        fit_metrics=fit_metrics,
        mlflow_run_id=run_id,
    )
