"""Path simulation endpoint.

Runs under the *physical* measure: the drift is whatever the calibration
produced, untouched. This endpoint answers "where might this asset actually
go", which is a different question from the risk-neutral one /price-option
asks, and overriding the drift with r - q here would silently turn the fan
chart into a pricing artefact.

The full path array is never returned — tens of thousands of paths at daily
resolution is tens of megabytes of JSON. What goes back is the terminal
distribution summarised, cross-sectional percentile bands, and a handful of
individual paths for texture, all thinned to the same ~50 dates.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from fastapi import APIRouter

from src.api.dependencies import get_simulator
from src.api.schemas import SimulateRequest, SimulationResponse
from src.api.utils import (
    downsample,
    enforce_simulation_budget,
    percentile_bands,
    terminal_summary,
)

router = APIRouter(tags=["simulation"])

N_SAMPLE_PATHS = 5


@router.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulateRequest) -> SimulationResponse:
    enforce_simulation_budget(request.n_simulations)

    simulator = get_simulator(request.model_params.model_type)

    started = time.perf_counter()
    paths = await asyncio.to_thread(
        simulator.simulate,
        request.S0,
        request.model_params,
        request.T,
        request.dt,
        request.n_simulations,
        seed=request.seed,
        variance_reduction=request.variance_reduction,
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000

    n_steps = paths.shape[1] - 1
    time_axis = np.arange(n_steps + 1, dtype=float) * request.dt

    # Sampled from the same Generator seed as the run itself where one was
    # given, so an identical request returns an identical picture — including
    # which paths were picked out of the fan.
    rng = np.random.default_rng(request.seed)
    sample_indices = rng.choice(
        paths.shape[0], size=min(N_SAMPLE_PATHS, paths.shape[0]), replace=False
    )

    return SimulationResponse(
        summary=terminal_summary(paths),
        percentile_bands=percentile_bands(paths),
        sample_paths=downsample(paths[sample_indices]).tolist(),
        time_axis=downsample(time_axis).tolist(),
        n_simulations=request.n_simulations,
        computation_time_ms=elapsed_ms,
    )
