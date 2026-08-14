"""Risk analytics endpoints.

Everything here runs under the *physical* measure — the drift stays whatever
calibration produced. VaR, CVaR and drawdown are statements about what the
asset is actually likely to do, so replacing the drift with r - q, as pricing
does, would answer a question nobody asked.

``r`` still appears in the requests, but only as the Sharpe ratio's benchmark
return. It never reaches a simulator.
"""
from __future__ import annotations

import asyncio

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import Engine

from src.api.dependencies import get_engine, get_simulator
from src.api.schemas import (
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    RiskMetricsRequest,
    RiskMetricsResponse,
    ScenarioFamily,
    ScenarioListResponse,
    ScenarioVariant,
)
from src.api.utils import confidence_key, enforce_simulation_budget, load_recent_closes
from src.calibration.utils import log_returns
from src.risk import (
    PRESET_SCENARIOS,
    PortfolioAsset,
    RiskMetrics,
    Scenario,
    apply_scenario,
    compute_risk_metrics,
    estimate_correlation_matrix,
    get_preset_scenario,
    simulate_portfolio,
)

router = APIRouter(prefix="/risk", tags=["risk"])

# The portfolio paths are normalized to start here, so P&L reads directly as a
# fraction of capital and assets of very different price levels contribute
# strictly in proportion to their weights.
PORTFOLIO_INITIAL_VALUE = 1.0


def _to_response(metrics: RiskMetrics) -> RiskMetricsResponse:
    """Re-key the engine's float-keyed dicts for JSON, sign convention intact."""
    return RiskMetricsResponse(
        var={confidence_key(level): value for level, value in metrics.var.items()},
        cvar={confidence_key(level): value for level, value in metrics.cvar.items()},
        max_drawdown={
            "mean": metrics.max_drawdown_mean,
            "percentile_95": metrics.max_drawdown_95th,
        },
        sharpe_ratio=metrics.sharpe_ratio,
        probability_of_loss=metrics.prob_of_loss,
        n_simulations=metrics.n_simulations,
    )


@router.post("/metrics", response_model=RiskMetricsResponse)
async def risk_metrics(request: RiskMetricsRequest) -> RiskMetricsResponse:
    """Single-asset VaR, CVaR, drawdown, Sharpe and probability of loss.

    Pass a scenario to get the stressed version of the same run: seed the two
    calls identically and the difference between them is the shock alone, not
    Monte Carlo noise.
    """
    enforce_simulation_budget(request.n_simulations)

    params = request.model_params
    scenario = _resolve_scenario(request, params.model_type)
    if scenario is not None:
        params = apply_scenario(params, scenario)

    simulator = get_simulator(params.model_type)
    paths = await asyncio.to_thread(
        simulator.simulate,
        request.S0,
        params,
        request.T,
        request.dt,
        request.n_simulations,
        seed=request.seed,
        variance_reduction=request.variance_reduction,
    )

    metrics = compute_risk_metrics(
        paths, request.S0, request.r, request.T, request.confidence_levels
    )
    return _to_response(metrics)


def _aligned_returns(
    tickers: list[str], lookback_days: int, engine: Engine
) -> dict[str, np.ndarray]:
    """Log returns for each ticker over the dates all of them share.

    estimate_correlation_matrix can only check that the series are the same
    *length*; lining them up by date is the caller's job, and doing it on the
    intersection of the trading calendars is the only way a correlation across
    tickers with different histories means anything.
    """
    closes_by_date: dict[str, dict] = {}
    for ticker in tickers:
        bars, _ = load_recent_closes(ticker, lookback_days, engine)
        closes_by_date[ticker] = {bar.date: bar.close for bar in bars}

    shared = set.intersection(*(set(series) for series in closes_by_date.values()))
    if len(shared) < 2:
        raise ValueError(
            "The requested tickers share fewer than two dates of stored history; "
            "a correlation cannot be estimated from them"
        )

    dates = sorted(shared)
    return {
        ticker: log_returns(np.array([series[day] for day in dates], dtype=float))
        for ticker, series in closes_by_date.items()
    }


def _resolve_scenario(
    request: RiskMetricsRequest | PortfolioRiskRequest, model_type: str
) -> Scenario | None:
    """The scenario to apply to one set of params, given its model family.

    A scenario names parameters, so it is model-specific: the preset lookup
    needs to know which family it is being applied to, and a custom shock
    naming ``v0`` will be rejected for a GBM asset by apply_scenario. That
    rejection is the honest outcome — the alternative is silently stressing
    only part of a mixed portfolio.
    """
    if request.scenario is not None and request.custom_scenario is not None:
        raise ValueError("Pass either 'scenario' or 'custom_scenario', not both")

    if request.custom_scenario is not None:
        return Scenario(**request.custom_scenario)

    if request.scenario is not None:
        return get_preset_scenario(request.scenario, model_type)

    return None


@router.post("/portfolio", response_model=PortfolioRiskResponse)
async def portfolio_risk(
    request: PortfolioRiskRequest, engine: Engine = Depends(get_engine)
) -> PortfolioRiskResponse:
    """Risk metrics for a weighted basket under an estimated correlation.

    Correlations come from stored history; the forward simulation comes from
    the per-asset params in the request. Assets are reordered alphabetically to
    match the row/column order estimate_correlation_matrix returns, since
    simulate_portfolio cannot detect a mismatched ordering itself.
    """
    enforce_simulation_budget(request.n_simulations)

    tickers = [asset.ticker for asset in request.assets]
    returns = await asyncio.to_thread(
        _aligned_returns, tickers, request.lookback_days, engine
    )
    correlation_matrix, ordered_tickers = estimate_correlation_matrix(returns)

    by_ticker = {asset.ticker: asset for asset in request.assets}
    assets = []
    for ticker in ordered_tickers:
        asset = by_ticker[ticker]
        params = asset.model_params
        scenario = _resolve_scenario(request, params.model_type)
        if scenario is not None:
            params = apply_scenario(params, scenario)

        assets.append(
            PortfolioAsset(
                ticker=ticker,
                weight=asset.weight,
                S0=asset.S0,
                simulator=get_simulator(params.model_type),
                params=params,
            )
        )

    result = await asyncio.to_thread(
        simulate_portfolio,
        assets,
        correlation_matrix,
        request.T,
        request.dt,
        request.n_simulations,
        PORTFOLIO_INITIAL_VALUE,
        request.seed,
    )

    metrics = compute_risk_metrics(
        result.portfolio_paths,
        PORTFOLIO_INITIAL_VALUE,
        request.r,
        request.T,
        request.confidence_levels,
    )

    scenario_applied = request.scenario
    if request.custom_scenario is not None:
        scenario_applied = Scenario(**request.custom_scenario).name

    return PortfolioRiskResponse(
        risk_metrics=_to_response(metrics),
        correlation_matrix=result.correlation_matrix.tolist(),
        tickers=result.tickers,
        scenario_applied=scenario_applied,
    )


@router.get("/scenarios", response_model=ScenarioListResponse)
async def list_scenarios() -> ScenarioListResponse:
    """Preset stress scenarios, grouped by family.

    A scenario names model parameters, so each family exists only for the
    model types it has a variant for — ``v0`` means nothing to GBM. The
    variants are listed so a client can tell which of its assets a scenario
    can actually be applied to.
    """
    families = []
    for key, by_model in PRESET_SCENARIOS.items():
        variants = [
            ScenarioVariant(
                model_type=model_type,
                name=scenario.name,
                description=scenario.description,
            )
            for model_type, scenario in sorted(by_model.items())
        ]
        families.append(
            ScenarioFamily(
                key=key,
                name=variants[0].name,
                model_types=[variant.model_type for variant in variants],
                variants=variants,
            )
        )

    return ScenarioListResponse(scenarios=families)
