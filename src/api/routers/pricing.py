"""Option pricing endpoint.

Runs under the *risk-neutral* measure. MCPricer does that override itself —
it simulates with mu = r - q regardless of the calibrated drift — so this
router never touches the drift; it only has to make sure the payoff arguments
and the histogram simulation agree with what the pricer did.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from fastapi import APIRouter

from src.api.dependencies import get_simulator
from src.api.schemas import PriceOptionRequest, PricingResponse
from src.api.utils import compute_histogram, enforce_simulation_budget, log_to_mlflow
from src.pricing import PAYOFF_REGISTRY, GreeksCalculator, MCPricer, bs_greeks
from src.pricing.black_scholes import bs_call, bs_put

router = APIRouter(tags=["pricing"])

# The two payoffs Black-Scholes prices exactly, and only under GBM — a
# jump-diffusion sigma looks like a Black-Scholes sigma but the closed form
# cannot see the jumps, so quoting it as a benchmark would invite the gap to
# be read as Monte Carlo error.
_BS_FORMULAS = {"european_call": (bs_call, "call"), "european_put": (bs_put, "put")}

# Greeks are bump-and-revalue with common random numbers, which needs a fixed
# seed to work at all. A request that did not pin one still gets a consistent
# set rather than nine independent samples differenced against each other.
_DEFAULT_GREEKS_SEED = 42


def _price_bundle(request: PriceOptionRequest) -> tuple:
    """Price, Greeks and a payoff sample — the whole blocking part, in one go.

    Called through ``asyncio.to_thread``. Keeping it a single function means
    one hop onto the worker thread instead of three, and keeps the argument
    plumbing (which payoff takes a strike, which takes a barrier) in one place.
    """
    simulator = get_simulator(request.model_params.model_type)
    pricer = MCPricer(simulator)

    payoff_kwargs = {} if request.barrier is None else {"barrier": request.barrier}
    common = dict(
        option_type=request.option_type,
        S0=request.S0,
        K=request.K,
        T=request.T,
        r=request.r,
        model_params=request.model_params,
        n_simulations=request.n_simulations,
        dt=request.dt,
        q=request.q,
        **payoff_kwargs,
    )

    result = pricer.price(
        **common, seed=request.seed, variance_reduction=request.variance_reduction
    )
    greeks = GreeksCalculator(pricer).compute(
        **common, seed=request.seed if request.seed is not None else _DEFAULT_GREEKS_SEED
    )

    # A second risk-neutral run purely to shape the payoff distribution: the
    # pricer returns the mean and its error, not the sample behind them. With
    # a seed given this reproduces the very paths that were priced; without
    # one it is an independent draw from the same distribution, which is all
    # a histogram needs.
    paths = simulator.simulate(
        request.S0,
        request.model_params,
        request.T,
        request.dt,
        request.n_simulations,
        seed=request.seed,
        variance_reduction=request.variance_reduction,
        mu=request.r - request.q,
    )
    kwargs = dict(payoff_kwargs)
    if request.K is not None:
        kwargs["K"] = request.K
    payoffs = PAYOFF_REGISTRY[request.option_type](paths, **kwargs)

    return result, greeks, payoffs


def _bs_benchmark(request: PriceOptionRequest, mc_price: float) -> dict | None:
    """Closed-form price and Greeks, or None where they do not apply."""
    formula_entry = _BS_FORMULAS.get(request.option_type)
    if formula_entry is None or request.model_params.model_type != "gbm":
        return None

    formula, flavour = formula_entry
    sigma = request.model_params.sigma
    price = formula(request.S0, request.K, request.T, request.r, sigma, request.q)
    greeks = bs_greeks(
        request.S0, request.K, request.T, request.r, sigma, request.q, flavour
    )

    return {
        "price": price,
        "greeks": greeks,
        # Deep out of the money the benchmark is ~0 and a relative error
        # against it is noise rather than a diagnostic.
        "relative_error": abs(mc_price - price) / price if price > 0 else None,
    }


@router.post("/price-option", response_model=PricingResponse)
async def price_option(request: PriceOptionRequest) -> PricingResponse:
    enforce_simulation_budget(request.n_simulations)

    started = time.perf_counter()
    result, greeks, payoffs = await asyncio.to_thread(_price_bundle, request)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    greeks_dict = greeks.model_dump()
    run_id = log_to_mlflow(
        experiment_name=f"pricing/{request.option_type}",
        params={
            "option_type": request.option_type,
            "S0": request.S0,
            "K": request.K,
            "T": request.T,
            "r": request.r,
            "model_type": request.model_params.model_type,
            "n_simulations": request.n_simulations,
        },
        metrics={"price": result.price, "std_error": result.std_error, **greeks_dict},
    )

    return PricingResponse(
        price=result.price,
        std_error=result.std_error,
        confidence_interval_95=result.confidence_interval_95,
        greeks=greeks_dict,
        bs_benchmark=_bs_benchmark(request, result.price),
        payoff_histogram=compute_histogram(np.asarray(payoffs)),
        n_simulations=request.n_simulations,
        computation_time_ms=elapsed_ms,
        mlflow_run_id=run_id,
    )
