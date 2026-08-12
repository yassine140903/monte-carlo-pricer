"""Monte Carlo option pricing on top of any simulator.

MCPricer takes a SimulatorBase by injection and never imports a concrete
simulator, so the same pricing path serves GBM, jump-diffusion and Heston.

Pricing happens under the risk-neutral measure: the calibrated real-world
drift in ``model_params`` is overridden with ``mu = r - q`` for the duration
of the simulation. The params object itself is never mutated.
"""
from __future__ import annotations

import time

import numpy as np
from pydantic import BaseModel

from src.pricing.black_scholes import bs_call, bs_put
from src.pricing.payoffs import PAYOFF_REGISTRY
from src.simulation.base import SimulatorBase

_BENCHMARKED_TYPES = {"european_call": bs_call, "european_put": bs_put}


class PricingResult(BaseModel):
    price: float
    std_error: float
    confidence_interval_95: tuple[float, float]
    n_simulations: int
    computation_time_ms: float
    bs_benchmark: float | None = None
    bs_relative_error: float | None = None


class MCPricer:
    def __init__(self, simulator: SimulatorBase) -> None:
        self.simulator = simulator

    def price(
        self,
        option_type: str,
        S0: float,
        K: float | None,
        T: float,
        r: float,
        model_params: BaseModel,
        n_simulations: int = 50_000,
        dt: float = 1 / 252,
        q: float = 0.0,
        seed: int | None = None,
        variance_reduction: str | None = None,
        **payoff_kwargs,
    ) -> PricingResult:
        """Price ``option_type`` by Monte Carlo under the risk-neutral measure.

        ``K`` is passed to the payoff only when it is not None, so floating
        strike lookbacks can be priced with ``K=None``. Extra payoff arguments
        (``barrier=...``) come through ``payoff_kwargs``.
        """
        payoff_fn = self._lookup(option_type)
        started = time.perf_counter()

        paths = self.simulator.simulate(
            S0,
            model_params,
            T,
            dt,
            n_simulations,
            seed=seed,
            variance_reduction=variance_reduction,
            mu=r - q,
        )

        kwargs = dict(payoff_kwargs)
        if K is not None:
            kwargs["K"] = K
        payoffs = payoff_fn(paths, **kwargs)

        discount = np.exp(-r * T)
        price = float(discount * np.mean(payoffs))
        std_error = discount * self._payoff_standard_error(payoffs, variance_reduction)
        elapsed_ms = (time.perf_counter() - started) * 1_000

        benchmark, relative_error = self._benchmark(
            option_type, price, S0, K, T, r, model_params, q
        )

        return PricingResult(
            price=price,
            std_error=std_error,
            confidence_interval_95=(price - 1.96 * std_error, price + 1.96 * std_error),
            n_simulations=n_simulations,
            computation_time_ms=elapsed_ms,
            bs_benchmark=benchmark,
            bs_relative_error=relative_error,
        )

    @staticmethod
    def _payoff_standard_error(payoffs: np.ndarray, variance_reduction: str | None) -> float:
        """Standard error of the mean payoff, honouring the sampling scheme.

        The textbook std(payoffs, ddof=1)/sqrt(n) assumes independent draws.
        Variance reduction deliberately breaks that:

        - antithetic — the estimator is really the mean of n/2 *independent*
          pair averages, and each pair is negatively correlated internally.
          Treating the 2n halves as independent ignores that correlation and
          reports an error roughly 1.8x too large, hiding the entire benefit.
          The pair-average estimator below is exact.
        - stratified — one draw per stratum, so the draws are negatively
          dependent by construction and no simple unbiased estimator exists.
          The i.i.d. formula is kept, which *overstates* the error (~2x in
          practice). Conservative rather than wrong-way, but it means a
          stratified CI is wider than the run actually earned.
        """
        n = len(payoffs)
        if n < 2:
            return 0.0

        if variance_reduction == "antithetic":
            # mirrors the pairing in variance_reduction.generate_samples:
            # row i is paired with row i + half. An odd n leaves a tail row
            # unpaired, which is simply excluded from the variance estimate.
            half = (n + 1) // 2
            n_pairs = n - half
            if n_pairs >= 2:
                pair_means = 0.5 * (payoffs[:n_pairs] + payoffs[half : half + n_pairs])
                return float(np.std(pair_means, ddof=1) / np.sqrt(n_pairs))

        return float(np.std(payoffs, ddof=1) / np.sqrt(n))

    @staticmethod
    def _lookup(option_type: str):
        try:
            return PAYOFF_REGISTRY[option_type]
        except KeyError:
            raise ValueError(
                f"Unknown option_type {option_type!r}; "
                f"expected one of {sorted(PAYOFF_REGISTRY)}"
            ) from None

    @staticmethod
    def _benchmark(
        option_type: str,
        price: float,
        S0: float,
        K: float | None,
        T: float,
        r: float,
        model_params: BaseModel,
        q: float,
    ) -> tuple[float | None, float | None]:
        """Closed-form reference for vanillas on a model that has a ``sigma``.

        Only exact for GBM. Jump-diffusion also carries a ``sigma``, but
        Black-Scholes cannot see its jumps, so there the benchmark is a
        reference point rather than the truth — expect a real gap.
        Heston has no ``sigma`` attribute and is skipped.
        """
        formula = _BENCHMARKED_TYPES.get(option_type)
        sigma = getattr(model_params, "sigma", None)
        if formula is None or sigma is None or K is None:
            return None, None

        benchmark = formula(S0, K, T, r, sigma, q)
        if benchmark <= 0:
            # deep out of the money: a relative error against ~0 is noise
            return benchmark, None
        return benchmark, abs(price - benchmark) / benchmark
