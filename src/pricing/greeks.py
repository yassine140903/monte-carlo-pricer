"""Monte Carlo Greeks by bump-and-revalue with common random numbers.

Every repricing runs on the same seed, so the bumped and base simulations use
identical draws. Without that the Monte Carlo noise — orders of magnitude
larger than the effect of a 1% bump — swamps the finite difference entirely.
This is why ``seed`` is required here rather than optional.

Bump sizes follow the locked M4 spec: S0 +/-1%, sigma +/-0.01 absolute
(v0 for Heston), T -1/252 forward difference, r +/-10bp.

Units match src/pricing/black_scholes.py: vega per 1.0 of volatility, theta
per year, rho per 1.0 of rate.
"""
from __future__ import annotations

from pydantic import BaseModel

from src.pricing.pricer import MCPricer
from src.simulation.utils import resolve_n_steps

_SPOT_BUMP_FRACTION = 0.01
_VOL_BUMP = 0.01
_RATE_BUMP = 0.001
_THETA_STEP = 1 / 252


class GreeksResult(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class GreeksCalculator:
    def __init__(self, pricer: MCPricer) -> None:
        self.pricer = pricer

    def compute(
        self,
        option_type: str,
        S0: float,
        K: float | None,
        T: float,
        r: float,
        model_params: BaseModel,
        n_simulations: int = 50_000,
        seed: int = 42,
        q: float = 0.0,
        dt: float = 1 / 252,
        **payoff_kwargs,
    ) -> GreeksResult:
        def reprice(
            *,
            S0: float = S0,
            T: float = T,
            r: float = r,
            params: BaseModel = model_params,
            dt: float = dt,
        ) -> float:
            return self.pricer.price(
                option_type,
                S0,
                K,
                T,
                r,
                params,
                n_simulations=n_simulations,
                dt=dt,
                q=q,
                seed=seed,
                **payoff_kwargs,
            ).price

        base = reprice()

        spot_bump = _SPOT_BUMP_FRACTION * S0
        price_up = reprice(S0=S0 + spot_bump)
        price_down = reprice(S0=S0 - spot_bump)

        delta = (price_up - price_down) / (2 * spot_bump)
        gamma = (price_up - 2 * base + price_down) / spot_bump**2

        vol_up = reprice(params=self._bump_volatility(model_params, _VOL_BUMP))
        vol_down = reprice(params=self._bump_volatility(model_params, -_VOL_BUMP))
        vega = (vol_up - vol_down) / (2 * _VOL_BUMP)

        rate_up = reprice(r=r + _RATE_BUMP)
        rate_down = reprice(r=r - _RATE_BUMP)
        rho = (rate_up - rate_down) / (2 * _RATE_BUMP)

        # Forward difference only — there is no such thing as bumping time
        # forward past expiry. The shortened horizon still has to be a horizon
        # the simulator will accept, i.e. at least one step of size dt, so the
        # guard tests T - _THETA_STEP against dt rather than against zero.
        shortened = T - _THETA_STEP
        if shortened >= dt:
            theta = (reprice(T=shortened, dt=self._matching_dt(T, dt, shortened)) - base) / (
                _THETA_STEP
            )
        else:
            theta = 0.0

        return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    @staticmethod
    def _matching_dt(T: float, dt: float, shortened_T: float) -> float:
        """The step size that gives ``shortened_T`` the same step *count* as T.

        Theta is the one bump that moves T, and the simulators derive
        n_steps = round(T/dt) and draw an (n_simulations, n_steps) block of
        normals. Shortening T by a day normally drops n_steps by one, which
        reshapes that block and reshuffles every draw after the first row —
        so the two prices are effectively independent samples, and dividing
        their difference by 1/252 amplifies the noise 252-fold. Theta then
        comes out as noise that happens to have the right sign.

        Holding the step count fixed and shrinking dt instead keeps the draws
        bit-identical, which is the whole point of common random numbers. The
        cost is a marginally finer grid on the bumped run — immaterial for
        GBM, which is exact at any step size, and far smaller than the noise
        it removes for the discretized models.
        """
        return shortened_T / resolve_n_steps(T, dt)

    @staticmethod
    def _bump_volatility(params: BaseModel, bump: float) -> BaseModel:
        """A copy of ``params`` with its volatility shifted by ``bump``.

        GBM and jump-diffusion carry ``sigma`` directly. Heston's spot
        volatility lives in ``v0``, which is a *variance*, so the bump is
        applied to sqrt(v0) and squared back — a 1 vol point bump there means
        the same thing it does for GBM. Floored at zero, since a negative
        variance is not a state the simulator can start from.
        """
        if hasattr(params, "sigma"):
            return params.model_copy(update={"sigma": max(params.sigma + bump, 1e-8)})
        if hasattr(params, "v0"):
            bumped_vol = max(params.v0**0.5 + bump, 0.0)
            return params.model_copy(update={"v0": bumped_vol**2})
        raise TypeError(
            f"{type(params).__name__} has neither 'sigma' nor 'v0' to bump for vega"
        )
