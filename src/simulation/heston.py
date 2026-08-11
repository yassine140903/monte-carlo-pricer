"""Heston stochastic-volatility path simulation via Andersen's Quadratic
Exponential (QE) scheme.

A plain Euler discretization of the CIR variance process goes negative
whenever the Feller condition is violated, which is exactly the regime the
calibrator in src/calibration/heston.py is allowed to return. The QE scheme
instead moment-matches the *exact* conditional distribution of v_{t+dt} given
v_t with one of two proxy distributions, chosen per path per step:

- psi <= psi_crit — a squared normal a*(b + Z)**2, which fits the moderately
  dispersed case (psi is the squared coefficient of variation of v_{t+dt}).
- psi > psi_crit — a point mass at zero plus an exponential tail, which fits
  the highly dispersed case where the conditional density spikes at zero.

Both proxies are non-negative by construction, so variance never needs
truncation and the price paths stay well defined. The switching rule uses
Andersen's recommended psi_crit = 1.5.

The scheme loops over time steps (each step depends on the previous variance)
but is fully vectorized across simulations.

Reference: Andersen (2008), "Simple and Efficient Simulation of the Heston
Stochastic Volatility Model".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.calibration.heston import HestonParams
from src.simulation.base import SimulatorBase
from src.simulation.utils import make_rng, resolve_n_steps
from src.simulation.variance_reduction import generate_samples

PSI_CRIT = 1.5

# Central discretization of the log-price integral (Andersen section 4.3).
_GAMMA_1 = 0.5
_GAMMA_2 = 0.5

_EPS = 1e-12


class HestonSimulator(SimulatorBase):
    def simulate(
        self,
        S0: float,
        params: HestonParams,
        T: float,
        dt: float,
        n_simulations: int,
        seed: int | None = None,
        variance_reduction: str | None = None,
        mu: float = 0.0,
    ) -> np.ndarray:
        """Price paths of shape (n_simulations, n_steps+1).

        HestonParams carries no drift — the calibrator fits the variance term
        structure only — so ``mu`` is passed separately and defaults to 0,
        i.e. the risk-neutral dynamics at a zero rate. Pricing code should
        pass the risk-free rate (less any dividend yield).
        """
        prices, _ = self.simulate_with_variance(
            S0, params, T, dt, n_simulations, seed, variance_reduction, mu
        )
        return prices

    def simulate_with_variance(
        self,
        S0: float,
        params: HestonParams,
        T: float,
        dt: float,
        n_simulations: int,
        seed: int | None = None,
        variance_reduction: str | None = None,
        mu: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Both the price paths and the variance paths that produced them.

        Same contract as ``simulate()``, but also returns the variance paths
        (same shape, column 0 is v0) for callers that need the volatility
        state itself — variance derivatives, diagnostics, or asserting
        non-negativity.
        """
        self._validate_inputs(S0, n_simulations)
        self._validate_params(params)
        n_steps = resolve_n_steps(T, dt)
        rng = make_rng(seed)

        # One draw block per step for the variance, one for the price. Drawing
        # them together means variance reduction applies to both: negating a
        # normal maps its uniform u to 1 - u, which is the antithetic pair for
        # the inverse-transform draws the variance step uses.
        samples = generate_samples(rng, (n_simulations, 2 * n_steps), variance_reduction)
        variance_uniforms = np.clip(norm.cdf(samples[:, :n_steps]), _EPS, 1.0 - _EPS)
        price_normals = samples[:, n_steps:]

        variance_paths = np.empty((n_simulations, n_steps + 1))
        log_increments = np.empty((n_simulations, n_steps))
        variance_paths[:, 0] = params.v0

        k0, k1, k2, k3, k4 = self._log_price_constants(params, dt)

        for step in range(n_steps):
            v_now = variance_paths[:, step]
            v_next = self._qe_variance_step(v_now, variance_uniforms[:, step], params, dt)
            variance_paths[:, step + 1] = v_next

            log_increments[:, step] = self._log_price_increment(
                v_now, v_next, price_normals[:, step], mu, dt, (k0, k1, k2, k3, k4)
            )

        prices = self._paths_from_log_increments(S0, log_increments)
        return prices, variance_paths

    # ------------------------------------------------------------------
    # QE variance step
    # ------------------------------------------------------------------

    @staticmethod
    def _qe_moments(
        v: np.ndarray, params: HestonParams, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact conditional mean and variance of v_{t+dt} given v_t = ``v``."""
        decay = np.exp(-params.kappa * dt)
        xi2 = params.xi**2

        m = params.theta + (v - params.theta) * decay
        s2 = (
            v * xi2 * decay / params.kappa * (1 - decay)
            + params.theta * xi2 / (2 * params.kappa) * (1 - decay) ** 2
        )
        return np.maximum(m, _EPS), np.maximum(s2, 0.0)

    @staticmethod
    def _quadratic_branch(m: np.ndarray, psi: np.ndarray, z: np.ndarray) -> np.ndarray:
        """v_next = a * (b + Z)**2, moment-matched to (m, psi). Valid for psi <= 2."""
        # Clipped so the branch stays finite where it will be discarded by the
        # np.where below (2/psi - 1 goes negative once psi > 2).
        psi = np.minimum(psi, PSI_CRIT)
        inv_psi = 2.0 / psi
        b2 = inv_psi - 1.0 + np.sqrt(inv_psi) * np.sqrt(inv_psi - 1.0)
        a = m / (1.0 + b2)
        return a * (np.sqrt(b2) + z) ** 2

    @staticmethod
    def _exponential_branch(m: np.ndarray, psi: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Point mass p at zero plus an exponential tail, matched to (m, psi).

        Valid for psi >= 1; below that p would go negative.
        """
        psi = np.maximum(psi, PSI_CRIT)
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        return np.where(u <= p, 0.0, np.log((1.0 - p) / (1.0 - u)) / beta)

    def _qe_variance_step(
        self,
        v: np.ndarray,
        u: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> np.ndarray:
        """One QE step for every simulation at once.

        A single uniform drives both branches — inverted to a normal for the
        quadratic one — so switching branch does not change how many random
        numbers a path consumes and paths stay reproducible step by step.
        """
        m, s2 = self._qe_moments(v, params, dt)
        psi = s2 / m**2

        quadratic = self._quadratic_branch(m, psi, norm.ppf(u))
        exponential = self._exponential_branch(m, psi, u)

        return np.where(psi <= PSI_CRIT, quadratic, exponential)

    # ------------------------------------------------------------------
    # Log-price step
    # ------------------------------------------------------------------

    @staticmethod
    def _log_price_constants(
        params: HestonParams, dt: float
    ) -> tuple[float, float, float, float, float]:
        """K0..K4 of Andersen's log-price discretization.

        They fold the correlated variance-driven part of the log price into
        terms in v_t and v_{t+dt}, leaving an orthogonal normal shock whose
        variance is (K3*v_t + K4*v_{t+dt}).
        """
        rho, xi, kappa, theta = params.rho, params.xi, params.kappa, params.theta
        xi = max(xi, _EPS)

        k0 = -rho * kappa * theta * dt / xi
        k1 = _GAMMA_1 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2 = _GAMMA_2 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3 = _GAMMA_1 * dt * (1.0 - rho**2)
        k4 = _GAMMA_2 * dt * (1.0 - rho**2)
        return k0, k1, k2, k3, k4

    @staticmethod
    def _log_price_increment(
        v_now: np.ndarray,
        v_next: np.ndarray,
        z: np.ndarray,
        mu: float,
        dt: float,
        constants: tuple[float, float, float, float, float],
    ) -> np.ndarray:
        k0, k1, k2, k3, k4 = constants
        variance = np.maximum(k3 * v_now + k4 * v_next, 0.0)
        return mu * dt + k0 + k1 * v_now + k2 * v_next + np.sqrt(variance) * z

    @staticmethod
    def _validate_params(params: HestonParams) -> None:
        if params.kappa <= 0:
            raise ValueError("kappa must be positive")
        if params.theta < 0 or params.v0 < 0:
            raise ValueError("theta and v0 must be non-negative")
        if params.xi <= 0:
            raise ValueError("xi must be positive")
        if not -1.0 <= params.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
