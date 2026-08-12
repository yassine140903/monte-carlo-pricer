"""Tests for src/pricing/. No DB or network access needed.

The Monte Carlo tests lean on Black-Scholes as ground truth wherever a closed
form exists, and on pathwise inequalities (a knocked-out option cannot be
worth more than the vanilla it is carved from) where one does not.
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from src.calibration.gbm import GBMParams
from src.calibration.heston import HestonParams
from src.calibration.jump_diffusion import JumpDiffusionParams
from src.pricing.black_scholes import bs_call, bs_greeks, bs_put
from src.pricing.greeks import GreeksCalculator, GreeksResult
from src.pricing.payoffs import (
    PAYOFF_REGISTRY,
    asian_call,
    asian_put,
    barrier_ko_call,
    barrier_ko_put,
    european_call,
    european_put,
    lookback_call,
    lookback_put,
)
from src.pricing.pricer import MCPricer, PricingResult
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.jump_diffusion import JumpDiffusionSimulator

# textbook setup: S=K=100, 1y, 5% rates, 20% vol
S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
DT = 1 / 252

GBM_PARAMS = GBMParams(mu=0.15, sigma=SIGMA)  # real-world drift, overridden to r-q
JD_PARAMS = JumpDiffusionParams(mu=0.15, sigma=SIGMA, lambda_j=1.0, mu_j=-0.03, sigma_j=0.08)
HESTON_PARAMS = HestonParams(
    kappa=2.0, theta=SIGMA**2, xi=0.3, rho=-0.6, v0=SIGMA**2, feller_satisfied=True
)


@pytest.fixture()
def pricer():
    return MCPricer(GBMSimulator())


# --------------------------------------------------------------------------
# payoffs
# --------------------------------------------------------------------------


class TestPayoffs:
    # rows: rising path, falling path, path that spikes then returns
    PATHS = np.array(
        [
            [100.0, 110.0, 120.0, 130.0],
            [100.0, 90.0, 80.0, 70.0],
            [100.0, 140.0, 95.0, 105.0],
        ]
    )

    def test_european_call(self):
        np.testing.assert_allclose(european_call(self.PATHS, 100.0), [30.0, 0.0, 5.0])

    def test_european_put(self):
        np.testing.assert_allclose(european_put(self.PATHS, 100.0), [0.0, 30.0, 0.0])

    def test_asian_call_averages_excluding_s0(self):
        # row 0 averages 110,120,130 -> 120; including S0 would give 115
        np.testing.assert_allclose(asian_call(self.PATHS, 100.0), [20.0, 0.0, 13.333333], rtol=1e-5)

    def test_asian_put(self):
        np.testing.assert_allclose(asian_put(self.PATHS, 100.0), [0.0, 20.0, 0.0])

    def test_asian_below_european_when_average_below_terminal(self):
        rising = self.PATHS[:1]
        assert asian_call(rising, 100.0)[0] < european_call(rising, 100.0)[0]

    def test_barrier_ko_call_zero_when_breached(self):
        # barrier 125: row 0 breaches at 130, row 2 breaches at 140
        payoffs = barrier_ko_call(self.PATHS, 100.0, barrier=125.0)
        np.testing.assert_allclose(payoffs, [0.0, 0.0, 0.0])

        # barrier 135: only row 2 breaches, row 0 survives and pays
        payoffs = barrier_ko_call(self.PATHS, 100.0, barrier=135.0)
        np.testing.assert_allclose(payoffs, [30.0, 0.0, 0.0])

    def test_barrier_ko_put_zero_when_breached(self):
        payoffs = barrier_ko_put(self.PATHS, 100.0, barrier=75.0)
        np.testing.assert_allclose(payoffs, [0.0, 0.0, 0.0])

        payoffs = barrier_ko_put(self.PATHS, 100.0, barrier=65.0)
        np.testing.assert_allclose(payoffs, [0.0, 30.0, 0.0])

    def test_barrier_monitoring_excludes_s0(self):
        """A barrier sitting exactly at S0 must not knock every path out
        before the option has started: t=0 is a known constant, not an
        observation. Row 0 rises away from 100 and has to survive.
        """
        rising = self.PATHS[:1]
        assert barrier_ko_put(rising, 100.0, barrier=100.0)[0] == 0.0  # survives, but OTM
        assert barrier_ko_put(rising, 130.0, barrier=100.0)[0] == 0.0  # survives, ITM at 130

        deep_itm_put = barrier_ko_put(rising, 200.0, barrier=100.0)
        assert deep_itm_put[0] == 70.0  # 200 - 130, paid because S0 did not knock it out

    def test_lookback_monitoring_excludes_s0(self):
        """Row 1 falls from S0=100; its minimum over the monitored window is
        70, not 100, so including S0 would understate the payoff.
        """
        falling = self.PATHS[1:2]
        assert lookback_call(falling)[0] == 0.0  # 70 - 70
        assert lookback_put(falling)[0] == 20.0  # 90 - 70

    def test_lookback_call_uses_realized_minimum(self):
        np.testing.assert_allclose(lookback_call(self.PATHS), [20.0, 0.0, 10.0])

    def test_lookback_put_uses_realized_maximum(self):
        np.testing.assert_allclose(lookback_put(self.PATHS), [0.0, 20.0, 35.0])

    def test_lookback_payoffs_are_non_negative(self):
        assert np.all(lookback_call(self.PATHS) >= 0)
        assert np.all(lookback_put(self.PATHS) >= 0)

    def test_registry_covers_every_payoff(self):
        assert set(PAYOFF_REGISTRY) == {
            "european_call",
            "european_put",
            "asian_call",
            "asian_put",
            "barrier_ko_call",
            "barrier_ko_put",
            "lookback_call",
            "lookback_put",
        }

    @pytest.mark.parametrize("name, fn", sorted(PAYOFF_REGISTRY.items()))
    def test_registry_entries_return_one_value_per_path(self, name, fn):
        kwargs = {} if name.startswith("lookback") else {"K": 100.0}
        if name.startswith("barrier"):
            kwargs["barrier"] = 200.0 if "call" in name else 1.0
        assert fn(self.PATHS, **kwargs).shape == (3,)


# --------------------------------------------------------------------------
# Black-Scholes
# --------------------------------------------------------------------------


class TestBlackScholes:
    def test_known_call_value(self):
        assert bs_call(S0, K, T, R, SIGMA) == pytest.approx(10.4506, abs=1e-4)

    def test_known_put_value(self):
        assert bs_put(S0, K, T, R, SIGMA) == pytest.approx(5.5735, abs=1e-4)

    @pytest.mark.parametrize("q", [0.0, 0.03])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_put_call_parity(self, q, strike):
        call = bs_call(S0, strike, T, R, SIGMA, q)
        put = bs_put(S0, strike, T, R, SIGMA, q)
        parity = S0 * np.exp(-q * T) - strike * np.exp(-R * T)
        assert call - put == pytest.approx(parity, abs=1e-10)

    def test_dividend_yield_lowers_a_call(self):
        assert bs_call(S0, K, T, R, SIGMA, q=0.05) < bs_call(S0, K, T, R, SIGMA, q=0.0)

    @pytest.mark.parametrize("spot, expected_call, expected_put", [(120.0, 20.0, 0.0), (80.0, 0.0, 20.0)])
    def test_zero_maturity_returns_intrinsic(self, spot, expected_call, expected_put):
        assert bs_call(spot, K, 0.0, R, SIGMA) == expected_call
        assert bs_put(spot, K, 0.0, R, SIGMA) == expected_put

    def test_non_positive_sigma_raises(self):
        for sigma in (0.0, -0.1):
            with pytest.raises(ValueError, match="sigma must be positive"):
                bs_call(S0, K, T, R, sigma)
            with pytest.raises(ValueError, match="sigma must be positive"):
                bs_put(S0, K, T, R, sigma)

    def test_negative_maturity_raises(self):
        with pytest.raises(ValueError, match="T must be non-negative"):
            bs_call(S0, K, -1.0, R, SIGMA)

    def test_greeks_atm_call(self):
        greeks = bs_greeks(S0, K, T, R, SIGMA, option_type="call")
        assert set(greeks) == {"delta", "gamma", "vega", "theta", "rho"}
        assert 0.5 < greeks["delta"] < 0.7
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0
        assert greeks["theta"] < 0  # long options decay
        assert greeks["rho"] > 0  # calls gain as rates rise

    def test_greeks_atm_put(self):
        greeks = bs_greeks(S0, K, T, R, SIGMA, option_type="put")
        assert -0.5 < greeks["delta"] < -0.3
        assert greeks["rho"] < 0

    def test_gamma_and_vega_match_across_call_and_put(self):
        call = bs_greeks(S0, K, T, R, SIGMA, option_type="call")
        put = bs_greeks(S0, K, T, R, SIGMA, option_type="put")
        assert call["gamma"] == pytest.approx(put["gamma"])
        assert call["vega"] == pytest.approx(put["vega"])

    def test_delta_difference_is_one(self):
        """Parity differentiated in S: delta_call - delta_put = e^{-qT}."""
        call = bs_greeks(S0, K, T, R, SIGMA, q=0.02, option_type="call")
        put = bs_greeks(S0, K, T, R, SIGMA, q=0.02, option_type="put")
        assert call["delta"] - put["delta"] == pytest.approx(np.exp(-0.02 * T))

    def test_delta_against_finite_difference(self):
        bump = 1e-5
        numeric = (bs_call(S0 + bump, K, T, R, SIGMA) - bs_call(S0 - bump, K, T, R, SIGMA)) / (
            2 * bump
        )
        assert bs_greeks(S0, K, T, R, SIGMA)["delta"] == pytest.approx(numeric, rel=1e-6)

    def test_vega_against_finite_difference(self):
        bump = 1e-6
        numeric = (bs_call(S0, K, T, R, SIGMA + bump) - bs_call(S0, K, T, R, SIGMA - bump)) / (
            2 * bump
        )
        assert bs_greeks(S0, K, T, R, SIGMA)["vega"] == pytest.approx(numeric, rel=1e-5)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type must be"):
            bs_greeks(S0, K, T, R, SIGMA, option_type="straddle")

    def test_greeks_at_expiry(self):
        assert bs_greeks(120.0, K, 0.0, R, SIGMA)["delta"] == 1.0
        assert bs_greeks(80.0, K, 0.0, R, SIGMA)["delta"] == 0.0
        assert bs_greeks(120.0, K, 0.0, R, SIGMA, option_type="put")["delta"] == 0.0


# --------------------------------------------------------------------------
# MCPricer
# --------------------------------------------------------------------------


class TestMCPricer:
    def test_european_call_matches_black_scholes(self, pricer):
        result = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7)
        assert result.bs_benchmark == pytest.approx(bs_call(S0, K, T, R, SIGMA))
        assert result.bs_relative_error < 0.01

    def test_european_put_matches_black_scholes(self, pricer):
        result = pricer.price("european_put", S0, K, T, R, GBM_PARAMS, seed=7)
        assert result.bs_relative_error < 0.01

    def test_calibrated_drift_is_ignored_in_favour_of_risk_neutral(self, pricer):
        """The real-world mu in the params must not leak into the price: two
        very different drifts have to give the same option value.
        """
        bullish = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7).price
        bearish = pricer.price(
            "european_call", S0, K, T, R, GBM_PARAMS.model_copy(update={"mu": -0.40}), seed=7
        ).price
        assert bullish == pytest.approx(bearish)

    def test_dividend_yield_is_applied(self, pricer):
        with_dividend = pricer.price(
            "european_call", S0, K, T, R, GBM_PARAMS, seed=7, q=0.05
        )
        assert with_dividend.bs_benchmark == pytest.approx(bs_call(S0, K, T, R, SIGMA, q=0.05))
        assert with_dividend.bs_relative_error < 0.01

    def test_put_call_parity_within_error(self, pricer):
        call = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=3)
        put = pricer.price("european_put", S0, K, T, R, GBM_PARAMS, seed=3)
        parity = S0 - K * np.exp(-R * T)
        tolerance = 4 * (call.std_error + put.std_error)
        assert abs((call.price - put.price) - parity) < tolerance

    def test_confidence_interval_brackets_the_price(self, pricer):
        result = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7)
        low, high = result.confidence_interval_95
        assert low < result.price < high
        assert high - low == pytest.approx(2 * 1.96 * result.std_error)

    def test_confidence_interval_covers_black_scholes(self, pricer):
        """Coverage, not a single draw: a 95% interval is supposed to miss
        about one run in twenty, so asserting one seed lands inside it tests
        that seed's luck rather than the interval.
        """
        analytic = bs_call(S0, K, T, R, SIGMA)
        hits = 0
        for seed in range(20):
            low, high = pricer.price(
                "european_call", S0, K, T, R, GBM_PARAMS, n_simulations=20_000, seed=seed
            ).confidence_interval_95
            hits += low < analytic < high
        assert hits >= 16

    def test_result_metadata(self, pricer):
        result = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, n_simulations=5_000, seed=7)
        assert isinstance(result, PricingResult)
        assert result.n_simulations == 5_000
        assert result.computation_time_ms > 0
        assert result.std_error > 0

    def test_asian_is_cheaper_than_european(self, pricer):
        asian = pricer.price("asian_call", S0, K, T, R, GBM_PARAMS, seed=7).price
        european = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7).price
        assert asian < european

    def test_knock_out_is_cheaper_than_european(self, pricer):
        knock_out = pricer.price(
            "barrier_ko_call", S0, K, T, R, GBM_PARAMS, seed=7, barrier=130.0
        ).price
        european = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7).price
        assert 0 < knock_out < european

    def test_lower_barrier_is_cheaper(self, pricer):
        near = pricer.price("barrier_ko_call", S0, K, T, R, GBM_PARAMS, seed=7, barrier=115.0).price
        far = pricer.price("barrier_ko_call", S0, K, T, R, GBM_PARAMS, seed=7, barrier=160.0).price
        assert near < far

    def test_knock_out_put_is_cheaper_than_european(self, pricer):
        knock_out = pricer.price(
            "barrier_ko_put", S0, K, T, R, GBM_PARAMS, seed=7, barrier=70.0
        ).price
        european = pricer.price("european_put", S0, K, T, R, GBM_PARAMS, seed=7).price
        assert 0 < knock_out < european

    def test_lookback_is_dearer_than_european(self, pricer):
        lookback = pricer.price("lookback_call", S0, None, T, R, GBM_PARAMS, seed=7).price
        european = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7).price
        assert lookback > european

    def test_lookback_needs_no_strike(self, pricer):
        """K=None must reach a payoff that takes no strike, without error."""
        result = pricer.price("lookback_put", S0, None, T, R, GBM_PARAMS, seed=7)
        assert result.price > 0
        assert result.bs_benchmark is None
        assert result.bs_relative_error is None

    def test_unknown_option_type_raises(self, pricer):
        with pytest.raises(ValueError, match="Unknown option_type 'swaption'"):
            pricer.price("swaption", S0, K, T, R, GBM_PARAMS, seed=7)

    def test_seed_reproducibility(self, pricer):
        first = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=99)
        second = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=99)
        assert first.price == second.price
        assert first.std_error == second.std_error

    def test_different_seeds_give_different_prices(self, pricer):
        first = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=1)
        second = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=2)
        assert first.price != second.price

    def test_antithetic_tightens_the_confidence_interval(self, pricer):
        naive = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, seed=7)
        antithetic = pricer.price(
            "european_call", S0, K, T, R, GBM_PARAMS, seed=7, variance_reduction="antithetic"
        )
        assert antithetic.std_error < naive.std_error
        width = lambda res: res.confidence_interval_95[1] - res.confidence_interval_95[0]
        assert width(antithetic) < width(naive)

    def test_antithetic_is_closer_to_black_scholes_on_average(self, pricer):
        """The reported error is one thing; the realized error is the point."""

        def mean_abs_error(method):
            errors = [
                abs(
                    pricer.price(
                        "european_call",
                        S0,
                        K,
                        T,
                        R,
                        GBM_PARAMS,
                        n_simulations=5_000,
                        dt=1 / 12,
                        seed=seed,
                        variance_reduction=method,
                    ).price
                    - bs_call(S0, K, T, R, SIGMA)
                )
                for seed in range(30)
            ]
            return float(np.mean(errors))

        assert mean_abs_error("antithetic") < mean_abs_error(None)

    def test_more_simulations_shrink_the_error(self, pricer):
        small = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, n_simulations=2_000, seed=7)
        large = pricer.price("european_call", S0, K, T, R, GBM_PARAMS, n_simulations=50_000, seed=7)
        assert large.std_error < small.std_error

    def test_prices_with_the_jump_diffusion_simulator(self):
        pricer = MCPricer(JumpDiffusionSimulator())
        result = pricer.price("european_call", S0, K, T, R, JD_PARAMS, seed=7)
        assert result.price > 0
        # jumps add value that Black-Scholes cannot see, so the benchmark is
        # reported but is not expected to match
        assert result.bs_benchmark == pytest.approx(bs_call(S0, K, T, R, SIGMA))
        assert result.price > result.bs_benchmark

    def test_prices_with_the_heston_simulator(self):
        pricer = MCPricer(HestonSimulator())
        result = pricer.price("european_call", S0, K, T, R, HESTON_PARAMS, seed=7)
        assert result.price > 0
        # Heston params carry no sigma, so no closed form is offered
        assert result.bs_benchmark is None
        assert result.bs_relative_error is None

    def test_heston_at_zero_vol_of_vol_matches_black_scholes(self):
        flat = HESTON_PARAMS.model_copy(update={"xi": 1e-8, "rho": 0.0})
        result = MCPricer(HestonSimulator()).price(
            "european_call", S0, K, T, R, flat, n_simulations=100_000, seed=7
        )
        assert result.price == pytest.approx(bs_call(S0, K, T, R, SIGMA), rel=0.02)


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------


@pytest.fixture()
def greeks(pricer):
    return GreeksCalculator(pricer).compute(
        "european_call", S0, K, T, R, GBM_PARAMS, n_simulations=50_000, seed=42
    )


class TestGreeksCalculator:
    ANALYTIC = bs_greeks(S0, K, T, R, SIGMA, option_type="call")

    def test_returns_a_greeks_result(self, greeks):
        assert isinstance(greeks, GreeksResult)

    @pytest.mark.parametrize("name", ["delta", "gamma", "vega", "theta", "rho"])
    def test_matches_black_scholes_within_five_percent(self, greeks, name):
        analytic = self.ANALYTIC[name]
        assert getattr(greeks, name) == pytest.approx(analytic, rel=0.05)

    def test_atm_call_delta(self, greeks):
        # a spot-ATM call with a year to run and 5% rates sits well above 0.5;
        # 0.5 is the forward-ATM, zero-rate intuition
        assert 0.5 < greeks.delta < 0.7
        assert greeks.delta == pytest.approx(self.ANALYTIC["delta"], rel=0.05)

    def test_signs(self, greeks):
        assert greeks.gamma > 0
        assert greeks.vega > 0
        assert greeks.theta < 0
        assert greeks.rho > 0

    def test_common_random_numbers_make_it_reproducible(self, pricer):
        calculator = GreeksCalculator(pricer)
        kwargs = dict(n_simulations=20_000, seed=11)
        first = calculator.compute("european_call", S0, K, T, R, GBM_PARAMS, **kwargs)
        second = calculator.compute("european_call", S0, K, T, R, GBM_PARAMS, **kwargs)
        assert first == second

    def test_common_random_numbers_actually_matter(self, pricer):
        """Without shared draws the bump is buried in noise. Re-deriving delta
        from independently seeded prices should be visibly worse than the
        common-random-numbers estimate.
        """
        bump = 0.01 * S0
        independent_up = pricer.price(
            "european_call", S0 + bump, K, T, R, GBM_PARAMS, n_simulations=20_000, seed=1
        ).price
        independent_down = pricer.price(
            "european_call", S0 - bump, K, T, R, GBM_PARAMS, n_simulations=20_000, seed=2
        ).price
        independent_delta = (independent_up - independent_down) / (2 * bump)

        common = GreeksCalculator(pricer).compute(
            "european_call", S0, K, T, R, GBM_PARAMS, n_simulations=20_000, seed=1
        ).delta

        analytic = self.ANALYTIC["delta"]
        assert abs(common - analytic) < abs(independent_delta - analytic)

    def test_put_greeks(self, pricer):
        result = GreeksCalculator(pricer).compute(
            "european_put", S0, K, T, R, GBM_PARAMS, n_simulations=50_000, seed=42
        )
        analytic = bs_greeks(S0, K, T, R, SIGMA, option_type="put")
        assert result.delta == pytest.approx(analytic["delta"], rel=0.05)
        assert result.rho == pytest.approx(analytic["rho"], rel=0.05)
        assert result.delta < 0
        assert result.rho < 0

    def test_theta_skipped_for_sub_step_maturity(self, pricer):
        result = GreeksCalculator(pricer).compute(
            "european_call",
            S0,
            K,
            1 / 504,
            R,
            GBM_PARAMS,
            n_simulations=2_000,
            seed=42,
            dt=1 / 2520,
        )
        assert result.theta == 0.0

    def test_theta_skipped_when_the_bump_would_leave_no_steps(self, pricer):
        """T of exactly one theta step: the bumped horizon is 0, which is not
        a horizon the simulator can run. Must skip rather than raise.
        """
        result = GreeksCalculator(pricer).compute(
            "european_call", S0, K, 1 / 252, R, GBM_PARAMS, n_simulations=2_000, seed=42
        )
        assert result.theta == 0.0
        assert result.delta > 0

    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_theta_is_stable_across_seeds(self, pricer, seed):
        """Regression on common random numbers for the theta bump.

        Shortening T normally drops a step and reshuffles the draws, which
        turns theta into 252x-amplified noise — this same assertion returned
        -18.4 against an analytic -10.5 before the step count was pinned.
        Parametrizing over seeds is the point: noise would not survive it.
        """
        result = GreeksCalculator(pricer).compute(
            "european_call", S0, K, 0.25, R, GBM_PARAMS, n_simulations=20_000, seed=seed
        )
        analytic = bs_greeks(S0, K, 0.25, R, SIGMA)["theta"]
        assert result.theta == pytest.approx(analytic, rel=0.05)

    def test_theta_bump_preserves_the_step_count(self, pricer):
        from src.simulation.utils import resolve_n_steps

        calculator = GreeksCalculator(pricer)
        matched_dt = calculator._matching_dt(T, DT, T - 1 / 252)
        assert resolve_n_steps(T - 1 / 252, matched_dt) == resolve_n_steps(T, DT)

    def test_works_on_a_path_dependent_payoff(self, pricer):
        """Bump-and-revalue needs no closed form, which is the point of it."""
        result = GreeksCalculator(pricer).compute(
            "barrier_ko_call",
            S0,
            K,
            T,
            R,
            GBM_PARAMS,
            n_simulations=20_000,
            seed=42,
            barrier=130.0,
        )
        # an up-and-out call loses value as spot approaches the barrier, so
        # its delta is far below the vanilla's and can even turn negative
        assert result.delta < self.ANALYTIC["delta"]

    def test_heston_vega_bumps_v0(self, pricer):
        calculator = GreeksCalculator(MCPricer(HestonSimulator()))
        result = calculator.compute(
            "european_call", S0, K, T, R, HESTON_PARAMS, n_simulations=20_000, seed=42
        )
        assert result.vega > 0
        assert result.delta == pytest.approx(0.6, abs=0.15)

    def test_bump_volatility_leaves_params_untouched(self, pricer):
        calculator = GreeksCalculator(pricer)
        original = GBM_PARAMS.model_copy()
        calculator.compute("european_call", S0, K, T, R, GBM_PARAMS, n_simulations=2_000, seed=42)
        assert GBM_PARAMS == original

    def test_bump_volatility_picks_the_right_field(self, pricer):
        calculator = GreeksCalculator(pricer)

        bumped_gbm = calculator._bump_volatility(GBM_PARAMS, 0.01)
        assert bumped_gbm.sigma == pytest.approx(SIGMA + 0.01)

        bumped_heston = calculator._bump_volatility(HESTON_PARAMS, 0.01)
        # v0 is a variance, so a one-point vol bump squares back up
        assert bumped_heston.v0 == pytest.approx((SIGMA + 0.01) ** 2)
        assert bumped_heston.kappa == HESTON_PARAMS.kappa

    def test_bump_volatility_rejects_params_without_a_vol(self, pricer):
        class DriftOnlyParams(BaseModel):
            mu: float

        with pytest.raises(TypeError, match="neither 'sigma' nor 'v0'"):
            GreeksCalculator(pricer)._bump_volatility(DriftOnlyParams(mu=0.1), 0.01)
