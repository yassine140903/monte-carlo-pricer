"""Tests for src/risk/scenarios.py.

Two things are being checked: that the parameter algebra is exactly what it
claims (multipliers then overrides, original untouched, unknown names
rejected), and that the presets actually stress a portfolio — a scenario that
applied cleanly but left the risk metrics unchanged would be useless.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.gbm import GBMParams
from src.calibration.heston import HestonParams
from src.calibration.jump_diffusion import JumpDiffusionParams
from src.risk.metrics import compute_risk_metrics
from src.risk.scenarios import (
    MODEL_TYPES,
    PRESET_SCENARIOS,
    Scenario,
    apply_scenario,
    get_preset_scenario,
    list_preset_scenarios,
)
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.jump_diffusion import JumpDiffusionSimulator

DT = 1 / 252

GBM_PARAMS = GBMParams(mu=0.10, sigma=0.20)
JD_PARAMS = JumpDiffusionParams(
    mu=0.10, sigma=0.20, lambda_j=1.0, mu_j=-0.03, sigma_j=0.08
)
HESTON_PARAMS = HestonParams(
    kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04, feller_satisfied=True
)

PARAMS_BY_MODEL = {
    "gbm": GBM_PARAMS,
    "jump_diffusion": JD_PARAMS,
    "heston": HESTON_PARAMS,
}
SIMULATOR_BY_MODEL = {
    "gbm": GBMSimulator(),
    "jump_diffusion": JumpDiffusionSimulator(),
    "heston": HestonSimulator(),
}


# --------------------------------------------------------------------------
# The parameter algebra
# --------------------------------------------------------------------------


class TestApplyScenario:
    def test_multiplier_scales_the_named_parameter_only(self):
        scenario = Scenario(
            name="x", description="", multipliers={"sigma": 2.0}
        )

        stressed = apply_scenario(GBM_PARAMS, scenario)

        assert stressed.sigma == pytest.approx(0.40)
        assert stressed.mu == pytest.approx(0.10)

    def test_override_pins_an_absolute_value(self):
        scenario = Scenario(name="x", description="", overrides={"sigma": 0.8})

        stressed = apply_scenario(GBM_PARAMS, scenario)

        assert stressed.sigma == pytest.approx(0.80)
        assert stressed.mu == pytest.approx(0.10)

    def test_override_wins_over_a_multiplier_on_the_same_key(self):
        scenario = Scenario(
            name="x",
            description="",
            multipliers={"sigma": 2.0},
            overrides={"sigma": 0.5},
        )

        assert apply_scenario(GBM_PARAMS, scenario).sigma == pytest.approx(0.50)

    def test_multipliers_and_overrides_on_different_keys_both_land(self):
        scenario = Scenario(
            name="x",
            description="",
            multipliers={"sigma": 3.0},
            overrides={"mu": -0.4},
        )

        stressed = apply_scenario(GBM_PARAMS, scenario)

        assert stressed.sigma == pytest.approx(0.60)
        assert stressed.mu == pytest.approx(-0.40)

    def test_empty_scenario_is_a_faithful_copy(self):
        stressed = apply_scenario(GBM_PARAMS, Scenario(name="x", description=""))

        assert stressed == GBM_PARAMS
        assert stressed is not GBM_PARAMS

    def test_the_original_params_are_never_mutated(self):
        original = GBMParams(mu=0.10, sigma=0.20)
        scenario = Scenario(
            name="x",
            description="",
            multipliers={"sigma": 5.0},
            overrides={"mu": -0.9},
        )

        apply_scenario(original, scenario)

        assert original.mu == pytest.approx(0.10)
        assert original.sigma == pytest.approx(0.20)

    def test_returns_the_same_params_type(self):
        scenario = Scenario(name="x", description="", multipliers={"v0": 2.0})

        assert isinstance(apply_scenario(HESTON_PARAMS, scenario), HestonParams)

    def test_non_float_fields_survive_the_round_trip(self):
        """HestonParams carries a bool alongside its floats; model_dump and
        reconstruction must not disturb it."""
        scenario = Scenario(name="x", description="", multipliers={"v0": 2.0})

        assert apply_scenario(HESTON_PARAMS, scenario).feller_satisfied is True

    def test_rejects_an_unknown_multiplier_key(self):
        scenario = Scenario(name="x", description="", multipliers={"nonsense": 2.0})

        with pytest.raises(
            ValueError, match="Unknown parameter 'nonsense' in multipliers for GBMParams"
        ):
            apply_scenario(GBM_PARAMS, scenario)

    def test_rejects_an_unknown_override_key(self):
        scenario = Scenario(name="x", description="", overrides={"nonsense": 2.0})

        with pytest.raises(
            ValueError, match="Unknown parameter 'nonsense' in overrides for GBMParams"
        ):
            apply_scenario(GBM_PARAMS, scenario)

    def test_a_parameter_of_another_model_counts_as_unknown(self):
        """v0 is a Heston parameter; asking GBM for it is a mistake worth
        raising rather than quietly ignoring."""
        scenario = Scenario(name="x", description="", multipliers={"v0": 4.0})

        with pytest.raises(ValueError, match="Unknown parameter 'v0'"):
            apply_scenario(GBM_PARAMS, scenario)


# --------------------------------------------------------------------------
# The preset registry
# --------------------------------------------------------------------------


class TestPresets:
    @pytest.mark.parametrize("scenario_name", list(PRESET_SCENARIOS))
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_every_scenario_covers_every_model(self, scenario_name, model_type):
        scenario = get_preset_scenario(scenario_name, model_type)

        assert isinstance(scenario, Scenario)
        assert scenario.name
        assert scenario.description

    @pytest.mark.parametrize("scenario_name", list(PRESET_SCENARIOS))
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_every_preset_applies_to_its_model(self, scenario_name, model_type):
        params = PARAMS_BY_MODEL[model_type]

        stressed = apply_scenario(params, get_preset_scenario(scenario_name, model_type))

        assert isinstance(stressed, type(params))
        assert stressed != params

    @pytest.mark.parametrize("scenario_name", list(PRESET_SCENARIOS))
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_every_preset_actually_raises_volatility(self, scenario_name, model_type):
        """Whatever else a stress scenario does, it must not calm the market
        down. For Heston the volatility state is v0 rather than sigma."""
        params = PARAMS_BY_MODEL[model_type]
        stressed = apply_scenario(params, get_preset_scenario(scenario_name, model_type))

        field = "v0" if model_type == "heston" else "sigma"

        assert getattr(stressed, field) > getattr(params, field)

    def test_heston_presets_stay_inside_the_simulators_validity_bounds(self):
        """rho must remain in [-1, 1] and kappa, xi positive, or the stressed
        params would not simulate at all."""
        for scenario_name in PRESET_SCENARIOS:
            stressed = apply_scenario(
                HESTON_PARAMS, get_preset_scenario(scenario_name, "heston")
            )

            assert -1.0 <= stressed.rho <= 1.0
            assert stressed.kappa > 0
            assert stressed.xi > 0

    def test_unknown_scenario_name_lists_what_is_available(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_preset_scenario("apocalypse", "gbm")

    def test_unknown_model_type_lists_what_is_available(self):
        with pytest.raises(KeyError, match="no variant for model type"):
            get_preset_scenario("vol_spike", "binomial_tree")

    def test_list_preset_scenarios_reports_every_scenario_and_model(self):
        listing = list_preset_scenarios()

        assert set(listing) == set(PRESET_SCENARIOS)
        for model_types in listing.values():
            assert model_types == sorted(MODEL_TYPES)

    def test_market_crash_turns_the_gbm_drift_negative(self):
        stressed = apply_scenario(GBM_PARAMS, get_preset_scenario("market_crash", "gbm"))

        assert stressed.mu < 0

    def test_market_crash_makes_jump_diffusion_jumps_negative_and_frequent(self):
        stressed = apply_scenario(
            JD_PARAMS, get_preset_scenario("market_crash", "jump_diffusion")
        )

        assert stressed.mu_j < 0
        assert stressed.lambda_j > JD_PARAMS.lambda_j


# --------------------------------------------------------------------------
# End to end: stressed simulations really are worse
# --------------------------------------------------------------------------


def simulate(model_type: str, params, seed: int = 42, **kwargs) -> np.ndarray:
    # lambda_j reaches 5x under market_crash, so the single-jump-per-step
    # approximation needs a step below the daily one to stay valid.
    dt = DT / 4 if model_type == "jump_diffusion" else DT
    return SIMULATOR_BY_MODEL[model_type].simulate(
        100.0, params, 1.0, dt, 10_000, seed=seed, **kwargs
    )


class TestStressedMetricsAreWorse:
    @pytest.mark.parametrize("scenario_name", list(PRESET_SCENARIOS))
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_stress_deepens_tail_loss_and_drawdown(self, scenario_name, model_type):
        params = PARAMS_BY_MODEL[model_type]
        stressed_params = apply_scenario(
            params, get_preset_scenario(scenario_name, model_type)
        )

        baseline = compute_risk_metrics(
            simulate(model_type, params), S0=100.0, r=0.02, T=1.0
        )
        stressed = compute_risk_metrics(
            simulate(model_type, stressed_params), S0=100.0, r=0.02, T=1.0
        )

        assert stressed.cvar[0.95] <= baseline.cvar[0.95]
        assert stressed.var[0.95] <= baseline.var[0.95]
        assert stressed.max_drawdown_mean >= baseline.max_drawdown_mean

    def test_a_crash_is_worse_than_a_vol_spike_for_gbm(self):
        """The crash scenario adds a deeply negative drift on top of the vol
        increase, so it must dominate the pure vol spike."""
        spike = apply_scenario(GBM_PARAMS, get_preset_scenario("vol_spike", "gbm"))
        crash = apply_scenario(GBM_PARAMS, get_preset_scenario("market_crash", "gbm"))

        spike_metrics = compute_risk_metrics(
            simulate("gbm", spike), S0=100.0, r=0.02, T=1.0
        )
        crash_metrics = compute_risk_metrics(
            simulate("gbm", crash), S0=100.0, r=0.02, T=1.0
        )

        assert crash_metrics.prob_of_loss > spike_metrics.prob_of_loss
        assert crash_metrics.cvar[0.95] < spike_metrics.cvar[0.95]

    def test_stress_flows_through_a_calibrated_model(self):
        """The realistic path: calibrate from data, stress the fitted params,
        compare. Nothing about the scenario depends on how mu and sigma were
        arrived at."""
        from src.calibration.gbm import GBMCalibrator

        rng = np.random.default_rng(0)
        prices = 100.0 * np.exp(
            np.cumsum(rng.normal(0.0003, 0.012, 1_000))
        )
        params = GBMCalibrator().calibrate(prices, dt=DT)

        stressed_params = apply_scenario(
            params, get_preset_scenario("vol_spike", "gbm")
        )

        assert stressed_params.sigma == pytest.approx(2.5 * params.sigma)

        baseline = compute_risk_metrics(
            simulate("gbm", params), S0=100.0, r=0.02, T=1.0
        )
        stressed = compute_risk_metrics(
            simulate("gbm", stressed_params), S0=100.0, r=0.02, T=1.0
        )

        assert stressed.cvar[0.95] < baseline.cvar[0.95]
