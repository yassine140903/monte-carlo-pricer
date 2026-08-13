"""Stress testing by transforming calibrated parameters.

A scenario is a declarative edit to a params object — multiply this, set that
— rather than a separate set of dynamics. Stressing a portfolio therefore
reuses the ordinary calibrate/simulate path unchanged: apply the scenario,
simulate with the returned params, compare the metrics.

Because a scenario names parameters, it is specific to a model family: ``v0``
means nothing to GBM. The presets below are keyed by scenario *and* model
type, and :func:`apply_scenario` rejects a name the params object does not
carry rather than silently ignoring it.
"""
from __future__ import annotations

from pydantic import BaseModel

MODEL_TYPES = ("gbm", "jump_diffusion", "heston")


class Scenario(BaseModel):
    """A named parameter shock.

    ``multipliers`` scale a parameter relative to its calibrated value, which
    keeps a scenario meaningful across tickers of very different volatility.
    ``overrides`` pin an absolute value, for parameters where the calibrated
    level is not the right anchor — a crash drift is -50% whatever the asset
    was doing before.
    """

    name: str
    description: str
    overrides: dict[str, float] = {}
    multipliers: dict[str, float] = {}


def apply_scenario(params: BaseModel, scenario: Scenario) -> BaseModel:
    """A new params object with the scenario applied. ``params`` is untouched.

    Multipliers are applied first and overrides second, so naming a parameter
    in both pins it to the override — the absolute statement is the more
    specific one, and letting the multiplier win instead would make the
    override unreachable.
    """
    param_dict = params.model_dump()
    model_name = type(params).__name__

    for key, multiplier in scenario.multipliers.items():
        _require_param(param_dict, key, model_name, "multipliers")
        param_dict[key] *= multiplier

    for key, value in scenario.overrides.items():
        _require_param(param_dict, key, model_name, "overrides")
        param_dict[key] = value

    return type(params)(**param_dict)


def _require_param(
    param_dict: dict[str, float], key: str, model_name: str, field: str
) -> None:
    if key not in param_dict:
        raise ValueError(f"Unknown parameter '{key}' in {field} for {model_name}")


PRESET_SCENARIOS: dict[str, dict[str, Scenario]] = {
    "vol_spike": {
        "gbm": Scenario(
            name="Volatility Spike",
            description="2008-style regime: volatility spikes to 2.5x calibrated level",
            multipliers={"sigma": 2.5},
        ),
        "jump_diffusion": Scenario(
            name="Volatility Spike",
            description="2008-style regime: elevated vol and jump intensity",
            multipliers={"sigma": 2.0, "lambda_j": 3.0},
        ),
        "heston": Scenario(
            name="Volatility Spike",
            description="2008-style regime: variance jumps to 4x, long-term vol elevated",
            multipliers={"v0": 4.0, "theta": 2.5, "xi": 1.5},
        ),
    },
    "market_crash": {
        "gbm": Scenario(
            name="Market Crash",
            description="Sudden crash: vol doubles, drift deeply negative",
            multipliers={"sigma": 2.0},
            overrides={"mu": -0.5},
        ),
        "jump_diffusion": Scenario(
            name="Market Crash",
            description=(
                "Crash with large negative jumps: high jump intensity, "
                "negative jump mean"
            ),
            multipliers={"sigma": 1.5, "lambda_j": 5.0},
            overrides={"mu_j": -0.15, "sigma_j": 0.10},
        ),
        "heston": Scenario(
            name="Market Crash",
            description=(
                "Crash: variance spikes, mean-reversion weakened, vol-of-vol surges"
            ),
            multipliers={"v0": 6.0, "xi": 2.0},
            overrides={"kappa": 0.5, "rho": -0.9},
        ),
    },
}


def get_preset_scenario(scenario_name: str, model_type: str) -> Scenario:
    """Look up a preset by scenario name and model family."""
    if scenario_name not in PRESET_SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'; available: "
            f"{sorted(PRESET_SCENARIOS)}"
        )

    by_model = PRESET_SCENARIOS[scenario_name]
    if model_type not in by_model:
        raise KeyError(
            f"Scenario '{scenario_name}' has no variant for model type "
            f"'{model_type}'; available: {sorted(by_model)}"
        )

    return by_model[model_type]


def list_preset_scenarios() -> dict[str, list[str]]:
    """Every preset scenario mapped to the model types it supports."""
    return {name: sorted(by_model) for name, by_model in PRESET_SCENARIOS.items()}
