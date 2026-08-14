"""Request and response models for the HTTP layer.

These are the wire contract and nothing else: they validate shapes and ranges,
but every model here is deliberately free of behaviour. The params models
themselves come straight from src/calibration/ — the API does not maintain a
second copy of them — joined into a discriminated union on ``model_type`` so
one request body can carry any of the three model families.
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from src.calibration import GBMParams, HestonParams, JumpDiffusionParams
from src.data.models import StockData
from src.pricing import PAYOFF_REGISTRY

ModelType = Literal["gbm", "jump_diffusion", "heston"]
VarianceReduction = Literal["antithetic", "stratified"]

# Discriminated on the constant ``model_type`` tag each params model carries,
# so a wrong-shaped body is rejected against the one model it claims to be
# rather than reported as three simultaneous failures.
ModelParams = Annotated[
    Annotated[GBMParams, Tag("gbm")]
    | Annotated[JumpDiffusionParams, Tag("jump_diffusion")]
    | Annotated[HestonParams, Tag("heston")],
    Discriminator("model_type"),
]

DEFAULT_DT = 1 / 252
DEFAULT_LOOKBACK_DAYS = 756
DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class CalibrateRequest(BaseModel):
    ticker: str
    model_type: ModelType
    lookback_days: int = Field(default=DEFAULT_LOOKBACK_DAYS, gt=1)
    window_days: int | None = Field(default=None, gt=1)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, v: str) -> str:
        return v.upper()


class SimulateRequest(BaseModel):
    """``n_simulations`` is bounded at request time rather than here — an
    over-large run is answered with 413, which a schema constraint could only
    report as 422. See api.utils.enforce_simulation_budget.
    """

    S0: float = Field(gt=0)
    model_params: ModelParams
    T: float = Field(gt=0)
    dt: float = Field(default=DEFAULT_DT, gt=0)
    n_simulations: int = Field(default=10_000, gt=0)
    variance_reduction: VarianceReduction | None = None
    seed: int | None = None


class PriceOptionRequest(BaseModel):
    """``K`` is optional only because the floating-strike lookbacks in
    PAYOFF_REGISTRY have no strike; every other payoff requires it, and
    ``barrier`` is required by exactly the knock-out payoffs. Both are checked
    against the requested ``option_type`` rather than left to fail inside the
    payoff function.
    """

    option_type: str
    S0: float = Field(gt=0)
    K: float | None = Field(default=None, gt=0)
    T: float = Field(gt=0)
    r: float
    q: float = 0.0
    model_params: ModelParams
    dt: float = Field(default=DEFAULT_DT, gt=0)
    n_simulations: int = Field(default=50_000, gt=0)
    barrier: float | None = Field(default=None, gt=0)
    seed: int | None = None
    variance_reduction: VarianceReduction | None = None

    @field_validator("option_type")
    @classmethod
    def _known_payoff(cls, v: str) -> str:
        if v not in PAYOFF_REGISTRY:
            raise ValueError(
                f"Unknown option_type {v!r}; expected one of {sorted(PAYOFF_REGISTRY)}"
            )
        return v

    @model_validator(mode="after")
    def _payoff_arguments_match(self) -> PriceOptionRequest:
        is_lookback = self.option_type.startswith("lookback")
        if is_lookback and self.K is not None:
            raise ValueError(
                f"{self.option_type} is a floating-strike payoff and takes no K"
            )
        if not is_lookback and self.K is None:
            raise ValueError(f"{self.option_type} requires a strike K")

        needs_barrier = self.option_type.startswith("barrier")
        if needs_barrier and self.barrier is None:
            raise ValueError(f"{self.option_type} requires a barrier level")
        if not needs_barrier and self.barrier is not None:
            raise ValueError(f"{self.option_type} does not take a barrier level")
        return self


class RiskMetricsRequest(BaseModel):
    """``r`` is the Sharpe benchmark only — the paths themselves are simulated
    under the physical measure, using the drift in ``model_params``.
    """

    S0: float = Field(gt=0)
    model_params: ModelParams
    T: float = Field(gt=0)
    dt: float = Field(default=DEFAULT_DT, gt=0)
    n_simulations: int = Field(default=10_000, gt=0)
    confidence_levels: list[float] = Field(default_factory=lambda: list(DEFAULT_CONFIDENCE_LEVELS))
    r: float = 0.0
    seed: int | None = None
    variance_reduction: VarianceReduction | None = None

    @field_validator("confidence_levels")
    @classmethod
    def _levels_in_range(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("confidence_levels must not be empty")
        for level in v:
            if not 0.0 < level < 1.0:
                raise ValueError(
                    f"confidence level must lie strictly in (0, 1), got {level}"
                )
        return v


class PortfolioAssetInput(BaseModel):
    ticker: str
    weight: float
    S0: float = Field(gt=0)
    model_params: ModelParams

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, v: str) -> str:
        return v.upper()


class PortfolioRiskRequest(BaseModel):
    """A basket simulated under one estimated correlation structure.

    Correlations are estimated from the last ``lookback_days`` of stored
    closes, so every ticker listed must already be in the database.
    """

    assets: list[PortfolioAssetInput] = Field(min_length=2)
    T: float = Field(gt=0)
    dt: float = Field(default=DEFAULT_DT, gt=0)
    n_simulations: int = Field(default=10_000, gt=0)
    confidence_levels: list[float] = Field(default_factory=lambda: list(DEFAULT_CONFIDENCE_LEVELS))
    lookback_days: int = Field(default=DEFAULT_LOOKBACK_DAYS, gt=1)
    scenario: str | None = None
    custom_scenario: dict | None = None
    r: float = 0.0
    seed: int | None = None

    @field_validator("assets")
    @classmethod
    def _unique_tickers(cls, v: list[PortfolioAssetInput]) -> list[PortfolioAssetInput]:
        tickers = [asset.ticker.upper() for asset in v]
        if len(set(tickers)) != len(tickers):
            raise ValueError("Each ticker may appear at most once in a portfolio")
        return v


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class CalibrateResponse(BaseModel):
    model_type: ModelType
    params: ModelParams
    fit_metrics: dict[str, float]
    mlflow_run_id: str | None = None


class SimulationResponse(BaseModel):
    """``percentile_bands``, ``sample_paths`` and ``time_axis`` are thinned to
    the same dates, so they can be plotted against one another directly.
    """

    summary: dict[str, float]
    percentile_bands: dict[str, list[float]]
    sample_paths: list[list[float]]
    time_axis: list[float]
    n_simulations: int
    computation_time_ms: float


class PricingResponse(BaseModel):
    price: float
    std_error: float
    confidence_interval_95: tuple[float, float]
    greeks: dict[str, float]
    bs_benchmark: dict | None = None
    payoff_histogram: dict[str, list[float]]
    n_simulations: int
    computation_time_ms: float
    mlflow_run_id: str | None = None


class RiskMetricsResponse(BaseModel):
    """``var`` and ``cvar`` are keyed by confidence level as a percentage
    string ("95", "99") — JSON object keys cannot be floats.

    Both keep the sign convention of src/risk/metrics.py: a loss is negative.
    """

    var: dict[str, float]
    cvar: dict[str, float]
    max_drawdown: dict[str, float]
    sharpe_ratio: float
    probability_of_loss: float
    n_simulations: int


class PortfolioRiskResponse(BaseModel):
    risk_metrics: RiskMetricsResponse
    correlation_matrix: list[list[float]]
    tickers: list[str]
    scenario_applied: str | None = None


class ScenarioVariant(BaseModel):
    model_type: ModelType
    name: str
    description: str


class ScenarioFamily(BaseModel):
    """One preset scenario and the per-model variants it is defined for."""

    key: str
    name: str
    model_types: list[str]
    variants: list[ScenarioVariant]


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioFamily]


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    run_count: int


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentSummary]


class OHLCVResponse(BaseModel):
    ticker: str
    start_date: date | None = None
    end_date: date | None = None
    row_count: int
    bars: list[StockData]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    context: dict = Field(default_factory=dict)
