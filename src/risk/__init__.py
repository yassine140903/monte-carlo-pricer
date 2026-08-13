"""Risk analytics: VaR/CVaR and drawdown metrics, correlated multi-asset
portfolio simulation, and parameter stress scenarios. Pure NumPy/SciPy and
Pydantic — no pandas, no MLflow. Logging is the caller's responsibility, as
in src/simulation/ and src/pricing/.
"""
from __future__ import annotations

from src.risk.metrics import (
    RiskMetrics,
    compute_cvar,
    compute_max_drawdown,
    compute_prob_of_loss,
    compute_risk_metrics,
    compute_sharpe,
    compute_var,
)
from src.risk.portfolio import (
    PortfolioAsset,
    PortfolioResult,
    estimate_correlation_matrix,
    simulate_portfolio,
)
from src.risk.scenarios import (
    PRESET_SCENARIOS,
    Scenario,
    apply_scenario,
    get_preset_scenario,
    list_preset_scenarios,
)

__all__ = [
    "RiskMetrics",
    "compute_var",
    "compute_cvar",
    "compute_max_drawdown",
    "compute_sharpe",
    "compute_prob_of_loss",
    "compute_risk_metrics",
    "PortfolioAsset",
    "PortfolioResult",
    "estimate_correlation_matrix",
    "simulate_portfolio",
    "Scenario",
    "apply_scenario",
    "get_preset_scenario",
    "list_preset_scenarios",
    "PRESET_SCENARIOS",
]
