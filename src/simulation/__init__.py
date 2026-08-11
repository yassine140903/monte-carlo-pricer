"""Path simulation: GBM, Merton jump-diffusion and Heston (QE scheme) Monte
Carlo engines, plus variance reduction. Pure NumPy/SciPy — no pandas, no
MLflow. Simulators are stateless; every call gets its own RNG.
"""
from __future__ import annotations

from src.simulation.base import SimulatorBase
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.jump_diffusion import JumpDiffusionSimulator
from src.simulation.utils import (
    make_rng,
    path_statistics,
    resolve_n_steps,
    terminal_values,
)
from src.simulation.variance_reduction import generate_samples

__all__ = [
    "SimulatorBase",
    "GBMSimulator",
    "JumpDiffusionSimulator",
    "HestonSimulator",
    "generate_samples",
    "make_rng",
    "resolve_n_steps",
    "path_statistics",
    "terminal_values",
]
