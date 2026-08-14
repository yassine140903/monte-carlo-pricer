"""HTTP routers, one module per resource.

Each module exposes a ``router`` that main.py mounts. Routers delegate: they
parse, call one module under src/, and serialize. Any arithmetic that is not
reshaping a response for the wire belongs in the module being called, not here.
"""
from __future__ import annotations

from src.api.routers import (
    calibration,
    data,
    mlflow_runs,
    pricing,
    risk,
    simulation,
)

__all__ = ["data", "calibration", "simulation", "pricing", "risk", "mlflow_runs"]
