"""FastAPI dependencies and the simulator factory.

The DB engine is built once in the app lifespan and handed out from
``app.state`` — a per-request engine would rebuild the connection pool on
every call, which defeats the point of pooling. Tests override
:func:`get_engine` to point at their own database.
"""
from __future__ import annotations

from fastapi import Request
from sqlalchemy import Engine

from src.calibration import (
    CalibratorBase,
    GBMCalibrator,
    HestonCalibrator,
    JumpDiffusionCalibrator,
)
from src.simulation import (
    GBMSimulator,
    HestonSimulator,
    JumpDiffusionSimulator,
    SimulatorBase,
)

_SIMULATORS: dict[str, type[SimulatorBase]] = {
    "gbm": GBMSimulator,
    "jump_diffusion": JumpDiffusionSimulator,
    "heston": HestonSimulator,
}

_CALIBRATORS: dict[str, type[CalibratorBase]] = {
    "gbm": GBMCalibrator,
    "jump_diffusion": JumpDiffusionCalibrator,
    "heston": HestonCalibrator,
}


def get_engine(request: Request) -> Engine:
    """The shared SQLAlchemy engine created during app startup."""
    return request.app.state.engine


def get_simulator(model_type: str) -> SimulatorBase:
    """A fresh simulator for ``model_type``.

    Simulators are stateless, so returning a new instance costs nothing and
    keeps concurrent requests from sharing an object at all.
    """
    try:
        simulator_cls = _SIMULATORS[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of {sorted(_SIMULATORS)}"
        ) from None
    return simulator_cls()


def get_calibrator(model_type: str) -> CalibratorBase:
    """A fresh calibrator for ``model_type``.

    Unlike the simulators these *are* stateful — ``goodness_of_fit()`` reads
    what ``calibrate()`` stored — so a per-request instance is required, not
    merely cheap.
    """
    try:
        calibrator_cls = _CALIBRATORS[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of {sorted(_CALIBRATORS)}"
        ) from None
    return calibrator_cls()
