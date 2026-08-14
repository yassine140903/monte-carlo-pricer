"""FastAPI layer over the calibration, simulation, pricing and risk modules.

This package contains no quantitative logic of its own. It validates request
shapes, calls into src/, translates domain errors into HTTP statuses, and
thins large arrays down to something a browser can plot.

Run it with ``uvicorn src.api.main:app``.
"""
from __future__ import annotations
