"""The FastAPI application: wiring, CORS, lifespan and error translation.

The API owns no quantitative logic. Every endpoint parses a request, calls
into src/data, src/calibration, src/simulation, src/pricing or src/risk, and
serializes what comes back. The one thing it does own is how a domain error
becomes an HTTP status:

- ValueError — the request was well formed but the numbers do not work
  (weights that do not sum to 1, a horizon shorter than one step). 422.
- KeyError — something named in the request does not exist: an unknown
  ticker, scenario or model type. 404.
- SimulationTooLargeError — a valid request that is simply too big. 413.

Domain modules raise the plain built-ins; only the size limit, which is an API
policy rather than a modelling constraint, gets its own exception type.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routers import calibration, data, mlflow_runs, pricing, risk, simulation
from src.api.schemas import ErrorResponse
from src.api.utils import SimulationTooLargeError
from src.data.storage import get_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the connection pool once for the process, dispose it on shutdown.

    ``create_engine`` does not connect eagerly, so a database that is down at
    startup delays the failure to the first query that needs it rather than
    preventing the app from booting.
    """
    app.state.engine = get_engine()
    try:
        yield
    finally:
        app.state.engine.dispose()


def _error_response(
    status_code: int, error: str, detail: str, context: dict | None = None
) -> JSONResponse:
    payload = ErrorResponse(error=error, detail=detail, context=context or {})
    return JSONResponse(status_code=status_code, content=payload.model_dump())


_HTTP_ERROR_NAMES = {
    404: "not_found",
    405: "method_not_allowed",
    413: "simulation_too_large",
    422: "invalid_request",
}


def _jsonable_errors(exc: RequestValidationError) -> list[dict]:
    """Validation errors with the offending input stripped out.

    ``ValidationError.errors()`` embeds the raw input under ``ctx``/``input``,
    which for a simulation body can be a large object and is not always JSON
    serializable. Only the location, message and type are reported.
    """
    return [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in exc.errors()
    ]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Monte Carlo Options Pricing API",
        description=(
            "Calibration, path simulation, option pricing and risk analytics "
            "over the GBM, Merton jump-diffusion and Heston models."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Wide open for now: the service is read-only-ish compute with no auth and
    # no user data, and the frontend origin is not yet fixed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SimulationTooLargeError)
    async def _too_large(request: Request, exc: SimulationTooLargeError) -> JSONResponse:
        return _error_response(
            413,
            "simulation_too_large",
            str(exc),
            {"n_simulations": exc.n_simulations, "limit": exc.limit},
        )

    @app.exception_handler(KeyError)
    async def _not_found(request: Request, exc: KeyError) -> JSONResponse:
        # KeyError stringifies with its own quotes; args[0] is the message.
        detail = str(exc.args[0]) if exc.args else "Not found"
        return _error_response(404, "not_found", detail)

    @app.exception_handler(ValueError)
    async def _invalid(request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(422, "invalid_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            422,
            "validation_error",
            "Request body failed validation",
            {"errors": _jsonable_errors(exc)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(
            exc.status_code,
            _HTTP_ERROR_NAMES.get(exc.status_code, "http_error"),
            str(exc.detail),
        )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    for router in (data, calibration, simulation, pricing, risk, mlflow_runs):
        app.include_router(router.router)

    return app


app = create_app()
