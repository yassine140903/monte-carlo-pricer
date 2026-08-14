"""Fixtures for the API tests.

The storage layer is stubbed rather than hit for real. These tests are about
the HTTP contract — status codes, response shapes, which module gets called —
and none of that is better tested by round-tripping through Postgres, which
tests/test_storage.py already covers directly.

MLflow is pointed at a temporary local file store, so logging genuinely
happens (and can be asserted on) without a tracking server running.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.data import storage
from src.data.models import StockData, TickerMetadata

# Correlated by construction, so the portfolio endpoint gets a correlation
# matrix with real off-diagonal structure rather than noise around zero.
FAKE_TICKERS = ("AAPL", "GOOGL", "MSFT")
N_BARS = 800
START_PRICES = {"AAPL": 180.0, "GOOGL": 140.0, "MSFT": 400.0}

GBM_PARAMS = {"model_type": "gbm", "mu": 0.08, "sigma": 0.25}
JUMP_PARAMS = {
    "model_type": "jump_diffusion",
    "mu": 0.08,
    "sigma": 0.2,
    "lambda_j": 5.0,
    "mu_j": -0.02,
    "sigma_j": 0.05,
}
HESTON_PARAMS = {
    "model_type": "heston",
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.6,
    "v0": 0.04,
    "feller_satisfied": True,
}


def _build_market_data() -> dict[str, list[StockData]]:
    rng = np.random.default_rng(20240814)

    market = rng.standard_normal(N_BARS)
    start = date(2021, 1, 4)

    data: dict[str, list[StockData]] = {}
    for index, ticker in enumerate(FAKE_TICKERS):
        idiosyncratic = rng.standard_normal(N_BARS)
        # ~0.6 loading on a common factor: enough correlation to be visible,
        # not so much that the matrix goes singular.
        returns = 0.0003 + 0.012 * (0.6 * market + 0.8 * idiosyncratic)
        closes = START_PRICES[ticker] * np.exp(np.cumsum(returns))

        bars = []
        for offset, close in enumerate(closes):
            # Weekday-only calendar, so the three tickers share trading days.
            day = start + timedelta(days=offset + 2 * (offset // 5))
            close = float(close)
            bars.append(
                StockData(
                    ticker=ticker,
                    date=day,
                    open=close * 0.995,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=1_000_000 + index,
                )
            )
        data[ticker] = bars
    return data


@pytest.fixture(scope="session")
def market_data() -> dict[str, list[StockData]]:
    return _build_market_data()


@pytest.fixture()
def stub_storage(monkeypatch, market_data):
    """Replace the three storage reads the API makes with in-memory lookups."""

    def load_ohlcv(ticker, start_date=None, end_date=None, engine=None):
        bars = market_data.get(ticker.upper(), [])
        if start_date is not None:
            bars = [bar for bar in bars if bar.date >= start_date]
        if end_date is not None:
            bars = [bar for bar in bars if bar.date <= end_date]
        return list(bars)

    def get_available_tickers(engine=None):
        return sorted(market_data)

    def get_date_range(ticker, engine=None):
        bars = market_data.get(ticker.upper())
        if not bars:
            return None
        return TickerMetadata(
            ticker=ticker.upper(),
            earliest_date=bars[0].date,
            latest_date=bars[-1].date,
            row_count=len(bars),
        )

    monkeypatch.setattr(storage, "load_ohlcv", load_ohlcv)
    monkeypatch.setattr(storage, "get_available_tickers", get_available_tickers)
    monkeypatch.setattr(storage, "get_date_range", get_date_range)
    return market_data


@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory) -> str:
    return tmp_path_factory.mktemp("mlruns").as_uri()


@pytest.fixture(autouse=True)
def local_mlflow(monkeypatch, mlflow_tracking_uri):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", mlflow_tracking_uri)
    return mlflow_tracking_uri


@pytest.fixture()
def client(stub_storage) -> TestClient:
    """A TestClient with the app's lifespan run.

    Built inside the ``with`` block so startup and shutdown actually fire —
    ``get_engine`` reads app.state.engine, which only exists once the lifespan
    has run. SQLAlchemy does not connect on create_engine, so no database is
    needed for the engine itself to exist.
    """
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client
