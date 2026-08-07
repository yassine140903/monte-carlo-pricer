"""Tests for src/data/fetcher.py.

The "valid ticker" test hits the real yfinance API (network required). The
append-logic test stubs yfinance so it can assert on the requested date
range without depending on network/market state.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data import fetcher
from src.data.models import StockData
from src.data.storage import save_ohlcv


@pytest.mark.network
def test_fetch_ohlcv_returns_valid_data_for_aapl(engine):
    rows = fetcher.fetch_ohlcv("AAPL", engine=engine)

    assert len(rows) > 0
    assert all(isinstance(row, StockData) for row in rows)
    assert all(row.ticker == "AAPL" for row in rows)
    assert all(row.high >= row.low for row in rows)
    assert all(row.volume >= 0 for row in rows)


@pytest.mark.network
def test_fetch_ohlcv_handles_invalid_ticker_gracefully(engine):
    rows = fetcher.fetch_ohlcv("THISISNOTAREALTICKER123", engine=engine)

    assert rows == []


def test_fetch_ohlcv_appends_only_missing_dates(engine, monkeypatch):
    ticker = "AAPL"
    existing_latest = date(2024, 1, 5)
    save_ohlcv(
        [
            StockData(
                ticker=ticker,
                date=existing_latest,
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1_000,
            )
        ],
        engine=engine,
    )

    captured_kwargs = {}

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end, auto_adjust, actions):
            captured_kwargs["start"] = start
            captured_kwargs["end"] = end
            new_date = existing_latest + timedelta(days=1)
            return pd.DataFrame(
                {
                    "Open": [110.0],
                    "High": [112.0],
                    "Low": [109.0],
                    "Close": [111.0],
                    "Volume": [2_000],
                },
                index=pd.DatetimeIndex([pd.Timestamp(new_date)]),
            )

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)

    rows = fetcher.fetch_ohlcv(ticker, engine=engine)

    assert captured_kwargs["start"] == (existing_latest + timedelta(days=1)).isoformat()
    assert len(rows) == 1
    assert rows[0].date == existing_latest + timedelta(days=1)


def test_fetch_ohlcv_skips_call_when_already_up_to_date(engine, monkeypatch):
    ticker = "AAPL"
    save_ohlcv(
        [
            StockData(
                ticker=ticker,
                date=date.today(),
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=1_000,
            )
        ],
        engine=engine,
    )

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("yfinance should not be called when already up to date")

    monkeypatch.setattr(fetcher.yf, "Ticker", _unexpected_call)

    rows = fetcher.fetch_ohlcv(ticker, engine=engine)

    assert rows == []


def test_fetch_ohlcv_handles_empty_response(engine, monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)

    rows = fetcher.fetch_ohlcv("AAPL", engine=engine)

    assert rows == []
