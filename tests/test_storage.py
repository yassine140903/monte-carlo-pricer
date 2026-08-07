"""Tests for src/data/storage.py. Requires a live Postgres instance
(see docker-compose.yml) matching the connection settings in .env.
"""
from __future__ import annotations

from datetime import date, timedelta

from src.data.models import StockData
from src.data.storage import (
    get_available_tickers,
    get_date_range,
    load_ohlcv,
    save_ohlcv,
)


def _bar(ticker: str, day: date, close: float = 100.0) -> StockData:
    return StockData(
        ticker=ticker,
        date=day,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1_000_000,
    )


def test_save_and_load_roundtrip(engine):
    day = date(2024, 1, 2)
    bar = _bar("AAPL", day, close=185.5)

    save_ohlcv([bar], engine=engine)
    loaded = load_ohlcv("AAPL", engine=engine)

    assert len(loaded) == 1
    assert loaded[0].ticker == "AAPL"
    assert loaded[0].date == day
    assert loaded[0].close == 185.5
    assert loaded[0].volume == 1_000_000


def test_save_ohlcv_upsert_handles_duplicates(engine):
    day = date(2024, 1, 2)
    original = _bar("AAPL", day, close=100.0)
    updated = _bar("AAPL", day, close=123.45)

    save_ohlcv([original], engine=engine)
    save_ohlcv([updated], engine=engine)  # same (ticker, date) -> should update, not error

    loaded = load_ohlcv("AAPL", engine=engine)
    assert len(loaded) == 1
    assert loaded[0].close == 123.45


def test_load_ohlcv_respects_date_bounds(engine):
    ticker = "MSFT"
    bars = [_bar(ticker, date(2024, 1, 1) + timedelta(days=i)) for i in range(5)]
    save_ohlcv(bars, engine=engine)

    loaded = load_ohlcv(
        ticker,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        engine=engine,
    )

    assert [b.date for b in loaded] == [date(2024, 1, 2), date(2024, 1, 3)]


def test_get_available_tickers(engine):
    save_ohlcv([_bar("AAPL", date(2024, 1, 2))], engine=engine)
    save_ohlcv([_bar("MSFT", date(2024, 1, 2))], engine=engine)
    save_ohlcv([_bar("GOOGL", date(2024, 1, 2))], engine=engine)

    tickers = get_available_tickers(engine=engine)

    assert tickers == ["AAPL", "GOOGL", "MSFT"]


def test_get_available_tickers_empty(engine):
    assert get_available_tickers(engine=engine) == []


def test_get_date_range(engine):
    ticker = "NVDA"
    bars = [_bar(ticker, date(2024, 1, 1) + timedelta(days=i)) for i in range(10)]
    save_ohlcv(bars, engine=engine)

    metadata = get_date_range(ticker, engine=engine)

    assert metadata is not None
    assert metadata.ticker == ticker
    assert metadata.earliest_date == date(2024, 1, 1)
    assert metadata.latest_date == date(2024, 1, 10)
    assert metadata.row_count == 10


def test_get_date_range_unknown_ticker_returns_none(engine):
    assert get_date_range("DOESNOTEXIST", engine=engine) is None
