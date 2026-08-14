"""Market data endpoints — thin reads over src/data/storage.py.

Nothing here fetches from yfinance: the API serves what has already been
ingested. Populating the database is the fetcher's job, run out of band.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Engine

from src.api.dependencies import get_engine
from src.api.schemas import OHLCVResponse
from src.data import storage
from src.data.models import TickerMetadata

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/tickers", response_model=list[TickerMetadata])
async def list_tickers(engine: Engine = Depends(get_engine)) -> list[TickerMetadata]:
    """Every ticker held in storage, with its date range and row count."""
    tickers = storage.get_available_tickers(engine)
    metadata = [storage.get_date_range(ticker, engine) for ticker in tickers]
    # get_date_range returns None for a ticker with no rows, which cannot
    # happen for a ticker that came out of the distinct query — unless a
    # concurrent delete landed between the two, so it is filtered rather than
    # allowed to become a null in the response.
    return [entry for entry in metadata if entry is not None]


@router.get("/{ticker}", response_model=OHLCVResponse)
async def get_ohlcv(
    ticker: str,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    engine: Engine = Depends(get_engine),
) -> OHLCVResponse:
    """Stored daily bars for ``ticker``, optionally bounded by a date range.

    An unknown ticker is a 404. A *known* ticker with no bars inside the
    requested window is not — that is an empty result for a valid question.
    """
    ticker = ticker.upper()
    if storage.get_date_range(ticker, engine) is None:
        raise KeyError(f"No stored data for ticker {ticker!r}")

    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must not be after end_date")

    bars = storage.load_ohlcv(ticker, start_date, end_date, engine)
    return OHLCVResponse(
        ticker=ticker,
        start_date=bars[0].date if bars else start_date,
        end_date=bars[-1].date if bars else end_date,
        row_count=len(bars),
        bars=bars,
    )
