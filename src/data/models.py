"""Pydantic schemas for OHLCV market data."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class StockData(BaseModel):
    """A single daily OHLCV bar for a ticker.

    All price fields are already split/dividend adjusted (yfinance
    auto_adjust=True) — there is no separate adj_close.
    """

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, v: str) -> str:
        return v.upper()


class TickerMetadata(BaseModel):
    """Summary of the data available for a ticker."""

    ticker: str
    earliest_date: date
    latest_date: date
    row_count: int = Field(ge=0)
