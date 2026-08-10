"""Pydantic schemas for OHLCV and options chain market data."""
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


class OptionData(BaseModel):
    """A single option contract quote, as of ``fetched_at``.

    Unlike OHLCV bars — which are fixed once a trading day closes — options
    quotes change every day, so every row is a dated snapshot and
    ``fetched_at`` is part of its identity.
    """

    ticker: str
    expiration_date: date
    strike: float
    option_type: str
    bid: float
    ask: float
    last_price: float
    implied_volatility: float
    volume: int | None = None
    open_interest: int | None = None
    fetched_at: date

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, v: str) -> str:
        return v.upper()

    @field_validator("option_type")
    @classmethod
    def _normalise_option_type(cls, v: str) -> str:
        normalised = v.lower()
        if normalised not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        return normalised


class TickerMetadata(BaseModel):
    """Summary of the data available for a ticker."""

    ticker: str
    earliest_date: date
    latest_date: date
    row_count: int = Field(ge=0)
