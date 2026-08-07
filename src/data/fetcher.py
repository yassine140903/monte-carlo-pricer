"""yfinance wrapper that incrementally fetches OHLCV data.

Fetch strategy: look up the latest date already stored for a ticker and only
request data after that (append-only). If nothing is stored yet, backfill
``DEFAULT_LOOKBACK_YEARS`` of history. Prices are always split/dividend
adjusted (``auto_adjust=True``) so there is no separate adj_close column.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy import Engine

from src.config import settings
from src.data.models import StockData
from src.data.storage import get_date_range

logger = logging.getLogger(__name__)


def _lookback_start_date(years: int = settings.DEFAULT_LOOKBACK_YEARS) -> date:
    return date.today() - timedelta(days=365 * years)


def _resolve_start_date(ticker: str, engine: Engine | None) -> date:
    """Return the first date we still need to fetch for this ticker."""
    metadata = get_date_range(ticker, engine=engine)
    if metadata is None:
        return _lookback_start_date()
    return metadata.latest_date + timedelta(days=1)


def _dataframe_to_stock_data(ticker: str, df: pd.DataFrame) -> list[StockData]:
    rows: list[StockData] = []
    for idx, record in df.iterrows():
        bar_date = idx.date() if hasattr(idx, "date") else idx
        try:
            rows.append(
                StockData(
                    ticker=ticker,
                    date=bar_date,
                    open=float(record["Open"]),
                    high=float(record["High"]),
                    low=float(record["Low"]),
                    close=float(record["Close"]),
                    volume=int(record["Volume"]),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping unparseable bar for %s on %s: %s", ticker, bar_date, exc)
    return rows


def fetch_ohlcv(ticker: str, engine: Engine | None = None) -> list[StockData]:
    """Fetch new OHLCV bars for ``ticker`` since the last stored date.

    Returns an empty list (never raises) for invalid/delisted tickers, empty
    responses from yfinance, or when there is nothing new to fetch (e.g. the
    most recent trading day is already stored, or today is a weekend/holiday
    with no new bars).
    """
    ticker = ticker.upper()
    start = _resolve_start_date(ticker, engine)
    today = date.today()

    if start > today:
        logger.info("%s is already up to date (latest stored date is recent).", ticker)
        return []

    try:
        history = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(today + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            actions=False,
        )
    except Exception as exc:  # yfinance raises a variety of exception types
        logger.warning("Failed to fetch data for %s: %s", ticker, exc)
        return []

    if history is None or history.empty:
        logger.info("No data returned for %s in range %s to %s.", ticker, start, today)
        return []

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(history.columns):
        logger.warning("Unexpected response shape for %s; missing columns.", ticker)
        return []

    history = history.dropna(subset=list(required_cols))
    if history.empty:
        return []

    return _dataframe_to_stock_data(ticker, history)
