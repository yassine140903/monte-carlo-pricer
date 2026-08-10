"""yfinance wrapper that fetches OHLCV bars and options chains.

OHLCV fetch strategy: look up the latest date already stored for a ticker and
only request data after that (append-only). If nothing is stored yet, backfill
``DEFAULT_LOOKBACK_YEARS`` of history. Prices are always split/dividend
adjusted (``auto_adjust=True``) so there is no separate adj_close column.

Options are different: there is no append-only logic because a chain is a
snapshot of the *current* market, not a fixed historical record. Every fetch
pulls the whole chain stamped with ``fetched_at=date.today()``, and the
composite primary key in storage handles deduplication.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy import Engine

from src.config import settings
from src.data.models import OptionData, StockData
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


_OPTION_PRICE_COLS = ("strike", "bid", "ask", "lastPrice", "impliedVolatility")


def _optional_int(value: object) -> int | None:
    """Coerce a yfinance volume/open-interest cell to int, or None if missing."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _quoted_price(value: object) -> float:
    """Coerce a bid/ask/last cell to float, treating a missing quote as 0.0.

    Deep out-of-the-money contracts legitimately have no bid, which yfinance
    reports as either 0.0 or NaN; both mean the same thing here.
    """
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _dataframe_to_option_data(
    ticker: str,
    df: pd.DataFrame,
    option_type: str,
    expiration: date,
    fetched_at: date,
) -> list[OptionData]:
    rows: list[OptionData] = []
    for _, record in df.iterrows():
        if pd.isna(record["strike"]) or pd.isna(record["impliedVolatility"]):
            logger.debug(
                "Skipping %s %s option with missing strike/IV for expiry %s.",
                ticker,
                option_type,
                expiration,
            )
            continue
        try:
            rows.append(
                OptionData(
                    ticker=ticker,
                    expiration_date=expiration,
                    strike=float(record["strike"]),
                    option_type=option_type,
                    bid=_quoted_price(record["bid"]),
                    ask=_quoted_price(record["ask"]),
                    last_price=_quoted_price(record["lastPrice"]),
                    implied_volatility=float(record["impliedVolatility"]),
                    volume=_optional_int(record.get("volume")),
                    open_interest=_optional_int(record.get("openInterest")),
                    fetched_at=fetched_at,
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Skipping unparseable %s contract for %s expiring %s: %s",
                option_type,
                ticker,
                expiration,
                exc,
            )
    return rows


def fetch_options_chain(ticker: str) -> list[OptionData]:
    """Fetch the full current options chain for ``ticker``.

    Walks every expiration yfinance advertises and returns both calls and puts
    as a snapshot stamped with today's date. Rows with a missing implied
    volatility are dropped (they are unusable for calibration); zero bids/asks
    are kept, since deep OTM contracts genuinely trade with no bid.

    Returns an empty list (never raises) for invalid/delisted tickers or when
    yfinance reports no listed options.
    """
    ticker = ticker.upper()
    fetched_at = date.today()

    try:
        yf_ticker = yf.Ticker(ticker)
        expirations = yf_ticker.options
    except Exception as exc:  # yfinance raises a variety of exception types
        logger.warning("Failed to list option expirations for %s: %s", ticker, exc)
        return []

    if not expirations:
        logger.info("No listed options returned for %s.", ticker)
        return []

    records: list[OptionData] = []
    for expiration in expirations:
        try:
            expiration_date = date.fromisoformat(expiration)
        except (TypeError, ValueError):
            logger.warning("Skipping unparseable expiration %r for %s.", expiration, ticker)
            continue

        try:
            chain = yf_ticker.option_chain(expiration)
        except Exception as exc:  # yfinance raises a variety of exception types
            logger.warning(
                "Failed to fetch %s option chain for expiry %s: %s", ticker, expiration, exc
            )
            continue

        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            if not set(_OPTION_PRICE_COLS).issubset(frame.columns):
                logger.warning(
                    "Unexpected %s chain shape for %s expiring %s; missing columns.",
                    option_type,
                    ticker,
                    expiration,
                )
                continue
            records.extend(
                _dataframe_to_option_data(
                    ticker, frame, option_type, expiration_date, fetched_at
                )
            )

    if not records:
        logger.info("No usable option contracts parsed for %s.", ticker)

    return records
