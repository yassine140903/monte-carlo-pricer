"""Postgres storage layer for OHLCV data, built on SQLAlchemy Core.

Deliberately uses Core (Table/select/insert) rather than the ORM — this is a
thin, explicit query-builder layer, not a domain model.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import (
    REAL,
    BigInteger,
    Column,
    Date,
    Engine,
    MetaData,
    Table,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import settings
from src.data.models import StockData, TickerMetadata

metadata = MetaData()

ohlcv_table = Table(
    "ohlcv",
    metadata,
    Column("ticker", Text, primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", REAL, nullable=False),
    Column("high", REAL, nullable=False),
    Column("low", REAL, nullable=False),
    Column("close", REAL, nullable=False),
    Column("volume", BigInteger, nullable=False),
)


def get_engine() -> Engine:
    """Create a pooled SQLAlchemy engine from application settings."""
    return create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        future=True,
    )


_default_engine: Engine | None = None


def _resolve_engine(engine: Engine | None) -> Engine:
    global _default_engine
    if engine is not None:
        return engine
    if _default_engine is None:
        _default_engine = get_engine()
    return _default_engine


def create_tables(engine: Engine | None = None) -> None:
    """Create the ohlcv table if it doesn't already exist."""
    engine = _resolve_engine(engine)
    metadata.create_all(engine)


def save_ohlcv(rows: list[StockData], engine: Engine | None = None) -> None:
    """Upsert a batch of OHLCV rows (insert, or update on ticker+date conflict)."""
    if not rows:
        return

    engine = _resolve_engine(engine)
    payload = [
        {
            "ticker": row.ticker,
            "date": row.date,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in rows
    ]

    stmt = pg_insert(ohlcv_table).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "date"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        },
    )

    with engine.begin() as conn:
        conn.execute(stmt)


def load_ohlcv(
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
    engine: Engine | None = None,
) -> list[StockData]:
    """Load OHLCV rows for a ticker, optionally bounded by [start_date, end_date]."""
    engine = _resolve_engine(engine)
    ticker = ticker.upper()

    stmt = select(ohlcv_table).where(ohlcv_table.c.ticker == ticker)
    if start_date is not None:
        stmt = stmt.where(ohlcv_table.c.date >= start_date)
    if end_date is not None:
        stmt = stmt.where(ohlcv_table.c.date <= end_date)
    stmt = stmt.order_by(ohlcv_table.c.date)

    with engine.connect() as conn:
        result = conn.execute(stmt)
        return [StockData(**row._mapping) for row in result]


def get_available_tickers(engine: Engine | None = None) -> list[str]:
    """Return the sorted list of distinct tickers present in storage."""
    engine = _resolve_engine(engine)
    stmt = select(ohlcv_table.c.ticker).distinct().order_by(ohlcv_table.c.ticker)
    with engine.connect() as conn:
        return [row.ticker for row in conn.execute(stmt)]


def get_date_range(ticker: str, engine: Engine | None = None) -> TickerMetadata | None:
    """Return earliest/latest date and row count for a ticker, or None if absent."""
    engine = _resolve_engine(engine)
    ticker = ticker.upper()

    stmt = select(
        func.min(ohlcv_table.c.date).label("earliest_date"),
        func.max(ohlcv_table.c.date).label("latest_date"),
        func.count().label("row_count"),
    ).where(ohlcv_table.c.ticker == ticker)

    with engine.connect() as conn:
        row = conn.execute(stmt).one()

    if row.row_count == 0 or row.earliest_date is None:
        return None

    return TickerMetadata(
        ticker=ticker,
        earliest_date=row.earliest_date,
        latest_date=row.latest_date,
        row_count=row.row_count,
    )
