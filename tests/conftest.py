"""Shared fixtures for tests that talk to the real Postgres instance
started by docker-compose (see docker-compose.yml -> postgres service).
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from src.data.storage import create_tables, get_engine, ohlcv_table


@pytest.fixture(scope="session")
def db_engine():
    engine = get_engine()
    create_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def engine(db_engine):
    """A DB engine with the ohlcv table truncated before each test."""
    with db_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {ohlcv_table.name}"))
    return db_engine
