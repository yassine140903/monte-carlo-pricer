"""First-run data seeding, so a fresh `docker-compose up` has something to price.

Run as ``python -m src.seed`` from the project root. The entrypoint calls this
before starting uvicorn.

Deliberately idempotent and best-effort: if any ticker is already stored the
whole thing is skipped, and a yfinance failure (offline, rate limited, market
data outage) is logged rather than raised. Seeding is a convenience, not a
precondition — the API boots fine against an empty database, it just has no
tickers to offer until someone ingests data.
"""
from __future__ import annotations

import logging
import sys
import time

from sqlalchemy.exc import OperationalError

from src.data.fetcher import fetch_ohlcv
from src.data.storage import create_tables, get_available_tickers, save_ohlcv

logger = logging.getLogger(__name__)

SEED_TICKERS = ("AAPL", "GOOGL", "MSFT")

# Compose gates the backend on the Postgres healthcheck, so the database is up
# by the time this runs. The retry is for the gap between "accepting
# connections" and "ready to serve", which pg_isready does not always cover.
_CONNECT_ATTEMPTS = 10
_CONNECT_BACKOFF_SECONDS = 2.0


def _wait_for_database() -> bool:
    """Create the tables, retrying while Postgres is still coming up."""
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            create_tables()
            return True
        except OperationalError as exc:
            logger.info(
                "Database not ready (attempt %d/%d): %s",
                attempt,
                _CONNECT_ATTEMPTS,
                exc.orig or exc,
            )
            time.sleep(_CONNECT_BACKOFF_SECONDS)
    logger.error("Database did not become reachable; skipping seed.")
    return False


def seed() -> None:
    if not _wait_for_database():
        return

    existing = get_available_tickers()
    if existing:
        logger.info("Database already has data for %s; skipping seed.", ", ".join(existing))
        return

    logger.info("Seeding market data for %s...", ", ".join(SEED_TICKERS))
    for ticker in SEED_TICKERS:
        # fetch_ohlcv resolves its own start date from what is already stored
        # (nothing, here) and falls back to DEFAULT_LOOKBACK_YEARS of history.
        rows = fetch_ohlcv(ticker)
        if not rows:
            logger.warning("No data returned for %s; continuing.", ticker)
            continue
        save_ohlcv(rows)
        logger.info("Seeded %s with %d bars.", ticker, len(rows))

    seeded = get_available_tickers()
    if seeded:
        logger.info("Seed complete: %s.", ", ".join(seeded))
    else:
        logger.warning(
            "Seed produced no data. The API will start, but no tickers are "
            "available until data is ingested."
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [seed] %(message)s",
        stream=sys.stdout,
    )
    seed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
