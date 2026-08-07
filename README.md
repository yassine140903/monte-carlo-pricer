# monte-carlo-pricer

Monte Carlo derivatives pricing engine: historical data ingestion, model
calibration, GPU/CPU simulation, pricing, and risk analytics, served behind a
FastAPI backend.

## Status

**M1 — Data Ingestion** is implemented: fetching daily OHLCV bars from
Yahoo Finance and persisting them to Postgres. Calibration, simulation,
pricing, risk, and the API/frontend layers are scaffolded but not yet
implemented.

## Project layout

```
src/
  config.py          Pydantic Settings (env-driven configuration)
  data/
    fetcher.py        yfinance wrapper, incremental fetch strategy
    storage.py         Postgres access (SQLAlchemy Core)
    models.py           StockData / TickerMetadata schemas
  calibration/        (M2)
  simulation/          (M3)
  pricing/               (M4)
  risk/                    (M5)
  api/                       FastAPI app + routers (later milestone)
tests/
frontend/              (later milestone)
mlflow/                 local MLflow artifact/tracking scratch space
```

## Setup

1. Copy `.env.example` to `.env` and adjust as needed.
2. Start Postgres and MLflow:

   ```
   docker compose up -d postgres mlflow
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create the database tables:

   ```
   python -c "from src.data.storage import create_tables; create_tables()"
   ```

## Fetching data

```python
from src.data.fetcher import fetch_ohlcv
from src.data.storage import save_ohlcv

rows = fetch_ohlcv("AAPL")
save_ohlcv(rows)
```

`fetch_ohlcv` only requests data after the latest date already stored for a
ticker (falling back to `DEFAULT_LOOKBACK_YEARS` of history on first fetch),
and returns split/dividend-adjusted prices (`auto_adjust=True`).

## Tests

Tests require a live Postgres instance (e.g. `docker compose up -d postgres`)
reachable via the settings in `.env`:

```
pytest
```

Tests marked `network` make real calls to yfinance; skip them with
`pytest -m "not network"` if you're offline.
