# monte-carlo-pricer

Monte Carlo derivatives pricing engine: historical data ingestion, model
calibration, GPU/CPU simulation, pricing, and risk analytics, served behind a
FastAPI backend.

## Status

**M1–M6** are implemented: data ingestion, model calibration, path simulation,
option pricing, risk analytics, and the FastAPI backend that serves them. The
frontend is still scaffolding.

## Project layout

```
src/
  config.py          Pydantic Settings (env-driven configuration)
  data/
    fetcher.py        yfinance wrapper, incremental fetch strategy
    storage.py         Postgres access (SQLAlchemy Core)
    models.py           StockData / TickerMetadata schemas
  calibration/        GBM, Merton jump-diffusion, Heston fitting
  simulation/          path engines + variance reduction
  pricing/               Monte Carlo pricing, payoffs, Black-Scholes, Greeks
  risk/                    VaR/CVaR, correlated portfolios, stress scenarios
  api/                       FastAPI app + routers
tests/
frontend/              (later milestone)
mlflow/                 local MLflow artifact/tracking scratch space
```

Everything under `src/calibration`, `src/simulation`, `src/pricing` and
`src/risk` is pure NumPy/SciPy: no pandas, no MLflow, no I/O. `src/api` is the
only layer that talks to the database or the tracking server.

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

## Running the API

```
uvicorn src.api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | What it does |
| --- | --- |
| `GET /health` | Liveness check |
| `GET /data/tickers` | Tickers held in storage, with date ranges |
| `GET /data/{ticker}` | Stored OHLCV bars, optionally date-bounded |
| `POST /calibrate` | Fit GBM / jump-diffusion / Heston to stored prices |
| `POST /simulate` | Path fan chart under the **physical** measure |
| `POST /price-option` | Monte Carlo price + Greeks under the **risk-neutral** measure |
| `POST /risk/metrics` | Single-asset VaR, CVaR, drawdown, Sharpe |
| `POST /risk/portfolio` | Correlated multi-asset risk, optionally stressed |
| `GET /risk/scenarios` | Available preset stress scenarios |
| `GET /mlflow/experiments` | Experiments logged to the tracking server |

Two things worth knowing about the contract:

- **Measure.** `/simulate` and the `/risk/*` endpoints keep the calibrated
  real-world drift; `/price-option` overrides it with `r - q`. Comparing a
  simulated mean against an option price will not line up, by design.
- **Sign.** VaR and CVaR come back as raw P&L quantiles, so a loss is
  *negative*. Flip the sign in the presentation layer if a report wants the
  usual positive figure.

Requests are capped at 100,000 simulated paths; over that the API answers
`413` rather than trying. MLflow logging is best-effort — if the tracking
server is unreachable the response still returns, with `mlflow_run_id: null`.

## Tests

Tests require a live Postgres instance (e.g. `docker compose up -d postgres`)
reachable via the settings in `.env`:

```
pytest
```

Tests marked `network` make real calls to yfinance; skip them with
`pytest -m "not network"` if you're offline.
