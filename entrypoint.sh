#!/bin/sh
set -e

# Seeding is best-effort by design (see src/seed.py): a yfinance outage should
# not stop the API from booting, so a non-zero exit here is logged and ignored.
echo "[entrypoint] Running first-run data seed..."
python -m src.seed || echo "[entrypoint] Seed failed; starting the API anyway."

echo "[entrypoint] Starting API on 0.0.0.0:8000..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
