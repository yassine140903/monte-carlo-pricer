"""Market data endpoints."""
from __future__ import annotations

from tests.test_api.conftest import FAKE_TICKERS, N_BARS


class TestListTickers:
    def test_returns_every_stored_ticker_with_metadata(self, client):
        response = client.get("/data/tickers")

        assert response.status_code == 200
        body = response.json()
        assert [entry["ticker"] for entry in body] == sorted(FAKE_TICKERS)
        assert all(entry["row_count"] == N_BARS for entry in body)

    def test_metadata_carries_the_stored_date_range(self, client, market_data):
        entry = next(e for e in client.get("/data/tickers").json() if e["ticker"] == "AAPL")

        bars = market_data["AAPL"]
        assert entry["earliest_date"] == bars[0].date.isoformat()
        assert entry["latest_date"] == bars[-1].date.isoformat()


class TestGetOHLCV:
    def test_returns_all_stored_bars_by_default(self, client):
        response = client.get("/data/AAPL")

        assert response.status_code == 200
        body = response.json()
        assert body["ticker"] == "AAPL"
        assert body["row_count"] == N_BARS
        assert len(body["bars"]) == N_BARS

    def test_bars_carry_the_full_ohlcv_row(self, client):
        bar = client.get("/data/AAPL").json()["bars"][0]

        assert set(bar) == {"ticker", "date", "open", "high", "low", "close", "volume"}
        assert bar["low"] <= bar["close"] <= bar["high"]

    def test_ticker_is_case_insensitive(self, client):
        assert client.get("/data/aapl").json()["ticker"] == "AAPL"

    def test_respects_the_date_bounds(self, client, market_data):
        bars = market_data["MSFT"]
        start, end = bars[10].date, bars[19].date

        body = client.get(
            "/data/MSFT", params={"start_date": start.isoformat(), "end_date": end.isoformat()}
        ).json()

        assert body["row_count"] == 10
        assert body["start_date"] == start.isoformat()
        assert body["end_date"] == end.isoformat()

    def test_unknown_ticker_is_404(self, client):
        response = client.get("/data/NOSUCH")

        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "not_found"
        assert "NOSUCH" in body["detail"]

    def test_a_window_with_no_bars_is_empty_not_404(self, client):
        """The ticker exists; the caller just asked about a quiet period."""
        response = client.get(
            "/data/AAPL", params={"start_date": "1990-01-01", "end_date": "1990-12-31"}
        )

        assert response.status_code == 200
        assert response.json()["row_count"] == 0

    def test_reversed_date_bounds_are_rejected(self, client):
        response = client.get(
            "/data/AAPL", params={"start_date": "2022-01-01", "end_date": "2021-01-01"}
        )

        assert response.status_code == 422
