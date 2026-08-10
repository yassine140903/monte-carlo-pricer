"""Tests for options chain schemas, storage and fetching.

Storage tests require a live Postgres instance (see docker-compose.yml)
matching the connection settings in .env. The tests marked ``network`` hit
the real yfinance API.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data import fetcher
from src.data.models import OptionData
from src.data.storage import (
    get_available_expirations,
    load_options_chain,
    save_options_chain,
)


def _contract(
    ticker: str = "AAPL",
    expiration: date = date(2024, 6, 21),
    strike: float = 190.0,
    option_type: str = "call",
    fetched_at: date = date(2024, 1, 2),
    last_price: float = 5.25,
) -> OptionData:
    return OptionData(
        ticker=ticker,
        expiration_date=expiration,
        strike=strike,
        option_type=option_type,
        bid=last_price - 0.25,
        ask=last_price + 0.25,
        last_price=last_price,
        implied_volatility=0.28,
        volume=1_500,
        open_interest=12_000,
        fetched_at=fetched_at,
    )


# --- Schema -----------------------------------------------------------------


def test_option_data_uppercases_ticker():
    contract = _contract(ticker="aapl")

    assert contract.ticker == "AAPL"


def test_option_data_lowercases_option_type():
    assert _contract(option_type="CALL").option_type == "call"
    assert _contract(option_type="Put").option_type == "put"


def test_option_data_rejects_unknown_option_type():
    with pytest.raises(ValidationError):
        _contract(option_type="straddle")


def test_option_data_allows_missing_volume_and_open_interest():
    contract = OptionData(
        ticker="AAPL",
        expiration_date=date(2024, 6, 21),
        strike=190.0,
        option_type="put",
        bid=0.0,
        ask=0.0,
        last_price=0.01,
        implied_volatility=0.5,
        volume=None,
        open_interest=None,
        fetched_at=date(2024, 1, 2),
    )

    assert contract.volume is None
    assert contract.open_interest is None


# --- Storage ----------------------------------------------------------------


def test_save_and_load_options_roundtrip(engine):
    contract = _contract()

    save_options_chain([contract], engine=engine)
    loaded = load_options_chain("AAPL", engine=engine)

    assert len(loaded) == 1
    assert loaded[0].ticker == "AAPL"
    assert loaded[0].expiration_date == date(2024, 6, 21)
    assert loaded[0].strike == 190.0
    assert loaded[0].option_type == "call"
    assert loaded[0].last_price == pytest.approx(5.25)
    assert loaded[0].volume == 1_500
    assert loaded[0].open_interest == 12_000
    assert loaded[0].fetched_at == date(2024, 1, 2)


def test_save_options_chain_upsert_handles_duplicates(engine):
    original = _contract(last_price=5.25)
    updated = _contract(last_price=7.75)

    save_options_chain([original], engine=engine)
    save_options_chain([updated], engine=engine)  # same key -> update, not error

    loaded = load_options_chain("AAPL", engine=engine)
    assert len(loaded) == 1
    assert loaded[0].last_price == pytest.approx(7.75)


def test_save_options_chain_empty_list_is_noop(engine):
    save_options_chain([], engine=engine)

    assert load_options_chain("AAPL", engine=engine) == []


def test_load_options_chain_filters_by_expiration(engine):
    near = date(2024, 6, 21)
    far = date(2024, 9, 20)
    save_options_chain(
        [
            _contract(expiration=near, strike=190.0),
            _contract(expiration=near, strike=195.0),
            _contract(expiration=far, strike=190.0),
        ],
        engine=engine,
    )

    loaded = load_options_chain("AAPL", expiration_date=near, engine=engine)

    assert len(loaded) == 2
    assert all(row.expiration_date == near for row in loaded)
    assert [row.strike for row in loaded] == [190.0, 195.0]


def test_load_options_chain_defaults_to_most_recent_snapshot(engine):
    old_snapshot = date(2024, 1, 2)
    new_snapshot = date(2024, 1, 3)
    save_options_chain(
        [
            _contract(fetched_at=old_snapshot, last_price=5.25),
            _contract(fetched_at=new_snapshot, last_price=6.50),
        ],
        engine=engine,
    )

    loaded = load_options_chain("AAPL", engine=engine)

    assert len(loaded) == 1
    assert loaded[0].fetched_at == new_snapshot
    assert loaded[0].last_price == pytest.approx(6.50)


def test_load_options_chain_filters_by_explicit_fetched_at(engine):
    old_snapshot = date(2024, 1, 2)
    new_snapshot = date(2024, 1, 3)
    save_options_chain(
        [
            _contract(fetched_at=old_snapshot, last_price=5.25),
            _contract(fetched_at=new_snapshot, last_price=6.50),
        ],
        engine=engine,
    )

    loaded = load_options_chain("AAPL", fetched_at=old_snapshot, engine=engine)

    assert len(loaded) == 1
    assert loaded[0].fetched_at == old_snapshot
    assert loaded[0].last_price == pytest.approx(5.25)


def test_load_options_chain_is_ticker_scoped(engine):
    save_options_chain([_contract(ticker="AAPL"), _contract(ticker="MSFT")], engine=engine)

    loaded = load_options_chain("msft", engine=engine)

    assert len(loaded) == 1
    assert loaded[0].ticker == "MSFT"


def test_load_options_chain_unknown_ticker_returns_empty(engine):
    assert load_options_chain("DOESNOTEXIST", engine=engine) == []


def test_get_available_expirations(engine):
    expirations = [date(2024, 6, 21), date(2024, 9, 20), date(2025, 1, 17)]
    save_options_chain(
        [_contract(expiration=exp) for exp in reversed(expirations)]
        + [_contract(expiration=expirations[0], strike=195.0)],  # duplicate expiry
        engine=engine,
    )

    assert get_available_expirations("AAPL", engine=engine) == expirations


def test_get_available_expirations_unknown_ticker_returns_empty(engine):
    assert get_available_expirations("DOESNOTEXIST", engine=engine) == []


# --- Fetcher ----------------------------------------------------------------


@pytest.mark.network
def test_fetch_options_chain_returns_valid_data_for_aapl():
    records = fetcher.fetch_options_chain("AAPL")

    assert len(records) > 0
    assert all(isinstance(record, OptionData) for record in records)
    assert all(record.ticker == "AAPL" for record in records)
    assert all(record.option_type in ("call", "put") for record in records)
    assert all(record.strike > 0 for record in records)
    assert all(record.fetched_at == date.today() for record in records)
    assert all(record.expiration_date >= date.today() - timedelta(days=7) for record in records)


@pytest.mark.network
def test_fetch_options_chain_handles_invalid_ticker_gracefully():
    assert fetcher.fetch_options_chain("THISISNOTAREALTICKER123") == []


def _fake_chain_frame(strikes, implied_vols=None, bids=None):
    implied_vols = [0.3] * len(strikes) if implied_vols is None else implied_vols
    bids = [1.0] * len(strikes) if bids is None else bids
    return pd.DataFrame(
        {
            "strike": strikes,
            "bid": bids,
            "ask": [b + 0.1 for b in bids],
            "lastPrice": [b + 0.05 for b in bids],
            "impliedVolatility": implied_vols,
            "volume": [100] * len(strikes),
            "openInterest": [1_000] * len(strikes),
        }
    )


class _FakeChain:
    def __init__(self, calls, puts):
        self.calls = calls
        self.puts = puts


def _install_fake_ticker(monkeypatch, expirations, chains):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.options = tuple(expirations)

        def option_chain(self, expiration):
            return chains[expiration]

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)


def test_fetch_options_chain_skips_rows_with_missing_implied_vol(monkeypatch):
    expiration = "2024-06-21"
    _install_fake_ticker(
        monkeypatch,
        [expiration],
        {
            expiration: _FakeChain(
                calls=_fake_chain_frame([190.0, 195.0], implied_vols=[0.3, float("nan")]),
                puts=_fake_chain_frame([190.0], implied_vols=[0.4]),
            )
        },
    )

    records = fetcher.fetch_options_chain("AAPL")

    assert len(records) == 2
    assert {(r.option_type, r.strike) for r in records} == {("call", 190.0), ("put", 190.0)}
    assert all(r.expiration_date == date(2024, 6, 21) for r in records)
    assert all(r.fetched_at == date.today() for r in records)


def test_fetch_options_chain_keeps_zero_bid_contracts(monkeypatch):
    expiration = "2024-06-21"
    _install_fake_ticker(
        monkeypatch,
        [expiration],
        {
            expiration: _FakeChain(
                calls=_fake_chain_frame([300.0], bids=[0.0]),
                puts=_fake_chain_frame([10.0], bids=[float("nan")]),
            )
        },
    )

    records = fetcher.fetch_options_chain("AAPL")

    assert len(records) == 2
    assert all(record.bid == 0.0 for record in records)


def test_fetch_options_chain_returns_empty_when_no_expirations(monkeypatch):
    _install_fake_ticker(monkeypatch, [], {})

    assert fetcher.fetch_options_chain("AAPL") == []


def test_fetch_options_chain_skips_expirations_that_fail(monkeypatch):
    good, bad = "2024-06-21", "2024-09-20"

    class FakeTicker:
        def __init__(self, symbol):
            self.options = (good, bad)

        def option_chain(self, expiration):
            if expiration == bad:
                raise RuntimeError("yfinance blew up")
            return _FakeChain(
                calls=_fake_chain_frame([190.0]),
                puts=pd.DataFrame(),
            )

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)

    records = fetcher.fetch_options_chain("AAPL")

    assert len(records) == 1
    assert records[0].expiration_date == date(2024, 6, 21)
