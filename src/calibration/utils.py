"""Pure-NumPy helpers shared by the calibrators: log returns, annualization,
and rolling volatility. No pandas anywhere in src/calibration/.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns ln(S_t / S_{t-1}), with any resulting NaNs dropped.

    Prices are used as given (no forward-filling) — a NaN in the input
    simply produces a NaN return, which is then dropped.
    """
    prices = np.asarray(prices, dtype=float)
    returns = np.log(prices[1:] / prices[:-1])
    return returns[~np.isnan(returns)]


def annualize_mean(daily_mean: float, dt: float = 1 / 252) -> float:
    return daily_mean / dt


def annualize_volatility(daily_vol: float, dt: float = 1 / 252) -> float:
    return daily_vol / np.sqrt(dt)


def rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """Annualized rolling std of ``returns`` over ``window``-sized windows.

    Output length is ``len(returns) - window + 1``. Pure NumPy — uses a
    strided view rather than pandas.
    """
    returns = np.asarray(returns, dtype=float)
    n_windows = len(returns) - window + 1
    if n_windows <= 0:
        return np.empty(0, dtype=float)

    windows = np.lib.stride_tricks.sliding_window_view(returns, window)
    daily_std = windows.std(axis=-1, ddof=1)
    return annualize_volatility(daily_std)
