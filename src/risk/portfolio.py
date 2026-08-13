"""Multi-asset portfolio simulation under an estimated correlation structure.

The assets are simulated by the ordinary single-asset engines in
src/simulation/ — each may use a different model — and correlation is imposed
from outside by handing every simulator its slice of one Cholesky-correlated
block of normals through ``z_extern``. Nothing about the correlation lives
inside the simulators, so a portfolio can mix GBM, jump-diffusion and Heston
assets freely.

This is a physical-measure simulation: the drift comes from each asset's
calibrated params, which is what VaR and CVaR should be measured under. Those
are statements about real-world risk, not option prices. Risk-neutral
portfolio simulation would additionally pass mu = r - q per asset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

from src.simulation.base import SimulatorBase
from src.simulation.utils import resolve_n_steps

_WEIGHT_TOLERANCE = 1e-6
_PSD_TOLERANCE = 1e-10
_RIDGE = 1e-10


@dataclass
class PortfolioAsset:
    """One holding: its weight, its spot, and the model calibrated to it.

    ``simulator`` and ``params`` are injected together and must agree —
    a HestonSimulator takes HestonParams. Different assets may use
    different models.
    """

    ticker: str
    weight: float
    S0: float
    simulator: SimulatorBase
    params: PydanticBaseModel


class PortfolioResult(PydanticBaseModel):
    """Arrays are kept as ndarrays, not nested lists — serializing them is the
    API layer's job, and doing it here would cost a full copy of every path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    portfolio_paths: np.ndarray
    asset_paths: dict[str, np.ndarray]
    weights: dict[str, float]
    correlation_matrix: np.ndarray
    tickers: list[str]


def estimate_correlation_matrix(
    returns: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Pearson correlation of aligned log-return series.

    Tickers are sorted alphabetically so the matrix is reproducible regardless
    of the caller's dict ordering, and the sorted list is returned alongside it
    as the authoritative row/column order.

    Aligning the series by date is the caller's responsibility — this function
    can only check that the lengths match, not that the dates do.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return series to estimate a correlation")

    tickers = sorted(returns)
    lengths = {ticker: len(returns[ticker]) for ticker in tickers}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All return series must have the same length, got {lengths}. "
            "Align the series by date before estimating correlation."
        )

    matrix = np.vstack([np.asarray(returns[ticker], dtype=float) for ticker in tickers])
    return np.corrcoef(matrix), tickers


def _validate_portfolio(
    assets: list[PortfolioAsset], correlation_matrix: np.ndarray
) -> None:
    if len(assets) < 2:
        raise ValueError(
            "simulate_portfolio needs at least two assets; simulate a single "
            "asset with its simulator directly"
        )

    total_weight = sum(asset.weight for asset in assets)
    if abs(total_weight - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(f"Asset weights must sum to 1.0, got {total_weight}")

    n_assets = len(assets)
    if correlation_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"correlation_matrix must have shape {(n_assets, n_assets)} for "
            f"{n_assets} assets, got {correlation_matrix.shape}"
        )
    if not np.allclose(correlation_matrix, correlation_matrix.T):
        raise ValueError("Correlation matrix is not symmetric")
    if not np.allclose(np.diag(correlation_matrix), 1.0):
        raise ValueError("Correlation matrix must have 1.0 on the diagonal")
    if not np.all(np.linalg.eigvalsh(correlation_matrix) >= -_PSD_TOLERANCE):
        raise ValueError("Correlation matrix is not positive semi-definite")


def _cholesky(correlation_matrix: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L @ L.T == correlation_matrix.

    A matrix that is positive *semi*-definite — one asset an exact linear
    combination of the others, or perfect correlation — passes the eigenvalue
    check but can still fail Cholesky, which needs strict positivity. A ridge
    far below any real correlation lifts the zero eigenvalue just enough.
    """
    try:
        return np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        ridged = correlation_matrix + _RIDGE * np.eye(correlation_matrix.shape[0])
        try:
            return np.linalg.cholesky(ridged)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Cholesky decomposition of the correlation matrix failed even "
                "after ridging; the matrix is too close to singular"
            ) from exc


def simulate_portfolio(
    assets: list[PortfolioAsset],
    correlation_matrix: np.ndarray,
    T: float,
    dt: float,
    n_simulations: int,
    initial_value: float = 1.0,
    seed: int | None = None,
) -> PortfolioResult:
    """Simulate a weighted basket of correlated assets.

    ``assets`` must be in the same order as the rows and columns of
    ``correlation_matrix`` — pair it with the ticker list returned by
    :func:`estimate_correlation_matrix`. Only the count can be validated here;
    a mismatched *ordering* is silently wrong, so build the two together.

    Args:
        assets: At least two holdings, weights summing to 1.
        correlation_matrix: Symmetric PSD, unit diagonal, ordered as ``assets``.
        T: Horizon in years.
        dt: Time step in years.
        n_simulations: Number of paths.
        initial_value: Level the portfolio starts at; paths scale from here.
        seed: Master seed. Drives the correlated normals, and is offset per
            asset for whatever internal randomness each model has left.

    Returns:
        PortfolioResult carrying the blended paths, the per-asset paths, and
        the correlation structure they were drawn under.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)
    _validate_portfolio(assets, correlation_matrix)

    n_assets = len(assets)
    n_steps = resolve_n_steps(T, dt)

    # One correlated block for the whole portfolio: draw independent normals,
    # then left-multiply by L so that across assets the shocks carry the target
    # correlation while each asset's own marginal stays standard normal.
    rng = np.random.default_rng(seed)
    z_independent = rng.standard_normal((n_assets, n_simulations, n_steps))

    correlated = _cholesky(correlation_matrix) @ z_independent.reshape(n_assets, -1)
    z_correlated = correlated.reshape(n_assets, n_simulations, n_steps)

    asset_paths: dict[str, np.ndarray] = {}
    return_paths = np.empty((n_assets, n_simulations, n_steps + 1))

    for index, asset in enumerate(assets):
        # The correlated normals are the whole of the price-driving randomness,
        # but jump counts and Heston's variance process still draw internally.
        # Offsetting the master seed keeps those reproducible without making
        # two assets share a stream.
        asset_seed = None if seed is None else seed + index

        paths = asset.simulator.simulate(
            S0=asset.S0,
            params=asset.params,
            T=T,
            dt=dt,
            n_simulations=n_simulations,
            seed=asset_seed,
            z_extern=z_correlated[index],
        )
        asset_paths[asset.ticker] = paths
        # Normalized to start at 1.0, so assets of very different price levels
        # contribute strictly in proportion to their weights.
        return_paths[index] = paths / asset.S0

    weights = np.array([asset.weight for asset in assets])
    portfolio_paths = initial_value * np.tensordot(weights, return_paths, axes=(0, 0))

    return PortfolioResult(
        portfolio_paths=portfolio_paths,
        asset_paths=asset_paths,
        weights={asset.ticker: asset.weight for asset in assets},
        correlation_matrix=correlation_matrix,
        tickers=[asset.ticker for asset in assets],
    )
