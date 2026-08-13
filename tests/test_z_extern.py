"""Tests for the ``z_extern`` hook on the simulators.

``z_extern`` lets src/risk/portfolio.py drive several simulators from one
block of correlated normals. The properties that matter are that it fully
determines the price-driving noise, that it composes with ``seed`` for the
randomness it does *not* cover (jumps, Heston's variance process), and that
leaving it off changes nothing at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.gbm import GBMParams
from src.calibration.heston import HestonParams
from src.calibration.jump_diffusion import JumpDiffusionParams
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.jump_diffusion import JumpDiffusionSimulator

DT = 1 / 252
T = 0.5
N_SIMS = 500
N_STEPS = 126
S0 = 100.0

GBM_PARAMS = GBMParams(mu=0.08, sigma=0.20)
JD_PARAMS = JumpDiffusionParams(mu=0.08, sigma=0.20, lambda_j=1.0, mu_j=-0.03, sigma_j=0.08)
HESTON_PARAMS = HestonParams(
    kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, v0=0.04, feller_satisfied=True
)

SIMULATORS = [
    (GBMSimulator(), GBM_PARAMS),
    (JumpDiffusionSimulator(), JD_PARAMS),
    (HestonSimulator(), HESTON_PARAMS),
]
SIMULATOR_IDS = ["gbm", "jump_diffusion", "heston"]


def make_z(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_SIMS, N_STEPS))


def simulate(simulator, params, **kwargs) -> np.ndarray:
    return simulator.simulate(S0, params, T, DT, N_SIMS, **kwargs)


# --------------------------------------------------------------------------
# Determinism and equivalence
# --------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize(
        "simulator, params", SIMULATORS, ids=SIMULATOR_IDS
    )
    def test_same_z_extern_and_seed_is_bitwise_reproducible(self, simulator, params):
        z = make_z()
        first = simulate(simulator, params, seed=7, z_extern=z)
        second = simulate(simulator, params, seed=7, z_extern=z)

        assert np.array_equal(first, second)

    def test_gbm_z_extern_needs_no_seed(self):
        """GBM's normals are all of its randomness, so z_extern alone pins the
        paths — passing no seed at all still reproduces them exactly."""
        z = make_z()
        first = simulate(GBMSimulator(), GBM_PARAMS, z_extern=z)
        second = simulate(GBMSimulator(), GBM_PARAMS, z_extern=z)

        assert np.array_equal(first, second)

    def test_gbm_z_extern_matches_internally_drawn_normals(self):
        """The drop-in property: feeding GBM the very normals it would have
        drawn for a seed reproduces that seed's paths. This is what makes
        z_extern a substitution rather than a different model."""
        internal = simulate(GBMSimulator(), GBM_PARAMS, seed=42)
        external = simulate(GBMSimulator(), GBM_PARAMS, z_extern=make_z(42))

        assert np.array_equal(internal, external)

    @pytest.mark.parametrize(
        "simulator, params", SIMULATORS, ids=SIMULATOR_IDS
    )
    def test_different_z_extern_gives_different_paths(self, simulator, params):
        same_seed = dict(seed=7)
        first = simulate(simulator, params, z_extern=make_z(1), **same_seed)
        second = simulate(simulator, params, z_extern=make_z(2), **same_seed)

        assert not np.allclose(first, second)


class TestSeedStillMattersForInternalRandomness:
    """JD and Heston keep randomness that z_extern does not cover, so there
    both arguments are live at once."""

    @pytest.mark.parametrize(
        "simulator, params",
        [(JumpDiffusionSimulator(), JD_PARAMS), (HestonSimulator(), HESTON_PARAMS)],
        ids=["jump_diffusion", "heston"],
    )
    def test_same_z_extern_different_seed_gives_different_paths(
        self, simulator, params
    ):
        z = make_z()
        first = simulate(simulator, params, seed=1, z_extern=z)
        second = simulate(simulator, params, seed=2, z_extern=z)

        assert not np.allclose(first, second)

    def test_gbm_seed_is_inert_under_z_extern(self):
        """The mirror image: GBM has no other randomness, so the seed cannot
        move the paths once z_extern is supplied."""
        z = make_z()
        first = simulate(GBMSimulator(), GBM_PARAMS, seed=1, z_extern=z)
        second = simulate(GBMSimulator(), GBM_PARAMS, seed=2, z_extern=z)

        assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "simulator, params", SIMULATORS, ids=SIMULATOR_IDS
    )
    @pytest.mark.parametrize("method", ["antithetic", "stratified"])
    def test_z_extern_with_variance_reduction_is_rejected(
        self, simulator, params, method
    ):
        with pytest.raises(ValueError, match="Cannot use z_extern with variance_reduction"):
            simulate(
                simulator, params, z_extern=make_z(), variance_reduction=method
            )

    @pytest.mark.parametrize(
        "simulator, params", SIMULATORS, ids=SIMULATOR_IDS
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (N_SIMS, N_STEPS + 5),
            (N_SIMS, N_STEPS - 1),
            (N_SIMS + 1, N_STEPS),
            (N_SIMS,),
        ],
        ids=["too_many_steps", "too_few_steps", "too_many_sims", "one_dimensional"],
    )
    def test_z_extern_shape_mismatch_is_rejected(self, simulator, params, shape):
        z = np.zeros(shape)
        with pytest.raises(ValueError, match="z_extern must have shape"):
            simulate(simulator, params, z_extern=z)

    def test_shape_error_names_expected_and_actual(self):
        z = np.zeros((N_SIMS, N_STEPS + 5))
        with pytest.raises(ValueError) as excinfo:
            simulate(GBMSimulator(), GBM_PARAMS, z_extern=z)

        message = str(excinfo.value)
        assert str((N_SIMS, N_STEPS)) in message
        assert str((N_SIMS, N_STEPS + 5)) in message


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.parametrize(
        "simulator, params", SIMULATORS, ids=SIMULATOR_IDS
    )
    def test_explicit_none_matches_omitting_the_argument(self, simulator, params):
        omitted = simulate(simulator, params, seed=42)
        explicit = simulate(simulator, params, seed=42, z_extern=None)

        assert np.array_equal(omitted, explicit)

    @pytest.mark.parametrize(
        "simulator, params", SIMULATORS, ids=SIMULATOR_IDS
    )
    @pytest.mark.parametrize("method", [None, "antithetic", "stratified"])
    def test_variance_reduction_unaffected_by_the_new_parameter(
        self, simulator, params, method
    ):
        omitted = simulate(simulator, params, seed=42, variance_reduction=method)
        explicit = simulate(
            simulator, params, seed=42, variance_reduction=method, z_extern=None
        )

        assert np.array_equal(omitted, explicit)


class TestComposesWithMu:
    """z_extern and the mu override control different things and must be
    usable together — risk-neutral portfolio simulation needs both."""

    def test_mu_still_shifts_the_drift_under_z_extern(self):
        z = make_z()
        low = simulate(GBMSimulator(), GBM_PARAMS, z_extern=z, mu=0.0)
        high = simulate(GBMSimulator(), GBM_PARAMS, z_extern=z, mu=0.5)

        assert np.mean(high[:, -1]) > np.mean(low[:, -1])

    def test_shared_z_extern_isolates_the_drift_difference(self):
        """With the noise held fixed, the log-ratio of two runs differing only
        in mu is exactly the drift gap — nothing else leaks through."""
        z = make_z()
        base = simulate(GBMSimulator(), GBM_PARAMS, z_extern=z, mu=0.0)
        shifted = simulate(GBMSimulator(), GBM_PARAMS, z_extern=z, mu=0.2)

        log_ratio = np.log(shifted[:, -1] / base[:, -1])
        assert np.allclose(log_ratio, 0.2 * T)
