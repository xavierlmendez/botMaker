"""The rank-one downdate solver against dense eigendecomposition and its own invariants.

The solver returns the top eigenvalues of ``diag(λ) - w wᵀ`` for many ``w`` at once by bisection
on the secular equation. Dense ``eigvalsh`` is the oracle; the fixtures supply the
repeated-eigenvalue (identity) and rank-one (all-ones) spectra that collapse interlacing intervals.
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from mllib.math.secular_equation import top_eigenvalues_of_rank_one_downdate

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_spectrum(name: str) -> np.ndarray:
    kernel = np.loadtxt(FIXTURES / name, delimiter=",")
    return np.sort(np.clip(np.linalg.eigvalsh(kernel), 0.0, None))[::-1]


def dense_top_eigenvalues(eigenvalues: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    columns = []
    for w in weights.T:
        spectrum = np.linalg.eigvalsh(np.diag(eigenvalues) - np.outer(w, w))
        columns.append(np.sort(spectrum)[::-1][:count])
    return np.array(columns).T.reshape(count, weights.shape[1])


def random_case(seed: int, rank: int, downdates: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    eigenvalues = np.sort(rng.uniform(0.0, 10.0, size=rank))[::-1]
    return eigenvalues, rng.standard_normal((rank, downdates))


def assert_matches_dense(eigenvalues, weights, count):
    got = top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, count)
    expected = dense_top_eigenvalues(eigenvalues, weights, count)
    scale = max(float(eigenvalues[0]), 1.0) if eigenvalues.size else 1.0
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12 * scale)


@pytest.mark.parametrize(("seed", "rank", "downdates", "count"), [(0, 6, 5, 3), (1, 12, 40, 12)])
def test_top_eigenvalues_match_dense_eigvalsh_on_random_spectra(seed, rank, downdates, count):
    eigenvalues, weights = random_case(seed, rank, downdates)
    assert_matches_dense(eigenvalues, weights, count)


@pytest.mark.parametrize(
    "fixture_name",
    ["identity_8x8.csv", "all_ones_8x8.csv", "rbf_chain_8x8.csv", "rbf_chain_10x10.csv"],
)
def test_repeated_and_rank_one_spectra_collapse_intervals_without_nan(fixture_name):
    eigenvalues = fixture_spectrum(fixture_name)
    rng = np.random.default_rng(3)
    weights = rng.standard_normal((eigenvalues.size, 7)) * 0.3
    got = top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, eigenvalues.size)
    assert np.all(np.isfinite(got))
    assert_matches_dense(eigenvalues, weights, eigenvalues.size)


def test_a_zero_downdate_leaves_the_spectrum_unchanged():
    eigenvalues, _ = random_case(5, 7, 1)
    weights = np.zeros((7, 3))
    got = top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, 7)
    np.testing.assert_allclose(got, np.tile(eigenvalues[:, None], (1, 3)), rtol=1e-12)


def test_count_zero_returns_an_empty_row_block_per_downdate():
    eigenvalues, weights = random_case(2, 4, 6)
    assert top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, 0).shape == (0, 6)


def test_the_last_eigenvalue_can_drop_below_the_parent_spectrum():
    # Subtracting more than the smallest eigenvalue pushes the last root under λ_r, into the
    # interval [λ_r - ‖w‖², λ_r]; the solver must search there, not clamp at λ_r.
    eigenvalues = np.array([3.0, 2.0, 0.5])
    weights = np.array([[0.5], [0.0], [2.0]])
    got = top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, 3)
    assert got[2, 0] < 0.5
    assert got[2, 0] > 0.5 - 4.25, "the root is strictly inside the bracket, not at its lower end"
    assert_matches_dense(eigenvalues, weights, 3)


@pytest.mark.parametrize(
    ("eigenvalues", "weights", "count", "message"),
    [
        (np.array([1.0, 2.0]), np.zeros((2, 1)), 1, "descending"),
        (np.array([1.0, -0.1]), np.zeros((2, 1)), 1, "non-negative"),
        (np.array([1.0, 0.5]), np.zeros((3, 1)), 1, "shape"),
        (np.array([1.0, 0.5]), np.zeros((2, 1)), 3, "count"),
        (np.array([[1.0, 0.5]]), np.zeros((2, 1)), 1, "1-D"),
    ],
)
def test_malformed_inputs_are_rejected(eigenvalues, weights, count, message):
    with pytest.raises(ValueError, match=message):
        top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, count)


@settings(max_examples=40, deadline=None)
@given(
    eigenvalues=hnp.arrays(
        np.float64, st.integers(1, 6), elements=st.floats(0.0, 5.0, allow_nan=False)
    ),
    weights=st.lists(st.floats(-2.0, 2.0, allow_nan=False), min_size=6, max_size=6),
)
def test_downdated_eigenvalues_interlace_the_parent_spectrum(eigenvalues, weights):
    eigenvalues = np.sort(eigenvalues)[::-1]
    rank = eigenvalues.size
    weights = np.array(weights[:rank])[:, None]
    got = top_eigenvalues_of_rank_one_downdate(eigenvalues, weights, rank)[:, 0]
    slack = 1e-9 * max(float(eigenvalues[0]), 1.0)
    for index in range(rank):
        below = eigenvalues[index] - weights[:, 0] @ weights[:, 0]
        lower = eigenvalues[index + 1] if index + 1 < rank else below
        assert lower - slack <= got[index] <= eigenvalues[index] + slack
    np.testing.assert_allclose(
        got, dense_top_eigenvalues(eigenvalues, weights, rank)[:, 0], rtol=1e-9, atol=slack
    )
