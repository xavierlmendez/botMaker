"""The ridge projector, the training arithmetic of the projector (BL-48 slice 2, D-31, D-35 (6)).

Four claims. `RidgeProjector(eps).residual` is, to the last bit, the ridge arithmetic
`training_loss` carried inline before the slice -- that body is pasted here as the oracle. The
ridge is the projector with filter factors s²/(s²+ε) in the singular basis of V, which the test
states as a closed form and checks. As ε goes to zero the ridge residual goes to the exact one on a
full-rank V, and on a V with a dependent column both stay finite and the gap shrinks monotonically
in ε -- the whole reason the ridge exists (D-31). And its knob is a knob: `describe` reads it, it is
frozen, and two projectors compare on it.

Deterministic: one fixed graph, seeded V; no clock, no unseeded randomness.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from mllib.describe import describe
from mllib.math.algorithms.two_hot_span.projectors import RidgeProjector
from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanConfig
from mllib.math.graph.two_hot_span_problem import SpanCost, incidence_matrix, roach_graph
from mllib.math.projector import ExactProjector

NODE_TOTAL = 20
SPANNING_COUNT = 3


def _old_ridge_residual(
    X_t: torch.Tensor, spanning_set_t: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """The ridge lines of `training_loss` as the optimizer had them before the slice, verbatim."""
    spanning_count = spanning_set_t.shape[1]
    gram = spanning_set_t.T @ spanning_set_t
    ridge = gram + epsilon * torch.eye(
        spanning_count, dtype=spanning_set_t.dtype, device=spanning_set_t.device
    )
    coefficients = torch.linalg.solve(ridge, spanning_set_t.T @ X_t)
    residual = X_t - spanning_set_t @ coefficients
    loss = torch.sum(residual * residual)
    return loss


@pytest.fixture(scope="module")
def X_t() -> torch.Tensor:
    matrix = incidence_matrix(roach_graph(5))
    assert matrix.shape[0] == NODE_TOTAL
    return torch.tensor(matrix, dtype=torch.float64)


@pytest.fixture(scope="module")
def full_rank() -> torch.Tensor:
    """A seeded float64 V whose three columns are independent."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(NODE_TOTAL, SPANNING_COUNT, dtype=torch.float64, generator=generator)


@pytest.fixture(scope="module")
def dependent(full_rank: torch.Tensor) -> torch.Tensor:
    """The same V with its last column a copy of its first: rank two, three columns."""
    copy = full_rank.clone()
    copy[:, -1] = copy[:, 0]
    return copy


# ------------------------------------------------------------------------------------------------
# The old arithmetic, to the last bit.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1e-6, 1e-1])
@pytest.mark.parametrize("kind", ["full_rank", "dependent"])
def test_the_ridge_projector_is_the_old_inline_ridge_to_the_last_bit(request, X_t, epsilon, kind):
    spanning_set = request.getfixturevalue(kind)
    assert float(RidgeProjector(epsilon).residual(X_t, spanning_set)) == float(
        _old_ridge_residual(X_t, spanning_set, epsilon)
    )


# ------------------------------------------------------------------------------------------------
# The knob.
# ------------------------------------------------------------------------------------------------


def test_describe_reads_epsilon_as_the_only_knob():
    assert describe(RidgeProjector)["params"] == ["epsilon"]


def test_the_default_epsilon_is_the_configs_default():
    assert RidgeProjector().epsilon == 1e-6
    assert RidgeProjector().epsilon == TwoHotSpanConfig().epsilon


def test_the_ridge_projector_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RidgeProjector().epsilon = 1e-3  # type: ignore[misc]


def test_two_ridge_projectors_compare_on_their_epsilon():
    assert RidgeProjector(1e-6) == RidgeProjector(1e-6)
    assert hash(RidgeProjector(1e-6)) == hash(RidgeProjector(1e-6))
    assert RidgeProjector(1e-6) != RidgeProjector(1e-3)


# ------------------------------------------------------------------------------------------------
# The filter factors.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1e-6, 1e-2, 1e-1])
@pytest.mark.parametrize("kind", ["full_rank", "dependent"])
def test_the_ridge_residual_is_the_filter_factor_closed_form(request, X_t, epsilon, kind):
    """With V = U diag(s) Wᵀ, P_ε = U diag(f) Uᵀ for f_i = s_i²/(s_i²+ε), so ‖X - P_ε X‖² =
    ‖X‖² - Σ_i (2 f_i - f_i²) ‖u_iᵀ X‖²: the term X gains back is what each direction keeps, twice,
    minus what keeping it twice double-counts. On the dependent V one s_i is rounding noise and its
    factor is zero, the collapsed-column case the docstring describes."""
    spanning_set = request.getfixturevalue(kind)
    left_singular, singular, _ = torch.linalg.svd(spanning_set, full_matrices=False)
    factors = singular * singular / (singular * singular + epsilon)
    projected_norms = torch.sum((left_singular.T @ X_t) ** 2, dim=1)
    kept = 2.0 * factors - factors * factors
    closed_form = torch.sum(X_t * X_t) - torch.sum(kept * projected_norms)
    # Observed relative error is ~2e-16 (singular values 4-5, ‖X‖² = 46); 1e-10 leaves four orders
    # of magnitude for a BLAS other than the one that ran this.
    assert float(RidgeProjector(epsilon).residual(X_t, spanning_set)) == pytest.approx(
        float(closed_form), rel=1e-10
    )


def test_the_ridge_residual_goes_to_the_exact_residual_as_epsilon_vanishes(X_t, full_rank):
    exact = ExactProjector().residual(X_t.numpy(), full_rank.numpy())
    assert float(RidgeProjector(1e-12).residual(X_t, full_rank)) == pytest.approx(exact, abs=1e-8)


def test_a_dependent_column_leaves_both_residuals_finite_and_the_gap_shrinking_in_epsilon(
    X_t, dependent
):
    """The reason the ridge exists: where the pseudo-inverse is exact but not smooth, the ridge is
    smooth and approaches it from above as ε falls."""
    exact = ExactProjector().residual(X_t.numpy(), dependent.numpy())
    assert np.isfinite(exact)
    gaps = []
    for epsilon in (1e-1, 1e-3, 1e-6):
        ridge = float(RidgeProjector(epsilon).residual(X_t, dependent))
        assert np.isfinite(ridge)
        gaps.append(ridge - exact)
    assert all(gap >= -1e-12 for gap in gaps)
    assert gaps[0] > gaps[1] > gaps[2]


# ------------------------------------------------------------------------------------------------
# SpanCost over a tensor X.
# ------------------------------------------------------------------------------------------------


def test_two_span_costs_on_the_same_tensor_x_are_equal_and_hash_alike(X_t):
    """Equality reads X off the gradient path, so a grad-tracked X compares instead of raising."""
    first = SpanCost(RidgeProjector(1e-6), X_t)
    second = SpanCost(RidgeProjector(1e-6), X_t.clone().requires_grad_(True))
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


# ------------------------------------------------------------------------------------------------
# On the gradient path.
# ------------------------------------------------------------------------------------------------


def test_the_ridge_residual_agrees_with_finite_differences(X_t, full_rank):
    projector = RidgeProjector(1e-6)
    leaf = full_rank.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda spanning_set: projector.residual(X_t, spanning_set), (leaf,), eps=1e-6, atol=1e-6
    )
