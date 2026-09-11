"""The projector as one concept with an exact reporting arithmetic (BL-48 slice 2, D-35 (6)).

Three claims, numpy only. `ExactProjector().residual` is, to the last bit, the `projector_residual`
function the problem module carried before the slice -- that old body is pasted here as the oracle,
so the equality is against the old code and not against the new. The residual is a residual: it is
never negative, it vanishes when X already lies in the span, and it does not see how the span is
written -- recombining or rescaling the columns of V leaves it alone. And `SpanCost` is the cost the
training loss starts from: the injected projector's residual of the X it holds, comparing by value.

Deterministic: one fixed graph, seeded V and recombinations; the hypothesis properties draw only
scalars and run with ``derandomize=True``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.describe import describe
from mllib.math.graph.two_hot_span_problem import SpanCost, incidence_matrix, roach_graph
from mllib.math.projector import AbstractProjector, ExactProjector

NODE_TOTAL = 20


def _old_projector_residual(X: np.ndarray, spanning_set: np.ndarray) -> float:
    """`projector_residual` as `two_hot_span_problem.py` had it before the slice, verbatim."""
    X = np.asarray(X, dtype=float)
    spanning_set = np.asarray(spanning_set, dtype=float)
    residual = X - spanning_set @ (np.linalg.pinv(spanning_set) @ X)
    return float(np.sum(residual * residual))


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    """The roach G₅ incidence matrix, twenty rows."""
    matrix = incidence_matrix(roach_graph(5))
    assert matrix.shape[0] == NODE_TOTAL
    return matrix


def seeded_spanning_set(column_count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((NODE_TOTAL, column_count))


def seeded_mixing(column_count: int, seed: int = 1) -> np.ndarray:
    """A unit-scale matrix with no structure, to be shrunk into a diagonally dominant mixing."""
    mixing = np.random.default_rng(seed).standard_normal((column_count, column_count))
    return mixing / (column_count * float(np.max(np.abs(mixing))))


# ------------------------------------------------------------------------------------------------
# The old function, to the last bit.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("column_count", [3, 18])
def test_the_exact_projector_is_the_old_projector_residual_to_the_last_bit(X, column_count):
    spanning_set = seeded_spanning_set(column_count)
    assert ExactProjector().residual(X, spanning_set) == _old_projector_residual(X, spanning_set)


# ------------------------------------------------------------------------------------------------
# The contract.
# ------------------------------------------------------------------------------------------------


def test_the_abstract_projector_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractProjector()  # type: ignore[abstract]


def test_the_exact_projector_has_no_knob_and_is_a_math_object():
    descriptor = describe(ExactProjector)
    assert descriptor["params"] == []
    assert descriptor["kind"] == "math"


def test_two_exact_projectors_are_equal_and_hash_alike():
    assert ExactProjector() == ExactProjector()
    assert hash(ExactProjector()) == hash(ExactProjector())


def test_the_exact_projector_is_frozen():
    # A frozen, slotted dataclass with no fields refuses a new attribute with TypeError rather than
    # FrozenInstanceError on CPython 3.12: the generated __setattr__ names the pre-slots class, and
    # `__slots__ = ()` is what actually blocks the write. The instance cannot change either way.
    with pytest.raises(TypeError):
        ExactProjector().anything = 1  # type: ignore[attr-defined]


# ------------------------------------------------------------------------------------------------
# A residual is a residual.
# ------------------------------------------------------------------------------------------------


def test_the_residual_is_never_negative(X):
    for seed in range(5):
        assert ExactProjector().residual(X, seeded_spanning_set(3, seed=seed)) >= 0.0


def test_the_residual_vanishes_when_x_already_lies_in_the_span(X):
    assert ExactProjector().residual(X, X) == pytest.approx(0.0, abs=1e-12)


def test_the_exact_projector_ignores_a_zero_column(X):
    """A zero column adds nothing to the span; the pseudo-inverse drops it, so the residual is the
    residual on the live columns. Compared to 1e-12 rather than exactly: the SVD inside `pinv`
    sees a different matrix and may round its last digits differently."""
    live = seeded_spanning_set(3)
    widened = np.hstack([live, np.zeros((NODE_TOTAL, 1))])
    assert ExactProjector().residual(X, widened) == pytest.approx(
        ExactProjector().residual(X, live), abs=1e-12
    )


def test_a_spanning_set_wider_than_n_explains_x(X):
    """Twenty-five random columns in twenty dimensions span everything, so nothing is left."""
    assert ExactProjector().residual(X, seeded_spanning_set(25)) == pytest.approx(0.0, abs=1e-12)


@settings(derandomize=True, deadline=None)
@given(strength=st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False))
def test_the_residual_does_not_see_how_the_span_is_written(strength):
    """V and V M span the same columns for invertible M, so the residual is the same number.

    ``I + strength * mixing`` is diagonally dominant for |strength| <= 0.5, hence invertible.
    """
    X = incidence_matrix(roach_graph(5))
    spanning_set = seeded_spanning_set(3)
    mixing = np.eye(3) + strength * seeded_mixing(3)
    assert ExactProjector().residual(X, spanning_set @ mixing) == pytest.approx(
        ExactProjector().residual(X, spanning_set), rel=1e-9
    )


@settings(derandomize=True, deadline=None)
@given(scale=st.floats(1e-2, 1e2, allow_nan=False, allow_infinity=False))
def test_the_residual_is_invariant_to_scaling_the_columns(scale):
    X = incidence_matrix(roach_graph(5))
    spanning_set = seeded_spanning_set(3)
    for sign in (1.0, -1.0):
        assert ExactProjector().residual(X, sign * scale * spanning_set) == pytest.approx(
            ExactProjector().residual(X, spanning_set), rel=1e-9
        )


# ------------------------------------------------------------------------------------------------
# SpanCost.
# ------------------------------------------------------------------------------------------------


def test_the_span_cost_is_the_injected_projectors_residual_of_the_x_it_holds(X):
    spanning_set = seeded_spanning_set(3)
    cost = SpanCost(ExactProjector(), X)
    assert cost.compute_cost(spanning_set) == ExactProjector().residual(X, spanning_set)


def test_two_span_costs_on_the_same_projector_and_x_are_equal_and_hash_alike(X):
    first, second = SpanCost(ExactProjector(), X), SpanCost(ExactProjector(), X.copy())
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_two_span_costs_on_different_x_are_not_equal(X):
    assert SpanCost(ExactProjector(), X) != SpanCost(ExactProjector(), 2.0 * X)


class _ZeroProjector(AbstractProjector):
    """A second projector arithmetic that explains everything, so this file stays torch-free."""

    def residual(self, X, spanning_set):
        return 0.0


def test_two_span_costs_on_different_projectors_are_not_equal(X):
    assert SpanCost(ExactProjector(), X) != SpanCost(_ZeroProjector(), X)
