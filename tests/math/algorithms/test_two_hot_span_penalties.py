"""The two-hot span penalties as injected objects (BL-48 slice 1, D-35).

Five claims. The training loss built from injected penalties equals, to the last bit, the training
loss the optimizer carried before the slice -- that old function is pasted here as the oracle, with
the two term functions it called, so the equality is against the old code and not against the new.
Each penalty's ``term`` is the pre-slice term body, and ``compute_penalty`` is exactly +/- weight
times it, for any weight. The training loss is a plain left-to-right sum. Each penalty's autograd
gradient agrees with finite differences, and its term is scale-invariant and per-column additive
where the arithmetic says it is. And the transitional adapter that builds penalties from a config's
weights builds nothing at a zero weight, which is what keeps a default run finite on a zero column.

Deterministic: one seeded V, one fixed graph, no clock; the hypothesis properties draw only scalars
and run with ``derandomize=True``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torch")

import torch

from mllib.describe import describe
from mllib.math.algorithms.two_hot_span.penalties import (
    CollisionPenalty,
    DiversityPenalty,
    EdgeProductAdjacencyPenalty,
    LaplacianAdjacencyPenalty,
)
from mllib.math.algorithms.two_hot_span_optimizer import (
    TwoHotSpanConfig,
    _penalties_from_config,
    training_loss,
)
from mllib.math.graph.two_hot_span_problem import graph_matrices, incidence_matrix, roach_graph
from mllib.math.regularization_function import AbstractRegularizationFunction

from .test_two_hot_span_refactor_baseline import CELLS

NODE_TOTAL = 20
SPANNING_COUNT = 3
EPSILON = 1e-6
REWARDS = ("collision", "laplacian", "edge_product")
PENALTY_NAMES = (*REWARDS, "diversity")


@pytest.fixture(scope="module")
def graph_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, L, A) of the roach G₅ instance, whose incidence matrix has twenty rows."""
    X = incidence_matrix(roach_graph(5))
    assert X.shape[0] == NODE_TOTAL
    laplacian, adjacency = graph_matrices(X)
    return X, laplacian, adjacency


@pytest.fixture(scope="module")
def spanning_set() -> torch.Tensor:
    """A seeded float64 V with no zero column, so every term is finite."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(NODE_TOTAL, SPANNING_COUNT, dtype=torch.float64, generator=generator)


def penalty_named(name: str, graph_data, weight: float) -> AbstractRegularizationFunction:
    _, laplacian, adjacency = graph_data
    if name == "collision":
        return CollisionPenalty(weight=weight)
    if name == "laplacian":
        return LaplacianAdjacencyPenalty(laplacian, weight=weight)
    if name == "edge_product":
        return EdgeProductAdjacencyPenalty(adjacency, weight=weight)
    return DiversityPenalty(weight=weight)


# ------------------------------------------------------------------------------------------------
# The oracle: the optimizer's training loss as it was on `main` before the slice, verbatim.
# ------------------------------------------------------------------------------------------------


def _old_adjacency_term(
    spanning_set_t: torch.Tensor,
    laplacian_t: torch.Tensor,
    adjacency_t: torch.Tensor,
    adjacency_form: str,
) -> torch.Tensor:
    squared = torch.sum(spanning_set_t * spanning_set_t, dim=0)
    if adjacency_form == "edge_product":
        absolute = torch.abs(spanning_set_t)
        numerator = torch.sum(absolute * (adjacency_t @ absolute), dim=0)
    else:
        numerator = torch.sum(spanning_set_t * (laplacian_t @ spanning_set_t), dim=0)
    return torch.sum(numerator / squared)


def _old_diversity_term(spanning_set_t: torch.Tensor) -> torch.Tensor:
    absolute = torch.abs(spanning_set_t)
    distribution = absolute / torch.sum(absolute, dim=0)
    load = torch.sum(distribution, dim=1)
    return torch.sum(load * load)


def _old_training_loss(
    X_t: torch.Tensor,
    spanning_set_t: torch.Tensor,
    collision_weight: float,
    epsilon: float,
    adjacency_weight: float = 0.0,
    laplacian_t: torch.Tensor | None = None,
    adjacency_t: torch.Tensor | None = None,
    adjacency_form: str = "laplacian",
    diversity_weight: float = 0.0,
) -> torch.Tensor:
    spanning_count = spanning_set_t.shape[1]
    gram = spanning_set_t.T @ spanning_set_t
    ridge = gram + epsilon * torch.eye(
        spanning_count, dtype=spanning_set_t.dtype, device=spanning_set_t.device
    )
    coefficients = torch.linalg.solve(ridge, spanning_set_t.T @ X_t)
    residual = X_t - spanning_set_t @ coefficients
    loss = torch.sum(residual * residual)
    if collision_weight != 0.0:
        squared = torch.sum(spanning_set_t * spanning_set_t, dim=0)
        absolute = torch.sum(torch.abs(spanning_set_t), dim=0)
        loss = loss - collision_weight * torch.sum(squared / (absolute * absolute))
    if adjacency_weight != 0.0:
        if laplacian_t is None or adjacency_t is None:
            raise ValueError("adjacency_weight != 0 needs both laplacian_t and adjacency_t")
        loss = loss - adjacency_weight * _old_adjacency_term(
            spanning_set_t, laplacian_t, adjacency_t, adjacency_form
        )
    if diversity_weight != 0.0:
        loss = loss + diversity_weight * _old_diversity_term(spanning_set_t)
    return loss


@pytest.mark.parametrize("cell", list(CELLS))
def test_the_training_loss_is_the_old_training_loss_to_the_last_bit(graph_data, spanning_set, cell):
    """Every cell of the refactor snapshot, through the adapter, against the pasted old function."""
    X, laplacian, adjacency = graph_data
    config = CELLS[cell]
    X_t = torch.tensor(X, dtype=torch.float64)
    laplacian_t = adjacency_t = None
    if config.adjacency_weight != 0.0:
        laplacian_t = torch.tensor(laplacian, dtype=torch.float64)
        adjacency_t = torch.tensor(adjacency, dtype=torch.float64)

    new = training_loss(X_t, spanning_set, config.epsilon, _penalties_from_config(config, X))

    old = _old_training_loss(
        X_t,
        spanning_set,
        config.collision_weight,
        config.epsilon,
        config.adjacency_weight,
        laplacian_t,
        adjacency_t,
        config.adjacency_form,
        config.diversity_weight,
    )
    assert float(new) == float(old)


# ------------------------------------------------------------------------------------------------
# Each term is the pre-slice term body; compute_penalty is exactly +/- weight times it.
# ------------------------------------------------------------------------------------------------


def old_collision_term(spanning_set_t: torch.Tensor) -> torch.Tensor:
    """The collision sum as `training_loss` wrote it inline before the slice."""
    squared = torch.sum(spanning_set_t * spanning_set_t, dim=0)
    absolute = torch.sum(torch.abs(spanning_set_t), dim=0)
    return torch.sum(squared / (absolute * absolute))


def test_the_collision_term_is_the_old_inline_sum_to_the_last_bit(spanning_set):
    assert float(CollisionPenalty().term(spanning_set)) == float(old_collision_term(spanning_set))


def test_the_laplacian_term_is_the_old_adjacency_term_to_the_last_bit(graph_data, spanning_set):
    _, laplacian, adjacency = graph_data
    expected = _old_adjacency_term(
        spanning_set, torch.tensor(laplacian), torch.tensor(adjacency), "laplacian"
    )
    assert float(LaplacianAdjacencyPenalty(laplacian).term(spanning_set)) == float(expected)


def test_the_edge_product_term_is_the_old_adjacency_term_to_the_last_bit(graph_data, spanning_set):
    _, laplacian, adjacency = graph_data
    expected = _old_adjacency_term(
        spanning_set, torch.tensor(laplacian), torch.tensor(adjacency), "edge_product"
    )
    assert float(EdgeProductAdjacencyPenalty(adjacency).term(spanning_set)) == float(expected)


def test_the_diversity_term_is_the_old_diversity_term_to_the_last_bit(spanning_set):
    assert float(DiversityPenalty().term(spanning_set)) == float(_old_diversity_term(spanning_set))


@pytest.mark.parametrize("name", REWARDS)
def test_a_reward_comes_back_negative(graph_data, spanning_set, name):
    assert float(penalty_named(name, graph_data, 2.0).compute_penalty(spanning_set)) < 0.0


def test_a_penalty_comes_back_positive(spanning_set):
    assert float(DiversityPenalty(weight=2.0).compute_penalty(spanning_set)) > 0.0


@pytest.mark.parametrize("name", REWARDS)
@settings(derandomize=True, deadline=None)
@given(weight=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False))
def test_a_reward_is_exactly_minus_weight_times_its_term_for_any_weight(
    graph_data, spanning_set, name, weight
):
    penalty = penalty_named(name, graph_data, weight)
    expected = -(weight * penalty.term(spanning_set))
    assert float(penalty.compute_penalty(spanning_set)) == float(expected)


@settings(derandomize=True, deadline=None)
@given(weight=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False))
def test_the_diversity_penalty_is_exactly_weight_times_its_term_for_any_weight(
    spanning_set, weight
):
    penalty = DiversityPenalty(weight=weight)
    expected = weight * penalty.term(spanning_set)
    assert float(penalty.compute_penalty(spanning_set)) == float(expected)


# ------------------------------------------------------------------------------------------------
# The training loss is a plain left-to-right sum.
# ------------------------------------------------------------------------------------------------


def test_the_training_loss_adds_each_penalty_left_to_right_exactly(graph_data, spanning_set):
    X, laplacian, adjacency = graph_data
    X_t = torch.tensor(X, dtype=torch.float64)
    penalties = (
        CollisionPenalty(weight=10.0),
        LaplacianAdjacencyPenalty(laplacian, weight=0.5),
        EdgeProductAdjacencyPenalty(adjacency, weight=0.3),
        DiversityPenalty(weight=10.0),
    )

    with_penalties = training_loss(X_t, spanning_set, EPSILON, penalties)

    expected = training_loss(X_t, spanning_set, EPSILON)
    for penalty in penalties:
        expected = expected + penalty.compute_penalty(spanning_set)
    assert float(with_penalties) == float(expected)


def test_the_training_loss_with_no_penalties_is_the_ridge_cost_alone(graph_data, spanning_set):
    X, _, _ = graph_data
    X_t = torch.tensor(X, dtype=torch.float64)
    assert float(training_loss(X_t, spanning_set, EPSILON)) == float(
        training_loss(X_t, spanning_set, EPSILON, ())
    )


# ------------------------------------------------------------------------------------------------
# Invariants of the terms: finite differences, scale invariance, per-column additivity.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PENALTY_NAMES)
def test_the_gradient_of_each_penalty_matches_finite_differences(graph_data, spanning_set, name):
    penalty = penalty_named(name, graph_data, 1.5)
    parameters = spanning_set.clone().double().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda parameters: penalty.compute_penalty(parameters),
        (parameters,),
        eps=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("name", REWARDS)
@settings(derandomize=True, deadline=None)
@given(scale=st.floats(1e-3, 1e3), negative=st.booleans())
def test_each_per_column_term_is_invariant_to_scaling_the_columns(
    graph_data, spanning_set, name, scale, negative
):
    """The division by ‖v_j‖₂² (or ‖v_j‖₁² for R) takes the scale, and the sign, back out."""
    penalty = penalty_named(name, graph_data, 1.0)
    factor = -scale if negative else scale
    assert float(penalty.term(factor * spanning_set)) == pytest.approx(
        float(penalty.term(spanning_set)), rel=1e-9
    )


@pytest.mark.parametrize("name", REWARDS)
@settings(derandomize=True, deadline=None)
@given(column_count=st.integers(2, 4))
def test_each_per_column_term_is_the_sum_of_its_columns_terms(graph_data, name, column_count):
    penalty = penalty_named(name, graph_data, 1.0)
    generator = torch.Generator().manual_seed(column_count)
    columns = [
        torch.randn(NODE_TOTAL, 1, dtype=torch.float64, generator=generator)
        for _ in range(column_count)
    ]

    together = float(penalty.term(torch.hstack(columns)))

    apart = sum(float(penalty.term(column)) for column in columns)
    assert together == pytest.approx(apart, rel=1e-12)


# ------------------------------------------------------------------------------------------------
# The contract: abstract base, frozen concretes, value equality, self-describing constructors.
# ------------------------------------------------------------------------------------------------


def test_the_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractRegularizationFunction()  # type: ignore[abstract]


@pytest.mark.parametrize("name", PENALTY_NAMES)
def test_every_penalty_is_a_regularization_function(graph_data, name):
    assert isinstance(penalty_named(name, graph_data, 1.0), AbstractRegularizationFunction)


@pytest.mark.parametrize("name", PENALTY_NAMES)
def test_every_penalty_is_frozen(graph_data, name):
    penalty = penalty_named(name, graph_data, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        penalty.weight = 2.0  # type: ignore[misc]


def test_describe_reads_the_knobs_off_the_constructor():
    assert describe(CollisionPenalty)["params"] == ["weight"]
    assert describe(LaplacianAdjacencyPenalty)["params"] == ["laplacian", "weight"]
    assert describe(EdgeProductAdjacencyPenalty)["params"] == ["adjacency", "weight"]
    assert describe(DiversityPenalty(weight=3.0))["kind"] == "math"


def test_an_adjacency_penalty_accepts_a_numpy_matrix_and_holds_a_float64_tensor(graph_data):
    _, laplacian, _ = graph_data
    penalty = LaplacianAdjacencyPenalty(laplacian)
    assert isinstance(penalty.laplacian, torch.Tensor)
    assert penalty.laplacian.dtype == torch.float64


def test_two_graph_penalties_on_the_same_graph_and_weight_are_equal_and_hash_alike(graph_data):
    _, laplacian, _ = graph_data
    one = LaplacianAdjacencyPenalty(laplacian, weight=0.5)
    other = LaplacianAdjacencyPenalty(laplacian.copy(), weight=0.5)
    assert one == other
    assert hash(one) == hash(other)
    assert len({one, other}) == 1


def test_two_graph_penalties_on_different_graphs_are_not_equal(graph_data):
    _, laplacian, _ = graph_data
    other_graph = laplacian.copy()
    other_graph[0, 0] += 1.0
    assert LaplacianAdjacencyPenalty(laplacian, 0.5) != LaplacianAdjacencyPenalty(other_graph, 0.5)


def test_two_graph_penalties_with_different_weights_are_not_equal(graph_data):
    _, laplacian, _ = graph_data
    assert LaplacianAdjacencyPenalty(laplacian, 0.5) != LaplacianAdjacencyPenalty(laplacian, 0.6)


def test_the_two_adjacency_forms_on_the_same_matrix_are_not_equal(graph_data):
    _, laplacian, _ = graph_data
    assert LaplacianAdjacencyPenalty(laplacian, 0.5) != EdgeProductAdjacencyPenalty(laplacian, 0.5)


def test_a_matrix_free_penalty_compares_on_its_weight():
    assert CollisionPenalty(1.0) == CollisionPenalty(1.0)
    assert CollisionPenalty(1.0) != CollisionPenalty(2.0)


# ------------------------------------------------------------------------------------------------
# The transitional adapter: a zero weight is an absent penalty, and that is what keeps a default
# run finite on a zero column.
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spanning_set_with_a_zero_column(spanning_set) -> torch.Tensor:
    zeroed = spanning_set.clone()
    zeroed[:, 1] = 0.0
    return zeroed


@pytest.mark.parametrize("name", PENALTY_NAMES)
def test_a_zero_weighted_penalty_on_a_zero_column_is_nan_which_is_why_the_adapter_omits_it(
    graph_data, spanning_set_with_a_zero_column, name
):
    penalty = penalty_named(name, graph_data, 0.0)
    assert torch.isnan(penalty.compute_penalty(spanning_set_with_a_zero_column))


def test_the_default_config_builds_no_penalty_so_its_loss_is_finite_on_a_zero_column(
    graph_data, spanning_set_with_a_zero_column
):
    X, _, _ = graph_data
    X_t = torch.tensor(X, dtype=torch.float64)
    penalties = _penalties_from_config(TwoHotSpanConfig(), X)
    assert penalties == ()
    assert torch.isfinite(training_loss(X_t, spanning_set_with_a_zero_column, EPSILON, penalties))


@pytest.mark.parametrize(
    ("adjacency_form", "adjacency_class"),
    [("laplacian", LaplacianAdjacencyPenalty), ("edge_product", EdgeProductAdjacencyPenalty)],
)
def test_every_weight_on_builds_the_three_penalties_in_the_losss_order(
    graph_data, adjacency_form, adjacency_class
):
    X, _, _ = graph_data
    config = TwoHotSpanConfig(
        collision_weight=10.0,
        adjacency_weight=0.3,
        adjacency_form=adjacency_form,
        diversity_weight=10.0,
    )

    penalties = _penalties_from_config(config, X)

    assert [type(penalty) for penalty in penalties] == [
        CollisionPenalty,
        adjacency_class,
        DiversityPenalty,
    ]
    assert [penalty.weight for penalty in penalties] == [10.0, 0.3, 10.0]
