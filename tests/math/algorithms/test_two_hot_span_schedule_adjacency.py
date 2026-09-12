"""Slice 2.4 — the learning-rate schedule and the optional per-column graph term.

Both are *experiments* and both are off by default, so the first thing this file pins is that a
default run is untouched (that test lives beside its siblings in
`test_two_hot_span_optimizer.py::test_the_defaulted_objects_reproduce_an_explicit_constant_run`).
What is here is the behaviour the knobs add: the shape of the schedule the optimizer actually ran
under, the two adjacency forms checked against numbers computed by hand on a graph small enough to
read, that switching the term on leaves the run's invariants standing, and that a misspelled name
fails at construction rather than 200 steps later.

Deterministic: every graph is built explicitly, every run carries its own seed, and the schedule
assertions are on closed-form values rather than on anything Adam did.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from mllib.math.algorithms.two_hot_span.optimizer import (
    TwoHotSpanOptimizer,
    TwoHotSpanSettings,
    training_cost,
)
from mllib.math.algorithms.two_hot_span.penalties import (
    EdgeProductAdjacencyPenalty,
    LaplacianAdjacencyPenalty,
)
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.graph.two_hot_span_problem import (
    brute_force_rcut,
    graph_matrices,
    incidence_matrix,
    laplacian_matrix,
    roach_graph,
    spectral_floor,
)
from mllib.math.learning_rate_schedule import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    WarmupCosineSchedule,
)

from .two_hot_span_support import problem_for, run, run_with_history

STEPS = 40
LEARNING_RATE = 0.05
FRACTION = 0.01


def weighted_graph() -> nx.Graph:
    """A five-vertex weighted graph: a triangle 0-1-2, a pendant 3 on 2, and 4 hung off 3.

    Small enough that d_i, w_ij and "is (i, j) an edge" are all readable off the picture, which is
    what makes the hand values in this file checkable without running anything.
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (0, 2, 0.5), (2, 3, 1.5), (3, 4, 4.0)])
    return graph


def unit_two_hot(node_total: int, first: int, second: int) -> np.ndarray:
    """The unit 2-hot column on the pair, entries +1/sqrt(2) and -1/sqrt(2)."""
    column = np.zeros((node_total, 1), dtype=np.float64)
    column[first, 0] = 1.0 / math.sqrt(2.0)
    column[second, 0] = -1.0 / math.sqrt(2.0)
    return column


def seeded_graph(node_total: int, seed: int) -> nx.Graph:
    """A connected seeded random graph with unit weights."""
    graph = nx.gnp_random_graph(node_total, 0.55, seed=seed)
    if not nx.is_connected(graph):
        graph = nx.connected_watts_strogatz_graph(node_total, 4, 0.3, seed=seed)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def schedule_named(name: str, **knobs):
    """The schedule object the old config name asked for; the names live in `mllib.ml` now."""
    fraction = float(knobs.get("final_learning_rate_fraction", 0.0))
    if name == "constant":
        return ConstantSchedule()
    if name == "linear":
        return LinearSchedule(final_fraction=fraction)
    if name == "cosine":
        return CosineSchedule(final_fraction=fraction)
    return WarmupCosineSchedule(
        warmup_steps=int(knobs.get("warmup_steps", 0)), final_fraction=fraction
    )


def run_with(**overrides) -> np.ndarray:
    """The learning-rate history of a short seeded run on the roach under the given schedule."""
    schedule = schedule_named(overrides.pop("learning_rate_schedule", "constant"), **overrides)
    _, history = run_with_history(
        roach_graph(5), 2, step_count=STEPS, learning_rate=LEARNING_RATE, seed=0, schedule=schedule
    )
    return history.learning_rate_history


# ------------------------------------------------------------------------------------------------
# The schedule.
# ------------------------------------------------------------------------------------------------


def test_the_constant_schedule_holds_the_learning_rate_at_every_step():
    history = run_with(learning_rate_schedule="constant")

    assert history.shape == (STEPS,)
    assert np.all(history == LEARNING_RATE)


def test_the_cosine_schedule_starts_at_the_learning_rate_and_ends_at_the_fraction():
    history = run_with(learning_rate_schedule="cosine", final_learning_rate_fraction=FRACTION)

    assert history[0] == pytest.approx(LEARNING_RATE, rel=1e-12)
    assert history[-1] == pytest.approx(LEARNING_RATE * FRACTION, rel=1e-12)


def test_the_cosine_schedule_never_rises():
    history = run_with(learning_rate_schedule="cosine", final_learning_rate_fraction=FRACTION)

    assert np.all(np.diff(history) <= 0.0)


def test_the_warmup_cosine_schedule_rises_over_the_warmup_and_falls_after_it():
    warmup = 8
    history = run_with(
        learning_rate_schedule="warmup_cosine",
        warmup_steps=warmup,
        final_learning_rate_fraction=FRACTION,
    )

    assert history[0] == 0.0
    # The rise spans steps 0..warmup: step `warmup` is where the cosine's own maximum sits.
    assert np.all(np.diff(history[: warmup + 1]) > 0.0)
    assert history[warmup] == pytest.approx(LEARNING_RATE, rel=1e-12)
    assert np.all(np.diff(history[warmup:]) < 0.0)
    assert history[-1] == pytest.approx(LEARNING_RATE * FRACTION, rel=1e-12)


def test_the_linear_schedule_is_linear_in_the_step():
    history = run_with(learning_rate_schedule="linear", final_learning_rate_fraction=FRACTION)
    span = LEARNING_RATE - LEARNING_RATE * FRACTION

    # Three points: the two ends and the midpoint, which a linear ramp puts exactly halfway.
    assert history[0] == pytest.approx(LEARNING_RATE, rel=1e-12)
    assert history[-1] == pytest.approx(LEARNING_RATE * FRACTION, rel=1e-12)
    middle = (STEPS - 1) // 2
    assert history[middle] == pytest.approx(LEARNING_RATE - span * middle / (STEPS - 1), rel=1e-12)


def test_a_scheduled_run_still_holds_the_zero_sum_constraint():
    result, history = run_with_history(
        roach_graph(5),
        2,
        step_count=STEPS,
        seed=0,
        collision_weight=1.0,
        schedule=WarmupCosineSchedule(warmup_steps=5, final_fraction=FRACTION),
    )

    assert result.max_zero_sum_violation <= 1e-12
    assert np.isfinite(history.loss_history).all()


# ------------------------------------------------------------------------------------------------
# The two adjacency forms, against values computed by hand.
# ------------------------------------------------------------------------------------------------


def test_the_recovered_laplacian_and_adjacency_are_the_graphs_own():
    graph = weighted_graph()
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))

    assert np.allclose(laplacian, laplacian_matrix(graph), atol=1e-12)
    assert np.allclose(adjacency, nx.to_numpy_array(graph, nodelist=sorted(graph), dtype=float))


@pytest.mark.parametrize(("first", "second"), [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)])
def test_the_laplacian_form_on_an_edge_pair_is_the_degree_sum_plus_twice_the_weight(first, second):
    graph = weighted_graph()
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))
    degrees = dict(graph.degree(weight="weight"))
    weight = graph[first][second]["weight"]

    column = torch.tensor(unit_two_hot(graph.number_of_nodes(), first, second))
    value = LaplacianAdjacencyPenalty(laplacian).term(column)

    assert float(value) == pytest.approx(
        (degrees[first] + degrees[second] + 2.0 * weight) / 2.0, abs=1e-12
    )


@pytest.mark.parametrize(("first", "second"), [(0, 3), (0, 4), (1, 3), (1, 4), (2, 4)])
def test_the_laplacian_form_off_an_edge_is_the_degree_sum_alone(first, second):
    """The degree bias the edge-product form does not carry: off an edge this is not 0."""
    graph = weighted_graph()
    assert not graph.has_edge(first, second)
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))
    degrees = dict(graph.degree(weight="weight"))

    column = torch.tensor(unit_two_hot(graph.number_of_nodes(), first, second))
    value = LaplacianAdjacencyPenalty(laplacian).term(column)

    assert float(value) == pytest.approx((degrees[first] + degrees[second]) / 2.0, abs=1e-12)


@pytest.mark.parametrize(("first", "second"), [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)])
def test_the_edge_product_form_on_an_edge_pair_is_the_edge_weight(first, second):
    graph = weighted_graph()
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))

    column = torch.tensor(unit_two_hot(graph.number_of_nodes(), first, second))
    value = EdgeProductAdjacencyPenalty(adjacency).term(column)

    assert float(value) == pytest.approx(graph[first][second]["weight"], abs=1e-12)


@pytest.mark.parametrize(("first", "second"), [(0, 3), (0, 4), (1, 3), (1, 4), (2, 4)])
def test_the_edge_product_form_off_an_edge_is_exactly_zero(first, second):
    graph = weighted_graph()
    assert not graph.has_edge(first, second)
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))

    column = torch.tensor(unit_two_hot(graph.number_of_nodes(), first, second))
    value = EdgeProductAdjacencyPenalty(adjacency).term(column)

    assert float(value) == pytest.approx(0.0, abs=1e-12)


def test_both_forms_are_scale_invariant():
    """‖v‖₂² is divided out, so a column and ten times it score the same."""
    graph = weighted_graph()
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))
    column = unit_two_hot(graph.number_of_nodes(), 1, 2)

    for penalty in (LaplacianAdjacencyPenalty(laplacian), EdgeProductAdjacencyPenalty(adjacency)):
        one = penalty.term(torch.tensor(column))
        ten = penalty.term(torch.tensor(10.0 * column))
        assert float(one) == pytest.approx(float(ten), rel=1e-12)


def test_the_term_sums_over_the_columns():
    graph = weighted_graph()
    laplacian, adjacency = graph_matrices(incidence_matrix(graph))
    node_total = graph.number_of_nodes()
    pair_columns = [unit_two_hot(node_total, 0, 1), unit_two_hot(node_total, 2, 3)]

    together = EdgeProductAdjacencyPenalty(adjacency).term(torch.tensor(np.hstack(pair_columns)))
    apart = sum(
        float(EdgeProductAdjacencyPenalty(adjacency).term(torch.tensor(column)))
        for column in pair_columns
    )

    assert float(together) == pytest.approx(apart, abs=1e-12)


def test_a_matrix_that_is_not_an_incidence_matrix_is_refused():
    """A recovered adjacency with a negative weight means the caller handed in something else."""
    with pytest.raises(ValueError, match="negative weight"):
        graph_matrices(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))


# ------------------------------------------------------------------------------------------------
# A run with the term switched on.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["laplacian", "edge_product"])
def test_a_run_with_the_graph_term_keeps_every_invariant(form):
    graph = seeded_graph(8, seed=5)
    optimum, _ = brute_force_rcut(graph, 2)
    floor = spectral_floor(graph, 2)

    result, history = run_with_history(
        graph,
        2,
        step_count=200,
        seed=7,
        collision_weight=1.0,
        adjacency_weight=1.0,
        adjacency_form=form,
        schedule=CosineSchedule(final_fraction=0.02),
    )

    assert result.max_zero_sum_violation <= 1e-12
    assert np.isfinite(history.loss_history).all()
    assert np.isfinite(result.spanning_set).all()
    assert result.rounded_cut >= optimum - 1e-10
    assert result.relaxed_objective >= floor - 1e-10


def test_the_graph_term_changes_the_run_it_is_switched_on_for():
    """A knob that made no difference would be a knob nobody could test the sign of."""
    graph = seeded_graph(8, seed=5)

    off = run(graph, 2, step_count=50, seed=7, collision_weight=1.0)
    on = run(graph, 2, step_count=50, seed=7, collision_weight=1.0, adjacency_weight=1.0)

    assert not np.array_equal(off.spanning_set, on.spanning_set)


# ------------------------------------------------------------------------------------------------
# Validation: each refusal lives on the object that owns the knob (D-35 (3)).
#
# The two name-based refusals the old config made — an unknown schedule name, an unknown adjacency
# form — are a composition root's: names live in `mllib.ml.projects.two_hot_span_composition` and
# are tested there. Negative penalty weights likewise: the penalty classes take any weight and the
# composition module refuses a negative one.
# ------------------------------------------------------------------------------------------------


def test_a_negative_warmup_is_refused_by_the_schedule():
    with pytest.raises(ValueError, match="warmup_steps must not be negative"):
        WarmupCosineSchedule(warmup_steps=-1)


def test_a_negative_learning_rate_is_refused_by_the_step_rule():
    with pytest.raises(ValueError, match="learning_rate must be finite and not negative"):
        AdamStepRule(-0.01)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
@pytest.mark.parametrize("schedule_class", [LinearSchedule, CosineSchedule])
def test_a_final_fraction_outside_the_unit_interval_is_refused_by_the_schedule(
    schedule_class, fraction
):
    with pytest.raises(ValueError, match="final_fraction"):
        schedule_class(final_fraction=fraction)


def test_a_warmup_as_long_as_the_run_is_refused_so_the_decay_cannot_silently_vanish():
    """The one cross-object check: the optimizer knows the budget, the schedule the warm-up."""
    graph = roach_graph(5)
    with pytest.raises(ValueError, match="warmup_steps"):
        TwoHotSpanOptimizer(
            problem_for(graph, 2),
            training_cost(problem_for(graph, 2)),
            step_rule=AdamStepRule(0.05, WarmupCosineSchedule(warmup_steps=30)),
            settings=TwoHotSpanSettings(step_count=30),
        )
    TwoHotSpanOptimizer(
        problem_for(graph, 2),
        training_cost(problem_for(graph, 2)),
        step_rule=AdamStepRule(0.05, WarmupCosineSchedule(warmup_steps=29)),
        settings=TwoHotSpanSettings(step_count=30),
    )
