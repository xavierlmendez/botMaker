"""The composition roots' one home: names and knobs to fresh math objects (D-35 (5), (8)).

Everything here is a mapping a command line needs and the math objects never see: a schedule
name to its class with its knobs, an adjacency form to a penalty over the problem's own matrix,
weights to the penalties the loss adds in order, and the lot to one optimizer. The refusals that
used to sit on a config dataclass live here now, so each is pinned here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.math.algorithms.two_hot_span.optimizer import (
    TwoHotSpanOptimizer,
    TwoHotSpanSettings,
    training_cost,
)
from mllib.math.algorithms.two_hot_span.penalties import (
    CollisionPenalty,
    DiversityPenalty,
    EdgeProductAdjacencyPenalty,
    LaplacianAdjacencyPenalty,
)
from mllib.math.algorithms.two_hot_span.projectors import RidgeProjector
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.graph.two_hot_span_problem import TwoHotSpanProblem, roach_graph
from mllib.math.learning_rate_schedule import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    WarmupCosineSchedule,
)
from mllib.ml.projects.two_hot_span_composition import (
    ADJACENCY_FORM_NAMES,
    DEFAULT_EPSILON,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STEP_COUNT,
    SCHEDULE_NAMES,
    adjacency_penalty_from_name,
    compose_two_hot_span,
    penalties_from_weights,
    schedule_from_name,
)

CLUSTER_COUNT = 2
STEP_COUNT = 30


def problem() -> TwoHotSpanProblem:
    return TwoHotSpanProblem(roach_graph(5), CLUSTER_COUNT, name="roach_g5")


# ------------------------------------------------------------------------------------------------
# Names to objects.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("constant", ConstantSchedule()),
        ("linear", LinearSchedule(final_fraction=0.2)),
        ("cosine", CosineSchedule(final_fraction=0.2)),
        ("warmup_cosine", WarmupCosineSchedule(warmup_steps=7, final_fraction=0.2)),
    ],
)
def test_each_schedule_name_maps_to_its_class_with_the_knobs_it_takes(name, expected):
    assert name in SCHEDULE_NAMES
    assert schedule_from_name(name, final_fraction=0.2, warmup_steps=7) == expected


def test_an_unknown_schedule_name_is_refused():
    with pytest.raises(ValueError, match="unknown schedule 'nope'"):
        schedule_from_name("nope")


def test_the_laplacian_form_is_a_penalty_over_the_problems_own_laplacian():
    instance = problem()
    penalty = adjacency_penalty_from_name("laplacian", instance, 0.5)

    assert penalty == LaplacianAdjacencyPenalty(instance.laplacian, weight=0.5)


def test_the_edge_product_form_is_a_penalty_over_the_problems_own_adjacency():
    instance = problem()
    penalty = adjacency_penalty_from_name("edge_product", instance, 0.5)

    assert penalty == EdgeProductAdjacencyPenalty(instance.adjacency, weight=0.5)


def test_an_unknown_adjacency_form_is_refused():
    assert "normalized" not in ADJACENCY_FORM_NAMES
    with pytest.raises(ValueError, match="unknown adjacency_form 'normalized'"):
        adjacency_penalty_from_name("normalized", problem(), 0.5)


# ------------------------------------------------------------------------------------------------
# Weights to penalties.
# ------------------------------------------------------------------------------------------------


def test_zero_weights_build_no_penalty():
    assert penalties_from_weights(problem()) == ()


@pytest.mark.parametrize("form", ADJACENCY_FORM_NAMES)
def test_every_weight_on_builds_the_three_penalties_in_the_losss_order(form):
    instance = problem()
    penalties = penalties_from_weights(
        instance,
        collision_weight=10.0,
        adjacency_weight=0.3,
        adjacency_form=form,
        diversity_weight=2.0,
    )

    graph_class = LaplacianAdjacencyPenalty if form == "laplacian" else EdgeProductAdjacencyPenalty
    assert [type(penalty) for penalty in penalties] == [
        CollisionPenalty,
        graph_class,
        DiversityPenalty,
    ]
    assert [penalty.weight for penalty in penalties] == [10.0, 0.3, 2.0]


@pytest.mark.parametrize("knob", ["collision_weight", "adjacency_weight", "diversity_weight"])
def test_a_negative_weight_is_refused_by_name(knob):
    with pytest.raises(ValueError, match=f"{knob} must not be negative"):
        penalties_from_weights(problem(), **{knob: -1.0})


def test_an_unknown_adjacency_form_is_refused_even_at_weight_zero():
    """The regression the review caught: a name is checked whether or not its term is built."""
    with pytest.raises(ValueError, match="unknown adjacency_form"):
        penalties_from_weights(problem(), adjacency_form="normalized")


# ------------------------------------------------------------------------------------------------
# The optimizer.
# ------------------------------------------------------------------------------------------------


def test_every_call_composes_a_fresh_optimizer_and_a_fresh_unbound_step_rule():
    instance = problem()
    first = compose_two_hot_span(instance, step_count=STEP_COUNT)
    second = compose_two_hot_span(instance, step_count=STEP_COUNT)

    assert first is not second
    assert first.step_rule is not second.step_rule
    assert not first.step_rule.is_bound and not second.step_rule.is_bound
    assert first.configuration == second.configuration


def test_a_negative_learning_rate_is_refused():
    with pytest.raises(ValueError, match="learning_rate must not be negative"):
        compose_two_hot_span(problem(), learning_rate=-0.1)


def test_a_composed_run_is_the_hand_built_optimizer_on_the_snapshot_cell_to_the_last_bit():
    """The refactor snapshot's `objective_only` cell, composed from knobs and built by hand."""
    instance = problem()
    composed = compose_two_hot_span(
        instance, step_count=STEP_COUNT, seed=0, collision_weight=10.0
    ).run()
    by_hand = TwoHotSpanOptimizer(
        instance,
        training_cost(instance),
        [CollisionPenalty(weight=10.0)],
        settings=TwoHotSpanSettings(step_count=STEP_COUNT, seed=0),
    ).run()

    assert np.array_equal(composed.spanning_set, by_hand.spanning_set)
    assert composed.relaxed_objective == by_hand.relaxed_objective
    assert composed.rounded_cut == by_hand.rounded_cut
    assert composed.final_training_loss == by_hand.final_training_loss
    assert np.array_equal(composed.labels, by_hand.labels)
    assert composed.configuration == by_hand.configuration


def test_the_defaults_are_the_objects_own():
    assert TwoHotSpanSettings().step_count == DEFAULT_STEP_COUNT
    assert AdamStepRule().learning_rate == DEFAULT_LEARNING_RATE
    assert RidgeProjector().epsilon == DEFAULT_EPSILON


def test_the_composed_configuration_is_plain_json_naming_every_knob():
    record = compose_two_hot_span(
        problem(),
        step_count=STEP_COUNT,
        schedule="warmup_cosine",
        warmup_steps=3,
        final_learning_rate_fraction=0.1,
        collision_weight=1.0,
        adjacency_weight=0.3,
        adjacency_form="edge_product",
        diversity_weight=2.0,
    ).configuration

    json.dumps(record)
    assert record["step_rule"]["schedule"] == {
        "name": "WarmupCosineSchedule",
        "warmup_steps": 3,
        "final_fraction": 0.1,
    }
    assert [penalty["name"] for penalty in record["penalties"]] == [
        "CollisionPenalty",
        "EdgeProductAdjacencyPenalty",
        "DiversityPenalty",
    ]
    assert record["cost"]["projector"]["epsilon"] == DEFAULT_EPSILON
