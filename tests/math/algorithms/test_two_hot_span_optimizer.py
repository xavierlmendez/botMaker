"""Tests for the two-hot span optimizer on `AbstractOptimizer` (BL-48 slice 4).

Deterministic: every graph is built or seeded explicitly and every run carries its own seed. The
optimizer is composed here from math objects alone (`two_hot_span_support`); the per-step
histories the old result carried are read off a recorder (D-35 (9)).
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.describe import describe
from mllib.math.algorithms.abstract_optimizer import StopReason
from mllib.math.algorithms.two_hot_span.optimizer import (
    TwoHotSpanOptimizer,
    TwoHotSpanSettings,
    training_cost,
)
from mllib.math.algorithms.two_hot_span.penalties import CollisionPenalty
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.graph.two_hot_span_problem import (
    brute_force_rcut,
    incidence_matrix,
    ratio_cut,
    roach_graph,
    spectral_floor,
    spectral_spanning_set,
)
from mllib.math.learning_rate_schedule import ConstantSchedule
from mllib.math.projector import ExactProjector

from .two_hot_span_support import HistoryRecorder, problem_for, run, run_with_history


def _seeded_graph(node_total: int, seed: int) -> nx.Graph:
    """A connected seeded random graph with unit weights."""
    graph = nx.gnp_random_graph(node_total, 0.55, seed=seed)
    if not nx.is_connected(graph):
        graph = nx.connected_watts_strogatz_graph(node_total, 4, 0.3, seed=seed)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def _within_block_forest(graph: nx.Graph, labels: np.ndarray) -> np.ndarray:
    """A 2-hot spanning set: a spanning forest of the blocks of `labels`."""
    node_total = graph.number_of_nodes()
    pairs: list[tuple[int, int]] = []
    for block in np.unique(labels):
        members = [int(v) for v in np.flatnonzero(labels == block)]
        pairs += list(nx.minimum_spanning_tree(graph.subgraph(members)).edges())
    spanning_set = np.zeros((node_total, len(pairs)), dtype=np.float64)
    for column, (high, low) in enumerate(pairs):
        spanning_set[high, column] = 1.0
        spanning_set[low, column] = -1.0
    return spanning_set


def test_the_defaulted_objects_reproduce_an_explicit_constant_run():
    """ "Off by default" means bit for bit, not nearly.

    An optimizer that leaves the step rule to its default and names one penalty must produce the
    same numbers in the same order as one that spells out `AdamStepRule(0.05, ConstantSchedule())`
    and the same penalty tuple — the default objective is the brief's §20 objective.
    """
    graph = roach_graph(5)
    silent_history, spelled_history = HistoryRecorder(), HistoryRecorder()
    problem = problem_for(graph, 2)
    silent = TwoHotSpanOptimizer(
        problem,
        training_cost(problem),
        [CollisionPenalty(weight=10.0)],
        settings=TwoHotSpanSettings(step_count=30, seed=0),
        recorder=silent_history,
    ).run()
    problem = problem_for(graph, 2)
    spelled_out = TwoHotSpanOptimizer(
        problem,
        training_cost(problem),
        (CollisionPenalty(weight=10.0),),
        AdamStepRule(0.05, ConstantSchedule()),
        settings=TwoHotSpanSettings(
            step_count=30, seed=0, normalize_columns=True, zero_sum_tolerance=1e-12
        ),
        recorder=spelled_history,
    ).run()

    assert np.array_equal(silent.spanning_set, spelled_out.spanning_set)
    assert np.array_equal(silent_history.loss_history, spelled_history.loss_history)
    assert np.array_equal(
        silent_history.learning_rate_history, spelled_history.learning_rate_history
    )
    assert np.all(silent_history.learning_rate_history == 0.05)
    assert silent.final_training_loss == spelled_out.final_training_loss
    assert silent.rounded_cut == spelled_out.rounded_cut


def test_zero_sum_holds_at_every_step():
    result = run(_seeded_graph(9, seed=3), 2, step_count=50, seed=3, collision_weight=0.5)
    assert result.max_zero_sum_violation <= 1e-12


def test_zero_sum_holds_without_column_normalization():
    result, history = run_with_history(
        _seeded_graph(9, seed=3),
        2,
        step_count=50,
        seed=3,
        collision_weight=0.5,
        normalize_columns=False,
    )
    assert result.max_zero_sum_violation <= 1e-12
    assert np.isfinite(history.loss_history).all()


def test_roach_run_stays_finite():
    result, history = run_with_history(
        roach_graph(5), 2, step_count=20, collision_weight=1.0, seed=0
    )
    assert history.loss_history.shape == (20,)
    assert result.steps_taken == 20
    assert np.isfinite(history.loss_history).all()
    assert np.isfinite(result.spanning_set).all()


def test_a_run_that_completes_its_budget_says_so():
    result = run(roach_graph(5), 2, step_count=20, collision_weight=1.0, seed=0)
    assert result.stop_reason is StopReason.STEP_BUDGET
    assert result.steps_taken == 20
    assert result.final_training_loss is not None


def test_reporting_path_never_uses_epsilon():
    graph = _seeded_graph(8, seed=1)
    X = incidence_matrix(graph)
    initial = spectral_spanning_set(graph, 2)
    small = run(graph, 2, step_count=0, epsilon=1e-6, initial=initial)
    large = run(graph, 2, step_count=0, epsilon=1e-1, initial=initial)
    assert small.relaxed_objective == large.relaxed_objective
    assert small.rounded_cut == large.rounded_cut
    assert small.relaxed_objective == ExactProjector().residual(X, small.spanning_set)
    assert "epsilon" not in inspect.signature(ExactProjector.residual).parameters
    assert describe(ExactProjector)["params"] == []


def test_reported_cut_is_above_the_brute_force_optimum_and_the_floor():
    graph = _seeded_graph(8, seed=5)
    optimum, _ = brute_force_rcut(graph, 2)
    floor = spectral_floor(graph, 2)
    result = run(graph, 2, step_count=200, seed=7, collision_weight=1.0)
    assert result.rounded_cut >= optimum - 1e-10
    assert result.relaxed_objective >= floor - 1e-10


def test_spectral_initialisation_holds_the_floor():
    """At step 0 the floor is exact; Adam's normalized step then drifts at the scale of the
    learning rate, not of the (vanishing) gradient, so the 1e-8 claim only holds at a small lr.
    """
    graph = _seeded_graph(8, seed=1)
    initial = spectral_spanning_set(graph, 2)
    floor = spectral_floor(graph, 2)

    settled = run(graph, 2, step_count=0, initial=initial)
    assert abs(settled.relaxed_objective - floor) <= 1e-12

    crawled = run(graph, 2, step_count=20, learning_rate=1e-5, initial=initial)
    assert crawled.relaxed_objective - floor <= 1e-8

    walked = run(graph, 2, step_count=20, learning_rate=0.05, initial=initial)
    assert walked.relaxed_objective - floor <= 1e-2


def test_collision_penalty_raises_the_mean_collision_measure():
    graph = _seeded_graph(8, seed=2)
    start = run(graph, 2, step_count=0, seed=2, collision_weight=5.0)
    finish = run(graph, 2, step_count=300, seed=2, collision_weight=5.0)
    assert finish.collision_measures.mean() > start.collision_measures.mean()


def test_a_non_finite_initial_spanning_set_stops_the_run_before_any_step():
    """A stop is a result with a reason, not a raise (D-35 (9))."""
    graph = _seeded_graph(8, seed=1)
    initial = spectral_spanning_set(graph, 2)
    initial[0, 0] = np.nan

    result = run(graph, 2, step_count=5, initial=initial)

    assert result.stop_reason is StopReason.NON_FINITE
    assert "initial spanning set" in result.stop_detail
    assert result.steps_taken == 0
    assert result.final_training_loss is None


def test_two_hot_forest_initialisation_reports_the_ratio_cut():
    graph = _seeded_graph(8, seed=5)
    _, labels = brute_force_rcut(graph, 2)
    initial = _within_block_forest(graph, labels)
    result = run(graph, 2, step_count=0, initial=initial)
    assert result.rounded_cut == pytest.approx(ratio_cut(graph, labels), abs=1e-10)
    assert (result.labels == labels).all()


def test_seeded_run_reports_both_the_cut_and_the_brute_force_optimum(capsys):
    """The plan's S6 clause: the report states both Ê and the n <= 8 brute-force optimum."""
    graph = _seeded_graph(8, seed=5)
    optimum, optimal_labels = brute_force_rcut(graph, 2)
    result = run(graph, 2, step_count=200, seed=11, collision_weight=1.0)

    report = (
        f"seeded 200-step run on a seeded n = 8 graph (seed 5, K = 2): "
        f"Ehat = {result.rounded_cut:.10f}, brute-force optimum = {optimum:.10f}, "
        f"Ehat - optimum = {result.rounded_cut - optimum:.10f}, "
        f"labels = {result.labels.tolist()}, optimal labels = {optimal_labels.tolist()}"
    )
    with capsys.disabled():
        print(report)

    assert result.rounded_cut >= optimum - 1e-10, report
