"""Tests for slice 2.2 — the two-hot span PyTorch optimizer.

Deterministic: every graph is built or seeded explicitly and every run carries its own seed.
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.math.algorithms.two_hot_span_optimizer import (
    NonFiniteLoss,
    TwoHotSpanConfig,
    fit_two_hot_span,
)
from mllib.math.graph.two_hot_span_problem import (
    brute_force_rcut,
    incidence_matrix,
    projector_residual,
    ratio_cut,
    roach_graph,
    spectral_floor,
    spectral_spanning_set,
)


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


def test_the_default_configuration_reproduces_an_explicit_constant_run():
    """The experimental knobs are off by default, and "off" means bit for bit, not nearly.

    The schedule and the graph term are slice 2.4's experiment and the diversity term is slice
    2.5's (`docs/plans/2026-09-two-hot-span.md`). The default objective has to stay the brief's §20
    objective, so a config that names none of them and a config that names them all at their
    neutral values must produce the same numbers in the same order — not merely the same answer to
    ten places.
    """
    X = incidence_matrix(roach_graph(5))
    silent = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=30, seed=0, collision_weight=10.0))
    spelled_out = fit_two_hot_span(
        X,
        2,
        TwoHotSpanConfig(
            step_count=30,
            seed=0,
            collision_weight=10.0,
            learning_rate_schedule="constant",
            warmup_steps=0,
            final_learning_rate_fraction=0.0,
            adjacency_weight=0.0,
            adjacency_form="laplacian",
            diversity_weight=0.0,
        ),
    )

    assert np.array_equal(silent.spanning_set, spelled_out.spanning_set)
    assert np.array_equal(silent.loss_history, spelled_out.loss_history)
    assert np.array_equal(silent.learning_rate_history, spelled_out.learning_rate_history)
    assert np.all(silent.learning_rate_history == silent.config.learning_rate)


def test_zero_sum_holds_at_every_step():
    X = incidence_matrix(_seeded_graph(9, seed=3))
    run = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=50, seed=3, collision_weight=0.5))
    assert run.max_zero_sum_violation <= 1e-12


def test_zero_sum_holds_without_column_normalization():
    X = incidence_matrix(_seeded_graph(9, seed=3))
    config = TwoHotSpanConfig(step_count=50, seed=3, collision_weight=0.5, normalize_columns=False)
    run = fit_two_hot_span(X, 2, config)
    assert run.max_zero_sum_violation <= 1e-12
    assert np.isfinite(run.loss_history).all()


def test_roach_run_stays_finite():
    X = incidence_matrix(roach_graph(5))
    run = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=20, collision_weight=1.0, seed=0))
    assert run.loss_history.shape == (20,)
    assert np.isfinite(run.loss_history).all()
    assert np.isfinite(run.spanning_set).all()


def test_reporting_path_never_uses_epsilon():
    graph = _seeded_graph(8, seed=1)
    X = incidence_matrix(graph)
    initial = spectral_spanning_set(graph, 2)
    small = fit_two_hot_span(
        X, 2, TwoHotSpanConfig(step_count=0, epsilon=1e-6), initial_spanning_set=initial
    )
    large = fit_two_hot_span(
        X, 2, TwoHotSpanConfig(step_count=0, epsilon=1e-1), initial_spanning_set=initial
    )
    assert small.relaxed_objective == large.relaxed_objective
    assert small.rounded_cut == large.rounded_cut
    assert small.relaxed_objective == projector_residual(X, small.spanning_set)
    assert "epsilon" not in inspect.signature(projector_residual).parameters


def test_reported_cut_is_above_the_brute_force_optimum_and_the_floor():
    graph = _seeded_graph(8, seed=5)
    X = incidence_matrix(graph)
    optimum, _ = brute_force_rcut(graph, 2)
    floor = spectral_floor(graph, 2)
    run = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=200, seed=7, collision_weight=1.0))
    assert run.rounded_cut >= optimum - 1e-10
    assert run.relaxed_objective >= floor - 1e-10


def test_spectral_initialisation_holds_the_floor():
    """At step 0 the floor is exact; Adam's normalized step then drifts at the scale of the
    learning rate, not of the (vanishing) gradient, so the 1e-8 claim only holds at a small lr.
    """
    graph = _seeded_graph(8, seed=1)
    X = incidence_matrix(graph)
    initial = spectral_spanning_set(graph, 2)
    floor = spectral_floor(graph, 2)

    settled = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=0), initial_spanning_set=initial)
    assert abs(settled.relaxed_objective - floor) <= 1e-12

    crawled = fit_two_hot_span(
        X,
        2,
        TwoHotSpanConfig(step_count=20, learning_rate=1e-5, collision_weight=0.0),
        initial_spanning_set=initial,
    )
    assert crawled.relaxed_objective - floor <= 1e-8

    walked = fit_two_hot_span(
        X,
        2,
        TwoHotSpanConfig(step_count=20, learning_rate=0.05, collision_weight=0.0),
        initial_spanning_set=initial,
    )
    assert walked.relaxed_objective - floor <= 1e-2


def test_collision_penalty_raises_the_mean_collision_measure():
    X = incidence_matrix(_seeded_graph(8, seed=2))
    start = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=0, seed=2, collision_weight=5.0))
    finish = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=300, seed=2, collision_weight=5.0))
    assert finish.collision_measures.mean() > start.collision_measures.mean()


def test_non_finite_initial_spanning_set_raises():
    graph = _seeded_graph(8, seed=1)
    X = incidence_matrix(graph)
    initial = spectral_spanning_set(graph, 2)
    initial[0, 0] = np.nan
    with pytest.raises(NonFiniteLoss):
        fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=5), initial_spanning_set=initial)


def test_two_hot_forest_initialisation_reports_the_ratio_cut():
    graph = _seeded_graph(8, seed=5)
    X = incidence_matrix(graph)
    _, labels = brute_force_rcut(graph, 2)
    initial = _within_block_forest(graph, labels)
    run = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=0), initial_spanning_set=initial)
    assert run.rounded_cut == pytest.approx(ratio_cut(graph, labels), abs=1e-10)
    assert (run.labels == labels).all()


def test_seeded_run_reports_both_the_cut_and_the_brute_force_optimum(capsys):
    """The plan's S6 clause: the report states both Ê and the n <= 8 brute-force optimum."""
    graph = _seeded_graph(8, seed=5)
    X = incidence_matrix(graph)
    optimum, optimal_labels = brute_force_rcut(graph, 2)
    run = fit_two_hot_span(X, 2, TwoHotSpanConfig(step_count=200, seed=11, collision_weight=1.0))

    report = (
        f"seeded 200-step run on a seeded n = 8 graph (seed 5, K = 2): "
        f"Ehat = {run.rounded_cut:.10f}, brute-force optimum = {optimum:.10f}, "
        f"Ehat - optimum = {run.rounded_cut - optimum:.10f}, "
        f"labels = {run.labels.tolist()}, optimal labels = {optimal_labels.tolist()}"
    )
    with capsys.disabled():
        print(report)

    assert run.rounded_cut >= optimum - 1e-10, report
