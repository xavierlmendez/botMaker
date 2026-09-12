"""`TwoHotSpanProblem`: the instance object the optimizer is handed (BL-48 slice 4).

Everything it exposes is the module function it wraps, checked exactly on roach G₅; the object
adds the constraint vector (the ncut seam, BL-41), a configuration and the report. No torch.
"""

from __future__ import annotations

import numpy as np

from mllib.math.graph.two_hot_span_problem import (
    GraphInstance,
    TwoHotSpanProblem,
    TwoHotSpanReport,
    clustering_from_pairs,
    collision_measures,
    graph_matrices,
    incidence_matrix,
    roach_graph,
    rounded_cut,
    rounded_pairs,
    spanning_vector_count,
    spectral_floor,
    spectral_spanning_set,
)
from mllib.math.projector import ExactProjector

CLUSTER_COUNT = 2


def problem() -> TwoHotSpanProblem:
    return TwoHotSpanProblem(roach_graph(5), CLUSTER_COUNT, name="roach_g5")


def test_the_problem_counts_its_nodes_and_spanning_vectors():
    instance = problem()
    assert instance.node_count == 20
    assert instance.spanning_vector_count == spanning_vector_count(20, CLUSTER_COUNT)
    assert instance.cluster_count == CLUSTER_COUNT


def test_the_incidence_matrix_is_the_module_functions_exactly():
    instance = problem()
    assert np.array_equal(instance.X, incidence_matrix(roach_graph(5)))


def test_the_laplacian_and_adjacency_are_recovered_from_the_incidence_matrix():
    instance = problem()
    laplacian, adjacency = graph_matrices(instance.X)
    assert np.array_equal(instance.laplacian, laplacian)
    assert np.array_equal(instance.adjacency, adjacency)


def test_the_constraint_vector_is_ones_for_the_ratio_cut():
    instance = problem()
    assert np.array_equal(instance.constraint_vector, np.ones(20))
    assert instance.constraint_vector.dtype == np.float64


def test_the_configuration_names_the_instance():
    assert problem().configuration == {
        "name": "TwoHotSpanProblem",
        "graph": "roach_g5",
        "node_count": 20,
        "cluster_count": CLUSTER_COUNT,
    }


def test_from_instance_carries_the_name_and_the_cluster_count():
    instance = TwoHotSpanProblem.from_instance(GraphInstance("roach", roach_graph(5), 3))
    assert (instance.name, instance.cluster_count) == ("roach", 3)
    assert instance.spanning_vector_count == spanning_vector_count(20, 3)


def test_the_report_is_the_four_module_functions_on_the_same_spanning_set_exactly():
    instance = problem()
    spanning_set = np.random.default_rng(0).standard_normal((20, instance.spanning_vector_count))

    report = instance.report(spanning_set)

    assert isinstance(report, TwoHotSpanReport)
    assert report.relaxed_objective == ExactProjector().residual(instance.X, spanning_set)
    assert report.rounded_cut == rounded_cut(instance.X, spanning_set)
    assert np.array_equal(report.collision_measures, collision_measures(spanning_set))
    assert np.array_equal(report.labels, clustering_from_pairs(20, rounded_pairs(spanning_set)))


def test_the_spectral_floor_and_start_are_the_module_functions():
    instance = problem()
    assert instance.spectral_floor() == spectral_floor(roach_graph(5), CLUSTER_COUNT)
    assert np.array_equal(
        instance.spectral_spanning_set(), spectral_spanning_set(roach_graph(5), CLUSTER_COUNT)
    )


def test_a_non_finite_spanning_set_reports_nans_and_no_clustering_instead_of_raising():
    """D-35 (9): a run stopped for a non-finite value still returns; its report is NaNs."""
    problem = TwoHotSpanProblem(roach_graph(5), 2)
    report = problem.report(np.full((problem.node_count, problem.spanning_vector_count), np.nan))

    assert np.isnan(report.relaxed_objective) and np.isnan(report.rounded_cut)
    assert report.collision_measures.shape == (problem.spanning_vector_count,)
    assert np.isnan(report.collision_measures).all()
    assert np.array_equal(report.labels, np.full(problem.node_count, -1))
