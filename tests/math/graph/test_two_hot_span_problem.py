"""The two-hot span problem: the incidence identity, the rounded cut, and the spectral floor.

The load-bearing test is `test_rounded_cut_of_a_within_block_spanning_forest_equals_the_ratio_cut`:
it is the plan's acceptance line for slice 2.1, and every reported Ê rests on it. The oracle tests
check `brute_force_rcut` against an enumeration written independently here, so a bug would have to
be made twice in two different shapes to survive.
"""

import inspect
import itertools

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.describe import describe
from mllib.math.graph.two_hot_span_problem import (
    GraphInstance,
    brute_force_rcut,
    clustering_from_pairs,
    collision_measure,
    collision_measures,
    default_test_graphs,
    incidence_matrix,
    laplacian_matrix,
    ratio_cut,
    roach_g20_instance,
    roach_graph,
    round_columns,
    rounded_cut,
    rounded_pairs,
    spanning_vector_count,
    spectral_floor,
    spectral_spanning_set,
    two_triangles_instance,
)
from mllib.math.projector import ExactProjector


def random_weighted_graph(n: int, seed: int, extra_edges: int = 4) -> nx.Graph:
    """A connected weighted graph: a random spanning tree plus a few random chords."""
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for node in range(1, n):
        graph.add_edge(node, int(rng.integers(0, node)), weight=float(rng.uniform(0.5, 2.0)))
    for _ in range(extra_edges):
        first, second = int(rng.integers(0, n)), int(rng.integers(0, n))
        if first != second and not graph.has_edge(first, second):
            graph.add_edge(first, second, weight=float(rng.uniform(0.5, 2.0)))
    return graph


def random_partition_labels(n: int, cluster_count: int, rng: np.random.Generator) -> np.ndarray:
    """A labelling into exactly `cluster_count` non-empty blocks."""
    while True:
        labels = rng.integers(0, cluster_count, size=n)
        if len(np.unique(labels)) == cluster_count:
            return np.asarray(labels, dtype=int)


def random_within_block_forest(labels: np.ndarray, rng: np.random.Generator) -> list[tuple]:
    """A random spanning tree per block: n - K pairs in total, none of them crossing a block."""
    pairs = []
    for block in np.unique(labels):
        members = list(np.flatnonzero(labels == block))
        rng.shuffle(members)
        for position in range(1, len(members)):
            parent = members[int(rng.integers(0, position))]
            pairs.append((members[position], parent))
    return pairs


def two_hot_columns(n: int, pairs: list[tuple], rng: np.random.Generator) -> np.ndarray:
    """The pairs as ± scaled two-hot columns; the scales prove rounding is scale-invariant."""
    columns = np.zeros((n, len(pairs)))
    for column, (first, second) in enumerate(pairs):
        scale = float(rng.uniform(0.1, 10.0))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        columns[first, column] = sign * scale
        columns[second, column] = -sign * scale
    return columns


def random_zero_sum_spanning_set(n: int, cluster_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    columns = rng.standard_normal((n, spanning_vector_count(n, cluster_count)))
    return columns - columns.mean(axis=0, keepdims=True)


def as_partition(labels: np.ndarray) -> set[frozenset]:
    """A labelling as the set of its blocks, so two labellings compare up to relabelling."""
    return {frozenset(np.flatnonzero(labels == block).tolist()) for block in np.unique(labels)}


def test_incidence_matrix_times_its_transpose_is_the_weighted_laplacian():
    graph = random_weighted_graph(9, seed=11)
    X = incidence_matrix(graph)
    expected = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray().astype(float)
    assert X.shape == (9, graph.number_of_edges())
    assert np.allclose(X @ X.T, expected, atol=1e-12)
    assert np.allclose(laplacian_matrix(graph), expected, atol=1e-12)


def test_roach_graph_has_four_k_vertices_and_five_k_minus_two_edges():
    for rung_count in (2, 3, 5, 7):
        graph = roach_graph(rung_count)
        assert graph.number_of_nodes() == 4 * rung_count
        assert graph.number_of_edges() == 5 * rung_count - 2
        assert nx.is_connected(graph)
    assert roach_graph(5).number_of_nodes() == 20
    assert roach_graph(5).number_of_edges() == 23


def test_the_three_known_roach_cuts_score_four_fifteenths_two_fifths_and_one():
    """One antenna is the optimum; antennae-vs-ladder is only the best *balanced* cut."""
    rung_count = 5
    graph = roach_graph(rung_count)

    one_antenna = np.zeros(4 * rung_count, dtype=int)
    one_antenna[list(range(rung_count))] = 1
    assert ratio_cut(graph, one_antenna) == pytest.approx(4 / 15, abs=1e-12)

    antennae_vs_ladder = np.zeros(4 * rung_count, dtype=int)
    antennae = list(range(rung_count)) + list(range(2 * rung_count, 3 * rung_count))
    antennae_vs_ladder[antennae] = 1
    assert ratio_cut(graph, antennae_vs_ladder) == pytest.approx(2 / 5, abs=1e-12)

    top_vs_bottom = np.zeros(4 * rung_count, dtype=int)
    top_vs_bottom[list(range(2 * rung_count, 4 * rung_count))] = 1
    assert ratio_cut(graph, top_vs_bottom) == pytest.approx(1.0, abs=1e-12)


def test_the_roach_g20_instance_is_the_papers_eighty_node_cockroach():
    instance = roach_g20_instance()
    assert instance.name == "roach_g20"
    assert instance.cluster_count == 3
    assert instance.graph.number_of_nodes() == 80
    assert instance.graph.number_of_edges() == 98
    assert nx.is_connected(instance.graph)


def test_the_roach_g20_planted_labels_are_the_ladder_and_the_two_antennae():
    instance = roach_g20_instance()
    labels = instance.planted_labels
    assert sorted(np.flatnonzero(labels == 1)) == list(range(0, 20))
    assert sorted(np.flatnonzero(labels == 2)) == list(range(40, 60))
    assert sorted(np.flatnonzero(labels == 0)) == list(range(20, 40)) + list(range(60, 80))


def test_the_roach_g20_planted_ratio_cut_is_three_twentieths():
    """2/40 + 1/20 + 1/20: the ladder loses one edge to each antenna, each antenna one back."""
    instance = roach_g20_instance()
    assert ratio_cut(instance.graph, instance.planted_labels) == pytest.approx(0.15, abs=1e-12)


def test_the_roach_g20_antennae_versus_ladder_cut_is_one_tenth():
    """The K = 2 reference: 2/40 + 2/40, cheaper than the planted cut because it pays per block."""
    instance = roach_g20_instance()
    labels = np.where(instance.planted_labels == 0, 0, 1)
    assert ratio_cut(instance.graph, labels) == pytest.approx(0.10, abs=1e-12)


def test_removing_the_two_antenna_edges_leaves_the_three_planted_blocks_as_components():
    """The paper's reason for K = 3: the two antennae are disconnected once the ladder goes."""
    instance = roach_g20_instance()
    severed = instance.graph.copy()
    # 19-20 joins the top antenna to the top of the ladder, 59-60 the bottom one to the bottom.
    severed.remove_edges_from([(19, 20), (59, 60)])
    components = sorted(nx.connected_components(severed), key=min)
    # The ladder survives as one component: the twenty rungs still join its two halves.
    assert [sorted(component) for component in components] == [
        list(range(0, 20)),
        list(range(20, 40)) + list(range(60, 80)),
        list(range(40, 60)),
    ]
    blocks = {frozenset(np.flatnonzero(instance.planted_labels == label)) for label in (0, 1, 2)}
    assert {frozenset(component) for component in components} == blocks


def test_the_roach_g20_laplacian_reproduces_the_papers_three_smallest_eigenvalues():
    """He, Gu & Zhang 2012 report λ2 = 0.0057, λ3 = 0.0062, λ4 = 0.0246 for this graph."""
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix(roach_g20_instance().graph))
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-10)
    assert [round(float(value), 4) for value in eigenvalues[1:4]] == [0.0057, 0.0062, 0.0246]


def test_the_two_triangles_instance_is_six_nodes_and_seven_weighted_edges():
    """Two weight-1 triangles and the weight-0.1 bridge (2, 3) that joins them."""
    instance = two_triangles_instance()
    graph = instance.graph
    assert instance.name == "two_triangles"
    assert instance.cluster_count == 2
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (6, 7)
    assert nx.is_connected(graph)
    assert graph[2][3]["weight"] == pytest.approx(0.1, abs=1e-12)
    triangle_edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    assert all(graph[first][second]["weight"] == 1.0 for first, second in triangle_edges)
    assert sorted(graph.edges()) == sorted([*triangle_edges, (2, 3)])


def test_the_two_triangles_spectral_floor_is_the_number_every_run_is_read_against():
    """Σλ = 0.063771: the λ·r/2 = 20 collision ceiling at λ = 10 is three hundred times it."""
    instance = two_triangles_instance()
    assert spectral_floor(instance.graph, instance.cluster_count) == pytest.approx(
        0.063771, abs=1e-6
    )


def test_the_two_triangles_optimum_is_one_fifteenth_at_the_planted_labels():
    """Cutting the 0.1 bridge costs 0.1/3 + 0.1/3; n = 6 is inside the brute-force guard."""
    instance = two_triangles_instance()
    optimum, labels = brute_force_rcut(instance.graph, instance.cluster_count)
    assert optimum == pytest.approx(1 / 15, abs=1e-12)
    assert as_partition(labels) == as_partition(instance.planted_labels)
    assert ratio_cut(instance.graph, instance.planted_labels) == pytest.approx(1 / 15, abs=1e-12)


def test_rounded_cut_of_a_within_block_spanning_forest_equals_the_ratio_cut():
    """The plan's acceptance line for slice 2.1: Ê is a RatioCut, at machine precision."""
    for seed, (n, cluster_count) in enumerate([(7, 2), (8, 3), (9, 2), (10, 3), (11, 2), (11, 3)]):
        rng = np.random.default_rng(1000 + seed)
        graph = random_weighted_graph(n, seed=200 + seed)
        labels = random_partition_labels(n, cluster_count, rng)
        pairs = random_within_block_forest(labels, rng)
        assert len(pairs) == spanning_vector_count(n, cluster_count)
        spanning_set = two_hot_columns(n, pairs, rng)
        X = incidence_matrix(graph)
        assert rounded_cut(X, spanning_set) == pytest.approx(ratio_cut(graph, labels), abs=1e-10)
        recovered = clustering_from_pairs(n, rounded_pairs(spanning_set))
        assert as_partition(recovered) == as_partition(labels)


def test_degenerate_columns_round_to_zero_and_leave_the_ratio_cut_identity_intact():
    """A constant column has no pair to read off; it must round to zero, not to a 1-hot column."""
    n, cluster_count = 9, 3
    rng = np.random.default_rng(31)
    graph = random_weighted_graph(n, seed=32)
    labels = random_partition_labels(n, cluster_count, rng)
    pairs = random_within_block_forest(labels, rng)
    forest_columns = two_hot_columns(n, pairs, rng)
    zero_column = np.zeros((n, 1))
    constant_column = np.full((n, 1), 2.5)
    spanning_set = np.hstack([forest_columns, zero_column, constant_column])

    assert len(rounded_pairs(spanning_set)) == forest_columns.shape[1]
    assert np.array_equal(round_columns(spanning_set)[:, -2:], np.zeros((n, 2)))
    recovered = clustering_from_pairs(n, rounded_pairs(spanning_set))
    assert as_partition(recovered) == as_partition(labels)
    assert rounded_cut(incidence_matrix(graph), spanning_set) == pytest.approx(
        ratio_cut(graph, recovered), abs=1e-10
    )


def enumerate_ratio_cut_optimum(graph: nx.Graph, cluster_count: int) -> float:
    """An independent oracle: every surjective labelling, scored, minimised. Deliberately naive."""
    n = graph.number_of_nodes()
    best = np.inf
    for labels in itertools.product(range(cluster_count), repeat=n):
        if len(set(labels)) != cluster_count:
            continue
        best = min(best, ratio_cut(graph, np.asarray(labels, dtype=int)))
    return float(best)


def test_brute_force_rcut_matches_an_independent_enumeration():
    for seed, (n, cluster_count) in enumerate([(6, 2), (7, 2), (7, 3), (8, 2), (8, 3)]):
        graph = random_weighted_graph(n, seed=300 + seed)
        optimum, labels = brute_force_rcut(graph, cluster_count)
        assert len(np.unique(labels)) == cluster_count
        assert optimum == pytest.approx(
            enumerate_ratio_cut_optimum(graph, cluster_count), abs=1e-12
        )
        assert ratio_cut(graph, labels) == pytest.approx(optimum, abs=1e-12)


def test_the_roach_optimum_at_k_two_is_one_antenna_alone():
    """4/(3k) at k = 2, and the optimal blocks are one antenna against everything else."""
    optimum, labels = brute_force_rcut(roach_graph(2), 2)
    assert optimum == pytest.approx(4 / 6, abs=1e-12)
    assert as_partition(labels) in (
        {frozenset({0, 1}), frozenset({2, 3, 4, 5, 6, 7})},
        {frozenset({4, 5}), frozenset({0, 1, 2, 3, 6, 7})},
    )


def test_brute_force_rcut_refuses_graphs_above_the_node_guard():
    with pytest.raises(ValueError, match="guarded"):
        brute_force_rcut(random_weighted_graph(11, seed=4), 2)


def test_spectral_floor_is_zero_when_the_graph_has_exactly_k_components():
    graph = nx.disjoint_union_all([nx.complete_graph(4), nx.complete_graph(3), nx.cycle_graph(5)])
    assert spectral_floor(graph, 3) == pytest.approx(0.0, abs=1e-10)


def test_spectral_spanning_set_attains_the_spectral_floor():
    for seed, (n, cluster_count) in enumerate([(8, 2), (9, 3), (10, 2)]):
        graph = random_weighted_graph(n, seed=400 + seed)
        X = incidence_matrix(graph)
        spanning_set = spectral_spanning_set(graph, cluster_count)
        assert np.allclose(spanning_set.sum(axis=0), 0.0, atol=1e-10)
        assert ExactProjector().residual(X, spanning_set) == pytest.approx(
            spectral_floor(graph, cluster_count), abs=1e-10
        )


def test_spectral_spanning_set_refuses_a_disconnected_graph():
    with pytest.raises(ValueError, match="connected"):
        spectral_spanning_set(nx.disjoint_union_all([nx.complete_graph(3)] * 2), 1)


def test_projector_residual_and_rounded_cut_stay_above_the_spectral_floor():
    for seed, (n, cluster_count) in enumerate([(8, 2), (9, 3), (10, 2)]):
        graph = random_weighted_graph(n, seed=500 + seed)
        X = incidence_matrix(graph)
        spanning_set = random_zero_sum_spanning_set(n, cluster_count, seed=600 + seed)
        floor = spectral_floor(graph, cluster_count)
        assert ExactProjector().residual(X, spanning_set) >= floor - 1e-10
        assert rounded_cut(X, spanning_set) >= floor - 1e-10


def test_rounded_cut_never_beats_the_brute_force_optimum():
    """Merging blocks never lowers RatioCut, so no rounding can undercut the exact optimum."""
    for seed, (n, cluster_count) in enumerate([(7, 2), (8, 2), (8, 3)]):
        graph = random_weighted_graph(n, seed=700 + seed)
        X = incidence_matrix(graph)
        spanning_set = random_zero_sum_spanning_set(n, cluster_count, seed=800 + seed)
        optimum, _ = brute_force_rcut(graph, cluster_count)
        assert rounded_cut(X, spanning_set) >= optimum - 1e-10


def test_collision_measure_is_one_half_exactly_at_a_two_hot_vector():
    vector = np.zeros(6)
    vector[1], vector[4] = 3.0, -3.0
    assert collision_measure(vector) == pytest.approx(0.5, abs=1e-15)


def test_collision_measure_is_below_one_half_for_a_spread_zero_sum_vector():
    rng = np.random.default_rng(7)
    vector = rng.standard_normal(9)
    vector -= vector.mean()
    assert collision_measure(vector) < 0.5


def test_collision_measure_of_an_alternating_vector_is_one_over_n():
    n = 8
    vector = np.array([1.0 if index % 2 == 0 else -1.0 for index in range(n)])
    assert collision_measure(vector) == pytest.approx(1.0 / n, abs=1e-15)


def test_collision_measure_is_scale_invariant_and_zero_on_the_zero_vector():
    rng = np.random.default_rng(8)
    vector = rng.standard_normal(7)
    vector -= vector.mean()
    assert collision_measure(17.5 * vector) == pytest.approx(collision_measure(vector), abs=1e-14)
    assert collision_measure(np.zeros(5)) == 0.0
    assert collision_measures(np.column_stack([vector, 2.0 * vector])).shape == (2,)


def test_round_columns_leaves_a_two_hot_matrix_unchanged():
    spanning_set = np.zeros((5, 2))
    spanning_set[0, 0], spanning_set[3, 0] = 1.0, -1.0
    spanning_set[4, 1], spanning_set[2, 1] = 1.0, -1.0
    assert np.array_equal(round_columns(spanning_set), spanning_set)


def test_rounded_pairs_are_the_argmax_and_argmin_of_each_column():
    spanning_set = np.array([[0.2, -1.0], [-0.7, 0.3], [1.5, 0.9]])
    assert rounded_pairs(spanning_set) == [(2, 1), (2, 0)]


def test_default_test_graphs_are_the_six_named_connected_instances():
    """`two_triangles` is appended last, after `roach_g20`, by the same index-stability rule."""
    instances = default_test_graphs()
    assert [instance.name for instance in instances] == [
        "roach_g5",
        "karate",
        "planted_partition",
        "two_moons_knn",
        "roach_g20",
        "two_triangles",
    ]
    assert [instance.cluster_count for instance in instances] == [2, 2, 3, 2, 3, 2]
    assert instances[0].graph.number_of_nodes() == 20
    for instance in instances:
        assert isinstance(instance, GraphInstance)
        assert nx.is_connected(instance.graph)
        if instance.planted_labels is not None:
            assert len(np.unique(instance.planted_labels)) == instance.cluster_count


def test_spanning_vector_count_rejects_cluster_counts_outside_one_to_n_minus_one():
    assert spanning_vector_count(10, 3) == 7
    with pytest.raises(ValueError):
        spanning_vector_count(5, 5)
    with pytest.raises(ValueError):
        spanning_vector_count(5, 0)


def test_the_exact_projector_exposes_no_epsilon():
    """D-31 guard: the ridge is a training knob and must never reach the reporting path."""
    assert "epsilon" not in inspect.signature(ExactProjector.residual).parameters
    assert describe(ExactProjector)["params"] == []


@settings(deadline=None, max_examples=50, derandomize=True)
@given(
    entries=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=12,
    )
)
def test_collision_measure_never_exceeds_one_half_for_a_zero_sum_vector(entries: list[float]):
    vector = np.array(entries, dtype=float)
    vector -= vector.mean()
    assert collision_measure(vector) <= 0.5 + 1e-12
