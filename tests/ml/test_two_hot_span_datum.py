"""The datum half of slice 2.3: the spectral relaxation, rounded three ways.

These tests import only the numpy half of `two_hot_span_harness`, so they run in the default suite
without the optional torch group (D-31). The oracles are sklearn's own implementations of the same
three roundings, behind `importorskip`, and they check agreement on the quantity each rounding
actually optimises — labels, objective or inertia — never on a fixed number that depends on a seed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from mllib.math.graph.two_hot_span_problem import (
    brute_force_rcut,
    ratio_cut,
    roach_graph,
    two_triangles_instance,
)
from mllib.ml.projects.two_hot_span_harness import (
    datum_roundings,
    round_cluster_qr,
    round_discretize,
    round_kmeans,
    spectral_embedding,
)

ROUNDINGS = ("kmeans", "discretize", "cluster_qr")

# The roach graph's K = 2 RatioCut optimum is one antenna alone: one edge, 1/k + 1/(3k) = 4/15.
ROACH_OPTIMUM = 4.0 / 15.0
# Splitting the top path from the bottom path cuts all k rungs: k/(2k) + k/(2k) = 1.0. This is
# spectral bisection's documented failure on the roach, and cluster_qr reproduces it.
ROACH_TOP_VERSUS_BOTTOM = 1.0


def canonical(labels: np.ndarray) -> tuple[int, ...]:
    """A labelling's representative, so partitions can be compared up to relabelling."""
    remap: dict[int, int] = {}
    for value in labels:
        remap.setdefault(int(value), len(remap))
    return tuple(remap[int(value)] for value in labels)


def random_connected_weighted_graph(seed: int, node_count: int = 12) -> nx.Graph:
    """A seeded connected Erdos-Renyi graph with uniform (0.5, 1.5] weights."""
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        graph = nx.gnp_random_graph(node_count, 0.35, seed=seed * 100 + attempt)
        if nx.is_connected(graph):
            for source, target in graph.edges():
                graph[source][target]["weight"] = float(rng.uniform(0.5, 1.5))
            return graph
    raise AssertionError("no connected graph found")


RANDOM_GRAPHS = [(seed, random_connected_weighted_graph(seed)) for seed in range(5)]


def two_cliques_graph() -> nx.Graph:
    """Two disjoint 5-cliques, relabelled 0..9."""
    return nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))


def planted_graph() -> tuple[nx.Graph, np.ndarray]:
    """3 groups of 8, p_in 0.9, p_out 0.02, connected at this seed."""
    graph = nx.planted_partition_graph(3, 8, 0.9, 0.02, seed=1)
    assert nx.is_connected(graph)
    return graph, np.repeat(np.arange(3), 8)


def test_spectral_embedding_columns_are_orthonormal():
    graph = random_connected_weighted_graph(0)
    embedding = spectral_embedding(graph, 3)
    assert np.allclose(embedding.T @ embedding, np.eye(3), atol=1e-10)


def test_spectral_embedding_first_column_is_constant_on_a_connected_graph():
    graph = random_connected_weighted_graph(1)
    embedding = spectral_embedding(graph, 2)
    node_count = graph.number_of_nodes()
    assert np.allclose(embedding[:, 0], np.full(node_count, 1.0 / np.sqrt(node_count)), atol=1e-10)


def test_spectral_embedding_is_reproducible_across_calls():
    """The sign convention pins the columns, so a BLAS sign flip cannot change the result."""
    graph = random_connected_weighted_graph(2)
    assert np.array_equal(spectral_embedding(graph, 3), spectral_embedding(graph, 3))


@pytest.mark.parametrize("rounding", ROUNDINGS)
def test_rounding_recovers_two_disconnected_cliques(rounding):
    labels = datum_roundings(two_cliques_graph(), 2, seed=0)[rounding]
    assert canonical(labels) == canonical(np.repeat(np.arange(2), 5))


@pytest.mark.parametrize("rounding", ROUNDINGS)
def test_rounding_recovers_the_planted_partition(rounding):
    graph, planted = planted_graph()
    labels = datum_roundings(graph, 3, seed=0)[rounding]
    assert canonical(labels) == canonical(planted)


@pytest.mark.parametrize("cluster_count", [2, 3])
@pytest.mark.parametrize(("seed", "graph"), RANDOM_GRAPHS)
def test_rounding_labels_every_node_with_every_cluster(seed, graph, cluster_count):
    for labels in datum_roundings(graph, cluster_count, seed=seed).values():
        assert labels.shape == (graph.number_of_nodes(),)
        assert set(labels.tolist()) == set(range(cluster_count))


@pytest.mark.parametrize(("seed", "graph"), RANDOM_GRAPHS)
def test_round_cluster_qr_matches_the_sklearn_oracle(seed, graph):
    # A private sklearn module (scikit-learn 1.9.0), used as an oracle in tests only: nothing under
    # src/ imports it, and a version that moves or renames it skips this test, never the library.
    oracle = pytest.importorskip("sklearn.cluster._spectral")
    embedding = spectral_embedding(graph, 3)
    assert canonical(round_cluster_qr(embedding, 3)) == canonical(oracle.cluster_qr(embedding))


@pytest.mark.parametrize(("seed", "graph"), RANDOM_GRAPHS)
def test_round_discretize_reaches_the_sklearn_oracle_objective(seed, graph):
    """The comparison is on the objective, over a sweep of seeds, not on the labels.

    sklearn seeds its rotation from a legacy RandomState and we from a Generator, so the two
    alternations start in different places and settle in different local optima of the same
    Yu & Shi objective (the sum of singular values of the indicator against the row-normalised
    embedding). Neither the labels nor a single-seed objective is therefore robust; the best
    objective reached over a seed sweep is, and it agrees with sklearn's to 1e-6.
    """
    # A private sklearn module (scikit-learn 1.9.0), used as an oracle in tests only: nothing under
    # src/ imports it, and a version that moves or renames it skips this test, never the library.
    oracle = pytest.importorskip("sklearn.cluster._spectral")
    embedding = spectral_embedding(graph, 3)
    rows = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    def objective(labels):
        indicator = np.zeros((rows.shape[0], 3))
        indicator[np.arange(rows.shape[0]), np.asarray(labels)] = 1.0
        return float(np.linalg.svd(indicator.T @ rows, compute_uv=False).sum())

    mine = max(objective(round_discretize(embedding, 3, seed=sweep)) for sweep in range(10))
    theirs = max(objective(oracle.discretize(embedding, random_state=state)) for state in range(10))
    assert mine == pytest.approx(theirs, abs=1e-6)


@pytest.mark.parametrize(("seed", "graph"), RANDOM_GRAPHS)
def test_round_kmeans_inertia_is_at_most_the_sklearn_oracle(seed, graph):
    cluster_module = pytest.importorskip("sklearn.cluster")
    embedding = spectral_embedding(graph, 3)
    labels = round_kmeans(embedding, 3, seed=seed)
    centres = np.array([embedding[labels == label].mean(axis=0) for label in range(3)])
    inertia = float(((embedding - centres[labels]) ** 2).sum())
    oracle = cluster_module.KMeans(n_clusters=3, n_init=10, random_state=0).fit(embedding)
    assert inertia <= oracle.inertia_ + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_brute_force_rcut_is_at_most_every_rounding(seed):
    graph = random_connected_weighted_graph(seed, node_count=8)
    optimum, _ = brute_force_rcut(graph, 2)
    for labels in datum_roundings(graph, 2, seed=seed).values():
        assert optimum <= ratio_cut(graph, labels) + 1e-9


def test_all_three_roundings_reach_the_two_triangles_optimum():
    """What makes `two_triangles` an optimizer test: its datum is exact, so only Ê can be at fault.

    Every rounding lands the planted split at the brute-force optimum 1/15, so a run that misses it
    on this instance has missed it for reasons of its own.
    """
    instance = two_triangles_instance()
    optimum, _ = brute_force_rcut(instance.graph, instance.cluster_count)
    assert optimum == pytest.approx(1 / 15, abs=1e-12)
    roundings = datum_roundings(instance.graph, instance.cluster_count, seed=0)
    assert set(roundings) == set(ROUNDINGS)
    for name, labels in roundings.items():
        assert canonical(labels) == canonical(instance.planted_labels), name
        assert ratio_cut(instance.graph, labels) == pytest.approx(1 / 15, abs=1e-12), name


def test_roach_graph_roundings_split_the_optimum_from_the_documented_failure():
    """k-means and discretize find one antenna (4/15); cluster_qr finds top-vs-bottom (1.0)."""
    graph = roach_graph(5)
    cuts = {
        name: ratio_cut(graph, labels) for name, labels in datum_roundings(graph, 2, seed=0).items()
    }
    assert cuts["kmeans"] == pytest.approx(ROACH_OPTIMUM)
    assert cuts["discretize"] == pytest.approx(ROACH_OPTIMUM)
    assert cuts["cluster_qr"] == pytest.approx(ROACH_TOP_VERSUS_BOTTOM)
    for value in cuts.values():
        assert value >= ROACH_OPTIMUM - 1e-12
