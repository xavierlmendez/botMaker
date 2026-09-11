"""Two-hot spanning sets: the graph objects behind the rounded ratio cut.

The incidence matrix ``X`` of a weighted graph has one column per edge ``(i, j)``, ``i < j``, equal
to ``sqrt(w) (e_i - e_j)``, so that ``X Xᵀ = L`` is the unnormalized Laplacian. A *spanning set* is
an ``n-by-r`` matrix ``V`` with ``r = n - K`` columns, each of them zero-sum (``1ᵀ v = 0``); its
quality is the projector residual ``E(V) = ‖X - V V⁺ X‖²_F`` with ``V⁺`` the numpy pseudo-inverse.
The *collision measure* ``R(v) = ‖v‖₂² / ‖v‖₁²`` scores how close a single column is to being
2-hot, and *rounding* sends a column to the 2-hot vector ``e_argmax(v) - e_argmin(v)`` — except a
constant column, whose largest and smallest entries are equal: it has no pair to read off and
rounds to the zero column, which contributes no edge and does not change the span. The rounded
pairs are the edges of a "pair graph" whose connected components are the clustering, and the
*rounded cut* ``Ê = E(round_columns(V))`` is the headline number. The *spectral floor* ``Σλ`` is
the sum of the ``K`` smallest eigenvalues of ``L``.

Three facts are settled and are implemented here rather than re-derived:

1. For zero-sum ``v``, ``R(v) ≤ 1/2``, with equality exactly when ``v`` is 2-hot (two non-zeros of
   opposite sign and equal magnitude).
2. ``Ê`` equals the RatioCut ``Σ_B cut(B, B̄) / |B|`` of the component partition of the pair graph;
   when the rounded pairs form a within-block spanning forest with ``K`` components, that partition
   is the intended one.
3. The minimum of ``E(V)`` over zero-sum ``n-by-r`` matrices is ``Σλ``, attained by the eigenvectors
   of the ``r`` *largest* eigenvalues of ``L``. Hence ``E(V) ≥ Σλ`` and ``Ê ≥ Σλ`` always.

``VᵀV = I`` is never imposed here, anywhere, for any reason: the columns are only ever constrained
to be zero-sum, and the projector is the pseudo-inverse one, which is defined whatever the rank.
Everything in this module is the rcut form (``c = 1``, ``C = I``); the normalized cut is out of
scope and tracked as BL-41.

Reported numbers always come through the exact projector (`mllib.math.projector.ExactProjector`);
the ridge projector the training loop descends is a training knob and never reaches a report
(D-31). ``SpanCost`` here is the span residual as the cost an optimizer descends, with its
projector injected, so one cost class serves the report and the training loss.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from mllib.math.cost_function import AbstractCostFunction
from mllib.math.projector import AbstractProjector, ExactProjector

# Above this many vertices the set-partition enumeration stops being an oracle and starts being a
# search: the Bell number of 11 is already 678 570 labellings per cluster count.
BRUTE_FORCE_NODE_LIMIT = 10
# A vector whose largest entry is at or below this is the zero vector as far as the collision
# measure is concerned; R is undefined there and reported as 0.0.
ZERO_VECTOR_TOLERANCE = 1e-300


@dataclass(frozen=True, slots=True)
class GraphInstance:
    """A named graph with the cluster count it is meant to be cut into.

    Not named ``Test*``: pytest collects classes by that prefix. ``planted_labels`` is the ground
    truth when the generator has one, and ``None`` for graphs that only have a known best cut.
    """

    name: str
    graph: nx.Graph
    cluster_count: int
    planted_labels: np.ndarray | None = None


def node_count(graph: nx.Graph) -> int:
    """The number of vertices, which are relabelled 0..n-1 in sorted node order everywhere here."""
    return int(graph.number_of_nodes())


def spanning_vector_count(node_count: int, cluster_count: int) -> int:
    """r = n - K, the number of columns a spanning set has."""
    if cluster_count < 1:
        raise ValueError("cluster_count must be at least 1.")
    if cluster_count >= node_count:
        raise ValueError("cluster_count must be smaller than node_count.")
    return node_count - cluster_count


def _sorted_nodes(graph: nx.Graph) -> list:
    return sorted(graph.nodes())


def _sorted_edges(graph: nx.Graph, index_of: dict) -> list[tuple[int, int, float]]:
    """Edges as ``(i, j, w)`` with ``i < j`` in index space, ordered by ``(i, j)``."""
    edges = []
    for source, target, data in graph.edges(data=True):
        first, second = index_of[source], index_of[target]
        if first > second:
            first, second = second, first
        edges.append((first, second, float(data.get("weight", 1.0))))
    edges.sort(key=lambda edge: (edge[0], edge[1]))
    return edges


def incidence_matrix(graph: nx.Graph) -> np.ndarray:
    """X, of shape n-by-m: the column of edge (i, j) is sqrt(w) (e_i - e_j). Then X Xᵀ = L."""
    nodes = _sorted_nodes(graph)
    index_of = {node: index for index, node in enumerate(nodes)}
    edges = _sorted_edges(graph, index_of)
    matrix = np.zeros((len(nodes), len(edges)), dtype=float)
    for column, (first, second, weight) in enumerate(edges):
        root_weight = np.sqrt(weight)
        matrix[first, column] = root_weight
        matrix[second, column] = -root_weight
    return matrix


def laplacian_matrix(graph: nx.Graph) -> np.ndarray:
    """The unnormalized Laplacian D - A, built from the adjacency matrix. Equals X Xᵀ."""
    adjacency = nx.to_numpy_array(graph, nodelist=_sorted_nodes(graph), dtype=float)
    return np.diag(adjacency.sum(axis=1)) - adjacency


# The weighted adjacency is read back off the Laplacian, so it is only as symmetric and as
# non-negative as X Xᵀ came out in float64. Anything below this is arithmetic, not a wrong graph.
ADJACENCY_TOLERANCE = 1e-9


def graph_matrices(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(L, A) from the incidence matrix: L = X Xᵀ and A = diag(diag L) - L, checked on the way out.

    The adjacency is *recovered* rather than passed in, so a graph term is guaranteed to describe
    the same graph the residual is measured against. An A that comes back asymmetric or negative
    means the caller did not hand in an incidence matrix, which is a stop.
    """
    X = np.asarray(X, dtype=np.float64)
    laplacian = X @ X.T
    adjacency = np.diag(np.diag(laplacian)) - laplacian
    if not np.allclose(adjacency, adjacency.T, atol=ADJACENCY_TOLERANCE, rtol=0.0):
        raise ValueError("the recovered adjacency is not symmetric: X is not an incidence matrix")
    if float(adjacency.min()) < -ADJACENCY_TOLERANCE:
        raise ValueError("the recovered adjacency has a negative weight")
    return laplacian, adjacency


def _as_numpy(values: Any) -> np.ndarray:
    """A numpy view of an array in either arithmetic, off the gradient path."""
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


@dataclass(frozen=True, slots=True)
class SpanCost(AbstractCostFunction):
    """E(V) = ‖X - P_V X‖²_F: the span residual of the incidence matrix, the two-hot cost.

    The projector is injected, so the same cost reads exactly for a report
    (`ExactProjector`) and smoothly for training (`RidgeProjector`, D-31); ``X`` is held in the
    projector's arithmetic. Injected into the training loss, which adds the penalties to it.
    """

    projector: AbstractProjector
    X: Any = field(compare=False, repr=False)

    def compute_cost(self, parameters: Any) -> Any:
        """E(V) at the spanning set ``parameters``, through the injected projector."""
        return self.projector.residual(self.X, parameters)

    # Value semantics on X too: the generated tuple comparison cannot compare arrays.
    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.projector == other.projector and np.array_equal(
            _as_numpy(self.X), _as_numpy(other.X)
        )

    def __hash__(self) -> int:
        return hash((type(self), self.projector, tuple(np.shape(self.X))))


def collision_measure(vector: np.ndarray) -> float:
    """R(v) = ‖v‖₂² / ‖v‖₁². Scale-invariant; 0.0 for the zero vector, where R is undefined."""
    vector = np.asarray(vector, dtype=float).ravel()
    peak = float(np.max(np.abs(vector))) if vector.size else 0.0
    if peak <= ZERO_VECTOR_TOLERANCE:
        return 0.0
    # R is scale-invariant, so divide by the largest entry first: without it a vector of subnormal
    # entries squares to zero in both norms and the ratio comes back NaN.
    scaled = vector / peak
    one_norm = float(np.sum(np.abs(scaled)))
    return float(np.dot(scaled, scaled) / (one_norm * one_norm))


def collision_measures(spanning_set: np.ndarray) -> np.ndarray:
    """R(v_j) for every column: the per-column diagnostic, never the criterion."""
    spanning_set = np.asarray(spanning_set, dtype=float)
    return np.array(
        [collision_measure(spanning_set[:, column]) for column in range(spanning_set.shape[1])]
    )


def _column_pair(column: np.ndarray) -> tuple[int, int] | None:
    """(argmax, argmin) of one column, or None when the column is degenerate.

    A constant column — the zero column included — has its largest and smallest entries equal, and
    there is no pair to read off it. Writing +1 and -1 into the same cell would leave a 1-hot,
    non-zero-sum column and break the identity, so such a column rounds to zero instead.
    """
    high, low = int(np.argmax(column)), int(np.argmin(column))
    if column[high] == column[low]:
        return None
    return high, low


def round_columns(spanning_set: np.ndarray) -> np.ndarray:
    """Each column v → the 2-hot vector e_argmax(v) - e_argmin(v); a constant column → zero.

    A zero column does not change the span, so dropping the degenerate ones leaves
    ``Ê = RatioCut(components of the pair graph)`` intact.
    """
    spanning_set = np.asarray(spanning_set, dtype=float)
    rounded = np.zeros_like(spanning_set)
    for column in range(spanning_set.shape[1]):
        pair = _column_pair(spanning_set[:, column])
        if pair is None:
            continue
        high, low = pair
        rounded[high, column] = 1.0
        rounded[low, column] = -1.0
    return rounded


def rounded_pairs(spanning_set: np.ndarray) -> list[tuple[int, int]]:
    """(argmax, argmin) per column: the edges of the pair graph.

    Columns that round to zero contribute no pair; they are skipped, so the list can be shorter
    than the number of columns.
    """
    spanning_set = np.asarray(spanning_set, dtype=float)
    pairs = (_column_pair(spanning_set[:, column]) for column in range(spanning_set.shape[1]))
    return [pair for pair in pairs if pair is not None]


def clustering_from_pairs(node_count: int, pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    """Connected components of the pair graph, labelled 0.. in order of their smallest member."""
    pair_graph = nx.Graph()
    pair_graph.add_nodes_from(range(node_count))
    pair_graph.add_edges_from((int(high), int(low)) for high, low in pairs)
    labels = np.empty(node_count, dtype=int)
    components = sorted(nx.connected_components(pair_graph), key=min)
    for label, component in enumerate(components):
        for node in component:
            labels[node] = label
    return labels


def rounded_cut(X: np.ndarray, spanning_set: np.ndarray) -> float:
    """Ê, the headline number: the projector residual of the rounded spanning set."""
    return ExactProjector().residual(X, round_columns(spanning_set))


def ratio_cut(graph: nx.Graph, labels: np.ndarray) -> float:
    """Σ_B cut(B, B̄) / |B| over the blocks of ``labels``, with edge weights."""
    labels = np.asarray(labels).ravel()
    nodes = _sorted_nodes(graph)
    if labels.shape[0] != len(nodes):
        raise ValueError("labels must have one entry per node.")
    index_of = {node: index for index, node in enumerate(nodes)}
    cut_weight: dict[int, float] = {}
    for first, second, weight in _sorted_edges(graph, index_of):
        if labels[first] != labels[second]:
            cut_weight[labels[first]] = cut_weight.get(labels[first], 0.0) + weight
            cut_weight[labels[second]] = cut_weight.get(labels[second], 0.0) + weight
    total = 0.0
    for block in np.unique(labels):
        total += cut_weight.get(block, 0.0) / float(np.count_nonzero(labels == block))
    return float(total)


def spectral_floor(graph: nx.Graph, cluster_count: int) -> float:
    """Σλ, the sum of the K smallest eigenvalues of L: the minimum of E over zero-sum V."""
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix(graph))
    return float(np.sum(eigenvalues[:cluster_count]))


def spectral_spanning_set(graph: nx.Graph, cluster_count: int) -> np.ndarray:
    """The eigenvectors of the r largest eigenvalues of L: zero-sum, and E(V) = Σλ exactly.

    Connectivity is required, not decoration: a graph with c components has a c-dimensional null
    space, so at K < c one of the r largest eigenvectors is still a null-space vector and need not
    be orthogonal to 1 — the columns would silently stop being zero-sum.
    """
    spanning_vector_count(node_count(graph), cluster_count)
    if not nx.is_connected(graph):
        raise ValueError("graph must be connected")
    _, eigenvectors = np.linalg.eigh(laplacian_matrix(graph))
    return np.array(eigenvectors[:, cluster_count:], dtype=float)


def _restricted_growth_strings(length: int, block_count: int) -> Iterator[tuple[int, ...]]:
    """Every set partition of 0..length-1 into exactly ``block_count`` non-empty blocks, once.

    A restricted growth string has ``labels[0] = 0`` and ``labels[i] ≤ 1 + max(labels[:i])``, which
    names each partition by exactly one labelling — the canonical one. Each labelling is yielded as
    its own tuple: the working list is overwritten in place, so handing it out would hand out a
    value that changes underfoot.
    """
    labels = [0] * length
    if length < block_count:
        return

    def extend(position: int, used: int) -> Iterator[tuple[int, ...]]:
        if position == length:
            if used == block_count:
                yield tuple(labels)
            return
        # Prune: even opening a new block at every remaining position must reach block_count.
        if used + (length - position) < block_count:
            return
        for label in range(min(used + 1, block_count)):
            labels[position] = label
            yield from extend(position + 1, max(used, label + 1))

    yield from extend(0, 0)


def brute_force_rcut(graph: nx.Graph, cluster_count: int) -> tuple[float, np.ndarray]:
    """The exact RatioCut optimum over all partitions into exactly K non-empty blocks."""
    count = node_count(graph)
    spanning_vector_count(count, cluster_count)
    if count > BRUTE_FORCE_NODE_LIMIT:
        raise ValueError(
            f"brute_force_rcut is guarded to {BRUTE_FORCE_NODE_LIMIT} nodes; got {count}."
        )
    best_value = np.inf
    best_labels = np.zeros(count, dtype=int)
    for labels in _restricted_growth_strings(count, cluster_count):
        value = ratio_cut(graph, np.asarray(labels, dtype=int))
        if value < best_value:
            best_value = value
            best_labels = np.asarray(labels, dtype=int)
    return float(best_value), best_labels


def roach_graph(rung_count: int = 5) -> nx.Graph:
    """The Guattery-Miller roach: two paths on 2k vertices joined by k rungs at their far ends.

    Top path is 0..2k-1, bottom path is 2k..4k-1, and rung i joins top vertex k+i to bottom vertex
    3k+i. So n = 4k and m = 2(2k-1) + k = 5k-2. Three K = 2 cuts are worth knowing apart: severing
    one antenna (k vertices, one edge) gives 1/k + 1/(3k) = 4/(3k), and one antenna alone is the
    K = 2 RatioCut optimum (brute force at k = 2; exhaustive 2^19 bipartition scan at k = 5);
    severing both antennae from the ladder (2k vs 2k, two edges) gives 2/(2k) + 2/(2k) = 2/k and is
    the best *balanced* cut, the one spectral bisection is famously supposed to find and does not;
    splitting top path from bottom path cuts all k rungs for 1.0.
    """
    if rung_count < 1:
        raise ValueError("rung_count must be at least 1.")
    graph = nx.Graph()
    graph.add_nodes_from(range(4 * rung_count))
    for offset in (0, 2 * rung_count):
        for index in range(2 * rung_count - 1):
            graph.add_edge(offset + index, offset + index + 1)
    for index in range(rung_count):
        graph.add_edge(rung_count + index, 3 * rung_count + index)
    return graph


def roach_g20_instance() -> GraphInstance:
    """The eighty-node cockroach of He, Gu & Zhang 2012 (arXiv:1201.5767), Fig. 1-2, at K = 3.

    Their figure draws "80 nodes ... each suspension points representing a line of 16 nodes": two
    paths of 40 joined by rungs along 20 columns, with antennae of 20 hanging off the other end.
    That is ``roach_graph(20)`` in this repository's convention — n = 4k = 80, m = 5k - 2 = 98 — up
    to the mirror image, which changes no invariant measured here.

    Their point is why K = 3 rather than K = 2. The Fiedler vector cuts horizontally through all
    twenty rungs (their Fig. 2a, solid line), while "the ideal cut separates the graph into three
    parts" (the dashed line): remove the ladder — both paths' rung columns — and the two antennae
    are left disconnected from each other, so the ideal separation is three blocks, not two, and it
    is the *third* eigenvector that finds it (Fig. 2b). Hence the planted labels here: 0 for the
    ladder (top ``k..2k-1`` and bottom ``3k..4k-1``), 1 for the top antenna, 2 for the bottom one.

    The numbers to read a run against. The planted RatioCut is 2/40 + 1/20 + 1/20 = 0.15 — the
    ladder loses one edge to each antenna, and each antenna one to the ladder. Merging the two
    antennae into a single block, which no connected partition would do but which every K = 2
    method must, gives 2/40 + 2/40 = 0.10: *lower*, because RatioCut pays per block and the two
    antennae are cheaper counted once. So on this instance the K = 3 answer the paper wants is not
    the K = 2 RatioCut it is competing with, and the two cut values are reported side by side
    rather than either being called the optimum.
    """
    rung_count = 20
    labels = np.zeros(4 * rung_count, dtype=int)
    labels[0:rung_count] = 1
    labels[2 * rung_count : 3 * rung_count] = 2
    return GraphInstance("roach_g20", roach_graph(rung_count), 3, labels)


def two_triangles_graph() -> nx.Graph:
    """Two triangles joined by a weak bridge: 0-1-2 and 3-4-5 at weight 1, edge (2, 3) at 0.1.

    Weighted on purpose — the bridge being the cheap edge is what makes the cut obvious — so the
    edges carry ``weight`` attributes the way ``karate_graph`` does and ``incidence_matrix`` reads
    them straight off.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    for first, second in ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)):
        graph.add_edge(first, second, weight=1.0)
    graph.add_edge(2, 3, weight=0.1)
    return graph


def two_triangles_instance() -> GraphInstance:
    """The smallest weighted instance whose optimum is enumerated and whose datum is exact.

    n = 6, K = 2, planted labels [0, 0, 0, 1, 1, 1]. Every quantity a run is read against here is a
    *fact* rather than a reference: n = 6 is inside ``BRUTE_FORCE_NODE_LIMIT``, so the RatioCut
    optimum is enumerated and comes back as 1/15 = 0.066667 at exactly the planted labels; all
    three datum roundings (``kmeans``, ``discretize``, ``cluster_qr``, seed 0) also reach 1/15; and
    the spectral floor is Σλ = 0.063771. So when a run misses 1/15 on this graph, nothing about the
    objective, the rounding rule or the datum is in question — the optimizer is. Isolating the
    optimizer's failure from the objective's is the instance's whole job, and no other instance in
    ``default_test_graphs`` can do it: ``roach_g5`` and up are past the brute-force guard or have a
    datum that itself misses the optimum.

    It also fixes the scale of λ, which the larger instances hide. The collision reward is bounded
    by ``λ · r / 2`` (R(v) ≤ 1/2 per column, r = n - K = 4 columns), and the span term it competes
    with is bounded below by Σλ = 0.064. At the prototype's λ = 10 the reward's ceiling is 20 —
    three hundred times the floor — and the span term stops mattering at all: columns go 2-hot on
    whatever pairs are nearest and ignore the graph, E\\* jumps to 1.5-2.5, and Ê reaches 1/15 from
    none of three random seeds at 300 or 3000 steps (nor from the spectral initialisation, nor from
    any of ten restarts of the joint-factorization engine it was run beside). At λ = 0.1 the ceiling
    is 0.2, the same order as Σλ, and Ê = 1/15 is reached from the spectral initialisation and from
    two of ten random seeds. λ is
    therefore not scale-free, and a λ quoted without Σλ and r beside it says nothing.
    """
    return GraphInstance(
        "two_triangles", two_triangles_graph(), 2, np.array([0, 0, 0, 1, 1, 1], dtype=int)
    )


def karate_graph() -> nx.Graph:
    """Zachary's karate club: 34 vertices already labelled 0..33, two known factions.

    networkx ships it *weighted* (edge weights 1..7 from Zachary's original data), so every RatioCut
    reported on it is the weighted one.
    """
    return nx.karate_club_graph()


def planted_partition_graph(
    group_count: int, group_size: int, p_in: float, p_out: float, seed: int
) -> GraphInstance:
    """A seeded planted-partition (stochastic block) graph whose blocks are the planted labels."""
    graph = nx.planted_partition_graph(group_count, group_size, p_in, p_out, seed=seed)
    if not nx.is_connected(graph):
        raise ValueError("planted_partition_graph produced a disconnected graph; change the seed.")
    labels = np.repeat(np.arange(group_count), group_size)
    return GraphInstance("planted_partition", graph, group_count, labels)


def two_moons_knn_graph(
    sample_count: int, neighbor_count: int, noise: float, seed: int
) -> GraphInstance:
    """A symmetrised k-nearest-neighbour connectivity graph over the two-moons point cloud.

    ``make_moons`` and ``kneighbors_graph`` are the only sklearn calls in this module family, and
    they build data and a neighbourhood structure — no sklearn model class is imported anywhere.
    """
    from sklearn.datasets import make_moons
    from sklearn.neighbors import kneighbors_graph

    points, moon_ids = make_moons(n_samples=sample_count, noise=noise, random_state=seed)
    neighbors = kneighbors_graph(points, neighbor_count, mode="connectivity").toarray()
    adjacency = np.maximum(neighbors, neighbors.T)
    np.fill_diagonal(adjacency, 0.0)
    graph = nx.from_numpy_array((adjacency > 0).astype(float), edge_attr=None)
    if not nx.is_connected(graph):
        raise ValueError("two_moons_knn_graph produced a disconnected graph; change the seed.")
    return GraphInstance("two_moons_knn", graph, 2, np.asarray(moon_ids, dtype=int))


def default_test_graphs() -> tuple[GraphInstance, ...]:
    """The six instances the prototype reports on, roach first.

    ``roach_g20`` and then ``two_triangles`` are appended last on purpose: every index-based
    reference to this tuple written before one of them existed still names the instance it named.
    """
    karate = karate_graph()
    karate_labels = np.array(
        [0 if karate.nodes[node]["club"] == "Mr. Hi" else 1 for node in sorted(karate.nodes())],
        dtype=int,
    )
    instances = (
        GraphInstance("roach_g5", roach_graph(5), 2),
        GraphInstance("karate", karate, 2, karate_labels),
        planted_partition_graph(3, 8, 0.7, 0.05, seed=0),
        two_moons_knn_graph(40, 5, 0.08, seed=0),
        roach_g20_instance(),
        two_triangles_instance(),
    )
    for instance in instances:
        if not nx.is_connected(instance.graph):
            raise ValueError(f"{instance.name} is disconnected.")
    return instances


@dataclass(frozen=True, slots=True)
class TwoHotSpanReport:
    """The numbers a run delivers, every one through the exact arithmetic of this module (D-31)."""

    relaxed_objective: float
    rounded_cut: float
    collision_measures: np.ndarray
    labels: np.ndarray


class TwoHotSpanProblem:
    """The instance an optimizer moves a spanning set over: a graph, K, and the reporting arithmetic.

    Holds the incidence matrix, the graph's own Laplacian and adjacency for the graph penalties,
    and the vector every column must be orthogonal to — the ones vector for the ratio cut, which
    is the seam the ncut generalisation replaces with sqrt(d) (BL-41). ``report`` turns a spanning
    set into the delivered numbers. Injected into the optimizer (D-35 (3)); numpy only.
    """

    __slots__ = (
        "X",
        "adjacency",
        "cluster_count",
        "graph",
        "laplacian",
        "name",
        "node_count",
        "spanning_vector_count",
    )

    def __init__(self, graph: nx.Graph, cluster_count: int, *, name: str = "graph"):
        self.graph = graph
        self.cluster_count = int(cluster_count)
        self.name = str(name)
        self.node_count = node_count(graph)
        self.spanning_vector_count = spanning_vector_count(self.node_count, self.cluster_count)
        self.X = incidence_matrix(graph)
        self.laplacian, self.adjacency = graph_matrices(self.X)

    @classmethod
    def from_instance(cls, instance: GraphInstance) -> TwoHotSpanProblem:
        return cls(instance.graph, instance.cluster_count, name=instance.name)

    @property
    def constraint_vector(self) -> np.ndarray:
        """The vector the columns stay orthogonal to: ones for the ratio cut."""
        return np.ones(self.node_count, dtype=np.float64)

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "graph": self.name,
            "node_count": self.node_count,
            "cluster_count": self.cluster_count,
        }

    def spectral_floor(self) -> float:
        return spectral_floor(self.graph, self.cluster_count)

    def spectral_spanning_set(self) -> np.ndarray:
        return spectral_spanning_set(self.graph, self.cluster_count)

    def report(self, spanning_set: np.ndarray) -> TwoHotSpanReport:
        """E\\*, Ê, R(v_j) and the clustering of ``spanning_set``, through the exact projector.

        A spanning set that is not finite has no report: every number is NaN and every label -1,
        so a run stopped for a non-finite value still returns (D-35 (9)) instead of failing inside
        the pseudo-inverse.
        """
        spanning_set = np.asarray(spanning_set, dtype=float)
        if not np.isfinite(spanning_set).all():
            return TwoHotSpanReport(
                relaxed_objective=math.nan,
                rounded_cut=math.nan,
                collision_measures=np.full(spanning_set.shape[1], np.nan),
                labels=np.full(self.node_count, -1, dtype=int),
            )
        pairs = rounded_pairs(spanning_set)
        return TwoHotSpanReport(
            relaxed_objective=ExactProjector().residual(self.X, spanning_set),
            rounded_cut=rounded_cut(self.X, spanning_set),
            collision_measures=collision_measures(spanning_set),
            labels=clustering_from_pairs(self.node_count, pairs),
        )
