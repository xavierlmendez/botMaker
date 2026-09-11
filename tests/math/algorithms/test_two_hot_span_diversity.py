"""Slice 2.5 — the column-diversity term D(V) = Σ_i ℓ_i².

An experiment, off by default, so the first thing this file pins is that a default run is untouched
bit for bit. What is here after that is the arithmetic of the term itself, checked against values a
reader can compute with a pencil: on exact 2-hot columns D(V) is ¼ Σ_i deg_i² of the pair graph, and
on a spanning set whose columns spread their mass evenly over all n vertices it is exactly the r²/n
floor. Then the floor as a property on random zero-sum matrices, a run with ν > 0 that still holds
every invariant, and the validation.

Deterministic: every graph is built explicitly, every run and every draw carries its own seed, and
the hypothesis property is `derandomize=True`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

pytest.importorskip("torch")

import torch

from mllib.math.algorithms.two_hot_span_optimizer import (
    TwoHotSpanConfig,
    diversity_term,
    fit_two_hot_span,
)
from mllib.math.graph.two_hot_span_problem import (
    brute_force_rcut,
    collision_measures,
    incidence_matrix,
    roach_graph,
    spectral_floor,
)


def two_hot_columns(node_total: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """One exact unit 2-hot column per pair, +1/√2 on the first vertex and -1/√2 on the second."""
    matrix = np.zeros((node_total, len(pairs)), dtype=np.float64)
    for column, (first, second) in enumerate(pairs):
        matrix[first, column] = 1.0 / math.sqrt(2.0)
        matrix[second, column] = -1.0 / math.sqrt(2.0)
    return matrix


def pair_degree_sum_of_squares(node_total: int, pairs: list[tuple[int, int]]) -> float:
    """Σ_i deg_i² in the pair *multigraph*: a pair used twice counts twice at both endpoints."""
    degrees = np.zeros(node_total, dtype=np.float64)
    for first, second in pairs:
        degrees[first] += 1.0
        degrees[second] += 1.0
    return float(np.sum(degrees * degrees))


def seeded_graph(node_total: int, seed: int) -> nx.Graph:
    """A connected seeded random graph with unit weights."""
    graph = nx.gnp_random_graph(node_total, 0.55, seed=seed)
    if not nx.is_connected(graph):
        graph = nx.connected_watts_strogatz_graph(node_total, 4, 0.3, seed=seed)
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def term_of(matrix: np.ndarray) -> float:
    return float(diversity_term(torch.tensor(matrix, dtype=torch.float64)))


# ------------------------------------------------------------------------------------------------
# Off by default, bit for bit.
# ------------------------------------------------------------------------------------------------


def test_the_default_run_is_bit_identical_to_one_that_names_a_zero_diversity_weight():
    """Adding a field with a neutral default is a claim; `np.array_equal` is the proof of it."""
    X = incidence_matrix(roach_graph(5))
    base = TwoHotSpanConfig(step_count=30, seed=0, collision_weight=10.0)

    silent = fit_two_hot_span(X, 2, base)
    spelled_out = fit_two_hot_span(X, 2, replace(base, diversity_weight=0.0))

    assert np.array_equal(silent.spanning_set, spelled_out.spanning_set)
    assert np.array_equal(silent.loss_history, spelled_out.loss_history)


def test_the_diversity_term_changes_the_run_it_is_switched_on_for():
    """A knob that made no difference would be a knob nobody could test the sign of."""
    X = incidence_matrix(seeded_graph(8, seed=5))
    base = TwoHotSpanConfig(step_count=50, seed=7, collision_weight=1.0)

    off = fit_two_hot_span(X, 2, base)
    on = fit_two_hot_span(X, 2, replace(base, diversity_weight=1.0))

    assert not np.array_equal(off.spanning_set, on.spanning_set)


# ------------------------------------------------------------------------------------------------
# The hand values: ¼ Σ_i deg_i² on exact 2-hot columns.
# ------------------------------------------------------------------------------------------------


def test_disjoint_two_hot_pairs_give_the_smallest_load_a_pair_cover_can():
    """Three disjoint pairs on six vertices: every deg_i = 1, so D = ¼ · 6 = 1.5."""
    pairs = [(0, 1), (2, 3), (4, 5)]
    matrix = two_hot_columns(6, pairs)

    assert term_of(matrix) == pytest.approx(0.25 * pair_degree_sum_of_squares(6, pairs), abs=1e-12)
    assert term_of(matrix) == pytest.approx(1.5, abs=1e-12)


def test_a_path_of_two_hot_pairs_pays_for_its_interior_vertices():
    """The path 0-1-2-3: degrees (1, 2, 2, 1), so D = ¼(1 + 4 + 4 + 1) = 2.5."""
    pairs = [(0, 1), (1, 2), (2, 3)]
    matrix = two_hot_columns(4, pairs)

    assert term_of(matrix) == pytest.approx(0.25 * pair_degree_sum_of_squares(4, pairs), abs=1e-12)
    assert term_of(matrix) == pytest.approx(2.5, abs=1e-12)


def test_every_column_on_one_pair_is_the_worst_case_the_term_exists_to_punish():
    """Three columns all on (0, 1): degrees (3, 3), so D = ¼(9 + 9) = 4.5 — the pile-up.

    Three columns, three distinct pairs and three columns on one pair all put the same r = 3 units
    of mass on the graph. D separates them (1.5 < 2.5 < 4.5) because it reads *where* the mass went,
    which is exactly what no per-column score can see.
    """
    pairs = [(0, 1), (0, 1), (0, 1)]
    matrix = two_hot_columns(6, pairs)

    assert term_of(matrix) == pytest.approx(0.25 * pair_degree_sum_of_squares(6, pairs), abs=1e-12)
    assert term_of(matrix) == pytest.approx(4.5, abs=1e-12)


@pytest.mark.parametrize(("node_total", "column_count"), [(6, 4), (8, 6), (10, 3)])
def test_columns_that_spread_their_mass_evenly_sit_exactly_on_the_floor(node_total, column_count):
    """Alternating ±1 on all n coordinates: p_ij = 1/n, ℓ_i = r/n, D = r²/n and no lower."""
    alternating = np.array([1.0 if index % 2 == 0 else -1.0 for index in range(node_total)])
    matrix = np.tile(alternating[:, None], (1, column_count))
    assert matrix.sum() == pytest.approx(0.0, abs=1e-12)

    assert term_of(matrix) == pytest.approx(column_count**2 / node_total, abs=1e-12)


def test_the_term_is_scale_invariant_column_by_column():
    """p_j divides by ‖v_j‖₁, so rescaling one column alone cannot move D."""
    matrix = two_hot_columns(6, [(0, 1), (1, 2), (3, 4)])
    rescaled = matrix.copy()
    rescaled[:, 1] *= 17.0

    assert term_of(rescaled) == pytest.approx(term_of(matrix), rel=1e-12)


def test_the_term_is_the_pairwise_column_overlap_plus_the_collision_sum():
    """Σ_{j<k} p_jᵀ p_k = ½(D(V) - Σ_j R(v_j)) — the identity that says D couples the columns."""
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((9, 5))
    matrix = matrix - matrix.mean(axis=0)
    distribution = np.abs(matrix) / np.abs(matrix).sum(axis=0)

    overlap = sum(
        float(distribution[:, first] @ distribution[:, second])
        for first in range(5)
        for second in range(first + 1, 5)
    )

    collisions = float(np.sum(collision_measures(matrix)))
    assert overlap == pytest.approx(0.5 * (term_of(matrix) - collisions), abs=1e-12)


# ------------------------------------------------------------------------------------------------
# The floor, as a property.
# ------------------------------------------------------------------------------------------------


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    matrix=hnp.arrays(
        np.float64,
        st.tuples(st.integers(3, 9), st.integers(1, 6)),
        elements=st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    )
)
def test_the_diversity_term_is_never_below_the_uniform_floor(matrix):
    """D(V) ≥ r²/n for every zero-sum V, by Cauchy-Schwarz on Σ_i ℓ_i = r."""
    matrix = matrix - matrix.mean(axis=0)
    # A column that came out all-zero has no distribution at all; nudge it off the origin.
    zero_columns = np.abs(matrix).sum(axis=0) == 0.0
    matrix[0, zero_columns] = 1.0
    matrix[1, zero_columns] = -1.0

    node_total, column_count = matrix.shape
    assert term_of(matrix) >= column_count**2 / node_total - 1e-12


# ------------------------------------------------------------------------------------------------
# A run with the term switched on.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("diversity_weight", [0.3, 3.0])
def test_a_run_with_the_diversity_term_keeps_every_invariant(diversity_weight):
    graph = seeded_graph(8, seed=5)
    X = incidence_matrix(graph)
    optimum, _ = brute_force_rcut(graph, 2)
    floor = spectral_floor(graph, 2)
    config = TwoHotSpanConfig(
        step_count=200, seed=7, collision_weight=1.0, diversity_weight=diversity_weight
    )

    run = fit_two_hot_span(X, 2, config)

    assert run.max_zero_sum_violation <= 1e-12
    assert np.isfinite(run.loss_history).all()
    assert np.isfinite(run.spanning_set).all()
    assert run.rounded_cut >= optimum - 1e-10
    assert run.relaxed_objective >= floor - 1e-10


def test_a_heavier_diversity_weight_spreads_the_load_further():
    """The term's whole claim: raise ν and the heaviest vertex load comes down."""
    X = incidence_matrix(roach_graph(5))
    base = TwoHotSpanConfig(step_count=200, seed=0, collision_weight=3.0)

    def max_load(diversity_weight: float) -> float:
        run = fit_two_hot_span(X, 2, replace(base, diversity_weight=diversity_weight))
        absolute = np.abs(run.spanning_set)
        return float(np.max((absolute / absolute.sum(axis=0)).sum(axis=1)))

    assert max_load(10.0) < max_load(0.0)


# ------------------------------------------------------------------------------------------------
# Validation.
# ------------------------------------------------------------------------------------------------


def test_a_negative_diversity_weight_is_refused():
    with pytest.raises(ValueError, match="diversity_weight must not be negative"):
        TwoHotSpanConfig(diversity_weight=-0.5)


def test_the_harness_records_the_diversity_weight_it_ran_under():
    from mllib.math.graph.two_hot_span_problem import GraphInstance
    from mllib.ml.projects.two_hot_span_harness import run_graph

    report = run_graph(
        GraphInstance("tiny", seeded_graph(8, seed=1), 2),
        collision_weights=(1.0,),
        inits=("spectral",),
        step_count=5,
        diversity_weight=0.75,
    )

    assert report.config["diversity_weight"] == 0.75
