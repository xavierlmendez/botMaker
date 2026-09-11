"""The probes' graph table, after slice 2.6 put a K = 3 instance in it.

Both probes were written when every graph they knew was a bisection, so they carried a literal
``2`` for the cluster count and a literal ``10`` for the roach's path boundary. `roach_g20` is
K = 3 with n = 80, so both constants had to become properties of the instance. These tests are on
that table and on the boundary arithmetic; the probes' own runs are examples and are not run here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from two_hot_span_adjacency_probe import (
    CLUSTER_COUNTS,
    GRAPHS,
    build_graph,
    cross_path_pair_count,
    is_roach,
    references_for,
)


def test_the_probes_know_all_three_graphs_and_their_cluster_counts():
    assert GRAPHS == ("roach_g5", "karate", "roach_g20")
    assert CLUSTER_COUNTS == {"roach_g5": 2, "karate": 2, "roach_g20": 3}


def test_the_eighty_node_cockroach_is_built_the_way_the_harness_builds_it():
    graph = build_graph("roach_g20")
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (80, 98)


def test_both_roaches_are_roaches_and_karate_is_not():
    assert is_roach("roach_g5")
    assert is_roach("roach_g20")
    assert not is_roach("karate")


def test_each_graph_has_its_own_line_of_reference_cut_values():
    assert "0.2667" in references_for("roach_g5")
    assert "0.1500" in references_for("roach_g20")
    assert "2.5972" in references_for("karate")


@pytest.mark.parametrize(("node_count", "split"), [(20, 10), (80, 40)])
def test_a_pair_is_cross_path_exactly_when_it_straddles_the_half_way_boundary(node_count, split):
    """The boundary is n/2 = 2k, so the same count reads the k = 5 roach and the k = 20 one."""
    within_top = frozenset({0, split - 1})
    within_bottom = frozenset({split, node_count - 1})
    across = frozenset({split - 1, split})
    assert cross_path_pair_count([within_top, within_bottom], split) == 0
    assert cross_path_pair_count([across], split) == 1
    assert cross_path_pair_count([within_top, across, within_bottom], split) == 1


def test_the_k_five_boundary_no_longer_reads_the_k_twenty_roach():
    """A guard on the constant that moved: at n = 80 the old literal 10 is inside the top path."""
    inside_top_path = frozenset({5, 30})
    assert cross_path_pair_count([inside_top_path], 40) == 0
    assert cross_path_pair_count([inside_top_path], 10) == 1


def test_a_graph_without_reference_cut_values_is_refused_rather_than_given_karates():
    with pytest.raises(ValueError, match="unknown graph"):
        references_for("two_moons_knn")
