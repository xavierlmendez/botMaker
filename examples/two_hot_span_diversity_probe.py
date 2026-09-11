"""Probe: does coupling the columns — a vertex-load diversity term — make the rounding pick pairs?

Slice 2.5 of `docs/plans/2026-09-two-hot-span.md`, and an **experiment**, not a decision. The
default objective stays the brief's §20 objective; everything this script varies beyond λ and the
initialisation is off unless asked for.

The question, and why it is a new one. Slice 2.4's probe put two knobs on the run and both failed
for the same reason: `R(v_j)` scores a column on its own and so does the graph term, so the cheapest
way to collect either reward is for many columns to converge on the same few good pairs. The pair
graph then has almost no edges, the components multiply, and Ê — the RatioCut of that component
partition — explodes. Nothing coupled the columns.

D(V) = Σ_i ℓ_i² does. Each column carries the brief's §9 distribution p_j = |v_j| / ‖v_j‖₁; the
vertex load ℓ_i = Σ_j p_ij is how much of the r units of mass the whole spanning set puts on vertex
i, and Σ_i ℓ_i = r whatever V is. So D ≥ r²/n, with equality iff the load is flat, and on exact
2-hot columns D = ¼ Σ_i deg_i² in the pair graph. Added to the loss with weight ν it is a direct
penalty on the pile-up, and by the identity Σ_{j<k} p_jᵀ p_k = ½(D - Σ_j R(v_j)) it is precisely the
pairwise column overlap plus the collision sum.

ν is a *training* knob in the sense D-31 fixes for `epsilon`: it changes what Adam does and never
enters a reported number. Every E\\* and Ê below comes back through the exact numpy `pinv` projector
of `two_hot_span_problem`.

    uv run --group torch python examples/two_hot_span_diversity_probe.py
    uv run --group torch python examples/two_hot_span_diversity_probe.py --graphs roach_g5
    uv run --group torch python examples/two_hot_span_diversity_probe.py --graphs roach_g20

This is a composition root: it builds the graphs, assembles the configurations, runs
`fit_two_hot_span`, and prints two tables per graph. It computes no mathematics of its own beyond
counting pairs and reading the load off the final V — the graph helpers are imported from slice
2.4's probe rather than copied, so the two tables count a pair the same way.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

# The 2.4 probe is a sibling script, not a package module: put its directory on the path so its
# helpers can be imported rather than copied. Doing it here keeps `edge_pair_count` and
# `cross_path_pair_count` defined once, which is the point.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_hot_span_adjacency_probe import (
    CLUSTER_COUNTS,
    GRAPHS,
    build_graph,
    cross_path_pair_count,
    distinct_pairs,
    edge_pair_count,
    is_roach,
    references_for,
)

DEFAULT_STEP_COUNT = 300
DEFAULT_SEED = 0
DEFAULT_LEARNING_RATE = 0.05

COLLISION_WEIGHTS = (3.0, 10.0)
INITS = ("spectral", "random")

# (form, mu). "none" is mu = 0, where the graph term is guarded off and the form is never consulted.
# Only `edge_product` survives from 2.4: the Laplacian form carries a degree bias that 2.4 showed
# has nothing to do with the question, and this probe is about ν, not about re-running that grid.
ADJACENCIES: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("edge_product", 0.3),
    ("edge_product", 1.0),
)

DIVERSITY_WEIGHTS = (0.0, 0.3, 1.0, 3.0, 10.0)

# The step counts the winner is re-run at, to see whether 300 steps was converged or merely stopped.
CONVERGENCE_STEP_COUNTS = (1000, 3000)


def max_vertex_load(spanning_set: np.ndarray) -> float:
    """ℓ_max of the final V: the single vertex the spanning set leans on hardest.

    The flat load is r/n and the term's whole claim is that raising ν brings this number down, so it
    is the column that says whether ν did what it says on the tin — separately from whether that
    helped Ê, which is the other half of the table.
    """
    absolute = np.abs(spanning_set)
    return float(np.max((absolute / absolute.sum(axis=0)).sum(axis=1)))


def probe_row(
    name: str,
    graph: nx.Graph,
    X: np.ndarray,
    spectral: np.ndarray,
    *,
    cluster_count: int,
    collision_weight: float,
    init: str,
    form: str,
    adjacency_weight: float,
    diversity_weight: float,
    step_count: int,
    seed: int,
) -> dict[str, object]:
    """One cell of the grid: run it and read off everything the tables print."""
    from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanConfig, fit_two_hot_span

    config = TwoHotSpanConfig(
        step_count=step_count,
        learning_rate=DEFAULT_LEARNING_RATE,
        collision_weight=collision_weight,
        seed=seed,
        adjacency_weight=adjacency_weight,
        # `none` still has to name a legal form; at mu = 0 it is never read.
        adjacency_form="laplacian" if form == "none" else form,
        diversity_weight=diversity_weight,
    )
    run = fit_two_hot_span(X, cluster_count, config, spectral if init == "spectral" else None)
    pairs = distinct_pairs(run.spanning_set)
    return {
        "collision_weight": collision_weight,
        "init": init,
        "form": form,
        "adjacency_weight": adjacency_weight,
        "diversity_weight": diversity_weight,
        "step_count": step_count,
        "relaxed_objective": float(run.relaxed_objective),
        "rounded_cut": float(run.rounded_cut),
        "component_count": int(np.unique(run.labels).size),
        "distinct_pairs": len(pairs),
        "edge_pairs": edge_pair_count(graph, pairs),
        "cross_path_pairs": (
            cross_path_pair_count(pairs, graph.number_of_nodes() // 2) if is_roach(name) else None
        ),
        "max_vertex_load": max_vertex_load(run.spanning_set),
        "mean_collision": float(np.mean(run.collision_measures)),
        "labels": run.labels,
    }


def probe_rows(name: str, graph: nx.Graph, step_count: int, seed: int) -> list[dict[str, object]]:
    """Every cell of the λ x init x adjacency x ν grid, as one row each."""
    from mllib.math.graph.two_hot_span_problem import incidence_matrix, spectral_spanning_set

    cluster_count = CLUSTER_COUNTS[name]
    X = incidence_matrix(graph)
    spectral = spectral_spanning_set(graph, cluster_count)
    grid = itertools.product(COLLISION_WEIGHTS, INITS, ADJACENCIES, DIVERSITY_WEIGHTS)
    return [
        probe_row(
            name,
            graph,
            X,
            spectral,
            cluster_count=cluster_count,
            collision_weight=collision_weight,
            init=init,
            form=form,
            adjacency_weight=adjacency_weight,
            diversity_weight=diversity_weight,
            step_count=step_count,
            seed=seed,
        )
        for collision_weight, init, (form, adjacency_weight), diversity_weight in grid
    ]


def _header(roach: bool) -> str:
    return (
        f"  {'λ':>5} {'init':<9} {'form':<13} {'μ':>5} {'ν':>6} "
        f"{'E*':>11} {'Ê':>11} {'comp':>5} {'pairs':>6} {'edges':>6} "
        + (f"{'cross':>6} " if roach else "")
        + f"{'ℓ_max':>7} {'mean R':>7}"
    )


def _row_line(row: dict[str, object], roach: bool, marker: str) -> str:
    return (
        f" {marker}{float(row['collision_weight']):>5.1f} {row['init']:<9} "  # type: ignore[arg-type]
        f"{row['form']:<13} {float(row['adjacency_weight']):>5.1f} "  # type: ignore[arg-type]
        f"{float(row['diversity_weight']):>6.1f} "  # type: ignore[arg-type]
        f"{float(row['relaxed_objective']):>11.4f} "  # type: ignore[arg-type]
        f"{float(row['rounded_cut']):>11.4f} "  # type: ignore[arg-type]
        f"{int(row['component_count']):>5} "  # type: ignore[arg-type]
        f"{int(row['distinct_pairs']):>6} {int(row['edge_pairs']):>6} "  # type: ignore[arg-type]
        + (f"{int(row['cross_path_pairs']):>6} " if roach else "")  # type: ignore[arg-type]
        + f"{float(row['max_vertex_load']):>7.4f} "  # type: ignore[arg-type]
        + f"{float(row['mean_collision']):>7.4f}"  # type: ignore[arg-type]
    )


def format_table(name: str, graph: nx.Graph, rows: list[dict[str, object]]) -> str:
    """One table per graph: every cell, then the best Ê row, then the references to read it by."""
    from mllib.math.graph.two_hot_span_problem import ratio_cut, spectral_floor

    roach = is_roach(name)
    cluster_count = CLUSTER_COUNTS[name]
    vector_count = graph.number_of_nodes() - cluster_count
    best = min(range(len(rows)), key=lambda index: float(rows[index]["rounded_cut"]))
    lines = [
        f"{name}: n={graph.number_of_nodes()} m={graph.number_of_edges()} K={cluster_count} "
        f"r={vector_count}  Σλ (floor)={spectral_floor(graph, cluster_count):.6f}  "
        f"flat load r/n={vector_count / graph.number_of_nodes():.4f}",
        _header(roach),
    ]
    lines += [
        _row_line(row, roach, "*" if index == best else " ") for index, row in enumerate(rows)
    ]
    winner = rows[best]
    lines.append(
        f"  best Ê: {float(winner['rounded_cut']):.4f} at "  # type: ignore[arg-type]
        f"λ={float(winner['collision_weight']):g} {winner['init']} "  # type: ignore[arg-type]
        f"{winner['form']} μ={float(winner['adjacency_weight']):g} "  # type: ignore[arg-type]
        f"ν={float(winner['diversity_weight']):g}, "  # type: ignore[arg-type]
        f"{int(winner['component_count'])} components, "  # type: ignore[arg-type]
        f"RatioCut of its labels {ratio_cut(graph, np.asarray(winner['labels'])):.4f}"
    )
    lines.append(f"  reference: {references_for(name)}")
    if roach:
        for row in rows:
            if int(row["component_count"]) == cluster_count:  # type: ignore[arg-type]
                lines.append(
                    f"  K components at λ={float(row['collision_weight']):g} "  # type: ignore[arg-type]
                    f"{row['init']} {row['form']} μ={float(row['adjacency_weight']):g} "  # type: ignore[arg-type]
                    f"ν={float(row['diversity_weight']):g}: "  # type: ignore[arg-type]
                    f"{list(np.asarray(row['labels']).tolist())}"
                )
    return "\n".join(lines)


def convergence_rows(
    name: str, graph: nx.Graph, winner: dict[str, object], seed: int
) -> list[dict[str, object]]:
    """The winning cell re-run longer: was 300 steps converged, or only stopped?"""
    from mllib.math.graph.two_hot_span_problem import incidence_matrix, spectral_spanning_set

    cluster_count = CLUSTER_COUNTS[name]
    X = incidence_matrix(graph)
    spectral = spectral_spanning_set(graph, cluster_count)
    return [
        probe_row(
            name,
            graph,
            X,
            spectral,
            cluster_count=cluster_count,
            collision_weight=float(winner["collision_weight"]),  # type: ignore[arg-type]
            init=str(winner["init"]),
            form=str(winner["form"]),
            adjacency_weight=float(winner["adjacency_weight"]),  # type: ignore[arg-type]
            diversity_weight=float(winner["diversity_weight"]),  # type: ignore[arg-type]
            step_count=step_count,
            seed=seed,
        )
        for step_count in CONVERGENCE_STEP_COUNTS
    ]


def format_convergence_table(
    name: str, roach: bool, base: dict[str, object], rows: list[dict[str, object]]
) -> str:
    """The winner at 300 steps and at the longer runs, one line each, same columns."""
    lines = [
        f"{name}: is the best cell converged? "
        f"λ={float(base['collision_weight']):g} {base['init']} "  # type: ignore[arg-type]
        f"{base['form']} μ={float(base['adjacency_weight']):g} "  # type: ignore[arg-type]
        f"ν={float(base['diversity_weight']):g}",  # type: ignore[arg-type]
        f"  {'steps':>6}" + _header(roach)[2:],
    ]
    for row in [base, *rows]:
        lines.append(f"  {int(row['step_count']):>6}" + _row_line(row, roach, " ")[2:])  # type: ignore[arg-type]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the grid on every requested graph, print its table, then the convergence table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEP_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-convergence", action="store_true")
    arguments = parser.parse_args(argv)

    for name in arguments.graphs:
        graph = build_graph(name)
        started = time.monotonic()
        rows = probe_rows(name, graph, arguments.steps, arguments.seed)
        print(format_table(name, graph, rows))
        print(f"  {len(rows)} runs at {arguments.steps} steps in {time.monotonic() - started:.1f}s")
        print()
        if arguments.skip_convergence:
            continue
        winner = min(rows, key=lambda row: float(row["rounded_cut"]))  # type: ignore[arg-type]
        started = time.monotonic()
        longer = convergence_rows(name, graph, winner, arguments.seed)
        print(format_convergence_table(name, is_roach(name), winner, longer))
        print(f"  {len(longer)} longer runs in {time.monotonic() - started:.1f}s")
        print()


if __name__ == "__main__":
    main()
