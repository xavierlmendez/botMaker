"""Probe: can a learning-rate schedule or a graph term make the rounding pick the right pairs?

Slice 2.4 of `docs/plans/2026-09-two-hot-span.md`, and an **experiment**, not a decision. The
default objective stays the brief's §20 objective; everything this script varies beyond λ and the
initialisation is off unless asked for.

The question. Ê is the RatioCut of the partition the *rounding* produces, and the rounding reads one
pair per column — argmax against argmin. Nothing in `E(V) - λ Σ_j R(v_j)` says which pair: R rewards
a column for being nearly 2-hot and is indifferent to where its two hot entries sit, and the span
term is invariant to any rotation inside the span. So there are two obvious places to push. A
learning-rate schedule pushes on the *trajectory* — anneal and the columns settle instead of
oscillating. A per-column graph term pushes on the *pair* — reward a column whose two hot entries
are joined by an edge. Two forms of that reward are worth telling apart, and the difference is the
whole reason both are here:

* `laplacian`, vᵀ L v / ‖v‖₂² = (d_i + d_j + 2 w_ij)/2 on a unit 2-hot column. Larger on an edge —
  hence rewarded, subtracted from the loss — but it also pays for high degree whether or not the
  pair is an edge at all.
* `edge_product`, |v|ᵀ A |v| / ‖v‖₂² = w_ij on an edge and exactly 0 off one. The same preference
  for edges with none of the degree bias.

Both are *training* knobs in the sense D-31 fixes for `epsilon`: they change what Adam does and they
never enter a reported number. Every E\\* and Ê below comes back through the exact numpy `pinv`
projector of `two_hot_span_problem`.

    uv run --group torch python examples/two_hot_span_adjacency_probe.py
    uv run --group torch python examples/two_hot_span_adjacency_probe.py --graphs roach_g5
    uv run --group torch python examples/two_hot_span_adjacency_probe.py --graphs roach_g20

This is a composition root: it builds the graphs, assembles the configurations, runs
`fit_two_hot_span`, and prints one table per graph. It computes no mathematics of its own beyond
counting pairs.
"""

from __future__ import annotations

import argparse
import itertools
import time

import networkx as nx
import numpy as np

from mllib.math.graph.two_hot_span_problem import (
    incidence_matrix,
    karate_graph,
    ratio_cut,
    roach_graph,
    rounded_pairs,
    spectral_floor,
    spectral_spanning_set,
)

DEFAULT_STEP_COUNT = 300
DEFAULT_SEED = 0
DEFAULT_LEARNING_RATE = 0.05

ROACH_RUNG_COUNT = 5
ROACH_G20_RUNG_COUNT = 20

COLLISION_WEIGHTS = (3.0, 10.0)
INITS = ("spectral", "random")

# name -> the three schedule fields. The two annealed ones end at 0.001 from a 0.05 start, which is
# the fraction 0.02; `warmup_cosine` spends its first 30 steps climbing to 0.05 before that.
SCHEDULES: dict[str, dict[str, object]] = {
    "constant": {
        "learning_rate_schedule": "constant",
        "warmup_steps": 0,
        "final_learning_rate_fraction": 0.0,
    },
    "cosine": {
        "learning_rate_schedule": "cosine",
        "warmup_steps": 0,
        "final_learning_rate_fraction": 0.02,
    },
    "warmup30": {
        "learning_rate_schedule": "warmup_cosine",
        "warmup_steps": 30,
        "final_learning_rate_fraction": 0.02,
    },
}

ADJACENCY_WEIGHTS = (0.3, 1.0, 3.0)
# (form, mu). "none" is mu = 0, where the term is guarded off and the form is never consulted.
ADJACENCIES: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    *((form, weight) for form in ("laplacian", "edge_product") for weight in ADJACENCY_WEIGHTS),
)

# The numbers each table is read against, stated where they come from rather than recomputed.
ROACH_REFERENCES = (
    "optimum (one antenna) 4/15 = 0.2667   best balanced cut 0.4000   spectral bisection 1.0000"
)
KARATE_REFERENCES = "datum's best rounding 2.5972   planted (club) RatioCut 2.9412"
# He, Gu & Zhang 2012's eighty-node cockroach, at the K = 3 their Fig. 2a calls the ideal cut.
ROACH_G20_REFERENCES = (
    "planted (ladder + two antennae, K=3) 0.1500   antennae-vs-ladder (K=2) 0.1000   "
    "spectral bisection 1.0000"
)

GRAPHS = ("roach_g5", "karate", "roach_g20")

# K per graph: the roach of He, Gu & Zhang is the one instance here that is not a bisection, and
# every place the probes used to write a literal 2 reads this instead.
CLUSTER_COUNTS: dict[str, int] = {"roach_g5": 2, "karate": 2, "roach_g20": 3}


def build_graph(name: str) -> nx.Graph:
    """The graph a name stands for, built the way the harness builds it."""
    if name == "roach_g5":
        return roach_graph(ROACH_RUNG_COUNT)
    if name == "roach_g20":
        return roach_graph(ROACH_G20_RUNG_COUNT)
    if name == "karate":
        return karate_graph()
    raise ValueError(f"unknown graph {name!r}; known: {sorted(GRAPHS)}")


def is_roach(name: str) -> bool:
    """Whether a graph name is a roach, and so has a cross-path column to print."""
    return name.startswith("roach")


# The line of known cut values each table is read against, by name. A mapping and not a chain of
# ifs with a fallback: a graph added without its references would otherwise silently print karate's.
REFERENCES: dict[str, str] = {
    "roach_g5": ROACH_REFERENCES,
    "karate": KARATE_REFERENCES,
    "roach_g20": ROACH_G20_REFERENCES,
}


def references_for(name: str) -> str:
    """The line of known cut values a table is read against; an unknown name is an error."""
    try:
        return REFERENCES[name]
    except KeyError:
        raise ValueError(f"unknown graph {name!r}; known: {sorted(REFERENCES)}") from None


def distinct_pairs(spanning_set: np.ndarray) -> list[frozenset[int]]:
    """The unordered pairs the rounding read off the columns, each counted once.

    A column rounds to (argmax, argmin) and two columns can round to the same pair; the pair graph
    only ever sees it once, so the count that explains a component structure is the distinct one.
    """
    return list({frozenset(pair) for pair in rounded_pairs(spanning_set)})


def edge_pair_count(graph: nx.Graph, pairs: list[frozenset[int]]) -> int:
    """How many distinct rounded pairs are edges of G — what the graph term is trying to raise."""
    return sum(1 for pair in pairs if graph.has_edge(*sorted(pair)))


def cross_path_pair_count(pairs: list[frozenset[int]], path_split: int) -> int:
    """Roach only: pairs whose two endpoints sit on opposite paths of the ladder.

    A roach's top path is 0..2k-1 and its bottom path is 2k..4k-1, so a pair straddles the paths
    exactly when its endpoints fall on opposite sides of 2k = n/2. The boundary is passed in as
    ``n // 2`` rather than fixed at the k = 5 value of 10, so the same count reads the k = 20 roach.
    """
    total = 0
    for pair in pairs:
        first, second = sorted(pair)
        if (first < path_split) != (second < path_split):
            total += 1
    return total


def probe_rows(name: str, graph: nx.Graph, step_count: int, seed: int) -> list[dict[str, object]]:
    """Every cell of the λ x init x schedule x adjacency grid, as one row each."""
    from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanConfig, fit_two_hot_span

    cluster_count = CLUSTER_COUNTS[name]
    path_split = graph.number_of_nodes() // 2
    X = incidence_matrix(graph)
    spectral = spectral_spanning_set(graph, cluster_count)

    rows: list[dict[str, object]] = []
    grid = itertools.product(COLLISION_WEIGHTS, INITS, SCHEDULES.items(), ADJACENCIES)
    for collision_weight, init, (schedule_name, schedule), (form, adjacency_weight) in grid:
        config = TwoHotSpanConfig(
            step_count=step_count,
            learning_rate=DEFAULT_LEARNING_RATE,
            collision_weight=collision_weight,
            seed=seed,
            adjacency_weight=adjacency_weight,
            # `none` still has to name a legal form; at mu = 0 it is never read.
            adjacency_form="laplacian" if form == "none" else form,
            **schedule,  # type: ignore[arg-type]
        )
        run = fit_two_hot_span(X, cluster_count, config, spectral if init == "spectral" else None)
        pairs = distinct_pairs(run.spanning_set)
        rows.append(
            {
                "collision_weight": collision_weight,
                "init": init,
                "schedule": schedule_name,
                "form": form,
                "adjacency_weight": adjacency_weight,
                "relaxed_objective": float(run.relaxed_objective),
                "rounded_cut": float(run.rounded_cut),
                "component_count": int(np.unique(run.labels).size),
                "distinct_pairs": len(pairs),
                "edge_pairs": edge_pair_count(graph, pairs),
                "cross_path_pairs": (
                    cross_path_pair_count(pairs, path_split) if is_roach(name) else None
                ),
                "mean_collision": float(np.mean(run.collision_measures)),
                "labels": run.labels,
            }
        )
    return rows


def format_table(name: str, graph: nx.Graph, rows: list[dict[str, object]]) -> str:
    """One table per graph: every cell, then the best Ê row, then the references to read it by."""
    roach = is_roach(name)
    cluster_count = CLUSTER_COUNTS[name]
    header = (
        f"  {'λ':>5} {'init':<9} {'schedule':<9} {'form':<13} {'μ':>5} "
        f"{'E*':>11} {'Ê':>11} {'comp':>5} {'pairs':>6} {'edges':>6} "
        + (f"{'cross':>6} " if roach else "")
        + f"{'mean R':>7}"
    )
    best = min(range(len(rows)), key=lambda index: float(rows[index]["rounded_cut"]))
    lines = [
        f"{name}: n={graph.number_of_nodes()} m={graph.number_of_edges()} K={cluster_count} "
        f"r={graph.number_of_nodes() - cluster_count}  "
        f"Σλ (floor)={spectral_floor(graph, cluster_count):.6f}",
        header,
    ]
    for index, row in enumerate(rows):
        marker = "*" if index == best else " "
        lines.append(
            f" {marker}{float(row['collision_weight']):>5.1f} {row['init']:<9} "  # type: ignore[arg-type]
            f"{row['schedule']:<9} {row['form']:<13} "
            f"{float(row['adjacency_weight']):>5.1f} "  # type: ignore[arg-type]
            f"{float(row['relaxed_objective']):>11.4f} "  # type: ignore[arg-type]
            f"{float(row['rounded_cut']):>11.4f} "  # type: ignore[arg-type]
            f"{int(row['component_count']):>5} "  # type: ignore[arg-type]
            f"{int(row['distinct_pairs']):>6} {int(row['edge_pairs']):>6} "  # type: ignore[arg-type]
            + (f"{int(row['cross_path_pairs']):>6} " if roach else "")  # type: ignore[arg-type]
            + f"{float(row['mean_collision']):>7.4f}"  # type: ignore[arg-type]
        )
    winner = rows[best]
    lines.append(
        f"  best Ê: {float(winner['rounded_cut']):.4f} at "  # type: ignore[arg-type]
        f"λ={float(winner['collision_weight']):g} {winner['init']} "  # type: ignore[arg-type]
        f"{winner['schedule']} {winner['form']} μ={float(winner['adjacency_weight']):g}, "  # type: ignore[arg-type]
        f"{int(winner['component_count'])} components, "  # type: ignore[arg-type]
        f"RatioCut of its labels {ratio_cut(graph, np.asarray(winner['labels'])):.4f}"
    )
    lines.append(f"  reference: {references_for(name)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the grid on every requested graph and print one table each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEP_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args(argv)

    for name in arguments.graphs:
        graph = build_graph(name)
        started = time.monotonic()
        rows = probe_rows(name, graph, arguments.steps, arguments.seed)
        print(format_table(name, graph, rows))
        print(f"  {len(rows)} runs at {arguments.steps} steps in {time.monotonic() - started:.1f}s")
        print()


if __name__ == "__main__":
    main()
