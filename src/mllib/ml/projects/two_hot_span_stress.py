"""The two-hot span stress ladder: the prototype's tuned cell, run on graphs it was not tuned on.

Slice 2.7 of `docs/plans/2026-09-two-hot-span.md`, from the research seed
`~/develop/research/sessions/2026-09-18-two-hot-stress.md`. The prototype found one tuned cell —
roach G_5 at λ = 10, ν = 10, μ = 0.3 (`edge_product`), Ê = 4/15, the brute-force optimum — and one
cell is not a result. This module runs a **pre-registered** ladder of graphs and coupling settings
so that three questions get measured answers: does the coupling band transfer across n and cluster
strength, where does the optimizer stop reaching K components, and how do Ê - E\\* and Ê - Σλ scale.

**It is a runner, not an engine.** Every number here comes back through the numpy `pinv` path of
`two_hot_span_problem` and the datum of `two_hot_span_harness`, exactly as the harness computes
them; this module adds no mathematics of its own. `VᵀV = I` is never imposed; Ê is the headline,
E\\* and Σλ are reported beside it, R(v_j) is a per-column diagnostic; rcut only — ncut is BL-41.
The three engine modules are untouched by this slice, and the rung-0 oracle is what proves it: the
prototype's five JSON reports are reproduced to 1e-8 before any new graph is allowed to run.

**Pre-registration is the point.** The ladder, the coupling grid, λ, the learning rate, the step
budget and the checkpoints are constants in this file, fixed before the first run, and a test pins
the cell counts. A cell that reaches fewer than K components is reported as such and never dropped;
so is a cell that times out, runs out of memory budget or raises. Tuning outside the pre-registered
sweep would make the numbers unreportable, which is why there is no `--diversity-weight` flag.

**Why a process per cell.** Peak RSS is a per-cell number, and a pool worker that has already run a
2000-node cell reports that cell's high-water mark for the next one. `maxtasksperchild=1` on a spawn
context buys a fresh process — and therefore a fresh `ru_maxrss` — for every cell. The same fresh
process is what makes the memory budget meaningful: the scheduler admits a cell only when the
predicted peak of everything already running plus the newcomer fits.

**Why the recorder raises.** A cell that will not finish inside its cap has to be interrupted from
*inside* the training loop, and the loop belongs to the engine, which this slice may not change. The
checkpoint recorder is already called once per step (D-32's injected collaborator), so the deadline
is checked there and a `CellTimeout` propagates out through the engine untouched, with whatever
checkpoints had completed still in hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mllib.math.graph.two_hot_span_problem import (
    GraphInstance,
    clustering_from_pairs,
    default_test_graphs,
    incidence_matrix,
    node_count,
    planted_partition_graph,
    projector_residual,
    ratio_cut,
    roach_g20_instance,
    rounded_cut,
    rounded_pairs,
    spanning_vector_count,
    spectral_floor,
    spectral_spanning_set,
)
from mllib.math.recorder import AbstractStepRecorder
from mllib.ml.projects.two_hot_span_harness import ROUNDING_NAMES, datum_roundings

SCHEMA_VERSION = 1


def repository_root() -> Path:
    """The checkout this module lives in: src/mllib/ml/projects/<this file> is four levels down."""
    return Path(__file__).resolve().parents[4]


# The prototype's frozen reports, committed beside the tests. `--reports-dir` names where they live
# on Xavier's laptop; the cloud instance has no `~/Desktop`, and rung 2 runs there, so the oracle
# falls back to the tree's own copies rather than being skippable off the laptop. Running it on the
# Linux box is a second arithmetic path, which is the point of checking it there instead of
# assuming it.
FIXTURE_REPORTS_DIR = repository_root() / "tests" / "ml" / "fixtures"

# ---------------------------------------------------------------------------------------------
# The ladder, pre-registered. Changing anything below changes what "the stress run" means, so it
# changes with a plan row and a new results file, never in place.
# ---------------------------------------------------------------------------------------------

COLLISION_WEIGHT = 10.0
LEARNING_RATE = 0.05
STEP_COUNT = 3000
CHECKPOINT_STEPS = (300, 1000, 3000)
# (diversity ν, adjacency μ): the brief's objective, the roach band, and each half of the band on
# its own — the four that make the band's two terms separable in the results.
FIXED_COUPLINGS = ((0.0, 0.0), (10.0, 0.3), (10.0, 0.0), (0.0, 0.3))
FIXED_INITS = ("random", "spectral")
SWEEP_DIVERSITY = (3.0, 10.0, 30.0)
SWEEP_ADJACENCY = (0.1, 0.3, 1.0)
SWEEP_INIT = "random"

RUNG0_GRAPHS = ("roach_g5", "karate", "roach_g20")
RUNG0_SEEDS = (0, 1, 2)
RUNG1_SIZES = ((100, 2), (201, 3), (500, 5))
RUNG1_SEEDS = (0, 1, 2)
RUNG2_SIZES = ((1000, 4), (2000, 8))
RUNG2_SEEDS = (0,)
RUNG3_GRAPHS = ("polbooks", "football", "email_eu_core")
RUNG3_SEEDS = (0, 1, 2)
# Published orders, used only to price a cell's cap and memory before the loader has run; the
# graph record carries the n the loader actually returned.
RUNG3_EXPECTED_NODES = {"polbooks": 105, "football": 115, "email_eu_core": 1005}

# Planted partitions are specified by expected degree rather than by probability, so that "moderate"
# means the same signal at n = 100 and n = 2000: 12 edges within the block per vertex, d_out edges
# out of it per vertex.
WITHIN_BLOCK_DEGREE = 12.0
CROSS_BLOCK_DEGREE = {"clear": 1.0, "moderate": 3.0, "weak": 6.0}
RUNG1_STRENGTHS = ("clear", "moderate", "weak")
RUNG2_STRENGTH = "moderate"

# Measured on this machine (Apple M2 Pro, 2 torch threads) at the shipped step count; interpolated
# log-log in n, since the cost is dominated by the r x r solve and grows as a power of n.
SECONDS_PER_STEP = {100: 0.0016, 201: 0.0058, 500: 0.047, 1000: 0.30, 2000: 2.57}
CAP_OVERHEAD_SECONDS = 60.0
DEFAULT_CAP_MULTIPLIER = 4.0
# Peak RSS is dominated by the n x n dense Laplacian/adjacency pair and the n x r spanning set, all
# float64: quadratic in n, plus a fixed interpreter-and-torch floor.
RSS_FLOOR_GB = 0.3
RSS_QUADRATIC_GB = 0.45
DEFAULT_MEMORY_BUDGET_GB = 8.0
LAPTOP_WORKER_LIMIT = 4
# A worker that is killed outright — the OOM killer, a segfault — never sets its result, so the
# recorder's own deadline can never fire and the parent would wait for a process that is gone. The
# parent therefore keeps its own clock: a cell that has not returned within its cap plus this grace
# is presumed dead, recorded, and its pool rebuilt.
WORKER_GRACE_SECONDS = 120.0
WORKER_LOST_MESSAGE = "worker did not return within cap + grace; presumed killed"

# Ê and E\\* may not fall below the spectral floor; anything smaller than this is arithmetic.
INVARIANT_TOLERANCE = 1e-10
# Two checkpoints agree when Ê matches this closely and the labels are identical.
CONVERGENCE_TOLERANCE = 1e-9
ORACLE_TOLERANCE = 1e-8
TUNED_ROUNDED_CUT = 4.0 / 15.0
TUNED_CELL_CHECK = "tuned_cell_roach_g5"
# Every one of these four must have passed in a results file before rungs 1-3 may run: the three
# report checks say the engine is unchanged, and the tuned cell says the *runner* still reaches the
# prototype's answer with the ladder's own budget, coupling and checkpoints.
ORACLE_CHECKS = (*RUNG0_GRAPHS, TUNED_CELL_CHECK)

ENGINE_MODULES = (
    "src/mllib/math/graph/two_hot_span_problem.py",
    "src/mllib/math/algorithms/two_hot_span_optimizer.py",
    "src/mllib/ml/projects/two_hot_span_harness.py",
)

STATUSES = (
    "reached_k",
    "drifted",
    "not_at_k",
    "error",
    "timeout",
    "skipped_deadline",
    "over_budget",
)


class CellTimeout(RuntimeError):  # noqa: N818 - a timeout is not an "...Error" in this vocabulary
    """One cell passed its wall-clock cap. Raised from the recorder, inside the engine's loop."""


# ---------------------------------------------------------------------------------------------
# Specs: what a graph is, and what a cell is, before either has run.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSpec:
    """A graph the ladder names, as parameters rather than as an object.

    A spec is what crosses the process boundary: the worker rebuilds the graph from it, so a cell
    carries kilobytes instead of a pickled networkx graph, and the record carries every generator
    parameter a reader would need to rebuild it by hand.
    """

    rung: int
    name: str
    graph_seed: int
    kind: str
    generator: tuple[tuple[str, object], ...]
    expected_node_count: int

    @property
    def generator_dict(self) -> dict[str, object]:
        return dict(self.generator)

    @property
    def graph_key(self) -> str:
        return f"r{self.rung}|{self.name}|gs{self.graph_seed}"


@dataclass(frozen=True)
class CellSpec:
    """One pre-registered configuration: a graph-seed pair and a point of the coupling grid."""

    graph: GraphSpec
    diversity_weight: float
    adjacency_weight: float
    init: str

    @property
    def adjacency_form(self) -> str:
        """`edge_product` whenever μ is on: the form the probes found, with no degree bias."""
        return "edge_product" if self.adjacency_weight != 0.0 else "laplacian"

    @property
    def key(self) -> str:
        return (
            f"r{self.graph.rung}|{self.graph.name}|gs{self.graph.graph_seed}"
            f"|l{COLLISION_WEIGHT:g}|nu{self.diversity_weight:g}"
            f"|mu{self.adjacency_weight:g}|{self.init}"
        )


def planted_partition_parameters(
    node_total: int, cluster_count: int, strength: str
) -> tuple[int, float, float]:
    """(group_size, p_in, p_out) for one rung-1/2 point, from expected degrees rather than from p.

    Holding p fixed across n would make "moderate" mean a different signal at every size — the
    within-block degree would grow linearly with the group. Holding the *degrees* fixed is what
    makes the rungs comparable: every vertex expects 12 within-block edges and d_out cross-block
    edges whatever n is, so what changes up the ladder is the size, not the difficulty.
    """
    if strength not in CROSS_BLOCK_DEGREE:
        raise ValueError(
            f"unknown strength {strength!r}; expected one of {list(CROSS_BLOCK_DEGREE)}"
        )
    group_size = node_total // cluster_count
    p_in = WITHIN_BLOCK_DEGREE / (group_size - 1)
    p_out = CROSS_BLOCK_DEGREE[strength] / (node_total - group_size)
    return group_size, p_in, p_out


def _planted_spec(rung: int, node_total: int, cluster_count: int, strength: str, seed: int):
    group_size, p_in, p_out = planted_partition_parameters(node_total, cluster_count, strength)
    return GraphSpec(
        rung=rung,
        name=f"sbm_n{node_total}_k{cluster_count}_{strength}",
        graph_seed=seed,
        kind="planted_partition",
        generator=(
            ("n", node_total),
            ("cluster_count", cluster_count),
            ("group_size", group_size),
            ("p_in", p_in),
            ("p_out", p_out),
            ("strength", strength),
            ("within_block_degree", WITHIN_BLOCK_DEGREE),
            ("cross_block_degree", CROSS_BLOCK_DEGREE[strength]),
            ("seed", seed),
        ),
        expected_node_count=node_total,
    )


def graph_specs(rung: int) -> tuple[GraphSpec, ...]:
    """Every (graph, seed) pair of one rung, in the order the runner walks them."""
    if rung == 0:
        builtin = {"roach_g5": 20, "karate": 34, "roach_g20": 80}
        return tuple(
            GraphSpec(0, name, seed, "builtin", (("source", "default_test_graphs"),), builtin[name])
            for name in RUNG0_GRAPHS
            for seed in RUNG0_SEEDS
        )
    if rung == 1:
        return tuple(
            _planted_spec(1, node_total, cluster_count, strength, seed)
            for node_total, cluster_count in RUNG1_SIZES
            for strength in RUNG1_STRENGTHS
            for seed in RUNG1_SEEDS
        )
    if rung == 2:
        return tuple(
            _planted_spec(2, node_total, cluster_count, RUNG2_STRENGTH, seed)
            for node_total, cluster_count in RUNG2_SIZES
            for seed in RUNG2_SEEDS
        )
    if rung == 3:
        return tuple(
            GraphSpec(
                3,
                name,
                seed,
                "community",
                (("loader", "load_community_graph"), ("name", name)),
                RUNG3_EXPECTED_NODES[name],
            )
            for name in RUNG3_GRAPHS
            for seed in RUNG3_SEEDS
        )
    raise ValueError(f"unknown rung {rung!r}; the ladder has rungs 0-3")


def coupling_settings(rung: int) -> tuple[tuple[float, float, str], ...]:
    """(ν, μ, init) for a rung: the four fixed couplings everywhere, the 3x3 sweep on rung 1 only.

    The sweep's (10, 0.3, random) point *is* the roach band's fixed cell, so it is deduplicated
    rather than run twice: two identical cells would differ only in which one a reader quoted.
    """
    settings = [
        (diversity, adjacency, init)
        for diversity, adjacency in FIXED_COUPLINGS
        for init in FIXED_INITS
    ]
    if rung == 1:
        settings += [
            (diversity, adjacency, SWEEP_INIT)
            for diversity in SWEEP_DIVERSITY
            for adjacency in SWEEP_ADJACENCY
        ]
    seen: dict[tuple[float, float, str], None] = {}
    for setting in settings:
        seen.setdefault(setting, None)
    return tuple(seen)


def plan_cells(rungs: Sequence[int]) -> tuple[CellSpec, ...]:
    """Every cell of the requested rungs, deduplicated by key, in ladder order."""
    cells: list[CellSpec] = []
    seen: set[str] = set()
    for rung in rungs:
        for spec in graph_specs(rung):
            for diversity, adjacency, init in coupling_settings(rung):
                cell = CellSpec(spec, diversity, adjacency, init)
                if cell.key in seen:
                    continue
                seen.add(cell.key)
                cells.append(cell)
    return tuple(cells)


# ---------------------------------------------------------------------------------------------
# Cost models: how long a cell may take, and how much memory it is expected to want.
# ---------------------------------------------------------------------------------------------


def predicted_seconds_per_step(node_total: int) -> float:
    """Log-log interpolation of the measured table; the last slope carries it past n = 2000.

    Below the table the first entry is held flat rather than extrapolated down: the small graphs are
    seconds either way, and the cap's fixed overhead already dominates there.
    """
    sizes = sorted(SECONDS_PER_STEP)
    log_sizes = np.log([float(size) for size in sizes])
    log_seconds = np.log([SECONDS_PER_STEP[size] for size in sizes])
    target = float(np.log(float(node_total)))
    if target <= log_sizes[0]:
        return float(SECONDS_PER_STEP[sizes[0]])
    if target >= log_sizes[-1]:
        slope = (log_seconds[-1] - log_seconds[-2]) / (log_sizes[-1] - log_sizes[-2])
        return float(np.exp(log_seconds[-1] + slope * (target - log_sizes[-1])))
    return float(np.exp(np.interp(target, log_sizes, log_seconds)))


def cap_seconds(node_total: int, step_count: int, multiplier: float) -> float:
    """The wall-clock a cell is allowed: a multiple of the predicted run, plus fixed overhead.

    The overhead is not decoration. A cell pays for building its graph, its incidence matrix, its
    spectral init and its checkpoints' pinv projections before and between the steps the table
    prices, and on the small rungs that setup is most of the run.
    """
    return multiplier * step_count * predicted_seconds_per_step(node_total) + CAP_OVERHEAD_SECONDS


def predicted_rss_gb(node_total: int) -> float:
    """Predicted peak RSS: an interpreter-and-torch floor plus the dense n x n working set."""
    return RSS_FLOOR_GB + RSS_QUADRATIC_GB * (float(node_total) / 1000.0) ** 2


def admits_cell(running_gb: float, cell_gb: float, budget_gb: float) -> bool:
    """Does the newcomer fit beside what is already running?"""
    return running_gb + cell_gb <= budget_gb


def budget_rejection(
    cell: CellSpec, host: str, budget_gb: float, allow_rung_2: bool = False
) -> str | None:
    """The reason a cell can never start on this host, or ``None`` when it may be scheduled.

    Two refusals, both structural rather than transient: the laptop does not run rung 2 by default
    (the seed's "do not" list — a 2000-node cell was meant to be a cloud cell), and a cell whose own
    predicted peak exceeds the whole budget would never fit however long the scheduler waited.

    ``allow_rung_2`` lifts the first refusal and nothing else. It exists because the cloud path did
    not survive contact: the Linux oracle fails — the spectral-init rows differ under OpenBLAS's
    eigh basis, and one karate random-init row flips its labels at a near-tie — so rung 2 comes home
    to the machine whose arithmetic the prototype's numbers were established on. The memory rules
    are untouched by the flag: at an 8 GB budget the admission arithmetic still holds three
    2000-node cells (2.1 GB each) at a time, and a single cell above the budget is still refused.
    """
    if host == "laptop" and cell.graph.rung == 2 and not allow_rung_2:
        return (
            "rung 2 is not scheduled on the laptop host; pass --allow-rung-2 to run it here "
            "(the memory budget still applies), or use --host cloud"
        )
    cell_gb = predicted_rss_gb(cell.graph.expected_node_count)
    if cell_gb > budget_gb:
        return f"predicted peak {cell_gb:.2f} GB exceeds the whole memory budget {budget_gb:.2f} GB"
    return None


# ---------------------------------------------------------------------------------------------
# The checkpoint recorder.
# ---------------------------------------------------------------------------------------------


class CheckpointRecorder(AbstractStepRecorder):
    """Ê, E\\*, the clustering and the component count at pre-registered steps of one run.

    The engine hands over the step index, the training loss and V as it stands, and computes
    nothing for the recorder's sake (D-32). Everything reported here is therefore this class's own
    work through the numpy `pinv` path — the same functions the harness reports through, so a
    checkpoint at the final step is the run's own headline number and a test says so.

    The deadline lives here for the same reason: `record_step` is the only code of ours that runs
    inside the engine's loop, so it is the only place a cell can be stopped without touching the
    engine. Raising loses nothing — the checkpoints reached so far are on this object, and the
    caller reports them under the `timeout` status.
    """

    enabled = True

    def __init__(
        self,
        X: np.ndarray,
        checkpoint_steps: Sequence[int] = CHECKPOINT_STEPS,
        deadline: float | None = None,
    ) -> None:
        super().__init__()
        self.X = np.asarray(X, dtype=float)
        self.checkpoint_steps = tuple(int(step) for step in checkpoint_steps)
        self.deadline = deadline
        self.checkpoints: list[dict[str, object]] = []

    def record_step(
        self, step: int, training_loss: float, spanning_set: object, **extras: object
    ) -> None:
        """The engine's 0-based index after the step, so step `t` is checkpoint `t + 1`."""
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise CellTimeout(f"the cell passed its cap at step {step + 1}")
        step_number = step + 1
        if step_number not in self.checkpoint_steps:
            return
        current = np.asarray(spanning_set, dtype=float)
        labels = clustering_from_pairs(current.shape[0], rounded_pairs(current))
        self.checkpoints.append(
            {
                "step": int(step_number),
                "relaxed_objective": float(projector_residual(self.X, current)),
                "rounded_cut": float(rounded_cut(self.X, current)),
                "component_count": int(np.unique(labels).size),
                "labels": [int(label) for label in labels],
            }
        )

    def record_end(self, run: object, **extras: object) -> None:
        """The run's own numbers are read off the returned run; nothing extra is stored."""

    def describe_result(self, result: object) -> dict[str, object]:
        return {}


def checkpoints_agree(first: dict, second: dict) -> bool:
    """Two checkpoints are the same answer: Ê within tolerance and an identical clustering."""
    return abs(
        float(first["rounded_cut"]) - float(second["rounded_cut"])
    ) <= CONVERGENCE_TOLERANCE and list(first["labels"]) == list(second["labels"])


def convergence(checkpoints: Sequence[dict]) -> tuple[bool, int | None]:
    """(converged, steps_to_convergence) from a checkpoint history.

    Converged means the last two checkpoints are the same answer — the cheapest honest statement a
    three-point history supports. `steps_to_convergence` is the first checkpoint the run never left
    afterwards, which is `None` exactly when it never settled.
    """
    if len(checkpoints) < 2:
        return False, None
    if not checkpoints_agree(checkpoints[-2], checkpoints[-1]):
        return False, None
    for index, checkpoint in enumerate(checkpoints):
        if all(checkpoints_agree(checkpoint, later) for later in checkpoints[index + 1 :]):
            return True, int(checkpoint["step"])
    return True, int(checkpoints[-1]["step"])


def cell_status(checkpoints: Sequence[dict], cluster_count: int) -> tuple[str, int | None]:
    """(status, drift_onset_step) for a cell that finished its loop.

    Drift is the failure the seed cares about most: a run that found K components and then lost
    them says something different about the objective than one that never found them, so the two
    are different statuses and the onset step is recorded.
    """
    if not checkpoints:
        return "not_at_k", None
    counts = [int(checkpoint["component_count"]) for checkpoint in checkpoints]
    if counts[-1] == cluster_count:
        return "reached_k", None
    if cluster_count not in counts:
        return "not_at_k", None
    # The run ended away from K having reached it, so some checkpoint after the first arrival must
    # differ; the earliest one is the onset.
    first_reached = counts.index(cluster_count)
    onset = next(
        int(checkpoints[index]["step"])
        for index in range(first_reached + 1, len(counts))
        if counts[index] != cluster_count
    )
    return "drifted", onset


# ---------------------------------------------------------------------------------------------
# Building the graphs.
# ---------------------------------------------------------------------------------------------


def build_graph(spec: GraphSpec, data_dir: Path | None = None) -> GraphInstance:
    """The `GraphInstance` a spec names, named after the spec so records and keys agree."""
    if spec.kind == "builtin":
        if spec.name == "roach_g20":
            return roach_g20_instance()
        for instance in default_test_graphs():
            if instance.name == spec.name:
                return instance
        raise ValueError(f"{spec.name!r} is not a default test graph")
    if spec.kind == "planted_partition":
        parameters = spec.generator_dict
        instance = planted_partition_graph(
            int(parameters["cluster_count"]),
            int(parameters["group_size"]),
            float(parameters["p_in"]),
            float(parameters["p_out"]),
            int(parameters["seed"]),
        )
        return dataclasses.replace(instance, name=spec.name)
    if spec.kind == "community":
        # Imported here rather than at module scope: the rung-3 loaders are their own slice, and a
        # laptop run of rungs 0-2 must not fail at import because rung 3's module is not in yet.
        try:
            from mllib.math.graph.community_graphs import load_community_graph
        except ImportError as error:
            raise ImportError(
                "rung 3 needs mllib.math.graph.community_graphs.load_community_graph, which is "
                f"not importable: {error}"
            ) from error
        return load_community_graph(spec.name, data_dir)
    raise ValueError(f"unknown graph kind {spec.kind!r}")


def graph_record(spec: GraphSpec, data_dir: Path | None = None) -> dict[str, object]:
    """Everything measured on one (graph, seed) once: the floor, the datum, the planted cut.

    The datum is the comparison the whole ladder is read against, and it depends only on the graph
    and the seed — computing it inside each of the sixteen cells would spend an eigendecomposition
    per cell to get the same three numbers back.
    """
    started = time.monotonic()
    instance = build_graph(spec, data_dir)
    graph = instance.graph
    count = node_count(graph)
    cluster_count = instance.cluster_count
    datum = {}
    for name, labels in datum_roundings(graph, cluster_count, seed=spec.graph_seed).items():
        datum[name] = {
            "ratio_cut": float(ratio_cut(graph, labels)),
            "labels": [int(label) for label in labels],
        }
    best_datum_name = min(datum, key=lambda name: float(datum[name]["ratio_cut"]))
    generator = spec.generator_dict
    if spec.kind == "community":
        # The rung-3 loader hangs its own provenance — source urls, sha256s, and what it dropped to
        # reach a connected graph — off the networkx graph. A downloaded graph's generator is that
        # provenance, so it belongs in the same field a planted graph's parameters go in.
        generator.update(graph.graph)
    return {
        "record": "graph",
        "schema_version": SCHEMA_VERSION,
        "rung": spec.rung,
        "name": spec.name,
        "graph_seed": spec.graph_seed,
        "status": "ok",
        "node_count": int(count),
        "edge_count": int(graph.number_of_edges()),
        "cluster_count": int(cluster_count),
        "spanning_vector_count": int(spanning_vector_count(count, cluster_count)),
        "spectral_floor": float(spectral_floor(graph, cluster_count)),
        "datum": datum,
        "best_datum_name": best_datum_name,
        "planted_labels_ratio_cut": (
            None
            if instance.planted_labels is None
            else float(ratio_cut(graph, instance.planted_labels))
        ),
        "generator": _json_ready(generator),
        "seconds": float(time.monotonic() - started),
        "error": None,
    }


def failed_graph_record(spec: GraphSpec, message: str, seconds: float) -> dict[str, object]:
    """A generator that raised is a recorded fact, not a silently missing row."""
    return {
        "record": "graph",
        "schema_version": SCHEMA_VERSION,
        "rung": spec.rung,
        "name": spec.name,
        "graph_seed": spec.graph_seed,
        "status": "error",
        "node_count": None,
        "edge_count": None,
        "cluster_count": None,
        "spanning_vector_count": None,
        "spectral_floor": None,
        "datum": None,
        "best_datum_name": None,
        "planted_labels_ratio_cut": None,
        "generator": _json_ready(spec.generator_dict),
        "seconds": float(seconds),
        "error": message,
    }


def _json_ready(value: object) -> object:
    """numpy scalars and paths out, plain JSON in — checked here so `json.dumps` never surprises."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------------------------
# Running one cell, in a worker process.
# ---------------------------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """`ru_maxrss` is bytes on macOS and kilobytes on Linux; the record is always megabytes."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def worker_initializer(threads: int) -> None:
    """Pin the thread count before torch is imported in this fresh process."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(threads)


def _check_invariants(rounded: float, relaxed: float, floor: float) -> None:
    """Ê ≥ Σλ and E\\* ≥ Σλ, per cell. The harness checks the same two; its checker is private."""
    if relaxed < floor - INVARIANT_TOLERANCE:
        raise ValueError(f"E* = {relaxed!r} is below the spectral floor {floor!r}")
    if rounded < floor - INVARIANT_TOLERANCE:
        raise ValueError(f"rounded cut {rounded!r} is below the spectral floor {floor!r}")


def run_cell(cell: CellSpec, context: dict) -> dict[str, object]:
    """One cell, start to finish, in this process. Returns the record; raises nothing it can help.

    A cell that raises is a result: `NonFiniteLoss`, `ZeroSumViolation` and every `ValueError` the
    engine or the invariant check produces come back as an `error` record with the message and with
    whatever checkpoints completed, because the seed asks for every cell to be reported including
    the failures.
    """
    import torch

    from mllib.math.algorithms.two_hot_span_optimizer import (
        NonFiniteLoss,
        TwoHotSpanConfig,
        ZeroSumViolation,
        fit_two_hot_span,
    )

    torch.set_num_threads(int(context["threads"]))
    started = time.monotonic()
    step_count = int(context.get("step_count", STEP_COUNT))
    checkpoint_steps = tuple(context.get("checkpoint_steps", CHECKPOINT_STEPS))
    cap = float(context["cap_seconds"])
    floor = float(context["spectral_floor"])
    cluster_count = int(context["cluster_count"])
    recorder = CheckpointRecorder(np.zeros((1, 1)), checkpoint_steps)
    status = "error"
    message: str | None = None
    run = None
    try:
        instance = build_graph(cell.graph, context.get("data_dir"))
        X = incidence_matrix(instance.graph)
        recorder = CheckpointRecorder(X, checkpoint_steps, deadline=started + cap)
        initial = (
            spectral_spanning_set(instance.graph, cluster_count)
            if cell.init == "spectral"
            else None
        )
        config = TwoHotSpanConfig(
            step_count=step_count,
            learning_rate=LEARNING_RATE,
            collision_weight=COLLISION_WEIGHT,
            seed=cell.graph.graph_seed,
            adjacency_weight=cell.adjacency_weight,
            adjacency_form=cell.adjacency_form,
            diversity_weight=cell.diversity_weight,
        )
        run = fit_two_hot_span(X, cluster_count, config, initial, recorder=recorder)
        _check_invariants(run.rounded_cut, run.relaxed_objective, floor)
        status, drift_onset = cell_status(recorder.checkpoints, cluster_count)
    except CellTimeout as error:
        status, drift_onset, message = "timeout", None, str(error)
    except (NonFiniteLoss, ZeroSumViolation, ValueError, ImportError) as error:
        status, drift_onset, message = "error", None, f"{type(error).__name__}: {error}"
    converged, steps_to_convergence = convergence(recorder.checkpoints)
    record = cell_record_skeleton(cell, context, status)
    record.update(_final_numbers(run, recorder.checkpoints, floor))
    record.update(
        {
            "converged": bool(converged),
            "steps_to_convergence": steps_to_convergence,
            "drift_onset_step": drift_onset,
            "checkpoints": _json_ready(recorder.checkpoints),
            "seconds": float(time.monotonic() - started),
            "peak_rss_mb": _peak_rss_mb(),
            "error": message,
        }
    )
    return record


def _final_numbers(run, checkpoints: Sequence[dict], floor: float) -> dict[str, object]:
    """The headline numbers: the run's own when the loop finished, else its last checkpoint's.

    A timed-out or failed cell still says where it had got to, which is the whole reason the
    checkpoints exist; what it cannot say is the per-column diagnostics and the training loss, which
    the engine only assembles at the end, so those stay `None` rather than being guessed.
    """
    if run is not None:
        rounded, relaxed = float(run.rounded_cut), float(run.relaxed_objective)
        numbers: dict[str, object] = {
            "component_count": int(np.unique(run.labels).size),
            "labels": [int(label) for label in run.labels],
            "collision_measures": [float(value) for value in run.collision_measures],
            "max_zero_sum_violation": float(run.max_zero_sum_violation),
            "final_training_loss": (float(run.loss_history[-1]) if run.loss_history.size else None),
        }
    elif checkpoints:
        last = checkpoints[-1]
        rounded, relaxed = float(last["rounded_cut"]), float(last["relaxed_objective"])
        numbers = {
            "component_count": int(last["component_count"]),
            "labels": [int(label) for label in last["labels"]],
        }
    else:
        return {}
    numbers.update(
        {
            "relaxed_objective": relaxed,
            "rounded_cut": rounded,
            "rounded_cut_minus_relaxed": rounded - relaxed,
            "rounded_cut_minus_floor": rounded - floor,
        }
    )
    return numbers


def cell_record_skeleton(cell: CellSpec, context: dict, status: str) -> dict[str, object]:
    """The fields every cell record carries whether or not the cell ever ran."""
    return {
        "record": "cell",
        "schema_version": SCHEMA_VERSION,
        "key": cell.key,
        "rung": cell.graph.rung,
        "name": cell.graph.name,
        "graph_seed": cell.graph.graph_seed,
        "init_seed": cell.graph.graph_seed,
        "init": cell.init,
        "collision_weight": COLLISION_WEIGHT,
        "diversity_weight": cell.diversity_weight,
        "adjacency_weight": cell.adjacency_weight,
        "adjacency_form": cell.adjacency_form,
        "step_count": int(context.get("step_count", STEP_COUNT)),
        "learning_rate": LEARNING_RATE,
        "checkpoint_steps": list(context.get("checkpoint_steps", CHECKPOINT_STEPS)),
        "host": context.get("host", "laptop"),
        "allow_rung_2": bool(context.get("allow_rung_2", False)),
        "status": status,
        "converged": False,
        "steps_to_convergence": None,
        "drift_onset_step": None,
        "relaxed_objective": None,
        "rounded_cut": None,
        "rounded_cut_minus_relaxed": None,
        "rounded_cut_minus_floor": None,
        "component_count": None,
        "collision_measures": None,
        "labels": None,
        "max_zero_sum_violation": None,
        "final_training_loss": None,
        "checkpoints": [],
        "seconds": 0.0,
        "peak_rss_mb": None,
        "spectral_floor": context.get("spectral_floor"),
        "cluster_count": context.get("cluster_count"),
        "node_count": context.get("node_count"),
        "datum_ratio_cuts": context.get("datum_ratio_cuts"),
        "planted_labels_ratio_cut": context.get("planted_labels_ratio_cut"),
        "cap_seconds": float(context.get("cap_seconds", 0.0)),
        "predicted_rss_gb": predicted_rss_gb(cell.graph.expected_node_count),
        "error": None,
        "engine": context.get("engine"),
    }


def unrun_cell_record(
    cell: CellSpec, context: dict, status: str, message: str
) -> dict[str, object]:
    """A cell the scheduler refused: `over_budget` or `skipped_deadline`, with the reason."""
    record = cell_record_skeleton(cell, context, status)
    record["error"] = message
    return record


# ---------------------------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------------------------


def engine_provenance() -> dict[str, object]:
    """The commit and the three engine files' digests: what produced every number in the file.

    Recomputed once per run and copied into every cell record rather than written once at the top:
    a JSONL grown over several resumed runs has no single header, and a row that cannot name its
    own engine is a row nobody can freeze.
    """
    root = repository_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    digests = {}
    for relative in ENGINE_MODULES:
        path = root / relative
        digests[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        )
    return {"botmaker_commit": commit, "engine_sha256": digests}


# ---------------------------------------------------------------------------------------------
# The rung-0 oracle.
# ---------------------------------------------------------------------------------------------

ORACLE_ROW_FIELDS = (
    "rounded_cut",
    "relaxed_objective",
    "rounded_cut_minus_relaxed",
    "rounded_cut_minus_floor",
    "final_training_loss",
)


def compare_reports(produced: dict, reference: dict) -> tuple[float, list[str]]:
    """(max absolute difference, mismatched exact fields) between a fresh report and the frozen one.

    The numeric fields are compared with a tolerance and the discrete ones — the component count and
    the clustering — exactly, because a partition that differs in one vertex is a different answer
    however close its Ê is.
    """
    worst = 0.0
    mismatched: list[str] = []
    produced_rows, reference_rows = produced["rows"], reference["rows"]
    if len(produced_rows) != len(reference_rows):
        return float("inf"), ["rows"]
    for index, (row, expected) in enumerate(zip(produced_rows, reference_rows, strict=True)):
        for field in ORACLE_ROW_FIELDS:
            worst = max(worst, abs(float(row[field]) - float(expected[field])))
        measures, expected_measures = row["collision_measures"], expected["collision_measures"]
        if len(measures) != len(expected_measures):
            mismatched.append(f"rows[{index}].collision_measures")
        else:
            for value, expected_value in zip(measures, expected_measures, strict=True):
                worst = max(worst, abs(float(value) - float(expected_value)))
        if int(row["component_count"]) != int(expected["component_count"]):
            mismatched.append(f"rows[{index}].component_count")
        if list(row["labels"]) != list(expected["labels"]):
            mismatched.append(f"rows[{index}].labels")
    worst = max(worst, abs(float(produced["spectral_floor"]) - float(reference["spectral_floor"])))
    for name in ROUNDING_NAMES:
        worst = max(
            worst,
            abs(
                float(produced["datum"][name]["ratio_cut"])
                - float(reference["datum"][name]["ratio_cut"])
            ),
        )
    return worst, mismatched


def report_path(name: str, reports_dir: Path | str) -> Path:
    """The frozen report to compare against: the named directory first, the tree's fixture second.

    The returned path is the one the oracle record names, so a reader can always tell which of the
    two a verdict was reached against — a laptop run and a cloud run compare the same numbers, but
    not the same file, and a record that hid that would make the two runs indistinguishable.
    """
    preferred = Path(reports_dir) / f"{name}.json"
    if preferred.exists():
        return preferred
    return FIXTURE_REPORTS_DIR / f"two_hot_span_{name}_report.json"


def oracle_report_check(name: str, reports_dir: Path) -> dict[str, object]:
    """Re-run the harness on one prototype graph and compare it with the frozen report."""
    from mllib.ml.projects.two_hot_span_harness import report_to_dict, run_graph

    started = time.monotonic()
    instances = {instance.name: instance for instance in default_test_graphs()}
    instances["roach_g20"] = roach_g20_instance()
    path = report_path(name, reports_dir)
    if not path.exists():
        return {
            "record": "oracle",
            "schema_version": SCHEMA_VERSION,
            "check": name,
            "passed": False,
            "max_abs_difference": None,
            "mismatched_fields": ["report file is missing"],
            "checkpoints": [],
            "report_path": str(path),
            "seconds": float(time.monotonic() - started),
        }
    produced = report_to_dict(run_graph(instances[name]))
    with path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    worst, mismatched = compare_reports(produced, reference)
    return {
        "record": "oracle",
        "schema_version": SCHEMA_VERSION,
        "check": name,
        "passed": bool(worst <= ORACLE_TOLERANCE and not mismatched),
        "max_abs_difference": float(worst),
        "mismatched_fields": mismatched,
        "checkpoints": [],
        "report_path": str(path),
        "seconds": float(time.monotonic() - started),
    }


def oracle_tuned_cell_check() -> dict[str, object]:
    """The prototype's one tuned cell, re-run through the ladder's own recorder.

    The report checks say the engine is unchanged; this says the *runner* reaches the same place,
    with the ladder's step budget, its coupling and its checkpoints. Ê = 4/15 at K = 2 components,
    at all three checkpoints, is the prototype's headline result.
    """
    from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanConfig, fit_two_hot_span

    started = time.monotonic()
    instance = next(instance for instance in default_test_graphs() if instance.name == "roach_g5")
    X = incidence_matrix(instance.graph)
    recorder = CheckpointRecorder(X, CHECKPOINT_STEPS)
    config = TwoHotSpanConfig(
        step_count=STEP_COUNT,
        learning_rate=LEARNING_RATE,
        collision_weight=COLLISION_WEIGHT,
        adjacency_weight=0.3,
        adjacency_form="edge_product",
        diversity_weight=10.0,
        seed=0,
    )
    run = fit_two_hot_span(X, instance.cluster_count, config, None, recorder=recorder)
    mismatched = [
        f"checkpoint {checkpoint['step']} component_count"
        for checkpoint in recorder.checkpoints
        if int(checkpoint["component_count"]) != instance.cluster_count
    ]
    difference = abs(float(run.rounded_cut) - TUNED_ROUNDED_CUT)
    if len(recorder.checkpoints) != len(CHECKPOINT_STEPS):
        mismatched.append("checkpoint count")
    return {
        "record": "oracle",
        "schema_version": SCHEMA_VERSION,
        "check": TUNED_CELL_CHECK,
        "passed": bool(difference <= ORACLE_TOLERANCE and not mismatched),
        "max_abs_difference": float(difference),
        "mismatched_fields": mismatched,
        "checkpoints": _json_ready(recorder.checkpoints),
        "report_path": None,
        "seconds": float(time.monotonic() - started),
    }


def run_oracle(reports_dir: Path) -> list[dict[str, object]]:
    """The four rung-0 checks, in process: three frozen reports and the tuned cell."""
    records = [oracle_report_check(name, reports_dir) for name in RUNG0_GRAPHS]
    records.append(oracle_tuned_cell_check())
    return records


def oracle_passed(records: Iterable[dict]) -> bool:
    """Have all four rung-0 checks passed in this results file?

    All four, not the three report checks: a runner that reproduces the frozen reports but no longer
    reaches Ê = 4/15 on the tuned cell has changed something the reports cannot see — the ladder's
    own step budget, coupling or checkpoint path — and its numbers are not the prototype's.
    """
    passed = {
        record.get("check")
        for record in records
        if record.get("record") == "oracle" and record.get("passed")
    }
    return set(ORACLE_CHECKS).issubset(passed)


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------


def read_records(path: Path) -> list[dict]:
    """Every record already in the results file; an absent file is an empty run, not an error."""
    if not Path(path).exists():
        return []
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ResultsWriter:
    """Append-only JSONL plus a progress line, both flushed per record.

    Flushed per record because a run that is killed — by the deadline, by a laptop lid, by the
    scheduler's own memory guard — must leave every finished cell behind it readable.
    """

    def __init__(self, results_path: Path) -> None:
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        stem = self.results_path.stem.removesuffix("_results")
        self.progress_path = self.results_path.with_name(f"{stem}_progress.log")

    def write(self, record: dict) -> None:
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
            handle.flush()

    def progress(self, line: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}"
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")
            handle.flush()
        print(stamped, flush=True)


def _cell_progress_line(record: dict) -> str:
    def number(field: str, spec: str) -> str:
        value = record.get(field)
        return "n/a" if value is None else format(float(value), spec)

    return (
        f"{record['key']} {record['status']} "
        f"Ê={number('rounded_cut', '.6f')} E*={number('relaxed_objective', '.6f')} "
        f"comp={record.get('component_count')} {number('seconds', '.1f')}s "
        f"rss={number('peak_rss_mb', '.0f')}MB"
    )


def _cell_context(cell: CellSpec, summary: dict, arguments) -> dict:
    """What a worker needs that it should not recompute: the graph's datum, the cap, provenance."""
    return {
        "host": arguments.host,
        "allow_rung_2": bool(arguments.allow_rung_2),
        "threads": arguments.threads,
        "step_count": STEP_COUNT,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "data_dir": arguments.data_dir,
        "cap_seconds": cap_seconds(
            int(summary.get("node_count") or cell.graph.expected_node_count),
            STEP_COUNT,
            arguments.cap_multiplier,
        ),
        "spectral_floor": summary.get("spectral_floor"),
        "cluster_count": summary.get("cluster_count"),
        "node_count": summary.get("node_count"),
        "datum_ratio_cuts": summary.get("datum_ratio_cuts"),
        "planted_labels_ratio_cut": summary.get("planted_labels_ratio_cut"),
        "engine": summary.get("engine"),
    }


def _summary_from_graph_record(record: dict, engine: dict) -> dict:
    datum = record.get("datum") or {}
    return {
        "spectral_floor": record.get("spectral_floor"),
        "cluster_count": record.get("cluster_count"),
        "node_count": record.get("node_count"),
        "datum_ratio_cuts": {name: float(values["ratio_cut"]) for name, values in datum.items()}
        or None,
        "planted_labels_ratio_cut": record.get("planted_labels_ratio_cut"),
        "engine": engine,
        "status": record.get("status"),
        "error": record.get("error"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--rung", type=int, action="append", choices=(0, 1, 2, 3), dest="rungs", default=None
    )
    parser.add_argument("--host", choices=("laptop", "cloud"), default="laptop")
    parser.add_argument(
        "--allow-rung-2",
        action="store_true",
        help="run rung 2 on the laptop host; the memory budget still applies unchanged",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-budget-gb", type=float, default=DEFAULT_MEMORY_BUDGET_GB)
    parser.add_argument("--cap-multiplier", type=float, default=DEFAULT_CAP_MULTIPLIER)
    parser.add_argument("--only", default=None, help="comma-separated substrings of cell keys")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run cells already recorded")
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path.home() / "Desktop/BotMaker/two-hot-span-reports",
        help="the prototype's frozen reports; a report absent here falls back to the tree's fixture",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser


def select_cells(cells: Sequence[CellSpec], only: str | None, done: set[str]) -> list[CellSpec]:
    """The cells this invocation will actually consider: filtered by `--only`, minus what is done."""
    chosen = list(cells)
    if only:
        needles = [part.strip() for part in only.split(",") if part.strip()]
        chosen = [cell for cell in chosen if any(needle in cell.key for needle in needles)]
    return [cell for cell in chosen if cell.key not in done]


def main(argv: list[str] | None = None) -> int:
    """Run the ladder: oracle first, then every planned cell, one JSON line per finished cell."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    rungs = sorted(set(arguments.rungs)) if arguments.rungs else [0, 1, 2, 3]
    workers = arguments.workers
    if arguments.host == "laptop":
        workers = min(workers, LAPTOP_WORKER_LIMIT)

    planned = plan_cells(rungs)
    if arguments.dry_run:
        for cell in planned:
            print(cell.key)
        for rung in rungs:
            print(f"rung {rung}: {sum(1 for cell in planned if cell.graph.rung == rung)} cells")
        print(f"total: {len(planned)} cells")
        return 0

    writer = ResultsWriter(arguments.results)
    existing = read_records(arguments.results)
    done = {
        record["key"] for record in existing if record.get("record") == "cell" and "key" in record
    }
    if arguments.force:
        done = set()

    if 0 in rungs:
        for record in run_oracle(arguments.reports_dir):
            writer.write(record)
            existing.append(record)
            writer.progress(
                f"oracle {record['check']}: {'PASS' if record['passed'] else 'FAIL'} "
                f"max|Δ|={record['max_abs_difference']} mismatched={record['mismatched_fields']}"
            )
    if not oracle_passed(existing):
        writer.progress("refusing to run: the rung-0 oracle has not passed in this results file")
        return 1

    cells = select_cells(planned, arguments.only, done)
    writer.progress(f"{len(cells)} cells to run of {len(planned)} planned (host={arguments.host})")
    engine = engine_provenance()
    started = time.monotonic()
    deadline = None if arguments.hours is None else started + arguments.hours * 3600.0
    context = multiprocessing.get_context("spawn")
    summaries: dict[str, dict] = {}
    for record in existing:
        if record.get("record") == "graph":
            key = f"r{record['rung']}|{record['name']}|gs{record['graph_seed']}"
            summaries[key] = _summary_from_graph_record(record, engine)

    runnable: list[tuple[CellSpec, dict]] = []
    for cell in cells:
        rejection = budget_rejection(
            cell, arguments.host, arguments.memory_budget_gb, arguments.allow_rung_2
        )
        if rejection is not None:
            record = unrun_cell_record(
                cell,
                # The engine still belongs on a row that never ran: an `over_budget` rung-2 row is
                # the record that says *this* engine could not be run here, on this host.
                _cell_context(
                    cell,
                    {**summaries.get(cell.graph.graph_key, {}), "engine": engine},
                    arguments,
                ),
                "over_budget",
                rejection,
            )
            writer.write(record)
            writer.progress(f"{cell.key} over_budget: {rejection}")
            continue
        summary = summaries.get(cell.graph.graph_key)
        if summary is None:
            graph_started = time.monotonic()
            try:
                record = graph_record(cell.graph, arguments.data_dir)
            except Exception as error:
                record = failed_graph_record(
                    cell.graph,
                    f"{type(error).__name__}: {error}",
                    time.monotonic() - graph_started,
                )
            writer.write(record)
            writer.progress(f"graph {cell.graph.graph_key}: {record['status']}")
            summary = _summary_from_graph_record(record, engine)
            summaries[cell.graph.graph_key] = summary
        if summary.get("status") == "error":
            record = unrun_cell_record(
                cell, _cell_context(cell, summary, arguments), "error", str(summary.get("error"))
            )
            writer.write(record)
            writer.progress(f"{cell.key} error: {summary.get('error')}")
            continue
        runnable.append((cell, _cell_context(cell, summary, arguments)))

    _schedule(runnable, writer, workers, arguments, context, deadline)
    writer.progress(f"done in {time.monotonic() - started:.1f}s")
    return 0


def _schedule(
    runnable: list[tuple[CellSpec, dict]],
    writer: ResultsWriter,
    workers: int,
    arguments,
    context,
    deadline: float | None,
) -> None:
    """Submit cells while the memory budget and the worker count allow, reaping as they finish.

    The pool is only ever handed as many tasks as may run at once, so "submitted" means "started"
    and the running total the budget is measured against is the set of live processes. A cell that
    does not fit beside what is running waits rather than being refused — `budget_rejection` has
    already turned the cells that could *never* fit into `over_budget` records. Anything still
    pending when `--hours` runs out is recorded `skipped_deadline` rather than dropped.
    """
    if not runnable:
        return
    pending = list(runnable)
    slots = max(workers, 1)
    pool = _new_pool(context, slots, arguments.threads)
    try:
        inflight: list[tuple[object, CellSpec, dict, float, float]] = []
        while pending or inflight:
            expired = deadline is not None and time.monotonic() > deadline
            if not expired:
                _submit(pool, pending, inflight, slots, arguments.memory_budget_gb)
            if expired and pending:
                for cell, cell_context in pending:
                    record = unrun_cell_record(
                        cell, cell_context, "skipped_deadline", "the --hours guard stopped the run"
                    )
                    writer.write(record)
                    writer.progress(f"{cell.key} skipped_deadline")
                pending = []
            still: list[tuple[object, CellSpec, dict, float, float]] = []
            stranded: list[tuple[object, CellSpec, dict, float, float]] = []
            for entry in inflight:
                handle, cell, cell_context, _, submitted = entry
                if not handle.ready():
                    grace = float(cell_context["cap_seconds"]) + WORKER_GRACE_SECONDS
                    (stranded if time.monotonic() - submitted > grace else still).append(entry)
                    continue
                try:
                    record = handle.get()
                except Exception as error:
                    record = unrun_cell_record(
                        cell, cell_context, "error", f"{type(error).__name__}: {error}"
                    )
                writer.write(record)
                writer.progress(_cell_progress_line(record))
            reaped = len(inflight) - len(still) - len(stranded)
            inflight = still
            if stranded:
                for _, cell, cell_context, _, _ in stranded:
                    record = unrun_cell_record(cell, cell_context, "timeout", WORKER_LOST_MESSAGE)
                    writer.write(record)
                    writer.progress(_cell_progress_line(record))
                # Terminating takes the *other* live workers with it, and those cells have no record
                # yet, so they go back to the head of the queue rather than being lost with the pool.
                pending = [
                    (cell, cell_context) for _, cell, cell_context, _, _ in inflight
                ] + pending
                inflight = []
                pool.terminate()
                pool = _new_pool(context, slots, arguments.threads)
                writer.progress(f"pool rebuilt after {len(stranded)} lost worker(s)")
            if inflight and reaped == 0:
                time.sleep(0.2)
    finally:
        pool.terminate()


def _new_pool(context, slots: int, threads: int):
    """A fresh spawn pool, one cell per process so every cell reports its own peak RSS."""
    return context.Pool(
        processes=slots,
        maxtasksperchild=1,
        initializer=worker_initializer,
        initargs=(threads,),
    )


def _submit(pool, pending: list, inflight: list, slots: int, budget_gb: float) -> None:
    """Start as many pending cells as the worker count and the memory budget allow.

    The queue is scanned rather than blocked on: a 2000-node cell that does not fit beside what is
    running must not hold back the 100-node cells behind it, or a budget meant to protect the
    machine would quietly serialise the whole grid behind its largest cell.
    """
    running_gb = sum(entry[3] for entry in inflight)
    while pending and len(inflight) < slots:
        chosen = None
        for index, (cell, _) in enumerate(pending):
            cell_gb = predicted_rss_gb(cell.graph.expected_node_count)
            if not inflight or admits_cell(running_gb, cell_gb, budget_gb):
                chosen = (index, cell_gb)
                break
        if chosen is None:
            return
        index, cell_gb = chosen
        cell, cell_context = pending.pop(index)
        handle = pool.apply_async(run_cell, (cell, cell_context))
        inflight.append((handle, cell, cell_context, cell_gb, time.monotonic()))
        running_gb += cell_gb


if __name__ == "__main__":
    raise SystemExit(main())
