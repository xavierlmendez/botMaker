"""The two-hot span rcut harness: what the relaxation buys, measured against a datum.

Slice 2.3 of `docs/plans/2026-09-two-hot-span.md`, with a fifth graph added in slice 2.6. For each
of the five test graphs the harness sweeps the collision weight λ across two initialisations, runs
the torch optimizer of slice 2.2, and reports one JSON file and one text block per graph.

**The judge.** Ê, the rounded cut, is the headline number: it is the RatioCut of the partition the
run actually produces, and it is the only number a comparison may be made on. E\\*, the relaxed
objective, and Σλ, the spectral floor, are *both* reported beside it — E\\* alone flatters the
relaxation (it is a lower bound on nothing the user wants) and Σλ says how much of the gap is
inherent to r = n - K zero-sum vectors rather than to the optimiser. R(v_j) is a per-column
diagnostic of how nearly 2-hot each column is and is **never** the criterion: a run can drive mean
R to 0.49 and still return the wrong partition, because being 2-hot is a per-column property and
nothing couples the columns.

**The datum.** The comparison is against the λ = 0 member of the same family: the spectral
relaxation of the unnormalized Laplacian — the K eigenvectors of its K smallest eigenvalues —
rounded three ways (Lloyd, Yu & Shi's discretisation, Damle-Minden-Ying's cluster QR), on the same
graphs, in the same report. That is the point of the datum: it is not a different method brought in
to lose, it is what this objective already reduces to when the collision reward is switched off.
Hand-written numpy throughout — the library imports no sklearn model class, and `KMeans` is one; the
sklearn implementations appear only as oracles inside the tests.

VᵀV = I is never imposed, anywhere. This is the gradient track only; the alternating analytical
solver is parked. rcut only — ncut is BL-41. Every reported number comes back through the exact
numpy `pinv` projector of `two_hot_span_problem` (D-31, P-2); the training ridge never reaches a
report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx
import numpy as np

from mllib.math.graph.two_hot_span_problem import (
    BRUTE_FORCE_NODE_LIMIT,
    GraphInstance,
    brute_force_rcut,
    default_test_graphs,
    incidence_matrix,
    laplacian_matrix,
    node_count,
    ratio_cut,
    spanning_vector_count,
    spectral_floor,
    spectral_spanning_set,
)

DEFAULT_COLLISION_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_STEP_COUNT = 300
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_INITS = ("random", "spectral")
DEFAULT_EPSILON = 1e-6
DEFAULT_SEED = 0
# Slice 2.4's experiment, passed straight through to `TwoHotSpanConfig`. These are its defaults, so
# a harness run that names none of them is the run slice 2.3 shipped, and the objective stays the
# brief's §20 objective.
DEFAULT_SCHEDULE = "constant"
DEFAULT_WARMUP_STEPS = 0
DEFAULT_FINAL_LR_FRACTION = 0.0
DEFAULT_ADJACENCY_WEIGHT = 0.0
DEFAULT_ADJACENCY_FORM = "laplacian"
# Slice 2.5's experiment, same pass-through and the same default-off contract.
DEFAULT_DIVERSITY_WEIGHT = 0.0

# An invariant that fails by less than this is arithmetic, not a wrong answer.
INVARIANT_TOLERANCE = 1e-10

_MAX_LLOYD_ITERATIONS = 300
_LLOYD_RESTARTS = 10
_DISCRETIZE_ITERATIONS = 20
_DISCRETIZE_TOLERANCE = 1e-7

ROUNDING_NAMES = ("kmeans", "discretize", "cluster_qr")


# --------------------------------------------------------------------------------------------
# The datum: the spectral relaxation, rounded three ways.
# --------------------------------------------------------------------------------------------


def _canonical(labels: np.ndarray) -> np.ndarray:
    """Relabel in order of first appearance so a labelling has one representation."""
    _, first_index = np.unique(labels, return_index=True)
    order = labels[np.sort(first_index)]
    remap = {value: position for position, value in enumerate(order)}
    return np.array([remap[value] for value in labels], dtype=int)


def spectral_embedding(graph: nx.Graph, cluster_count: int) -> np.ndarray:
    """n x K eigenvectors of the K smallest eigenvalues of L, ascending, sign-normalised."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix(graph))
    embedding = eigenvectors[:, np.argsort(eigenvalues, kind="stable")[:cluster_count]]
    dominant = np.argmax(np.abs(embedding), axis=0)
    signs = np.sign(embedding[dominant, np.arange(embedding.shape[1])])
    return embedding * np.where(signs == 0.0, 1.0, signs)


def _kmeans_plus_plus(embedding: np.ndarray, cluster_count: int, rng) -> np.ndarray:
    """k-means++ seeding: first centre uniform, the rest with probability proportional to D^2."""
    centres = [embedding[rng.integers(embedding.shape[0])]]
    for _ in range(cluster_count - 1):
        distances = np.min(
            ((embedding[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        total = distances.sum()
        weights = (
            np.full(embedding.shape[0], 1.0 / embedding.shape[0])
            if total <= 0
            else distances / total
        )
        centres.append(embedding[rng.choice(embedding.shape[0], p=weights)])
    return np.array(centres)


def _lloyd(embedding: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, float]:
    """Lloyd's algorithm to convergence; empty clusters are re-seeded to the farthest point."""
    assignments = None
    for _ in range(_MAX_LLOYD_ITERATIONS):
        squared = ((embedding[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        updated = squared.argmin(axis=1)
        if assignments is not None and np.array_equal(updated, assignments):
            break
        assignments = updated
        for centre in range(centres.shape[0]):
            members = embedding[assignments == centre]
            if members.size:
                centres[centre] = members.mean(axis=0)
            else:
                centres[centre] = embedding[squared.min(axis=1).argmax()]
    squared = ((embedding[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return squared.argmin(axis=1), float(squared.min(axis=1).sum())


def round_kmeans(embedding: np.ndarray, cluster_count: int, seed: int) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding, 10 restarts, lowest inertia wins."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(_LLOYD_RESTARTS):
        labels, inertia = _lloyd(embedding, _kmeans_plus_plus(embedding, cluster_count, rng))
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return _canonical(best_labels)


def _discretization_rotation_init(rows: np.ndarray, cluster_count: int, rng) -> np.ndarray:
    """Yu & Shi initialisation: a random row, then the rows most orthogonal to the picks so far."""
    rotation = np.zeros((cluster_count, cluster_count))
    rotation[:, 0] = rows[rng.integers(rows.shape[0])]
    accumulated = np.zeros(rows.shape[0])
    for column in range(1, cluster_count):
        accumulated += np.abs(rows @ rotation[:, column - 1])
        rotation[:, column] = rows[accumulated.argmin()]
    return rotation


def round_discretize(embedding: np.ndarray, cluster_count: int, seed: int) -> np.ndarray:
    """Yu & Shi 2003 multiclass spectral clustering: alternate argmax discretisation and SVD."""
    rng = np.random.default_rng(seed)
    rows = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    rotation = _discretization_rotation_init(rows, cluster_count, rng)
    labels = np.zeros(rows.shape[0], dtype=int)
    previous = 0.0
    for _ in range(_DISCRETIZE_ITERATIONS + 1):
        labels = (rows @ rotation).argmax(axis=1)
        indicator = np.zeros((rows.shape[0], cluster_count))
        indicator[np.arange(rows.shape[0]), labels] = 1.0
        left, singular_values, right = np.linalg.svd(indicator.T @ rows)
        objective = float(singular_values.sum())
        if abs(objective - previous) <= _DISCRETIZE_TOLERANCE * max(abs(objective), 1.0):
            break
        previous = objective
        rotation = right.T @ left.T
    return _canonical(labels)


def _pivoted_qr_columns(matrix: np.ndarray, pivot_count: int) -> np.ndarray:
    """Column-pivoted modified Gram-Schmidt: the first pivot_count pivot column indices."""
    residual = matrix.astype(float).copy()
    norms = (residual**2).sum(axis=0)
    pivots = []
    for _ in range(pivot_count):
        pivot = int(np.argmax(norms))
        pivots.append(pivot)
        direction = residual[:, pivot] / np.linalg.norm(residual[:, pivot])
        residual -= np.outer(direction, direction @ residual)
        norms = (residual**2).sum(axis=0)
        norms[pivots] = -np.inf
    return np.array(pivots, dtype=int)


def round_cluster_qr(embedding: np.ndarray, cluster_count: int) -> np.ndarray:
    """Damle-Minden-Ying 2019: pivoted QR of the embedding transpose, SVD rotation, argmax |.|."""
    pivots = _pivoted_qr_columns(embedding.T, cluster_count)
    left, _, right = np.linalg.svd(embedding[pivots, :].T)
    return _canonical(np.abs(embedding @ (left @ right)).argmax(axis=1))


def datum_roundings(graph: nx.Graph, cluster_count: int, seed: int) -> dict[str, np.ndarray]:
    """The spectral embedding rounded by all three methods."""
    embedding = spectral_embedding(graph, cluster_count)
    return {
        "kmeans": round_kmeans(embedding, cluster_count, seed),
        "discretize": round_discretize(embedding, cluster_count, seed),
        "cluster_qr": round_cluster_qr(embedding, cluster_count),
    }


# --------------------------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One (λ, init) cell of the sweep. Ê is the headline; the rest is context for reading it.

    ``final_training_loss`` is the last value of the *ridge* training loss
    ``E_ridge(V) - λ Σ_j R(v_j)``. It is a training diagnostic and is never E: it is computed
    through the regularized projector and it carries the collision reward, so it is neither
    comparable to E\\* nor to Ê. The field is named for what it is and must stay so named.
    """

    collision_weight: float
    init: str
    relaxed_objective: float
    rounded_cut: float
    rounded_cut_minus_relaxed: float
    rounded_cut_minus_floor: float
    component_count: int
    collision_measures: tuple[float, ...]
    labels: tuple[int, ...]
    max_zero_sum_violation: float
    final_training_loss: float


@dataclass(frozen=True, slots=True)
class GraphReport:
    """Everything measured on one graph: the sweep, the datum, and the references to read against."""

    name: str
    node_count: int
    edge_count: int
    cluster_count: int
    spanning_vector_count: int
    spectral_floor: float
    rows: tuple[SweepRow, ...]
    best_row_index: int
    datum: dict[str, dict[str, object]]
    best_datum_name: str
    brute_force_optimum: float | None
    brute_force_labels: tuple[int, ...] | None
    planted_labels_ratio_cut: float | None
    single_weight_gives_k_components: dict[float, bool]
    config: dict[str, object]


def _initial_spanning_set(graph: nx.Graph, cluster_count: int, init: str) -> np.ndarray | None:
    """The starting V for an init name; ``None`` means the optimizer's own seeded randn."""
    if init == "random":
        return None
    if init == "spectral":
        return spectral_spanning_set(graph, cluster_count)
    raise ValueError(f"unknown init {init!r}; expected 'random' or 'spectral'")


def run_graph(
    instance: GraphInstance,
    *,
    collision_weights: tuple[float, ...] = DEFAULT_COLLISION_WEIGHTS,
    inits: tuple[str, ...] = DEFAULT_INITS,
    step_count: int = DEFAULT_STEP_COUNT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon: float = DEFAULT_EPSILON,
    seed: int = DEFAULT_SEED,
    learning_rate_schedule: str = DEFAULT_SCHEDULE,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    final_learning_rate_fraction: float = DEFAULT_FINAL_LR_FRACTION,
    adjacency_weight: float = DEFAULT_ADJACENCY_WEIGHT,
    adjacency_form: str = DEFAULT_ADJACENCY_FORM,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
) -> GraphReport:
    """Sweep λ x init on one graph and assemble the report, checking every invariant on the way."""
    # torch is an optional group (D-31); the datum half of this module runs without it.
    from mllib.math.algorithms.two_hot_span_optimizer import TwoHotSpanConfig, fit_two_hot_span

    graph = instance.graph
    cluster_count = instance.cluster_count
    count = node_count(graph)
    vector_count = spanning_vector_count(count, cluster_count)
    X = incidence_matrix(graph)
    floor = spectral_floor(graph, cluster_count)

    optimum: float | None = None
    optimum_labels: tuple[int, ...] | None = None
    if count <= BRUTE_FORCE_NODE_LIMIT:
        best_value, best_labels = brute_force_rcut(graph, cluster_count)
        optimum, optimum_labels = best_value, tuple(int(label) for label in best_labels)

    # One config for the whole sweep, with only λ replaced per row: the recorded config is then the
    # object the rows actually ran under, not a second one built to look like it.
    base_config = TwoHotSpanConfig(
        step_count=step_count,
        learning_rate=learning_rate,
        epsilon=epsilon,
        seed=seed,
        learning_rate_schedule=learning_rate_schedule,
        warmup_steps=warmup_steps,
        final_learning_rate_fraction=final_learning_rate_fraction,
        adjacency_weight=adjacency_weight,
        adjacency_form=adjacency_form,
        diversity_weight=diversity_weight,
    )

    rows: list[SweepRow] = []
    for collision_weight in collision_weights:
        for init in inits:
            config = replace(base_config, collision_weight=float(collision_weight))
            run = fit_two_hot_span(
                X, cluster_count, config, _initial_spanning_set(graph, cluster_count, init)
            )
            _check_row_invariants(run.rounded_cut, run.relaxed_objective, floor, optimum)
            rows.append(
                SweepRow(
                    collision_weight=float(collision_weight),
                    init=init,
                    relaxed_objective=float(run.relaxed_objective),
                    rounded_cut=float(run.rounded_cut),
                    rounded_cut_minus_relaxed=float(run.rounded_cut - run.relaxed_objective),
                    rounded_cut_minus_floor=float(run.rounded_cut - floor),
                    component_count=int(np.unique(run.labels).size),
                    collision_measures=tuple(float(value) for value in run.collision_measures),
                    labels=tuple(int(label) for label in run.labels),
                    max_zero_sum_violation=float(run.max_zero_sum_violation),
                    final_training_loss=float(run.loss_history[-1])
                    if run.loss_history.size
                    else 0.0,
                )
            )

    # Ties go to the first row, so the earliest λ (and within it the earliest init) wins.
    best_row_index = int(min(range(len(rows)), key=lambda index: rows[index].rounded_cut))

    datum: dict[str, dict[str, object]] = {}
    for name, labels in datum_roundings(graph, cluster_count, seed).items():
        datum[name] = {
            "ratio_cut": float(ratio_cut(graph, labels)),
            "labels": tuple(int(label) for label in labels),
        }
    best_datum_name = min(datum, key=lambda name: float(datum[name]["ratio_cut"]))

    return GraphReport(
        name=instance.name,
        node_count=count,
        edge_count=int(graph.number_of_edges()),
        cluster_count=cluster_count,
        spanning_vector_count=vector_count,
        spectral_floor=float(floor),
        rows=tuple(rows),
        best_row_index=best_row_index,
        datum=datum,
        best_datum_name=best_datum_name,
        brute_force_optimum=optimum,
        brute_force_labels=optimum_labels,
        planted_labels_ratio_cut=(
            None
            if instance.planted_labels is None
            else float(ratio_cut(graph, instance.planted_labels))
        ),
        single_weight_gives_k_components=_single_weight_gives_k_components(
            rows, collision_weights, cluster_count
        ),
        config={
            "step_count": base_config.step_count,
            "learning_rate": base_config.learning_rate,
            "epsilon": base_config.epsilon,
            "seed": base_config.seed,
            "normalize_columns": base_config.normalize_columns,
            "learning_rate_schedule": base_config.learning_rate_schedule,
            "warmup_steps": base_config.warmup_steps,
            "final_learning_rate_fraction": base_config.final_learning_rate_fraction,
            "adjacency_weight": base_config.adjacency_weight,
            "adjacency_form": base_config.adjacency_form,
            "diversity_weight": base_config.diversity_weight,
            "collision_weights": [float(weight) for weight in collision_weights],
            "inits": list(inits),
        },
    )


def _check_row_invariants(
    rounded: float, relaxed: float, floor: float, optimum: float | None
) -> None:
    """Ê ≥ Σλ, E\\* ≥ Σλ always, and Ê ≥ the brute-force optimum when there is one. A stop."""
    if relaxed < floor - INVARIANT_TOLERANCE:
        raise AssertionError(f"E* = {relaxed!r} is below the spectral floor {floor!r}")
    if rounded < floor - INVARIANT_TOLERANCE:
        raise AssertionError(f"rounded cut {rounded!r} is below the spectral floor {floor!r}")
    if optimum is not None and rounded < optimum - INVARIANT_TOLERANCE:
        raise AssertionError(
            f"rounded cut {rounded!r} is below the brute-force RatioCut optimum {optimum!r}"
        )


def _single_weight_gives_k_components(
    rows: list[SweepRow], collision_weights: tuple[float, ...], cluster_count: int
) -> dict[float, bool]:
    """Per λ: does the best init at that λ round to exactly K components? (The BL-41 question.)

    BL-41 opens when "a single λ gives exactly K components on at least three of the four *original*
    test graphs" — `roach_g5`, `karate`, `planted_partition`, `two_moons_knn`. `roach_g20` (slice
    2.6, K = 3, He-Gu-Zhang 2012) is an added instance reported alongside and outside that
    criterion: the criterion was set against those four and does not move because a fifth graph
    arrived. This function is unchanged by that — it answers the question per graph, and it is a
    reader who applies the three-of-four count across the reports.

    The tie-break to the *lowest-Ê* init is deliberate rather than incidental: Ê is the
    headline, so the row that wins on Ê is the clustering the report puts forward for that λ. An
    init that happened to land on K components but lost on Ê is not the reported clustering, and
    counting it here would answer a question about the sweep instead of about the answer.
    """
    answer: dict[float, bool] = {}
    for collision_weight in collision_weights:
        at_weight = [row for row in rows if row.collision_weight == float(collision_weight)]
        if not at_weight:
            continue
        best = min(at_weight, key=lambda row: row.rounded_cut)
        answer[float(collision_weight)] = best.component_count == cluster_count
    return answer


# --------------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------------


def report_to_dict(report: GraphReport) -> dict:
    """The report as plain JSON types: floats as floats, labels as lists, λ keys as strings."""
    return {
        "name": report.name,
        "node_count": int(report.node_count),
        "edge_count": int(report.edge_count),
        "cluster_count": int(report.cluster_count),
        "spanning_vector_count": int(report.spanning_vector_count),
        "spectral_floor": float(report.spectral_floor),
        "best_row_index": int(report.best_row_index),
        "rows": [
            {
                "collision_weight": float(row.collision_weight),
                "init": row.init,
                "relaxed_objective": float(row.relaxed_objective),
                "rounded_cut": float(row.rounded_cut),
                "rounded_cut_minus_relaxed": float(row.rounded_cut_minus_relaxed),
                "rounded_cut_minus_floor": float(row.rounded_cut_minus_floor),
                "component_count": int(row.component_count),
                "collision_measures": [float(value) for value in row.collision_measures],
                "labels": [int(label) for label in row.labels],
                "max_zero_sum_violation": float(row.max_zero_sum_violation),
                "final_training_loss": float(row.final_training_loss),
            }
            for row in report.rows
        ],
        "datum": {
            name: {
                "ratio_cut": float(values["ratio_cut"]),
                "labels": [int(label) for label in values["labels"]],  # type: ignore[union-attr]
            }
            for name, values in report.datum.items()
        },
        "best_datum_name": report.best_datum_name,
        "brute_force_optimum": (
            None if report.brute_force_optimum is None else float(report.brute_force_optimum)
        ),
        "brute_force_labels": (
            None
            if report.brute_force_labels is None
            else [int(label) for label in report.brute_force_labels]
        ),
        "planted_labels_ratio_cut": (
            None
            if report.planted_labels_ratio_cut is None
            else float(report.planted_labels_ratio_cut)
        ),
        "single_weight_gives_k_components": {
            # JSON object keys are strings; the λ they name is the float in the sweep rows.
            repr(float(weight)): bool(value)
            for weight, value in report.single_weight_gives_k_components.items()
        },
        "config": dict(report.config),
    }


def write_report(report: GraphReport, output_dir: Path) -> Path:
    """Write ``<output_dir>/<name>.json`` and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report_to_dict(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_report(report: GraphReport) -> str:
    """One header block, one line per sweep row, then the best row's per-column R and labels."""
    datum_text = "  ".join(
        f"{name}={float(report.datum[name]['ratio_cut']):.6f}"
        for name in ROUNDING_NAMES
        if name in report.datum
    )
    lines = [
        f"{report.name}: n={report.node_count} m={report.edge_count} "
        f"K={report.cluster_count} r={report.spanning_vector_count}  "
        f"Σλ (floor)={report.spectral_floor:.6f}  "
        f"brute force={_optional(report.brute_force_optimum)}  "
        f"planted RatioCut={_optional(report.planted_labels_ratio_cut)}",
        f"  datum RatioCut: {datum_text}  (best: {report.best_datum_name})",
        f"  {'':1} {'λ':>6} {'init':<9} {'E*':>12} {'Ê':>12} {'Ê-E*':>12} {'Ê-Σλ':>12} "
        f"{'comp':>5}  R min/mean/max",
    ]
    for index, row in enumerate(report.rows):
        measures = np.asarray(row.collision_measures, dtype=float)
        marker = "*" if index == report.best_row_index else " "
        lines.append(
            f"  {marker} {row.collision_weight:>6.2f} {row.init:<9} "
            f"{row.relaxed_objective:>12.6f} {row.rounded_cut:>12.6f} "
            f"{row.rounded_cut_minus_relaxed:>12.6f} {row.rounded_cut_minus_floor:>12.6f} "
            f"{row.component_count:>5}  "
            f"{measures.min():.4f}/{measures.mean():.4f}/{measures.max():.4f}"
        )
    best = report.rows[report.best_row_index]
    lines.append(
        f"  best row (* λ={best.collision_weight:g} {best.init}) R(v_j): "
        + " ".join(f"{value:.4f}" for value in best.collision_measures)
    )
    lines.append(f"  best row labels: {list(best.labels)}")
    lines.append(
        "  exactly K components at λ: "
        + " ".join(
            f"{weight:g}={'yes' if value else 'no'}"
            for weight, value in report.single_weight_gives_k_components.items()
        )
    )
    return "\n".join(lines)


def run_default_suite(output_dir: Path, **kwargs) -> list[GraphReport]:
    """Run every default test graph and write one JSON report each."""
    reports = []
    for instance in default_test_graphs():
        report = run_graph(instance, **kwargs)
        write_report(report, Path(output_dir))
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> None:
    """Run the sweep over the requested graphs, write the JSON reports and print each block."""
    # Same reason as `run_graph`'s import: the allowed schedule and form names live beside the
    # config that validates against them, in the torch module (D-31), and only the CLI needs them.
    from mllib.math.algorithms.two_hot_span_optimizer import (
        ADJACENCY_FORMS,
        LEARNING_RATE_SCHEDULES,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEP_COUNT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--graphs", nargs="+", default=None, help="subset of graph names to run")
    # The experimental knobs of slices 2.4 and 2.5. Defaults are the config's own, so omitting them
    # all leaves the sweep exactly as slice 2.3 ran it.
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, choices=LEARNING_RATE_SCHEDULES)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--final-lr-fraction", type=float, default=DEFAULT_FINAL_LR_FRACTION)
    parser.add_argument("--adjacency-weight", type=float, default=DEFAULT_ADJACENCY_WEIGHT)
    parser.add_argument("--adjacency-form", default=DEFAULT_ADJACENCY_FORM, choices=ADJACENCY_FORMS)
    parser.add_argument("--diversity-weight", type=float, default=DEFAULT_DIVERSITY_WEIGHT)
    arguments = parser.parse_args(argv)

    instances = default_test_graphs()
    if arguments.graphs is not None:
        known = {instance.name for instance in instances}
        unknown = set(arguments.graphs) - known
        if unknown:
            parser.error(f"unknown graph name(s): {sorted(unknown)}; known: {sorted(known)}")
        instances = tuple(
            instance for instance in instances if instance.name in set(arguments.graphs)
        )

    for instance in instances:
        report = run_graph(
            instance,
            step_count=arguments.steps,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            learning_rate_schedule=arguments.schedule,
            warmup_steps=arguments.warmup_steps,
            final_learning_rate_fraction=arguments.final_lr_fraction,
            adjacency_weight=arguments.adjacency_weight,
            adjacency_form=arguments.adjacency_form,
            diversity_weight=arguments.diversity_weight,
        )
        write_report(report, arguments.output_dir)
        print(format_report(report))


if __name__ == "__main__":
    main()
