"""The sweep half of slice 2.3: the report a run produces, and what it must never omit.

torch is an optional dependency group (D-31), so the whole file skips without it; the datum tests
live in `test_two_hot_span_datum.py` and run either way. The configurations here are deliberately
tiny (20 steps, two λ, one init) — the assertions are on the shape and the invariants of a report,
not on the numbers a longer run would reach.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.math.graph.two_hot_span_problem import (
    GraphInstance,
    ratio_cut,
    roach_graph,
)
from mllib.ml.projects.two_hot_span_harness import (
    DEFAULT_COLLISION_WEIGHTS,
    DEFAULT_INITS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STEP_COUNT,
    ROUNDING_NAMES,
    format_report,
    main,
    report_to_dict,
    run_default_suite,
    run_graph,
    write_report,
)

SMALL_WEIGHTS = (0.0, 1.0)
SMALL_INITS = ("spectral",)
SMALL_STEPS = 20


def seeded_graph(node_count: int, seed: int) -> nx.Graph:
    """A seeded connected Erdos-Renyi graph, unweighted."""
    for attempt in range(100):
        graph = nx.gnp_random_graph(node_count, 0.45, seed=seed * 100 + attempt)
        if nx.is_connected(graph):
            return graph
    raise AssertionError("no connected graph found")


def small_report(instance: GraphInstance):
    return run_graph(
        instance,
        collision_weights=SMALL_WEIGHTS,
        inits=SMALL_INITS,
        step_count=SMALL_STEPS,
    )


@pytest.fixture(scope="module")
def roach_report():
    return small_report(GraphInstance("roach_g5", roach_graph(5), 2))


@pytest.fixture(scope="module")
def tiny_report():
    return small_report(GraphInstance("tiny", seeded_graph(8, seed=1), 2))


def test_the_report_names_the_graph_it_actually_solved(roach_report):
    assert roach_report.name == "roach_g5"
    assert (roach_report.node_count, roach_report.edge_count) == (20, 23)
    assert roach_report.cluster_count == 2
    assert roach_report.spanning_vector_count == 18


def test_every_row_carries_the_three_objectives_and_both_differences(roach_report):
    for row in roach_report.rows:
        assert row.rounded_cut_minus_relaxed == pytest.approx(
            row.rounded_cut - row.relaxed_objective
        )
        assert row.rounded_cut_minus_floor == pytest.approx(
            row.rounded_cut - roach_report.spectral_floor
        )


def test_no_row_falls_below_the_spectral_floor(roach_report):
    for row in roach_report.rows:
        assert row.rounded_cut >= roach_report.spectral_floor - 1e-10
        assert row.relaxed_objective >= roach_report.spectral_floor - 1e-10


def test_the_collision_measure_is_reported_per_column(roach_report):
    for row in roach_report.rows:
        assert len(row.collision_measures) == roach_report.spanning_vector_count
        # R(v) <= 1/2 for a zero-sum column, with equality only at a 2-hot one.
        assert all(0.0 <= value <= 0.5 + 1e-12 for value in row.collision_measures)


def test_every_row_labels_every_node_and_counts_its_own_components(roach_report):
    for row in roach_report.rows:
        assert len(row.labels) == roach_report.node_count
        assert row.component_count == len(set(row.labels))


def test_the_rounded_cut_of_every_row_is_the_ratio_cut_of_its_own_labels(roach_report):
    """The identity that makes the whole report readable: Ê *is* RatioCut(the reported clustering).

    Ê is computed as a projector residual through pinv, RatioCut by summing cut weights over the
    blocks — two different routes to the same number. If they ever parted, every Ê in the table
    would stop being a cut and start being an unlabelled residual.
    """
    graph = roach_graph(5)
    for row in roach_report.rows:
        assert row.rounded_cut == pytest.approx(ratio_cut(graph, np.asarray(row.labels)), abs=1e-10)


def test_the_sweep_covers_every_requested_weight_and_init(roach_report):
    assert {row.collision_weight for row in roach_report.rows} == set(SMALL_WEIGHTS)
    assert {row.init for row in roach_report.rows} == set(SMALL_INITS)
    assert len(roach_report.rows) == len(SMALL_WEIGHTS) * len(SMALL_INITS)


def test_the_datum_is_the_same_three_roundings_on_the_same_graph(roach_report):
    assert set(roach_report.datum) == set(ROUNDING_NAMES) == {"kmeans", "discretize", "cluster_qr"}
    for values in roach_report.datum.values():
        assert isinstance(values["ratio_cut"], float)
        assert len(values["labels"]) == roach_report.node_count
    best = roach_report.datum[roach_report.best_datum_name]["ratio_cut"]
    assert best == min(values["ratio_cut"] for values in roach_report.datum.values())


def test_the_spectral_floor_is_reported_and_is_the_eigenvalue_sum(roach_report):
    eigenvalues = np.linalg.eigvalsh(nx.laplacian_matrix(roach_graph(5)).toarray().astype(float))
    assert roach_report.spectral_floor == pytest.approx(float(eigenvalues[:2].sum()), abs=1e-10)


def test_the_best_row_is_the_one_with_the_smallest_rounded_cut(roach_report):
    best = roach_report.rows[roach_report.best_row_index]
    assert best.rounded_cut == min(row.rounded_cut for row in roach_report.rows)


def test_a_tie_on_the_rounded_cut_goes_to_the_first_row(roach_report):
    values = [row.rounded_cut for row in roach_report.rows]
    assert roach_report.best_row_index == values.index(min(values))


def test_the_per_weight_component_answer_covers_every_weight(roach_report):
    answer = roach_report.single_weight_gives_k_components
    assert set(answer) == set(SMALL_WEIGHTS)
    assert all(isinstance(value, bool) for value in answer.values())


def test_a_small_graph_carries_the_brute_force_optimum_and_no_row_beats_it(tiny_report):
    assert tiny_report.brute_force_optimum is not None
    assert tiny_report.brute_force_labels is not None
    assert len(tiny_report.brute_force_labels) == tiny_report.node_count
    for row in tiny_report.rows:
        assert row.rounded_cut >= tiny_report.brute_force_optimum - 1e-10


def test_a_graph_above_the_brute_force_guard_reports_no_optimum(roach_report):
    assert roach_report.node_count == 20
    assert roach_report.brute_force_optimum is None
    assert roach_report.brute_force_labels is None


def test_a_planted_labelling_is_reported_as_a_reference_ratio_cut():
    graph = seeded_graph(9, seed=2)
    planted = np.array([0] * 5 + [1] * 4)
    report = small_report(GraphInstance("planted_tiny", graph, 2, planted))
    assert report.planted_labels_ratio_cut is not None
    assert report.planted_labels_ratio_cut >= report.brute_force_optimum - 1e-10


def test_a_graph_without_planted_labels_reports_none(roach_report):
    assert roach_report.planted_labels_ratio_cut is None


def test_the_report_records_the_configuration_it_ran_under(roach_report):
    config = roach_report.configuration
    assert config["step_count"] == SMALL_STEPS
    assert config["learning_rate"] == DEFAULT_LEARNING_RATE
    assert config["collision_weights"] == list(SMALL_WEIGHTS)
    assert config["inits"] == list(SMALL_INITS)
    assert set(config) == {
        "step_count",
        "learning_rate",
        "epsilon",
        "seed",
        "normalize_columns",
        "learning_rate_schedule",
        "warmup_steps",
        "final_learning_rate_fraction",
        "adjacency_weight",
        "adjacency_form",
        "diversity_weight",
        "collision_weights",
        "inits",
    }
    assert isinstance(config["normalize_columns"], bool)


def test_every_row_carries_the_configuration_it_ran_under_and_why_it_stopped(roach_report):
    """D-35 (8): a row without its configuration cannot be reproduced; (9): a stop is a reason."""
    for row in roach_report.rows:
        configuration = row.configuration
        assert configuration["name"] == "TwoHotSpanOptimizer"
        assert configuration["step_count"] == SMALL_STEPS
        assert configuration["step_rule"]["learning_rate"] == DEFAULT_LEARNING_RATE
        assert configuration["problem"]["graph"] == roach_report.name
        assert configuration["initial_spanning_set"] == "given"
        weights = [penalty["weight"] for penalty in configuration["penalties"]]
        assert weights == ([] if row.collision_weight == 0.0 else [row.collision_weight])
        assert row.stop_reason == "step_budget"
        assert row.stop_detail


def test_the_report_reads_normalize_columns_off_the_rows_it_ran(roach_report):
    """D-35 (4): assembled, not declared - the report-level knob is the rows' own."""
    assert (
        roach_report.configuration["normalize_columns"]
        == (roach_report.rows[0].configuration["normalize_columns"])
    )


def test_a_stopped_row_writes_its_numbers_as_null_and_is_never_the_best_row(monkeypatch):
    """A run stopped for a non-finite start is a row (D-35 (9)); its NaNs cannot be JSON and
    cannot win. The first cell's start is made non-finite through the composition the harness
    calls, which is the seam a composition root has."""
    import mllib.ml.projects.two_hot_span_composition as composition

    real_compose = composition.compose_two_hot_span
    calls = {"count": 0}

    def compose_with_a_bad_first_start(problem, **knobs):
        calls["count"] += 1
        if calls["count"] == 1:
            knobs["initial_spanning_set"] = np.full(
                (problem.node_count, problem.spanning_vector_count), np.nan
            )
        return real_compose(problem, **knobs)

    monkeypatch.setattr(composition, "compose_two_hot_span", compose_with_a_bad_first_start)
    report = small_report(GraphInstance("roach_g5", roach_graph(5), 2))

    stopped, finished = report.rows[0], report.rows[1]
    assert stopped.stop_reason == "non_finite"
    assert finished.stop_reason == "step_budget"
    assert report.best_row_index == 1
    written = report_to_dict(report)
    assert written["rows"][0]["rounded_cut"] is None
    assert written["rows"][0]["relaxed_objective"] is None
    assert written["rows"][0]["collision_measures"] == [None] * 18
    assert "NaN" not in json.dumps(written)


def test_the_harness_records_the_diversity_weight_it_ran_under():
    report = run_graph(
        GraphInstance("tiny", seeded_graph(8, seed=1), 2),
        collision_weights=(1.0,),
        inits=("spectral",),
        step_count=5,
        diversity_weight=0.75,
    )

    assert report.configuration["diversity_weight"] == 0.75
    # And the row itself carries the assembled record of the objects that ran (D-35 (4), (8)).
    penalties = report.rows[0].configuration["penalties"]
    assert {"name": "DiversityPenalty", "weight": 0.75} in penalties


def test_the_report_records_the_experimental_knobs_at_their_defaults(roach_report):
    """The six experimental knobs are recorded; a report naming none of them ran without them."""
    config = roach_report.configuration
    assert config["learning_rate_schedule"] == "constant"
    assert config["warmup_steps"] == 0
    assert config["final_learning_rate_fraction"] == 0.0
    assert config["adjacency_weight"] == 0.0
    assert config["adjacency_form"] == "laplacian"
    assert config["diversity_weight"] == 0.0


def test_the_report_records_the_experimental_knobs_it_was_asked_for():
    report = run_graph(
        GraphInstance("tiny", seeded_graph(8, seed=1), 2),
        collision_weights=(1.0,),
        inits=SMALL_INITS,
        step_count=SMALL_STEPS,
        learning_rate_schedule="warmup_cosine",
        warmup_steps=3,
        final_learning_rate_fraction=0.02,
        adjacency_weight=0.5,
        adjacency_form="edge_product",
        diversity_weight=0.25,
    )
    assert report.configuration["learning_rate_schedule"] == "warmup_cosine"
    assert report.configuration["warmup_steps"] == 3
    assert report.configuration["final_learning_rate_fraction"] == 0.02
    assert report.configuration["adjacency_weight"] == 0.5
    assert report.configuration["adjacency_form"] == "edge_product"
    assert report.configuration["diversity_weight"] == 0.25
    for row in report.rows:
        assert row.rounded_cut >= report.brute_force_optimum - 1e-10
        # The row's own record names the objects those knobs became.
        assert row.configuration["step_rule"]["schedule"] == {
            "name": "WarmupCosineSchedule",
            "warmup_steps": 3,
            "final_fraction": 0.02,
        }
        assert [penalty["name"] for penalty in row.configuration["penalties"]] == [
            "CollisionPenalty",
            "EdgeProductAdjacencyPenalty",
            "DiversityPenalty",
        ]


def test_main_accepts_the_experimental_flags_and_writes_them_into_the_report(tmp_path):
    main(
        [
            "--output-dir",
            str(tmp_path),
            "--steps",
            "5",
            "--graphs",
            "roach_g5",
            "--schedule",
            "cosine",
            "--final-lr-fraction",
            "0.01",
            "--adjacency-weight",
            "0.5",
            "--adjacency-form",
            "edge_product",
            "--diversity-weight",
            "0.25",
        ]
    )
    with (tmp_path / "roach_g5.json").open(encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["configuration"]["learning_rate_schedule"] == "cosine"
    assert written["configuration"]["final_learning_rate_fraction"] == 0.01
    assert written["configuration"]["adjacency_weight"] == 0.5
    assert written["configuration"]["adjacency_form"] == "edge_product"
    assert written["configuration"]["diversity_weight"] == 0.25
    assert written["rows"][0]["configuration"]["step_rule"]["schedule"]["name"] == "CosineSchedule"
    assert written["rows"][0]["stop_reason"] == "step_budget"


def test_main_rejects_a_schedule_name_that_does_not_exist(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--steps", "5", "--schedule", "exponential"])


def test_the_dictionary_form_holds_no_numpy_types(roach_report):
    encoded = json.dumps(report_to_dict(roach_report))
    assert "numpy" not in encoded
    restored = json.loads(encoded)
    assert restored["name"] == "roach_g5"


def test_the_written_file_round_trips_the_headline_number(roach_report, tmp_path):
    path = write_report(roach_report, tmp_path)
    assert path.name == "roach_g5.json"
    with path.open(encoding="utf-8") as handle:
        restored = json.load(handle)
    best = roach_report.rows[roach_report.best_row_index]
    assert restored["rows"][roach_report.best_row_index]["rounded_cut"] == pytest.approx(
        best.rounded_cut
    )


def test_the_written_file_labels_the_training_loss_as_training(roach_report, tmp_path):
    """The ridge loss carries epsilon and the collision reward; it must never be read as E."""
    with write_report(roach_report, tmp_path).open(encoding="utf-8") as handle:
        restored = json.load(handle)
    for row in restored["rows"]:
        assert "final_training_loss" in row
        assert row["relaxed_objective"] != row["final_training_loss"]


def test_the_printed_block_carries_the_name_the_floor_and_the_best_marker(roach_report):
    text = format_report(roach_report)
    assert "roach_g5" in text
    assert "Σλ" in text or "floor" in text
    assert "*" in text
    for name in ROUNDING_NAMES:
        assert name in text


def test_main_writes_one_file_for_one_requested_graph(tmp_path):
    main(["--output-dir", str(tmp_path), "--steps", "5", "--graphs", "roach_g5"])
    written = sorted(tmp_path.glob("*.json"))
    assert [path.name for path in written] == ["roach_g5.json"]


def test_main_rejects_a_graph_name_that_does_not_exist(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--steps", "5", "--graphs", "not_a_graph"])


def test_the_declared_defaults_are_the_ones_the_plan_sweeps():
    assert DEFAULT_COLLISION_WEIGHTS == (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    assert DEFAULT_INITS == ("random", "spectral")
    assert (DEFAULT_STEP_COUNT, DEFAULT_LEARNING_RATE) == (300, 0.05)


def test_an_unknown_initialisation_name_is_rejected():
    with pytest.raises(ValueError, match="unknown init"):
        run_graph(
            GraphInstance("tiny", seeded_graph(8, seed=4), 2),
            collision_weights=(0.0,),
            inits=("orthogonal",),
            step_count=5,
        )


def test_the_default_suite_covers_all_six_named_graphs(tmp_path):
    """`roach_g20` is the fifth and `two_triangles` the sixth; the suite must run what it reports.

    `two_triangles` is also the first default-suite graph inside `BRUTE_FORCE_NODE_LIMIT`, so it is
    the only row where the suite's `Ê ≥ optimum` guard has an optimum to check rather than `None`.
    """
    reports = run_default_suite(
        tmp_path,
        collision_weights=SMALL_WEIGHTS,
        inits=SMALL_INITS,
        step_count=SMALL_STEPS,
    )
    assert [report.name for report in reports] == [
        "roach_g5",
        "karate",
        "planted_partition",
        "two_moons_knn",
        "roach_g20",
        "two_triangles",
    ]
    assert sorted(path.name for path in tmp_path.glob("*.json")) == sorted(
        f"{report.name}.json" for report in reports
    )
    roach_g20 = reports[-2]
    assert (roach_g20.node_count, roach_g20.edge_count, roach_g20.cluster_count) == (80, 98, 3)
    assert roach_g20.planted_labels_ratio_cut == pytest.approx(0.15, abs=1e-12)

    two_triangles = reports[-1]
    assert (two_triangles.node_count, two_triangles.edge_count, two_triangles.cluster_count) == (
        6,
        7,
        2,
    )
    assert two_triangles.brute_force_optimum == pytest.approx(1 / 15, abs=1e-12)
    assert two_triangles.planted_labels_ratio_cut == pytest.approx(1 / 15, abs=1e-12)
    for row in two_triangles.rows:
        assert row.rounded_cut >= two_triangles.brute_force_optimum - 1e-12


def test_main_writes_a_report_for_the_eighty_node_cockroach(tmp_path):
    main(["--output-dir", str(tmp_path), "--steps", "5", "--graphs", "roach_g20"])
    with (tmp_path / "roach_g20.json").open(encoding="utf-8") as handle:
        restored = json.load(handle)
    assert (restored["node_count"], restored["edge_count"]) == (80, 98)
    assert restored["cluster_count"] == 3
    assert restored["spanning_vector_count"] == 77
    # n = 80 is over the brute-force guard, so the oracle is the planted cut and not an optimum.
    assert restored["brute_force_optimum"] is None
    assert restored["planted_labels_ratio_cut"] == pytest.approx(0.15, abs=1e-12)
