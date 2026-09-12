"""The stress ladder's engine-facing half: the recorder, one whole cell, and the rung-0 oracle.

torch is an optional dependency group (D-31), so the whole file skips without it; the ladder, the
statuses and the two scheduling refusals are pinned in `test_two_hot_span_stress_plan.py` and run
either way. The configurations here are deliberately tiny (20 and 30 steps) except the oracle,
which is the one place the *prototype's own* numbers are the assertion and therefore has to run the
prototype's own configuration.
"""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from mllib.math.graph.two_hot_span_problem import TwoHotSpanProblem, incidence_matrix, roach_graph
from mllib.ml.projects.two_hot_span_composition import compose_two_hot_span
from mllib.ml.projects.two_hot_span_stress import (
    ENGINE_MODULES,
    FIXTURE_REPORTS_DIR,
    RUNG0_GRAPHS,
    CellSpec,
    CellTimeout,
    CheckpointRecorder,
    GraphSpec,
    build_graph,
    engine_provenance,
    graph_record,
    oracle_report_check,
    plan_cells,
    report_path,
    run_cell,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "two_hot_span_roach_g5_report.json"

ROACH_SPEC = GraphSpec(
    rung=0,
    name="roach_g5",
    graph_seed=0,
    kind="builtin",
    generator=(("source", "default_test_graphs"),),
    expected_node_count=20,
)


def tuned_cell() -> CellSpec:
    """The prototype's tuned cell as the ladder plans it: λ = 10, ν = 10, μ = 0.3, random init."""
    return next(
        cell for cell in plan_cells([0]) if cell.key == "r0|roach_g5|gs0|l10|nu10|mu0.3|random"
    )


def short_context(cap_seconds: float = 600.0, step_count: int = 30) -> dict:
    """The context `main` would assemble for a roach cell, with a short step budget."""
    record = graph_record(ROACH_SPEC)
    return {
        "host": "laptop",
        "threads": 1,
        "step_count": step_count,
        "checkpoint_steps": [10, step_count],
        "data_dir": None,
        "cap_seconds": cap_seconds,
        "spectral_floor": record["spectral_floor"],
        "cluster_count": record["cluster_count"],
        "node_count": record["node_count"],
        "datum_ratio_cuts": {name: values["ratio_cut"] for name, values in record["datum"].items()},
        "planted_labels_ratio_cut": record["planted_labels_ratio_cut"],
        "engine": engine_provenance(),
    }


# ---------------------------------------------------------------------------------------------
# The checkpoint recorder.
# ---------------------------------------------------------------------------------------------


def test_the_final_checkpoint_is_the_run_the_engine_returns():
    """The whole ladder reads its Ê off checkpoints, so the last one must *be* the run's own Ê.

    Both sides go through the numpy pinv path, but only one of them goes through it from inside the
    training loop; if the recorder ever saw a different V than the one the engine finishes with,
    every intermediate checkpoint in the results file would be measuring something else.
    """
    problem = TwoHotSpanProblem(roach_graph(5), 2, name="roach_g5")
    recorder = CheckpointRecorder(problem.X, (10, 20))
    run = compose_two_hot_span(
        problem, step_count=20, learning_rate=0.05, collision_weight=10.0, seed=0, recorder=recorder
    ).run()

    assert [checkpoint["step"] for checkpoint in recorder.checkpoints] == [10, 20]
    final = recorder.checkpoints[-1]
    assert final["rounded_cut"] == pytest.approx(run.rounded_cut, abs=1e-12)
    assert final["relaxed_objective"] == pytest.approx(run.relaxed_objective, abs=1e-12)
    assert final["labels"] == [int(label) for label in run.labels]
    assert final["component_count"] == int(np.unique(run.labels).size)


def test_the_recorder_only_stores_the_steps_it_was_asked_for():
    X = incidence_matrix(roach_graph(5))
    recorder = CheckpointRecorder(X, (10,))
    spanning_set = np.asarray(np.eye(20, 18), dtype=float)
    for step in range(20):
        recorder.record_step(step, 0.0, spanning_set)
    assert [checkpoint["step"] for checkpoint in recorder.checkpoints] == [10]


def test_a_deadline_already_passed_raises_and_keeps_the_checkpoints_it_reached():
    X = incidence_matrix(roach_graph(5))
    recorder = CheckpointRecorder(X, (1, 2), deadline=None)
    spanning_set = np.asarray(np.eye(20, 18), dtype=float)
    recorder.record_step(0, 0.0, spanning_set)
    recorder.deadline = 0.0
    with pytest.raises(CellTimeout):
        recorder.record_step(1, 0.0, spanning_set)
    assert [checkpoint["step"] for checkpoint in recorder.checkpoints] == [1]


def test_a_cell_past_its_cap_is_recorded_as_a_timeout_with_what_it_completed():
    record = run_cell(tuned_cell(), short_context(cap_seconds=0.0, step_count=30))
    assert record["status"] == "timeout"
    assert record["error"].startswith("the cell passed its cap")
    assert record["checkpoints"] == []
    assert record["rounded_cut"] is None
    assert record["seconds"] > 0.0
    # D-35 (8): the record says what it ran under even though the run never returned.
    assert record["configuration"]["name"] == "TwoHotSpanOptimizer"
    assert record["configuration"]["step_count"] == 30


# ---------------------------------------------------------------------------------------------
# One whole cell.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_cell_record():
    return run_cell(tuned_cell(), short_context())


def test_a_finished_cell_names_the_configuration_it_actually_ran(short_cell_record):
    assert short_cell_record["key"] == "r0|roach_g5|gs0|l10|nu10|mu0.3|random"
    assert short_cell_record["record"] == "cell"
    assert short_cell_record["schema_version"] == 1
    assert short_cell_record["collision_weight"] == 10.0
    assert short_cell_record["diversity_weight"] == 10.0
    assert short_cell_record["adjacency_weight"] == 0.3
    assert short_cell_record["adjacency_form"] == "edge_product"
    assert short_cell_record["init"] == "random"
    assert short_cell_record["init_seed"] == 0
    assert short_cell_record["step_count"] == 30
    assert short_cell_record["checkpoint_steps"] == [10, 30]
    # The run's own assembled record (D-35 (4)): the classes and knobs that actually ran.
    configuration = short_cell_record["configuration"]
    assert configuration["name"] == "TwoHotSpanOptimizer"
    assert configuration["step_count"] == 30
    assert configuration["seed"] == 0
    assert configuration["problem"] == {
        "name": "TwoHotSpanProblem",
        "graph": "roach_g5",
        "node_count": 20,
        "cluster_count": 2,
    }
    assert configuration["cost"]["projector"] == {"name": "RidgeProjector", "epsilon": 1e-6}
    assert configuration["step_rule"]["name"] == "AdamStepRule"
    assert configuration["step_rule"]["learning_rate"] == 0.05
    assert configuration["step_rule"]["schedule"] == {"name": "ConstantSchedule"}
    assert [(penalty["name"], penalty["weight"]) for penalty in configuration["penalties"]] == [
        ("CollisionPenalty", 10.0),
        ("EdgeProductAdjacencyPenalty", 0.3),
        ("DiversityPenalty", 10.0),
    ]
    assert configuration["initial_spanning_set"] == "seeded"
    assert short_cell_record["stop_reason"] == "step_budget"
    assert short_cell_record["steps_taken"] == 30


def test_a_cell_whose_run_stops_early_is_a_stopped_record_with_its_reason_not_an_error():
    """A stop is a result (D-35 (9)): the cell keeps its numbers and says why the run ended."""
    context = short_context()
    context["zero_sum_tolerance"] = 0.0
    record = run_cell(tuned_cell(), context)

    assert record["status"] == "stopped"
    assert record["stop_reason"] == "constraint_violation"
    assert "exceeds" in record["stop_detail"]
    assert record["steps_taken"] == 0
    assert record["error"] is None
    assert record["rounded_cut"] is not None
    assert record["final_training_loss"] is None
    assert record["configuration"]["zero_sum_tolerance"] == 0.0


def test_a_cell_whose_start_is_not_finite_is_a_stopped_record_with_null_numbers(monkeypatch):
    """A NON_FINITE stop: no step completes, the report is NaN and the record writes it as null.

    `run_cell` takes its spectral start from the problem, so the start is made non-finite there;
    a knob on the context would make this patch-free.
    """
    monkeypatch.setattr(
        TwoHotSpanProblem,
        "spectral_spanning_set",
        lambda self: np.full((self.node_count, self.spanning_vector_count), np.nan),
    )
    cell = CellSpec(tuned_cell().graph, 10.0, 0.3, "spectral")
    record = run_cell(cell, short_context())

    assert record["status"] == "stopped"
    assert record["stop_reason"] == "non_finite"
    assert record["steps_taken"] == 0
    assert record["error"] is None
    assert record["relaxed_objective"] is None
    assert record["rounded_cut"] is None
    assert record["rounded_cut_minus_floor"] is None
    assert record["collision_measures"] == [None] * 18
    assert set(record["labels"]) == {-1}
    assert record["configuration"]["initial_spanning_set"] == "given"
    assert "NaN" not in json.dumps(record)


def test_a_finished_cell_carries_the_three_objectives_and_both_differences(short_cell_record):
    floor = short_cell_record["spectral_floor"]
    assert short_cell_record["rounded_cut"] >= floor - 1e-10
    assert short_cell_record["relaxed_objective"] >= floor - 1e-10
    assert short_cell_record["rounded_cut_minus_relaxed"] == pytest.approx(
        short_cell_record["rounded_cut"] - short_cell_record["relaxed_objective"]
    )
    assert short_cell_record["rounded_cut_minus_floor"] == pytest.approx(
        short_cell_record["rounded_cut"] - floor
    )


def test_a_finished_cell_carries_the_datum_it_is_read_against(short_cell_record):
    assert set(short_cell_record["datum_ratio_cuts"]) == {"kmeans", "discretize", "cluster_qr"}
    assert short_cell_record["node_count"] == 20
    assert short_cell_record["cluster_count"] == 2
    assert short_cell_record["peak_rss_mb"] > 0.0
    assert short_cell_record["cap_seconds"] == 600.0
    assert short_cell_record["predicted_rss_gb"] > 0.0


def test_a_finished_cell_names_the_engine_that_produced_it(short_cell_record):
    engine = short_cell_record["engine"]
    assert set(engine["engine_sha256"]) == set(ENGINE_MODULES)
    assert "src/mllib/math/algorithms/two_hot_span/optimizer.py" in engine["engine_sha256"]
    assert all(len(digest) == 64 for digest in engine["engine_sha256"].values())


def test_the_cells_checkpoints_end_on_the_numbers_the_cell_reports(short_cell_record):
    final = short_cell_record["checkpoints"][-1]
    assert final["step"] == 30
    assert final["rounded_cut"] == pytest.approx(short_cell_record["rounded_cut"], abs=1e-12)
    assert final["labels"] == short_cell_record["labels"]
    assert short_cell_record["status"] in ("reached_k", "drifted", "not_at_k")


def test_the_cell_record_is_plain_json_with_no_numpy_types(short_cell_record):
    def check(value):
        assert not isinstance(value, np.generic | np.ndarray), value
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(short_cell_record)
    assert json.loads(json.dumps(short_cell_record, sort_keys=True)) == short_cell_record


# ---------------------------------------------------------------------------------------------
# Graph building and the oracle.
# ---------------------------------------------------------------------------------------------


def test_a_planted_graph_is_named_after_the_spec_that_asked_for_it():
    spec = next(cell.graph for cell in plan_cells([1]) if "n100_k2_clear" in cell.key)
    instance = build_graph(spec)
    assert instance.name == "sbm_n100_k2_clear"
    assert instance.cluster_count == 2
    assert instance.graph.number_of_nodes() == 100


# The spectral start is a stationary point of the span term, so Adam's first step is a sign
# function of a gradient whose near-zero entries the BLAS decides: the fixtures written on this
# platform reproduce only on it (BL-50). The random-start snapshot has no such sensitivity.
FIXTURE_PLATFORM = "Darwin-arm64"
only_on_fixture_platform = pytest.mark.skipif(
    f"{platform.system()}-{platform.machine()}" != FIXTURE_PLATFORM,
    reason=f"BL-50: spectral-start fixture reproduces only on {FIXTURE_PLATFORM}",
)


@only_on_fixture_platform
def test_the_oracle_reproduces_the_prototypes_frozen_report(tmp_path):
    """Rung 0 is the runner's oracle: unchanged engine, unchanged numbers, to 1e-8.

    The fixture is the prototype's own `roach_g5.json`, committed beside this test. A failure here
    means the engine moved under the ladder, and no number the grid produces may be reported until
    it is explained.
    """
    shutil.copy(FIXTURE, tmp_path / "roach_g5.json")
    record = oracle_report_check("roach_g5", tmp_path)
    assert record["passed"] is True
    assert record["mismatched_fields"] == []
    assert record["max_abs_difference"] <= 1e-8
    assert record["report_path"] == str(tmp_path / "roach_g5.json")


@only_on_fixture_platform
@pytest.mark.parametrize("name", RUNG0_GRAPHS)
def test_the_oracle_passes_from_the_tree_alone_when_the_reports_directory_is_absent(name, tmp_path):
    """The cloud instance has no `~/Desktop`, and rung 2 runs there behind this same oracle.

    Comparing against the committed fixtures is not a convenience: it is what lets the oracle run
    on a second machine at all, and a second machine is a second arithmetic path — the one thing a
    bit-for-bit claim about the engine cannot be asserted about without checking.
    """
    record = oracle_report_check(name, tmp_path / "missing")
    assert record["passed"] is True
    assert record["mismatched_fields"] == []
    assert record["max_abs_difference"] <= 1e-8
    assert record["report_path"] == str(FIXTURES / f"two_hot_span_{name}_report.json")


def test_the_resolved_report_prefers_the_named_directory_over_the_fixture(tmp_path):
    assert report_path("roach_g5", tmp_path) == FIXTURE_REPORTS_DIR / (
        "two_hot_span_roach_g5_report.json"
    )
    shutil.copy(FIXTURE, tmp_path / "roach_g5.json")
    assert report_path("roach_g5", tmp_path) == tmp_path / "roach_g5.json"


def test_a_report_missing_from_both_places_fails_the_oracle_rather_than_passing_it(tmp_path):
    record = oracle_report_check("planted_partition", tmp_path)
    assert record["passed"] is False
    assert record["mismatched_fields"] == ["report file is missing"]
