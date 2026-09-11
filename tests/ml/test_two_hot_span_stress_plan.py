"""The stress ladder's plan: the cells it will run, and the ones it refuses before running any.

Nothing here needs torch — the ladder, the keys, the statuses and the two refusals are decided
before a single step is taken, and the point of pinning them in a torch-free file is that the
pre-registration is checkable in a checkout that cannot run the grid at all.
"""

from __future__ import annotations

import json
import time

import pytest

from mllib.ml.projects.two_hot_span_stress import (
    CHECKPOINT_STEPS,
    COLLISION_WEIGHT,
    LEARNING_RATE,
    ORACLE_CHECKS,
    SCHEMA_VERSION,
    STEP_COUNT,
    WORKER_LOST_MESSAGE,
    ResultsWriter,
    _schedule,
    _submit,
    admits_cell,
    budget_rejection,
    cap_seconds,
    cell_status,
    convergence,
    coupling_settings,
    main,
    oracle_passed,
    plan_cells,
    planted_partition_parameters,
    predicted_rss_gb,
    predicted_seconds_per_step,
    read_records,
    select_cells,
)

# The ladder as pre-registered in the research seed. A change to any of these four numbers is a
# change to what "the stress run" means, and must arrive with a plan row and a fresh results file.
EXPECTED_CELL_COUNTS = {0: 72, 1: 432, 2: 16, 3: 72}


@pytest.mark.parametrize(("rung", "expected"), sorted(EXPECTED_CELL_COUNTS.items()))
def test_each_rung_plans_exactly_the_pre_registered_number_of_cells(rung, expected):
    assert len(plan_cells([rung])) == expected


def test_no_cell_key_repeats_across_the_whole_ladder():
    cells = plan_cells([0, 1, 2, 3])
    keys = [cell.key for cell in cells]
    assert len(keys) == len(set(keys)) == sum(EXPECTED_CELL_COUNTS.values())


def test_the_rung_one_sweep_and_the_roach_band_are_the_same_cell_run_once():
    """(ν, μ) = (10, 0.3) at random init is both a fixed setting and a sweep point."""
    settings = coupling_settings(1)
    assert settings.count((10.0, 0.3, "random")) == 1
    assert len(settings) == 16
    assert len(coupling_settings(0)) == len(coupling_settings(2)) == len(coupling_settings(3)) == 8


def test_the_cell_key_names_the_graph_seed_and_the_whole_coupling():
    cells = {cell.key: cell for cell in plan_cells([0])}
    key = "r0|roach_g5|gs0|l10|nu10|mu0.3|random"
    assert key in cells
    cell = cells[key]
    assert (cell.diversity_weight, cell.adjacency_weight, cell.init) == (10.0, 0.3, "random")
    assert cell.adjacency_form == "edge_product"


def test_the_adjacency_form_is_the_laplacian_only_when_the_reward_is_off():
    forms = {(cell.adjacency_weight != 0.0, cell.adjacency_form) for cell in plan_cells([0])}
    assert forms == {(False, "laplacian"), (True, "edge_product")}


def test_the_fixed_training_settings_are_the_ones_the_plan_states():
    assert (COLLISION_WEIGHT, LEARNING_RATE, STEP_COUNT) == (10.0, 0.05, 3000)
    assert CHECKPOINT_STEPS == (300, 1000, 3000)


@pytest.mark.parametrize(
    ("node_total", "cluster_count", "strength", "group_size", "p_in", "p_out"),
    [
        (100, 2, "clear", 50, 12.0 / 49.0, 1.0 / 50.0),
        (100, 2, "moderate", 50, 12.0 / 49.0, 3.0 / 50.0),
        (100, 2, "weak", 50, 12.0 / 49.0, 6.0 / 50.0),
        (201, 3, "moderate", 67, 12.0 / 66.0, 3.0 / 134.0),
        (500, 5, "weak", 100, 12.0 / 99.0, 6.0 / 400.0),
        (1000, 4, "moderate", 250, 12.0 / 249.0, 3.0 / 750.0),
        (2000, 8, "moderate", 250, 12.0 / 249.0, 3.0 / 1750.0),
    ],
)
def test_the_planted_partition_parameters_are_pinned_by_value(
    node_total, cluster_count, strength, group_size, p_in, p_out
):
    assert planted_partition_parameters(node_total, cluster_count, strength) == pytest.approx(
        (group_size, p_in, p_out)
    )


def test_every_planted_rung_covers_its_blocks_exactly():
    for cell in plan_cells([1, 2]):
        parameters = cell.graph.generator_dict
        assert parameters["group_size"] * parameters["cluster_count"] == parameters["n"]


# ---------------------------------------------------------------------------------------------
# Status and convergence, from synthetic checkpoint histories.
# ---------------------------------------------------------------------------------------------


def checkpoint(step, rounded_cut, labels):
    return {
        "step": step,
        "relaxed_objective": 0.0,
        "rounded_cut": rounded_cut,
        "component_count": len(set(labels)),
        "labels": list(labels),
    }


def test_a_run_holding_k_components_at_the_end_reached_k():
    history = [checkpoint(300, 1.0, [0, 0, 1]), checkpoint(3000, 0.5, [0, 0, 1])]
    assert cell_status(history, 2) == ("reached_k", None)


def test_a_run_that_finds_k_and_loses_it_drifted_from_the_first_checkpoint_that_differs():
    history = [
        checkpoint(300, 1.0, [0, 1, 2]),
        checkpoint(1000, 0.9, [0, 0, 1]),
        checkpoint(3000, 0.8, [0, 1, 2]),
    ]
    assert cell_status(history, 2) == ("drifted", 3000)


def test_drift_onset_is_the_first_departure_not_the_last():
    history = [
        checkpoint(300, 1.0, [0, 0, 1]),
        checkpoint(1000, 0.9, [0, 1, 2]),
        checkpoint(3000, 0.8, [0, 1, 2]),
    ]
    assert cell_status(history, 2) == ("drifted", 1000)


def test_a_run_that_never_finds_k_is_not_at_k():
    history = [checkpoint(300, 1.0, [0, 1, 2]), checkpoint(3000, 0.9, [0, 1, 2])]
    assert cell_status(history, 2) == ("not_at_k", None)


def test_a_cell_with_no_checkpoints_is_not_at_k():
    assert cell_status([], 2) == ("not_at_k", None)


def test_convergence_needs_the_last_two_checkpoints_to_be_the_same_answer():
    settled = [
        checkpoint(300, 0.5, [0, 0, 1]),
        checkpoint(1000, 0.5, [0, 0, 1]),
        checkpoint(3000, 0.5, [0, 0, 1]),
    ]
    assert convergence(settled) == (True, 300)


def test_a_run_that_changes_its_labels_at_the_last_checkpoint_has_not_converged():
    moving = [
        checkpoint(300, 0.5, [0, 0, 1]),
        checkpoint(1000, 0.5, [0, 0, 1]),
        checkpoint(3000, 0.5, [0, 1, 1]),
    ]
    assert convergence(moving) == (False, None)


def test_steps_to_convergence_is_the_checkpoint_the_run_never_left():
    late = [
        checkpoint(300, 0.9, [0, 1, 1]),
        checkpoint(1000, 0.5, [0, 0, 1]),
        checkpoint(3000, 0.5, [0, 0, 1]),
    ]
    assert convergence(late) == (True, 1000)


def test_a_rounded_cut_that_moves_below_tolerance_still_counts_as_settled():
    drifting = [
        checkpoint(1000, 0.5, [0, 0, 1]),
        checkpoint(3000, 0.5 + 1e-12, [0, 0, 1]),
    ]
    assert convergence(drifting)[0] is True


# ---------------------------------------------------------------------------------------------
# The cost models and the two refusals.
# ---------------------------------------------------------------------------------------------


def test_the_seconds_per_step_table_is_reproduced_at_its_own_points():
    for node_total, seconds in ((100, 0.0016), (500, 0.047), (2000, 2.57)):
        assert predicted_seconds_per_step(node_total) == pytest.approx(seconds, rel=1e-9)


def test_the_prediction_is_monotone_and_extrapolates_past_the_table():
    assert predicted_seconds_per_step(4000) > predicted_seconds_per_step(2000)
    assert predicted_seconds_per_step(20) == predicted_seconds_per_step(100)


def test_the_cap_is_the_multiplied_prediction_plus_the_fixed_overhead():
    assert cap_seconds(500, 3000, 4.0) == pytest.approx(4.0 * 3000 * 0.047 + 60.0)


def test_the_memory_budget_refuses_a_cell_that_does_not_fit_beside_what_runs():
    big = predicted_rss_gb(2000)
    assert admits_cell(0.0, big, 8.0) is True
    assert admits_cell(3 * big, big, 8.0) is False


def test_a_cell_larger_than_the_whole_budget_is_over_budget_rather_than_waiting_forever():
    cell = next(cell for cell in plan_cells([2]) if "n2000" in cell.key)
    assert budget_rejection(cell, "cloud", 1.0) is not None
    assert budget_rejection(cell, "cloud", 8.0) is None


def test_the_laptop_host_refuses_every_rung_two_cell_whatever_the_budget():
    for cell in plan_cells([2]):
        assert "laptop" in budget_rejection(cell, "laptop", 1024.0)
    assert budget_rejection(cell, "cloud", 1024.0) is None


def test_the_laptop_still_runs_the_other_rungs():
    for cell in plan_cells([0, 1]):
        assert budget_rejection(cell, "laptop", 8.0) is None


# ---------------------------------------------------------------------------------------------
# Resume, and the dry run.
# ---------------------------------------------------------------------------------------------


def test_a_cell_whose_key_is_already_recorded_is_not_planned_again(tmp_path):
    cells = plan_cells([0])
    results = tmp_path / "two_hot_stress_results.jsonl"
    results.write_text(
        json.dumps({"record": "cell", "key": cells[0].key, "status": "reached_k"}) + "\n",
        encoding="utf-8",
    )
    done = {record["key"] for record in read_records(results) if record["record"] == "cell"}
    remaining = select_cells(cells, None, done)
    assert cells[0].key not in {cell.key for cell in remaining}
    assert len(remaining) == len(cells) - 1
    # --force clears the done set, so the same cell is planned again.
    assert len(select_cells(cells, None, set())) == len(cells)


def test_only_keeps_the_cells_whose_key_contains_a_requested_substring():
    cells = plan_cells([0])
    chosen = select_cells(cells, "roach_g5|gs0,karate|gs2", set())
    assert {cell.graph.name for cell in chosen} == {"roach_g5", "karate"}
    assert all(("gs0" in cell.key) or ("gs2" in cell.key) for cell in chosen)


def test_a_dry_run_prints_every_key_and_writes_nothing(tmp_path, capsys):
    results = tmp_path / "two_hot_stress_results.jsonl"
    assert main(["--results", str(results), "--rung", "2", "--dry-run"]) == 0
    printed = capsys.readouterr().out.splitlines()
    keys = [line for line in printed if line.startswith("r2|")]
    assert keys == [cell.key for cell in plan_cells([2])]
    assert "total: 16 cells" in printed
    assert not results.exists()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------
# The scheduler's failure paths, driven with a stub pool: no cell ever runs here.
# ---------------------------------------------------------------------------------------------


class NeverReadyHandle:
    """What `apply_async` returns for a worker that is killed before it can set a result."""

    def ready(self) -> bool:
        return False

    def get(self):
        raise AssertionError("a handle that is never ready must never be got")


class StubPool:
    """A pool that accepts submissions, answers nothing, and remembers being terminated."""

    def __init__(self, terminations: list) -> None:
        self.submitted: list = []
        self._terminations = terminations

    def apply_async(self, function, arguments):
        self.submitted.append(arguments)
        return NeverReadyHandle()

    def terminate(self) -> None:
        self._terminations.append(self)


class StubContext:
    """Stands in for the spawn context, handing out stub pools and counting rebuilds."""

    def __init__(self) -> None:
        self.pools: list[StubPool] = []
        self.terminations: list[StubPool] = []

    def Pool(self, **kwargs) -> StubPool:  # noqa: N802 - the multiprocessing context's own name
        assert kwargs["maxtasksperchild"] == 1
        pool = StubPool(self.terminations)
        self.pools.append(pool)
        return pool


class StubArguments:
    threads = 1
    memory_budget_gb = 1024.0


def runnable_cells(count: int, cap_seconds_value: float = 0.0) -> list:
    cells = plan_cells([0])[:count]
    context = {
        "cap_seconds": cap_seconds_value,
        "step_count": STEP_COUNT,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "host": "laptop",
        "engine": {"botmaker_commit": "stub", "engine_sha256": {}},
    }
    return [(cell, dict(context)) for cell in cells]


def cell_records(results_path) -> list[dict]:
    return [record for record in read_records(results_path) if record["record"] == "cell"]


def test_a_worker_that_never_returns_is_recorded_and_its_pool_rebuilt(tmp_path, monkeypatch):
    """A killed worker sets no result, so `ready()` stays false and the parent must not wait on it.

    Without the parent's own clock the loop would spin forever on a process the OOM killer already
    took, and every cell behind it would be stranded with no record at all — the one outcome the
    seed's "report every cell" rule cannot survive.
    """
    monkeypatch.setattr(
        "mllib.ml.projects.two_hot_span_stress.WORKER_GRACE_SECONDS", 0.0, raising=True
    )
    writer = ResultsWriter(tmp_path / "two_hot_stress_results.jsonl")
    context = StubContext()
    runnable = runnable_cells(3)
    _schedule(list(runnable), writer, 2, StubArguments(), context, None)

    records = cell_records(writer.results_path)
    assert [record["status"] for record in records] == ["timeout"] * 3
    assert {record["key"] for record in records} == {cell.key for cell, _ in runnable}
    assert all(record["error"] == WORKER_LOST_MESSAGE for record in records)
    assert context.terminations, "the stranded pool was never terminated"
    assert len(context.pools) > 1, "the pool was never rebuilt, so the remaining cells were lost"


def test_a_deadline_already_passed_records_every_pending_cell_and_starts_none(tmp_path):
    writer = ResultsWriter(tmp_path / "two_hot_stress_results.jsonl")
    context = StubContext()
    runnable = runnable_cells(4)
    _schedule(list(runnable), writer, 2, StubArguments(), context, time.monotonic() - 1.0)

    records = cell_records(writer.results_path)
    assert [record["status"] for record in records] == ["skipped_deadline"] * 4
    assert all(record["error"] == "the --hours guard stopped the run" for record in records)
    assert context.pools[0].submitted == [], "a cell was started after the deadline"


# ---------------------------------------------------------------------------------------------
# The oracle gate.
# ---------------------------------------------------------------------------------------------


def oracle_record(check: str, passed: bool) -> dict:
    return {"record": "oracle", "schema_version": SCHEMA_VERSION, "check": check, "passed": passed}


def test_all_four_rung_zero_checks_are_needed_before_the_other_rungs_run():
    assert oracle_passed([oracle_record(check, True) for check in ORACLE_CHECKS]) is True


@pytest.mark.parametrize("missing", ORACLE_CHECKS)
def test_one_missing_oracle_check_closes_the_gate(missing):
    records = [oracle_record(check, True) for check in ORACLE_CHECKS if check != missing]
    assert oracle_passed(records) is False


@pytest.mark.parametrize("failed", ORACLE_CHECKS)
def test_one_failed_oracle_check_closes_the_gate(failed):
    records = [oracle_record(check, check != failed) for check in ORACLE_CHECKS]
    assert oracle_passed(records) is False


def test_a_failed_oracle_in_the_results_file_refuses_the_run(tmp_path, capsys):
    results = tmp_path / "two_hot_stress_results.jsonl"
    with results.open("w", encoding="utf-8") as handle:
        for check in ORACLE_CHECKS:
            handle.write(json.dumps(oracle_record(check, check != "karate")) + "\n")

    exit_code = main(
        ["--results", str(results), "--rung", "1", "--only", "no-such-cell-key-anywhere"]
    )

    assert exit_code == 1
    assert "refusing to run" in capsys.readouterr().out
    assert cell_records(results) == []


# ---------------------------------------------------------------------------------------------
# Rows that never ran still name the engine that could not run them.
# ---------------------------------------------------------------------------------------------


def test_every_over_budget_row_names_the_engine_and_the_host(tmp_path):
    """A laptop run of rung 2 records sixteen refusals, and a refusal is a result like any other.

    The graph is never built for these rows, so their engine can only come from the run itself; a
    row that could not say which engine it belongs to could not be frozen beside the ones that ran.
    """
    results = tmp_path / "two_hot_stress_results.jsonl"
    with results.open("w", encoding="utf-8") as handle:
        for check in ORACLE_CHECKS:
            handle.write(json.dumps(oracle_record(check, True)) + "\n")

    assert main(["--results", str(results), "--rung", "2", "--host", "laptop"]) == 0

    records = cell_records(results)
    assert len(records) == 16
    assert {record["status"] for record in records} == {"over_budget"}
    for record in records:
        assert record["engine"]["botmaker_commit"]
        assert set(record["engine"]["engine_sha256"]) and record["host"] == "laptop"
        assert "laptop" in record["error"]


# ---------------------------------------------------------------------------------------------
# --allow-rung-2: the laptop refusal lifts, the memory rules do not.
# ---------------------------------------------------------------------------------------------


def rung2_cell(node_total: int):
    return next(cell for cell in plan_cells([2]) if f"n{node_total}" in cell.key)


def test_the_flag_lifts_the_laptop_refusal_of_rung_two():
    cell = rung2_cell(2000)
    assert "--allow-rung-2" in budget_rejection(cell, "laptop", 8.0)
    assert budget_rejection(cell, "laptop", 8.0, True) is None


def test_the_flag_does_not_lift_the_single_cell_budget_refusal():
    """The flag says where rung 2 may run, never how much memory it may have.

    A cell whose own predicted peak is above the budget can never be admitted however long the
    scheduler waits, so it is still refused up front — with the flag on and off alike.
    """
    cell = rung2_cell(2000)
    assert budget_rejection(cell, "laptop", 1.0, True) is not None
    assert budget_rejection(cell, "cloud", 1.0) is not None


def test_the_admission_arithmetic_is_unchanged_by_the_flag():
    """Three 2000-node cells and two 1000-node cells fit an 8 GB budget; the rest wait.

    2.1 GB each for n = 2000 and 0.75 GB for n = 1000: 3 x 2.1 = 6.3, then the fourth 2000-node cell
    would reach 8.4 and is skipped over, and two 1000-node cells bring the total to 7.8. Sixteen
    cells are pending; five start.
    """
    slots = 16
    pending = [(rung2_cell(2000), {}) for _ in range(8)] + [
        (rung2_cell(1000), {}) for _ in range(8)
    ]
    inflight: list = []
    _submit(StubPool([]), pending, inflight, slots, 8.0)

    started = [cell.graph.expected_node_count for _, cell, _, _, _ in inflight]
    assert started == [2000, 2000, 2000, 1000, 1000]
    assert sum(predicted_rss_gb(size) for size in started) <= 8.0
    assert len(pending) == 11


def test_the_cell_record_states_whether_rung_two_was_allowed(tmp_path):
    results = tmp_path / "two_hot_stress_results.jsonl"
    with results.open("w", encoding="utf-8") as handle:
        for check in ORACLE_CHECKS:
            handle.write(json.dumps(oracle_record(check, True)) + "\n")

    assert main(["--results", str(results), "--rung", "2", "--host", "laptop"]) == 0
    records = cell_records(results)
    assert len(records) == 16
    assert all(record["allow_rung_2"] is False for record in records)
    assert all("--allow-rung-2" in record["error"] for record in records)
