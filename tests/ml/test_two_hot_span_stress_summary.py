"""The stress summary's merge, tables and freeze, on a synthetic results set.

The summary script is the reading end of the stress runner, so the fixture here is a hand-built
results set rather than a real run: two rungs, two graphs, two coupling settings, both
initialisations, and statuses spanning reached_k / drifted / not_at_k / error. Every expected
number below is worked out by hand from that fixture — the point of the tests is that the script
agrees with arithmetic done on paper, not that it agrees with itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from two_hot_span_stress_summary import (
    MISSING,
    freeze_text,
    graph_table,
    main,
    merge_result_files,
    render_summary,
    restrict_to_rung,
    rung_table,
    setting_table,
    sweep_grid_tables,
    write_frozen,
)

DATUM_CUTS = {"kmeans": 12.0, "discretize": 11.0, "cluster_qr": 13.0}


def make_graph(rung, name, graph_seed, *, kmeans, node_count, cluster_count):
    return {
        "record": "graph",
        "schema_version": 1,
        "rung": rung,
        "name": name,
        "graph_seed": graph_seed,
        "node_count": node_count,
        "edge_count": 4 * node_count,
        "cluster_count": cluster_count,
        "spanning_vector_count": cluster_count,
        "spectral_floor": 6.0,
        "datum": {
            "kmeans": {"ratio_cut": kmeans, "labels": [0, 1]},
            "discretize": {"ratio_cut": 11.0, "labels": [0, 1]},
            "cluster_qr": {"ratio_cut": 13.0, "labels": [0, 1]},
        },
        "best_datum_name": "discretize",
        "planted_labels_ratio_cut": 7.0,
        "generator": {"strength": 0.5},
        "seconds": 1.0,
    }


def make_cell(
    key,
    *,
    rung,
    name,
    graph_seed,
    init,
    nu,
    mu,
    status,
    rounded=None,
    relaxed=None,
    floor=6.0,
    components=None,
    seconds=1.0,
    rss=None,
    node_count=60,
    cluster_count=2,
):
    gap_relaxed = None if rounded is None or relaxed is None else rounded - relaxed
    gap_floor = None if rounded is None or floor is None else rounded - floor
    return {
        "record": "cell",
        "schema_version": 1,
        "key": key,
        "rung": rung,
        "name": name,
        "graph_seed": graph_seed,
        "init_seed": 0,
        "init": init,
        "collision_weight": 1.0,
        "diversity_weight": nu,
        "adjacency_weight": mu,
        "adjacency_form": "edge_product",
        "step_count": 300,
        "learning_rate": 0.05,
        "checkpoint_steps": [100, 200],
        "host": "laptop",
        "status": status,
        "converged": status == "reached_k",
        "steps_to_convergence": 120 if status == "reached_k" else None,
        "drift_onset_step": 40 if status == "drifted" else None,
        "relaxed_objective": relaxed,
        "rounded_cut": rounded,
        "rounded_cut_minus_relaxed": gap_relaxed,
        "rounded_cut_minus_floor": gap_floor,
        "component_count": components,
        "collision_measures": [0.1],
        "labels": [0, 1],
        "max_zero_sum_violation": 1e-9,
        "final_training_loss": 0.5,
        "checkpoints": [],
        "seconds": seconds,
        "peak_rss_mb": rss,
        "spectral_floor": floor,
        "cluster_count": cluster_count,
        "node_count": node_count,
        "datum_ratio_cuts": dict(DATUM_CUTS),
        "planted_labels_ratio_cut": 7.0,
        "cap_seconds": 600.0,
        "predicted_rss_gb": 0.5,
        "error": None,
        "engine": {"botmaker_commit": "abc1234", "engine_sha256": {"optimizer": "0" * 64}},
    }


def synthetic_records():
    """Two rungs; rung 1 is a 2 seeds x 2 settings x 2 inits block with mixed statuses."""
    graphs = [
        make_graph(1, "sbm_a", 0, kmeans=12.0, node_count=60, cluster_count=2),
        make_graph(1, "sbm_a", 1, kmeans=12.5, node_count=60, cluster_count=2),
        make_graph(2, "sbm_b", 0, kmeans=25.0, node_count=200, cluster_count=3),
    ]
    cells = [
        make_cell(
            "r1-s0-a-spectral",
            rung=1,
            name="sbm_a",
            graph_seed=0,
            init="spectral",
            nu=3.0,
            mu=0.1,
            status="reached_k",
            rounded=10.0,
            relaxed=8.0,
            components=2,
            seconds=1.0,
            rss=100.0,
        ),
        make_cell(
            "r1-s0-a-random",
            rung=1,
            name="sbm_a",
            graph_seed=0,
            init="random",
            nu=3.0,
            mu=0.1,
            status="drifted",
            rounded=8.0,
            relaxed=7.0,
            components=3,
            seconds=2.0,
            rss=110.0,
        ),
        make_cell(
            "r1-s1-a-spectral",
            rung=1,
            name="sbm_a",
            graph_seed=1,
            init="spectral",
            nu=3.0,
            mu=0.1,
            status="reached_k",
            rounded=11.0,
            relaxed=8.5,
            components=2,
            seconds=3.0,
            rss=120.0,
        ),
        make_cell(
            "r1-s1-a-random",
            rung=1,
            name="sbm_a",
            graph_seed=1,
            init="random",
            nu=3.0,
            mu=0.1,
            status="not_at_k",
            rounded=14.0,
            relaxed=10.0,
            components=1,
            seconds=4.0,
            rss=130.0,
        ),
        make_cell(
            "r1-s0-b-spectral",
            rung=1,
            name="sbm_a",
            graph_seed=0,
            init="spectral",
            nu=10.0,
            mu=0.3,
            status="reached_k",
            rounded=9.0,
            relaxed=7.0,
            components=2,
            seconds=5.0,
            rss=140.0,
        ),
        make_cell(
            "r1-s0-b-random",
            rung=1,
            name="sbm_a",
            graph_seed=0,
            init="random",
            nu=10.0,
            mu=0.3,
            status="error",
            seconds=0.5,
        ),
        make_cell(
            "r1-s1-b-spectral",
            rung=1,
            name="sbm_a",
            graph_seed=1,
            init="spectral",
            nu=10.0,
            mu=0.3,
            status="drifted",
            rounded=9.5,
            relaxed=7.5,
            components=2,
            seconds=6.0,
            rss=150.0,
        ),
        make_cell(
            "r1-s1-b-random",
            rung=1,
            name="sbm_a",
            graph_seed=1,
            init="random",
            nu=10.0,
            mu=0.3,
            status="not_at_k",
            rounded=13.0,
            relaxed=9.0,
            components=4,
            seconds=7.0,
            rss=160.0,
        ),
        make_cell(
            "r2-a-spectral",
            rung=2,
            name="sbm_b",
            graph_seed=0,
            init="spectral",
            nu=3.0,
            mu=0.1,
            status="reached_k",
            rounded=20.0,
            relaxed=18.0,
            floor=15.0,
            components=3,
            seconds=20.0,
            rss=500.0,
            node_count=200,
            cluster_count=3,
        ),
        make_cell(
            "r2-a-random",
            rung=2,
            name="sbm_b",
            graph_seed=0,
            init="random",
            nu=10.0,
            mu=0.3,
            status="error",
            floor=None,
            seconds=30.0,
            node_count=200,
            cluster_count=3,
        ),
    ]
    oracles = [
        {
            "record": "oracle",
            "schema_version": 1,
            "check": "relaxed_objective_matches_numpy",
            "passed": True,
            "max_abs_difference": 1.5e-12,
            "mismatched_fields": [],
            "checkpoints": 2,
            "report_path": "reports/oracle.md",
            "seconds": 0.25,
        },
        {
            "record": "oracle",
            "schema_version": 1,
            "check": "labels_match_numpy",
            "passed": False,
            "max_abs_difference": None,
            "mismatched_fields": ["labels"],
            "checkpoints": 2,
            "report_path": "reports/oracle.md",
            "seconds": 0.5,
        },
    ]
    return [*oracles, *graphs, *cells]


def table_rows(text):
    """The pipe rows of a rendered table, separators dropped."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def row_with(rows, prefix):
    """The one row whose leading cells are `prefix`."""
    return next(row for row in rows if row[: len(prefix)] == list(prefix))


def row_starting(text, first):
    for row in table_rows(text):
        if row[0] == first:
            return row
    raise AssertionError(f"no row starting {first!r}")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------------------


def test_merge_keeps_one_record_per_cell_key_and_counts_the_duplicates():
    records = synthetic_records()
    repeat = [dict(records[-1])]
    merged, conflicts, duplicates = merge_result_files([records, repeat])
    assert len(merged) == len(records)
    assert duplicates == 1
    assert conflicts == []


def test_a_duplicate_with_a_different_status_is_reported_as_a_conflict():
    records = synthetic_records()
    later = dict(next(r for r in records if r.get("key") == "r1-s0-a-spectral"))
    later["status"] = "not_at_k"
    merged, conflicts, duplicates = merge_result_files([records, [later]])
    assert duplicates == 1
    assert len(conflicts) == 1
    assert conflicts[0]["identity"] == ("cell", "r1-s0-a-spectral")
    assert conflicts[0]["differences"]["status"] == ("reached_k", "not_at_k")
    kept = next(r for r in merged if r.get("key") == "r1-s0-a-spectral")
    assert kept["status"] == "reached_k"


def test_prefer_later_keeps_the_later_record_on_a_conflict():
    records = synthetic_records()
    later = dict(next(r for r in records if r.get("key") == "r1-s0-a-spectral"))
    later["rounded_cut"] = 99.0
    merged, conflicts, _ = merge_result_files([records, [later]], prefer_later=True)
    assert len(conflicts) == 1
    kept = next(r for r in merged if r.get("key") == "r1-s0-a-spectral")
    assert kept["rounded_cut"] == 99.0


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def test_per_rung_counts_and_medians_are_the_hand_values():
    row = row_starting(rung_table(synthetic_records()), "1")
    # rung, cells, reached_k, drifted, not_at_k, stopped, error, timeout, skipped, over_budget,
    # converged
    assert row[:11] == ["1", "8", "3", "2", "2", "0", "1", "0", "0", "0", "3"]
    # Ê-E* over the seven cells with numbers: 1, 2, 2, 2, 2.5, 4, 4
    assert row[11:13] == ["2", "4"]
    # Ê-Σλ: 2, 3, 3.5, 4, 5, 7, 8
    assert row[13:15] == ["4", "8"]
    # seconds over all eight cells: 0.5, 1, 2, 3, 4, 5, 6, 7 ; RSS over the seven that reported
    assert row[15:] == ["3.5", "7", "130", "160", "60"]


def test_a_stopped_cell_counts_in_the_stopped_column_and_contributes_no_number():
    """One stopped record appended to the synthetic set, so the hand values above stay as read."""
    stopped = make_cell(
        "r1-s0-c-random",
        rung=1,
        name="sbm_a",
        graph_seed=0,
        init="random",
        nu=3.0,
        mu=0.1,
        status="stopped",
        seconds=0.25,
    )
    records = [*synthetic_records(), stopped]
    row = row_starting(rung_table(records), "1")

    # rung, cells, reached_k, drifted, not_at_k, stopped, error, timeout, skipped, over_budget,
    # converged: one more cell, one stopped, nothing else moves.
    assert row[:11] == ["1", "9", "3", "2", "2", "1", "1", "0", "0", "0", "3"]
    # The stopped cell has no numbers, so the Ê-E* and Ê-Σλ medians are the eight-cell ones.
    assert row[11:15] == ["2", "4", "4", "8"]


def test_a_rung_whose_error_cell_has_no_numbers_renders_the_missing_marker():
    table = setting_table(synthetic_records())
    row = row_starting(table, "2")  # rung 2, first setting row
    error_row = next(r for r in table_rows(table) if r[0] == "2" and r[3] == "random")
    assert error_row[4:7] == ["1", "0", "0"]
    assert error_row[7:10] == [MISSING, MISSING, MISSING]
    assert row is not None


def test_the_per_setting_reached_k_fraction_is_the_hand_fraction():
    rows = table_rows(setting_table(synthetic_records()))
    spectral_a = row_with(rows, ["1", "3", "0.1", "spectral"])
    assert spectral_a[4:8] == ["2", "2", "1", "2.25"]
    random_a = row_with(rows, ["1", "3", "0.1", "random"])
    assert random_a[4:8] == ["2", "0", "0", "2.5"]
    spectral_b = row_with(rows, ["1", "10", "0.3", "spectral"])
    # one of the two reached K, the other drifted
    assert spectral_b[4:8] == ["2", "1", "0.5", "2"]
    assert spectral_b[-1] == "1"


def test_the_per_setting_datum_ratio_is_e_hat_over_the_best_of_the_three_roundings():
    rows = table_rows(setting_table(synthetic_records()))
    spectral_b = row_with(rows, ["1", "10", "0.3", "spectral"])
    # best datum Ê is 11 (discretize); the two cells scored 9 and 9.5
    assert spectral_b[9] == f"{((9.0 / 11.0) + (9.5 / 11.0)) / 2:.4g}"


def test_the_graph_row_medians_the_seeded_roundings_and_names_the_best_cell_at_k():
    row = row_starting(graph_table(synthetic_records()), "1")
    assert row[1:3] == ["sbm_a", "2"]
    assert row[3:7] == ["12.25", "11", "13", "7"]  # kmeans median over seeds, then the rest
    assert row[7] == "9"  # best Ê among the four cells at K
    assert row[8] == "ν=10, μ=0.3, spectral"
    assert row[9] == "8"  # best Ê overall, from a drifted cell that is not at K
    assert row[10] == "4 / 8"


def test_the_sweep_grid_counts_reached_k_over_cells_summed_across_seeds():
    text = sweep_grid_tables(synthetic_records())
    assert "n = 60, K = 2, strength = 0.5" in text
    rows = table_rows(text)
    # random init only: one cell per seed in each square, neither of them reached K
    nu_three = [r for r in rows if r[0] == "3"]
    assert nu_three[0] == ["3", "0 / 2", MISSING, MISSING]  # counts grid
    assert nu_three[1] == ["3", "2.5", MISSING, MISSING]  # median Ê-E*, from gaps 1 and 4
    nu_ten = [r for r in rows if r[0] == "10"]
    assert nu_ten[0] == ["10", MISSING, "0 / 2", MISSING]
    assert nu_ten[1] == ["10", MISSING, "4", MISSING]  # the error cell contributes no number


def test_the_sweep_grid_leaves_out_spectral_init_cells_at_a_sweep_setting():
    """The sweep is a random-init sweep. A (ν, μ) square that the campaign also ran at the spectral
    init would otherwise hold twice the cells of its neighbours and compare nothing."""
    shared = dict(
        rung=1,
        name="sbm_a",
        graph_seed=0,
        nu=10.0,
        mu=0.3,
        status="reached_k",
        rounded=9.0,
        relaxed=7.0,
        components=2,
        seconds=1.0,
        rss=100.0,
    )
    records = [
        *synthetic_records(),
        make_cell("extra-random", init="random", **shared),
        make_cell("extra-spectral", init="spectral", **shared),
    ]
    text = sweep_grid_tables(records)
    assert "random init only" in text
    nu_ten = [r for r in table_rows(text) if r[0] == "10"]
    # the random cell joins the square and reaches K; the spectral one is not counted at all
    assert nu_ten[0] == ["10", MISSING, "1 / 3", MISSING]


def test_restricting_to_one_rung_drops_the_other_rungs_cells_but_keeps_the_oracle():
    shown = restrict_to_rung(synthetic_records(), 2)
    assert {r.get("rung") for r in shown if r.get("record") == "cell"} == {2}
    assert len(shown) == 2 + 1 + 2  # two oracle checks, one graph, two cells
    summary = render_summary(shown)
    assert "sbm_a" not in summary


def test_the_oracle_table_shows_a_missing_difference_as_the_dash():
    row = row_starting(render_summary(synthetic_records()), "labels_match_numpy")
    assert row[1:] == ["no", MISSING, "0.5"]


# --------------------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------------------


def test_freeze_writes_the_sorted_file_and_a_matching_sha256(tmp_path):
    records = synthetic_records()
    target = tmp_path / "merged.jsonl"
    digest = write_frozen(records, target)

    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    kinds = [line["record"] for line in lines]
    assert kinds == ["oracle"] * 2 + ["graph"] * 3 + ["cell"] * 10
    assert [line["check"] for line in lines[:2]] == [
        "labels_match_numpy",
        "relaxed_objective_matches_numpy",
    ]
    assert [(line["rung"], line["name"], line["graph_seed"]) for line in lines[2:5]] == [
        (1, "sbm_a", 0),
        (1, "sbm_a", 1),
        (2, "sbm_b", 0),
    ]
    assert [line["key"] for line in lines[5:]] == sorted(line["key"] for line in lines[5:])

    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == expected
    sidecar = (tmp_path / "merged.sha256").read_text(encoding="utf-8")
    assert sidecar == f"{expected}  merged.jsonl\n"


def test_the_frozen_json_holds_only_builtin_scalars(tmp_path):
    """A numpy scalar would either crash json.dumps or round-trip as a bare float; assert neither
    happens by checking every leaf against the builtin types the schema allows."""

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        else:
            assert value is None or type(value) in (bool, int, float, str), type(value)

    text = freeze_text(synthetic_records())
    for line in text.splitlines():
        check(json.loads(line))


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------


def test_main_merges_two_files_prints_the_tables_and_freezes(tmp_path, capsys):
    records = synthetic_records()
    first = write_jsonl(tmp_path / "laptop.jsonl", records[:8])
    second = write_jsonl(tmp_path / "cloud.jsonl", records[6:])
    frozen = tmp_path / "out.jsonl"
    markdown = tmp_path / "out.md"

    code = main(
        [
            "--results",
            str(first),
            str(second),
            "--freeze",
            str(frozen),
            "--markdown",
            str(markdown),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "duplicates: 2; conflicts: 0" in printed
    assert "### Per rung" in printed
    assert markdown.read_text(encoding="utf-8").startswith("### Merge")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert f"frozen sha256: {digest}" in printed
    assert len(frozen.read_text(encoding="utf-8").splitlines()) == len(records)


def test_main_refuses_to_freeze_while_a_conflict_stands(tmp_path, capsys):
    records = synthetic_records()
    conflicting = dict(next(r for r in records if r.get("key") == "r1-s0-a-spectral"))
    conflicting["status"] = "timeout"
    first = write_jsonl(tmp_path / "a.jsonl", records)
    second = write_jsonl(tmp_path / "b.jsonl", [conflicting])
    frozen = tmp_path / "out.jsonl"

    code = main(["--results", str(first), str(second), "--freeze", str(frozen)])
    captured = capsys.readouterr()
    assert code == 1
    assert not frozen.exists()
    assert "refusing --freeze" in captured.err
    assert "conflicts: 1" in captured.out

    code = main(["--results", str(first), str(second), "--freeze", str(frozen), "--prefer-later"])
    assert code == 0
    assert frozen.exists()


def test_main_restricts_the_tables_to_one_rung(tmp_path, capsys):
    path = write_jsonl(tmp_path / "all.jsonl", synthetic_records())
    assert main(["--results", str(path), "--rung", "1"]) == 0
    printed = capsys.readouterr().out
    assert "sbm_a" in printed
    assert "sbm_b" not in printed
