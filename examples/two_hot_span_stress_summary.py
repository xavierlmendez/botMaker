"""Summarise a two-hot span stress run: merge the results files, print the per-rung tables.

The stress runner (`mllib.ml.projects.two_hot_span_stress`) appends one JSON object per line to a
results file, and a campaign produces more than one of them: the laptop writes one, the cloud host
writes another that is fetched afterwards, and a re-run after a crash writes a third that overlaps
the first. This script is the *reading* end of that: it merges the files on the runner's own keys,
says where two files disagree about the same cell, and prints the tables that answer the questions
the campaign was run to answer — did the cells reach K components, how far above the relaxed
objective and the spectral floor did the rounding land, and what did it cost in wall-clock and
resident memory.

Merging is by identity, not by position: a cell is identified by its `key`, a graph by
(rung, name, graph_seed) and an oracle check by its `check` name. The first file named on the
command line wins a tie, because the convention in the campaign is to name the trusted run first.
A later duplicate that disagrees about the *status* or about Ê is not a tie — it means two runs of
the same cell produced different answers, which is a fact about the experiment and not a merge
detail — so those are collected into a conflicts list, printed, and they block `--freeze` until
`--prefer-later` says which side to keep.

`--freeze` writes the merged records back out in a fixed order with a sidecar `.sha256`, so that the
file a later analysis reads is the file this summary described, byte for byte.

    uv run python examples/two_hot_span_stress_summary.py --results results.jsonl
    uv run python examples/two_hot_span_stress_summary.py --results laptop.jsonl cloud.jsonl \\
        --freeze merged.jsonl --markdown summary.md

Every table is built by a pure function over a list of plain dicts — the same dicts `json.loads`
produces — so the tests feed synthetic records and never need the runner, torch, or a results file.
`main` only parses arguments, reads files, and prints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

MISSING = "—"

CELL_STATUSES: tuple[str, ...] = (
    "reached_k",
    "drifted",
    "not_at_k",
    "stopped",
    "error",
    "timeout",
    "skipped_deadline",
    "over_budget",
)

STATUS_HEADERS: dict[str, str] = {
    "reached_k": "reached_k",
    "drifted": "drifted",
    "not_at_k": "not_at_k",
    "stopped": "stopped",
    "error": "error",
    "timeout": "timeout",
    "skipped_deadline": "skipped",
    "over_budget": "over_budget",
}

DATUM_NAMES: tuple[str, ...] = ("kmeans", "discretize", "cluster_qr")

SWEEP_DIVERSITY_WEIGHTS: tuple[float, ...] = (3.0, 10.0, 30.0)
SWEEP_ADJACENCY_WEIGHTS: tuple[float, ...] = (0.1, 0.3, 1.0)
SWEEP_INIT = "random"  # the rung-1 sweep is pre-registered as a random-init sweep


# --------------------------------------------------------------------------------------
# reading and merging
# --------------------------------------------------------------------------------------


def parse_records(text: str) -> list[dict[str, Any]]:
    """Parse a JSONL body. Blank lines are skipped; the runner may end a file mid-flush."""
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def read_records(path: Path) -> list[dict[str, Any]]:
    return parse_records(path.read_text(encoding="utf-8"))


def record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    """The merge key: what makes two records the same record rather than two observations."""
    kind = record.get("record")
    if kind == "cell":
        return ("cell", record.get("key"))
    if kind == "graph":
        return ("graph", record.get("rung"), record.get("name"), record.get("graph_seed"))
    if kind == "oracle":
        return ("oracle", record.get("check"))
    return ("other", json.dumps(record, sort_keys=True))


def _conflict_fields(kind: str) -> tuple[str, ...]:
    """Which fields make a duplicate a disagreement rather than a repeat.

    For a cell the brief names status and Ê. A graph carries no status, so the analogous pair is
    which rounding won and what the planted labels scored; an oracle check disagrees when it
    passed in one file and failed in the other.
    """
    if kind == "cell":
        return ("status", "rounded_cut")
    if kind == "graph":
        return ("best_datum_name", "planted_labels_ratio_cut")
    if kind == "oracle":
        return ("passed",)
    return ()


def merge_result_files(
    files: Sequence[Sequence[dict[str, Any]]],
    *,
    prefer_later: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Merge records from several files by identity.

    Returns the merged records in first-seen order, the conflicts, and the number of duplicate
    records dropped (or replaced). The first file wins unless ``prefer_later``, which is the flag
    the operator sets when the *later* file is the trusted re-run.
    """
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    origin: dict[tuple[Any, ...], int] = {}
    conflicts: list[dict[str, Any]] = []
    duplicate_count = 0

    for file_index, records in enumerate(files):
        for record in records:
            identity = record_identity(record)
            if identity not in merged:
                merged[identity] = record
                origin[identity] = file_index
                continue

            duplicate_count += 1
            kept = merged[identity]
            differences = {
                field: (kept.get(field), record.get(field))
                for field in _conflict_fields(identity[0])
                if kept.get(field) != record.get(field)
            }
            if differences:
                conflicts.append(
                    {
                        "identity": identity,
                        "first_file": origin[identity],
                        "later_file": file_index,
                        "differences": differences,
                    }
                )
            if prefer_later:
                merged[identity] = record
                origin[identity] = file_index

    return list(merged.values()), conflicts, duplicate_count


def records_of(records: Iterable[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record") == kind]


def restrict_to_rung(records: Sequence[dict[str, Any]], rung: int | None) -> list[dict[str, Any]]:
    """Keep the oracle records and only the graphs and cells of one rung."""
    if rung is None:
        return list(records)
    kept = []
    for record in records:
        if record.get("record") == "oracle" or record.get("rung") == rung:
            kept.append(record)
    return kept


# --------------------------------------------------------------------------------------
# formatting primitives
# --------------------------------------------------------------------------------------


def format_number(value: Any, digits: int = 4) -> str:
    """Four significant figures, integers as integers, anything absent as an em dash."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return MISSING
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.{digits}g}"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A Markdown pipe table. Empty bodies still print a header, so the reader sees the zero."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _numbers(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool | str):
            continue
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            continue
        out.append(number)
    return out


def median_of(values: Iterable[Any]) -> float | None:
    numbers = _numbers(values)
    if not numbers:
        return None
    return float(np.median(np.asarray(numbers, dtype=float)))


def max_of(values: Iterable[Any]) -> float | None:
    numbers = _numbers(values)
    if not numbers:
        return None
    return float(np.max(np.asarray(numbers, dtype=float)))


def group_by(
    records: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Any]
) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return groups


def _sort_key(value: Any) -> tuple[int, Any]:
    """Sort mixed keys without comparing None to a number, or a str to an int."""
    if value is None:
        return (2, 0)
    if isinstance(value, bool | int | float):
        return (0, float(value))
    return (1, str(value))


def sorted_groups(groups: dict[Any, list[dict[str, Any]]]) -> list[Any]:
    return sorted(groups, key=lambda key: tuple(_sort_key(part) for part in _as_tuple(key)))


def _as_tuple(key: Any) -> tuple[Any, ...]:
    return key if isinstance(key, tuple) else (key,)


# --------------------------------------------------------------------------------------
# derived cell quantities
# --------------------------------------------------------------------------------------


def excess_over_relaxed(cell: dict[str, Any]) -> float | None:
    """Ê - E*, from the runner's field, or recomputed when the runner did not store it."""
    value = cell.get("rounded_cut_minus_relaxed")
    if value is not None:
        return float(value)
    rounded, relaxed = cell.get("rounded_cut"), cell.get("relaxed_objective")
    if rounded is None or relaxed is None:
        return None
    return float(rounded) - float(relaxed)


def excess_over_floor(cell: dict[str, Any]) -> float | None:
    """Ê - Σλ, the gap to the spectral floor."""
    value = cell.get("rounded_cut_minus_floor")
    if value is not None:
        return float(value)
    rounded, floor = cell.get("rounded_cut"), cell.get("spectral_floor")
    if rounded is None or floor is None:
        return None
    return float(rounded) - float(floor)


def best_datum_ratio_cut(cell_or_graph: dict[str, Any]) -> float | None:
    """The lowest RatioCut among the three spectral roundings — the bar a cell has to beat."""
    cuts = cell_or_graph.get("datum_ratio_cuts")
    if cuts is None:
        datum = cell_or_graph.get("datum") or {}
        cuts = {name: (datum.get(name) or {}).get("ratio_cut") for name in DATUM_NAMES}
    numbers = _numbers(cuts.values())
    return min(numbers) if numbers else None


def datum_ratio(cell: dict[str, Any]) -> float | None:
    """Ê divided by the best datum Ê: below 1 the optimiser beat every spectral rounding."""
    rounded = cell.get("rounded_cut")
    best = best_datum_ratio_cut(cell)
    if rounded is None or best is None or best == 0:
        return None
    return float(rounded) / best


def is_at_k(cell: dict[str, Any]) -> bool:
    """A cell is 'at K' when its rounding produced exactly the requested number of components."""
    components, clusters = cell.get("component_count"), cell.get("cluster_count")
    return components is not None and clusters is not None and components == clusters


def setting_label(cell: dict[str, Any]) -> str:
    return f"ν={format_number(cell.get('diversity_weight'))}, μ={format_number(cell.get('adjacency_weight'))}"


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def oracle_table(records: Sequence[dict[str, Any]]) -> str:
    """Table 1 — the parity checks between the torch run and the numpy reference."""
    rows = [
        [
            str(record.get("check", MISSING)),
            format_number(record.get("passed")),
            format_number(record.get("max_abs_difference")),
            format_number(record.get("seconds")),
        ]
        for record in records_of(records, "oracle")
    ]
    return "### Oracle checks\n\n" + render_table(
        ["check", "passed", "max abs diff", "seconds"], rows
    )


def rung_table(records: Sequence[dict[str, Any]]) -> str:
    """Table 2 — one row per rung: how the cells ended, how far off, and what they cost."""
    cells = records_of(records, "cell")
    groups = group_by(cells, lambda cell: cell.get("rung"))
    headers = (
        ["rung", "cells"]
        + [STATUS_HEADERS[status] for status in CELL_STATUSES]
        + [
            "converged",
            "med Ê-E*",
            "max Ê-E*",
            "med Ê-Σλ",
            "max Ê-Σλ",
            "med s",
            "max s",
            "med RSS MB",
            "max RSS MB",
            "max nodes",
        ]
    )
    rows = []
    for rung in sorted_groups(groups):
        group = groups[rung]
        relaxed_gaps = [excess_over_relaxed(cell) for cell in group]
        floor_gaps = [excess_over_floor(cell) for cell in group]
        row = [format_number(rung), str(len(group))]
        row += [
            str(sum(1 for cell in group if cell.get("status") == status))
            for status in CELL_STATUSES
        ]
        row += [
            str(sum(1 for cell in group if cell.get("converged") is True)),
            format_number(median_of(relaxed_gaps)),
            format_number(max_of(relaxed_gaps)),
            format_number(median_of(floor_gaps)),
            format_number(max_of(floor_gaps)),
            format_number(median_of(cell.get("seconds") for cell in group)),
            format_number(max_of(cell.get("seconds") for cell in group)),
            format_number(median_of(cell.get("peak_rss_mb") for cell in group)),
            format_number(max_of(cell.get("peak_rss_mb") for cell in group)),
            format_number(max_of(cell.get("node_count") for cell in group)),
        ]
        rows.append(row)
    return "### Per rung\n\n" + render_table(headers, rows)


def setting_table(records: Sequence[dict[str, Any]]) -> str:
    """Table 3 — rung x coupling setting (ν, μ) x initialisation."""
    cells = records_of(records, "cell")
    groups = group_by(
        cells,
        lambda cell: (
            cell.get("rung"),
            cell.get("diversity_weight"),
            cell.get("adjacency_weight"),
            cell.get("init"),
        ),
    )
    headers = [
        "rung",
        "ν",
        "μ",
        "init",
        "cells",
        "reached_k",
        "frac",
        "med Ê-E*",
        "med Ê-Σλ",
        "med Ê/datum",
        "drifted",
    ]
    rows = []
    for key in sorted_groups(groups):
        rung, diversity, adjacency, init = key
        group = groups[key]
        reached = sum(1 for cell in group if cell.get("status") == "reached_k")
        rows.append(
            [
                format_number(rung),
                format_number(diversity),
                format_number(adjacency),
                str(init if init is not None else MISSING),
                str(len(group)),
                str(reached),
                format_number(reached / len(group)),
                format_number(median_of(excess_over_relaxed(cell) for cell in group)),
                format_number(median_of(excess_over_floor(cell) for cell in group)),
                format_number(median_of(datum_ratio(cell) for cell in group)),
                str(sum(1 for cell in group if cell.get("status") == "drifted")),
            ]
        )
    return "### Per rung x coupling setting x init\n\n" + render_table(headers, rows)


def graph_table(records: Sequence[dict[str, Any]]) -> str:
    """Table 4 — one row per graph: the spectral baselines against the best cell.

    The three roundings are medians over the graph seeds because k-means is itself seeded, so a
    single seed's number is not the graph's number.
    """
    graphs = records_of(records, "graph")
    cells = records_of(records, "cell")
    graph_groups = group_by(graphs, lambda graph: (graph.get("rung"), graph.get("name")))
    cell_groups = group_by(cells, lambda cell: (cell.get("rung"), cell.get("name")))

    headers = [
        "rung",
        "graph",
        "seeds",
        "Ê kmeans",
        "Ê discretize",
        "Ê cluster_qr",
        "Ê planted",
        "best Ê at K",
        "best-at-K setting",
        "best Ê any",
        "cells at K / run",
    ]
    rows = []
    for key in sorted_groups(graph_groups):
        rung, name = key
        group = graph_groups[key]
        row = [format_number(rung), str(name), str(len(group))]
        for datum_name in DATUM_NAMES:
            row.append(
                format_number(
                    median_of(
                        ((graph.get("datum") or {}).get(datum_name) or {}).get("ratio_cut")
                        for graph in group
                    )
                )
            )
        row.append(
            format_number(median_of(graph.get("planted_labels_ratio_cut") for graph in group))
        )

        own_cells = cell_groups.get(key, [])
        at_k = [cell for cell in own_cells if is_at_k(cell) and cell.get("rounded_cut") is not None]
        if at_k:
            best = min(at_k, key=lambda cell: float(cell["rounded_cut"]))
            row.append(format_number(best.get("rounded_cut")))
            row.append(f"{setting_label(best)}, {best.get('init', MISSING)}")
        else:
            row.append(MISSING)
            row.append(MISSING)
        row.append(
            format_number(
                min(_numbers(cell.get("rounded_cut") for cell in own_cells), default=None)
            )
        )
        row.append(f"{len(at_k)} / {len(own_cells)}")
        rows.append(row)
    return "### Per graph\n\n" + render_table(headers, rows)


def _generator_strength(graph: dict[str, Any]) -> Any:
    generator = graph.get("generator") or {}
    for field in ("strength", "cluster_strength", "signal_strength"):
        if field in generator:
            return generator[field]
    return None


def sweep_grid_tables(records: Sequence[dict[str, Any]], rung: int = 1) -> str:
    """Table 5 — the rung-1 sweep as 3x3 grids of ν against μ, one pair per (n, K, strength).

    Random init only. The sweep is pre-registered as a random-init sweep, while the rest of the
    campaign runs some (ν, μ) squares at the spectral init as well. Letting those in would give a
    square that happens to be shared with the rest of the campaign twice the cells of its
    neighbours, and a grid whose squares count different populations compares nothing.

    The strength of the planted structure lives on the graph record's generator block, not on the
    cell, so the cells are joined back to their graph to find it.
    """
    strength_of: dict[tuple[Any, Any, Any], Any] = {}
    for graph in records_of(records, "graph"):
        identity = (graph.get("rung"), graph.get("name"), graph.get("graph_seed"))
        strength_of[identity] = _generator_strength(graph)

    cells = [
        cell
        for cell in records_of(records, "cell")
        if cell.get("rung") == rung and cell.get("init") == SWEEP_INIT
    ]
    if not cells:
        return ""

    def cell_key(cell: dict[str, Any]) -> tuple[Any, Any, Any]:
        identity = (cell.get("rung"), cell.get("name"), cell.get("graph_seed"))
        return (cell.get("node_count"), cell.get("cluster_count"), strength_of.get(identity))

    groups = group_by(cells, cell_key)
    blocks: list[str] = []
    for key in sorted_groups(groups):
        node_count, cluster_count, strength = key
        group = groups[key]
        by_setting = group_by(
            group, lambda cell: (cell.get("diversity_weight"), cell.get("adjacency_weight"))
        )
        headers = ["ν \\ μ"] + [format_number(mu) for mu in SWEEP_ADJACENCY_WEIGHTS]
        reached_rows = []
        gap_rows = []
        for nu in SWEEP_DIVERSITY_WEIGHTS:
            reached_row = [format_number(nu)]
            gap_row = [format_number(nu)]
            for mu in SWEEP_ADJACENCY_WEIGHTS:
                bucket = by_setting.get((nu, mu), [])
                if not bucket:
                    reached_row.append(MISSING)
                    gap_row.append(MISSING)
                    continue
                reached = sum(1 for cell in bucket if cell.get("status") == "reached_k")
                reached_row.append(f"{reached} / {len(bucket)}")
                gap_row.append(
                    format_number(median_of(excess_over_relaxed(cell) for cell in bucket))
                )
            reached_rows.append(reached_row)
            gap_rows.append(gap_row)
        title = (
            f"n = {format_number(node_count)}, K = {format_number(cluster_count)}, "
            f"strength = {format_number(strength)}"
        )
        blocks.append(
            f"#### Rung {rung} sweep ({SWEEP_INIT} init only) — {title}\n\nreached_k / cells\n\n"
            + render_table(headers, reached_rows)
            + "\n\nmedian Ê-E*\n\n"
            + render_table(headers, gap_rows)
        )
    return "\n\n".join(blocks)


def top_rung_cost_table(records: Sequence[dict[str, Any]]) -> str:
    """Table 6 — what the largest rung present cost, per graph."""
    cells = records_of(records, "cell")
    rungs = _numbers(cell.get("rung") for cell in cells)
    if not rungs:
        return ""
    top = max(rungs)
    top_cells = [
        cell for cell in cells if cell.get("rung") is not None and float(cell["rung"]) == top
    ]
    groups = group_by(top_cells, lambda cell: cell.get("name"))
    rows = []
    for name in sorted_groups(groups):
        group = groups[name]
        rows.append(
            [
                str(name),
                str(len(group)),
                format_number(median_of(cell.get("seconds") for cell in group)),
                format_number(max_of(cell.get("seconds") for cell in group)),
                format_number(median_of(cell.get("peak_rss_mb") for cell in group)),
                format_number(max_of(cell.get("peak_rss_mb") for cell in group)),
            ]
        )
    return f"### Top rung ({format_number(top)}) cost\n\n" + render_table(
        ["graph", "cells", "med s", "max s", "med RSS MB", "max RSS MB"], rows
    )


def render_conflicts(conflicts: Sequence[dict[str, Any]], duplicate_count: int) -> str:
    lines = [f"### Merge\n\nduplicates: {duplicate_count}; conflicts: {len(conflicts)}"]
    for conflict in conflicts:
        identity = ":".join(str(part) for part in conflict["identity"])
        differences = ", ".join(
            f"{field}: {first!r} (file {conflict['first_file']}) vs {later!r} "
            f"(file {conflict['later_file']})"
            for field, (first, later) in conflict["differences"].items()
        )
        lines.append(f"- {identity} — {differences}")
    return "\n".join(lines)


def render_summary(
    records: Sequence[dict[str, Any]],
    *,
    conflicts: Sequence[dict[str, Any]] = (),
    duplicate_count: int = 0,
) -> str:
    """All tables, in the brief's order, separated by one blank line."""
    blocks = [
        render_conflicts(conflicts, duplicate_count),
        oracle_table(records),
        rung_table(records),
        setting_table(records),
        graph_table(records),
        sweep_grid_tables(records),
        top_rung_cost_table(records),
    ]
    return "\n\n".join(block for block in blocks if block)


# --------------------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------------------


def freeze_order(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Oracle records, then graphs by (rung, name, graph_seed), then cells by key."""

    def graph_key(record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            _sort_key(part)
            for part in (record.get("rung"), record.get("name"), record.get("graph_seed"))
        )

    oracles = sorted(records_of(records, "oracle"), key=lambda record: str(record.get("check")))
    graphs = sorted(records_of(records, "graph"), key=graph_key)
    cells = sorted(records_of(records, "cell"), key=lambda record: str(record.get("key")))
    known = {"oracle", "graph", "cell"}
    others = [record for record in records if record.get("record") not in known]
    return [*oracles, *graphs, *cells, *others]


def freeze_text(records: Sequence[dict[str, Any]]) -> str:
    """The frozen body: one compact JSON object per line, in the fixed order."""
    lines = [json.dumps(record, sort_keys=True) for record in freeze_order(records)]
    return "".join(line + "\n" for line in lines)


def write_frozen(records: Sequence[dict[str, Any]], path: Path) -> str:
    """Write the merged file and its sidecar digest; return the hex digest."""
    text = freeze_text(records)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    stem = str(path)
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    Path(stem + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results", nargs="+", required=True, help="results JSONL files, trusted first"
    )
    parser.add_argument(
        "--freeze", default=None, help="write the merged JSONL and its .sha256 here"
    )
    parser.add_argument("--markdown", default=None, help="also write the tables to this file")
    parser.add_argument("--rung", type=int, default=None, help="restrict the tables to one rung")
    parser.add_argument(
        "--prefer-later",
        action="store_true",
        help="on a conflicting duplicate keep the later file's record, and allow --freeze",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    files = [read_records(Path(path)) for path in arguments.results]
    merged, conflicts, duplicate_count = merge_result_files(
        files, prefer_later=arguments.prefer_later
    )
    shown = restrict_to_rung(merged, arguments.rung)

    summary = render_summary(shown, conflicts=conflicts, duplicate_count=duplicate_count)
    print(summary)

    if arguments.markdown is not None:
        Path(arguments.markdown).write_text(summary + "\n", encoding="utf-8")

    if arguments.freeze is not None:
        if conflicts and not arguments.prefer_later:
            print(
                f"\nrefusing --freeze: {len(conflicts)} conflict(s); "
                "resolve them or pass --prefer-later",
                file=sys.stderr,
            )
            return 1
        digest = write_frozen(merged, Path(arguments.freeze))
        print(f"\nfrozen sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
