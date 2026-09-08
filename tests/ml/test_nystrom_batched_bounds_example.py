"""The committed output of ``examples/nystrom_batched_bounds.py`` agrees with the search baseline.

The example is the timing record for engine changes and its ``--no-timings`` output is committed
so a PR can show it byte-identical. This test does not run the example (it takes about 40 s); it
checks that the committed record and ``nystrom_search_snapshot.json`` tell the same story on the
cells they share, and that the example sweeps the bandwidth scales the plan names.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "nystrom_batched_bounds.py"
OUTPUT_PATH = REPOSITORY_ROOT / "examples" / "nystrom_batched_bounds.example_output.txt"
SNAPSHOT_PATH = HERE / "nystrom_search_snapshot.json"

# "  40  3       668  True  [7, 16, 32]" — a row of the search comparison block.
SEARCH_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(True|False)\s+\[(.*)\]$")
# "  0.25       363  [16, 17, 32]" — a row of the expansions-by-bandwidth block.
SCALE_ROW = re.compile(r"^\s*([\d.]+)\s+(\d+)\s+\[(.*)\]$")


@pytest.fixture(scope="module")
def example_module():
    spec = importlib.util.spec_from_file_location("nystrom_batched_bounds_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed_output() -> list[str]:
    return OUTPUT_PATH.read_text().splitlines()


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text())


def test_the_example_sweeps_the_five_bandwidth_scales_the_plan_names(example_module):
    assert example_module.EXPANSION_COUNT_SCALES == (0.25, 0.5, 1.0, 2.0, 4.0)


def test_the_example_runs_the_harness_at_its_default_scales(example_module):
    assert example_module.BANDWIDTH_SCALES == (0.25, 1.0, 4.0)


def test_every_committed_search_row_reports_the_same_search_as_the_baseline(
    committed_output, snapshot
):
    rows = [SEARCH_ROW.match(line) for line in committed_output]
    rows = [match for match in rows if match]
    assert len(rows) == 5, "the search comparison block should have five rows"
    for match in rows:
        n, k, expanded, same, landmarks = match.groups()
        assert same == "True", f"n={n}, k={k}: batched and per-child bounds disagreed"
        cell = snapshot[f"SPECTF:n={n},k={k},scale=1.0"]
        assert int(expanded) == cell["nodes_expanded"], (n, k)
        assert [int(index) for index in landmarks.split(", ")] in cell["accepted_states"], (n, k)


def test_the_committed_bandwidth_sweep_matches_the_baseline_where_they_share_a_cell(
    committed_output, snapshot
):
    rows = [SCALE_ROW.match(line) for line in committed_output]
    rows = [match for match in rows if match]
    assert [float(match.group(1)) for match in rows] == [0.25, 0.5, 1.0, 2.0, 4.0]
    for match in rows:
        scale, expanded, landmarks = match.groups()
        key = f"SPECTF:n=40,k=3,scale={float(scale)}"
        if key not in snapshot:
            continue
        assert int(expanded) == snapshot[key]["nodes_expanded"], scale
        assert [int(i) for i in landmarks.split(", ")] in snapshot[key]["accepted_states"], scale
