"""The two-hot span example against its committed recording.

`tests/visualization/fixtures/two_hot_span_roach_g5_30steps.json` is a regeneration guard, not a
sample: it is the exact document `record_graph("roach_g5", …, step_count=30, frame_every=5)` writes,
committed so that a change to the recorder, to the optimizer or to the document shape shows up here
as a diff rather than as a walkthrough that quietly says something else. Regenerate it deliberately,
with

    uv run --group torch python examples/two_hot_span_walkthrough.py \\
        --output-dir tests/visualization/fixtures --graph roach_g5 --steps 30 --frame-every 5

renaming the written `roach_g5.json` to the fixture's name and deleting the `roach_g5.html` written
beside it — the page is rendered *from* the recording, so only the recording is committed — and say
so in the PR.

`--frame-every 5` is the example's own default now; the flag is written out anyway, because this
command has to keep producing this fixture even if that default moves again.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from mllib.visualization.recording import Recording
from mllib.visualization.render import main as render_main
from tests.visualization.conftest import assert_same_recording, only_on_fixture_platform

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "two_hot_span_roach_g5_30steps.json"

sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from two_hot_span_walkthrough import (
    GRAPHS,
    ROACH_RUNG_COUNT,
    build_graph,
    layout_positions,
    main,
    record_graph,
)

# The configuration of slices 2.4 and 2.5 that reaches the roach optimum, as the docstring gives it.
REFERENCE_FLAGS = [
    "--graph",
    "roach_g5",
    "--init",
    "random",
    "--collision-weight",
    "10",
    "--adjacency-form",
    "edge_product",
    "--adjacency-weight",
    "0.3",
    "--diversity-weight",
    "10",
    "--output-name",
    "roach_g5_diversity",
]
# 4/15: the antenna cut, the roach's brute-force optimum.
REFERENCE_ROUNDED_CUT = 4.0 / 15.0

STEP_COUNT = 30
FRAME_EVERY = 5

EMBEDDED = re.compile(r'<script id="recording" type="application/json">(.*?)</script>', re.DOTALL)


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> tuple[Path, Path]:
    """Both artefacts the example writes for the fixture's cell, once for the whole module.

    The skip lives here rather than at the top of the module because torch is what *running* the
    optimizer needs (D-31). Where the example puts a node is the instance's geometry, and the tests
    below that ask only that question run whether or not the optional group is installed.
    """
    pytest.importorskip("torch")
    output_dir = tmp_path_factory.mktemp("two_hot_span_walkthrough")
    return record_graph(
        "roach_g5", output_dir, step_count=STEP_COUNT, frame_every=FRAME_EVERY, seed=0
    )


def test_the_example_writes_a_recording_and_a_page_side_by_side(written: tuple[Path, Path]):
    recording_path, page_path = written

    assert recording_path.name == "roach_g5.json"
    assert page_path == recording_path.with_suffix(".html")
    assert page_path.read_text().startswith("<!doctype html>")


def test_the_page_carries_the_recording_the_example_wrote_beside_it(written: tuple[Path, Path]):
    recording_path, page_path = written

    embedded = json.loads(EMBEDDED.search(page_path.read_text()).group(1))

    assert embedded == Recording.load(recording_path).to_dict()


def test_the_cli_renders_the_committed_fixture_through_the_two_hot_span_view(tmp_path):
    # The CLI is given the document and nothing else, so the kind in it is what chooses the view.
    rendered = render_main(["--recording", str(FIXTURE), "--output", str(tmp_path / "page.html")])
    page = rendered.read_text()

    assert 'var KIND = "two_hot_span";' in page
    assert "window.walkthroughViews[KIND] = function" in page


@only_on_fixture_platform
def test_the_example_reproduces_the_committed_recording(written: tuple[Path, Path]):
    assert_same_recording(written[0].read_text(), FIXTURE.read_text())


def test_the_reference_experiment_writes_its_own_pair_and_reaches_the_roach_optimum(tmp_path):
    """The 2.4/2.5 flags, in the combination the docstring names, land the antenna cut.

    `--steps 300 --frame-every 5` is the walkthrough's own default run length; it is written out
    because this run is judged on the partition it ends at, and that is a property of the schedule
    it was given rather than of the flags alone.
    """
    pytest.importorskip("torch")

    main(["--output-dir", str(tmp_path), "--steps", "300", "--frame-every", "5", *REFERENCE_FLAGS])

    recording_path = tmp_path / "roach_g5_diversity.json"
    assert recording_path.exists()
    assert (tmp_path / "roach_g5_diversity.html").exists()
    # The default run's name is not taken: --output-name puts the experiment beside it, not over it.
    assert not (tmp_path / "roach_g5.json").exists()

    recording = Recording.load(recording_path)
    assert recording.result["rounded_cut"] == pytest.approx(REFERENCE_ROUNDED_CUT, abs=1e-9)
    assert recording.result["component_count"] == 2

    problem = recording.problem
    assert problem["init"] == "random"
    assert problem["collision_weight"] == pytest.approx(10.0)
    assert problem["adjacency_weight"] == pytest.approx(0.3)
    assert problem["adjacency_form"] == "edge_product"
    assert problem["diversity_weight"] == pytest.approx(10.0)
    # The schedule was left alone, so it stays out of the document like every other off knob.
    assert "learning_rate_schedule" not in problem


def test_the_default_run_carries_no_experiment_key(written: tuple[Path, Path]):
    """What keeps the committed fixture byte-identical: an off knob is an absent key.

    The byte-for-byte test above is the guard; this one says *why* a default document still matches
    now that `problem` can carry the 2.4/2.5 knobs.
    """
    problem = json.loads(written[0].read_text())["problem"]

    assert problem["init"] == "spectral"
    assert not {"adjacency_weight", "adjacency_form", "diversity_weight"} & set(problem)
    assert "learning_rate_schedule" not in problem


def test_the_layout_places_every_node_and_every_edge(written: tuple[Path, Path]):
    document = json.loads(written[0].read_text())
    layout = document["problem"]["layout"]

    node_total = document["problem"]["n"]
    assert node_total == 4 * ROACH_RUNG_COUNT
    assert len(layout["positions"]) == node_total
    assert all(len(position) == 2 for position in layout["positions"])
    # The ladder: the top path sits at y = 1, the bottom path at y = 0, x along the path.
    assert [position[1] for position in layout["positions"]] == [1.0] * (2 * ROACH_RUNG_COUNT) + [
        0.0
    ] * (2 * ROACH_RUNG_COUNT)
    # n = 4k, m = 5k - 2 (`roach_graph`), and every edge names two of the nodes just placed.
    assert len(layout["edges"]) == 5 * ROACH_RUNG_COUNT - 2
    assert all(0 <= first < second < node_total for first, second in layout["edges"])


def test_the_seeded_spring_layout_places_karate_the_same_way_twice():
    # A recording is regenerated to be diffed against the committed one, so an unseeded layout
    # would make every regeneration differ in the one field that has nothing to do with the run.
    graph = build_graph("karate")

    assert layout_positions("karate", graph) == layout_positions("karate", build_graph("karate"))


def test_the_roach_layout_is_the_ladder_the_graph_is_named_after():
    positions = layout_positions("roach_g5", build_graph("roach_g5"))
    rung_count = ROACH_RUNG_COUNT

    assert len(positions) == 4 * rung_count
    # Two rows and nothing between them: the top path at y = 1, the bottom path at y = 0.
    assert sorted({y for _, y in positions}) == [0.0, 1.0]
    assert [y for _, y in positions] == [1.0] * (2 * rung_count) + [0.0] * (2 * rung_count)
    # x is the position along each path, so both rows run 0, 1, ... 2k - 1 from left to right.
    assert [x for x, _ in positions[: 2 * rung_count]] == [
        float(index) for index in range(2 * rung_count)
    ]
    assert [x for x, _ in positions[2 * rung_count :]] == [
        float(index) for index in range(2 * rung_count)
    ]


def test_the_ladder_layout_generalises_to_the_eighty_node_cockroach():
    """The k = 20 roach draws as a ladder too: k is read off n = 4k, not from a constant."""
    graph = build_graph("roach_g20")
    positions = layout_positions("roach_g20", graph)

    assert graph.number_of_nodes() == 80
    assert len(positions) == 80
    assert [y for _, y in positions] == [1.0] * 40 + [0.0] * 40
    assert [x for x, _ in positions[:40]] == [float(index) for index in range(40)]
    assert [x for x, _ in positions[40:]] == [float(index) for index in range(40)]


def test_the_walkthrough_knows_the_eighty_node_cockroach_at_k_three():
    assert GRAPHS["roach_g20"] == {"cluster_count": 3, "collision_weight": 10.0}


def test_the_walkthrough_runs_two_triangles_at_its_own_lambda_and_schedule():
    """λ = 0.1 against Σλ = 0.064, and a schedule of its own: 5000 steps at lr 0.01."""
    assert GRAPHS["two_triangles"] == {
        "cluster_count": 2,
        "collision_weight": 0.1,
        "learning_rate": 0.01,
        "step_count": 5000,
    }


def test_only_two_triangles_carries_a_schedule_of_its_own():
    """The guard on the older graphs: no entry of theirs overrides anything, so nothing moved."""
    for name in ("roach_g5", "karate", "roach_g20"):
        assert not {"learning_rate", "step_count"} & set(GRAPHS[name]), name


def test_the_two_triangles_layout_separates_the_triangles_with_the_bridge_between_them():
    positions = layout_positions("two_triangles", build_graph("two_triangles"))

    assert len(positions) == 6
    left, right = positions[:3], positions[3:]
    # The two triangles sit apart on x: every left node is left of every right node.
    assert max(x for x, _ in left) < min(x for x, _ in right)
    # Nodes 2 and 3 are the inner pair — the ends of the weak bridge — and it draws horizontally.
    assert positions[2][0] == max(x for x, _ in left)
    assert positions[3][0] == min(x for x, _ in right)
    assert positions[2][1] == positions[3][1]


def test_the_two_triangles_layout_is_handed_out_by_value():
    """A caller that mutates a position cannot move the next run's picture."""
    first = layout_positions("two_triangles", build_graph("two_triangles"))
    first[0][0] = 99.0

    assert layout_positions("two_triangles", build_graph("two_triangles"))[0][0] == 0.0
