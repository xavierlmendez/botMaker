"""The graph-search example against its committed recording, and against the animation it replaced.

`tests/visualization/fixtures/graph_traversal_bfs.json` is a snapshot in the sense CONTRIBUTING
§ "Behavioural baseline" means: the exact document the example writes for the BFS run, byte for
byte, regenerated **deliberately** —

    uv run python examples/graph_search_vs_networkx.py --output-dir tests/visualization/fixtures
    mv tests/visualization/fixtures/bfs.json tests/visualization/fixtures/graph_traversal_bfs.json
    rm tests/visualization/fixtures/{dfs.json,bfs.html,dfs.html}

— in a PR that says why the frames are *supposed* to have changed. Only the BFS document is
committed: the DFS run is the same code path over the same graph and its own test below checks the
part that differs, which is the order.

This example is also the one place where the recording has to agree with something older than
itself. `tests/math/algorithms/test_graph_search_example_fingerprint.py` holds the traversal the
deleted matplotlib animation walked; the recording's result must still be that list.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from mllib.visualization.recording import Recording
from mllib.visualization.render import main as render_main
from tests.math.algorithms.test_graph_search_example_fingerprint import (
    BFS_FINGERPRINT,
    DFS_FINGERPRINT,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "graph_search_vs_networkx.py"
FIXTURE_PATH = HERE / "fixtures" / "graph_traversal_bfs.json"

# This recording is a fifteen-node walk: kilobytes, not the megabyte the pre-commit
# `check-added-large-files` hook and D-19 allow. Holding it to a tighter bound than the shared one
# is what would catch a frame quietly starting to carry the whole graph.
MAXIMUM_FIXTURE_BYTES = 50_000

EMBEDDED = re.compile(
    r'<script id="(recording|layout)" type="application/json">(.*?)</script>', re.DOTALL
)


@pytest.fixture(scope="module")
def example_module():
    spec = importlib.util.spec_from_file_location("graph_search_vs_networkx_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def written(example_module, tmp_path_factory) -> Path:
    """The example run once, into a directory of its own; every test below reads what it wrote."""
    output_dir = tmp_path_factory.mktemp("graph-search-example")
    example_module.main(["--output-dir", str(output_dir)])
    return output_dir


def _embedded(page: str) -> dict[str, object]:
    return {
        name: json.loads(payload.replace("<\\/", "</")) for name, payload in EMBEDDED.findall(page)
    }


def test_the_example_writes_a_recording_and_a_page_for_each_algorithm(written):
    assert sorted(path.name for path in written.iterdir()) == [
        "bfs.html",
        "bfs.json",
        "dfs.html",
        "dfs.json",
    ]
    for name in ("bfs", "dfs"):
        assert (written / f"{name}.html").read_text().startswith("<!doctype html>")


@pytest.mark.parametrize("algorithm", ["bfs", "dfs"])
def test_each_page_carries_its_own_recording_verbatim(written, algorithm):
    # The page is the artefact that gets moved around, so the run has to be *in* it: a page that
    # referenced its recording would be a page that stops working the moment it is sent to anyone.
    recording = Recording.load(written / f"{algorithm}.json")
    page = (written / f"{algorithm}.html").read_text()

    assert _embedded(page)["recording"] == recording.to_dict()
    assert recording.problem["algorithm"] == algorithm


def test_the_recorded_traversals_are_the_ones_the_deleted_animation_walked(written):
    breadth = Recording.load(written / "bfs.json")
    depth = Recording.load(written / "dfs.json")

    assert breadth.result["traversal_order"] == BFS_FINGERPRINT
    assert depth.result["traversal_order"] == DFS_FINGERPRINT
    assert breadth.result["visit_count"] == len(BFS_FINGERPRINT)


def test_the_visits_in_the_frames_are_the_traversal_the_result_reports(written):
    breadth = Recording.load(written / "bfs.json")

    visits = [frame["node_id"] for frame in breadth.frames if not frame["end"]]
    assert visits == breadth.result["traversal_order"]
    assert breadth.frames[-1]["end"] is True
    assert breadth.frames[-1]["found"] is True


def test_the_committed_fixture_is_the_document_the_example_writes(written):
    assert (written / "bfs.json").read_text() == FIXTURE_PATH.read_text()


def test_the_committed_fixture_stays_small_enough_to_commit():
    text = FIXTURE_PATH.read_text()

    assert len(text.encode()) < MAXIMUM_FIXTURE_BYTES
    assert text.endswith("\n")


def test_the_cli_re_renders_the_recording_to_the_page_the_example_writes(written, tmp_path):
    rendered = render_main(
        ["--recording", str(written / "bfs.json"), "--output", str(tmp_path / "cli.html")]
    )

    assert rendered.read_text() == (written / "bfs.html").read_text()
