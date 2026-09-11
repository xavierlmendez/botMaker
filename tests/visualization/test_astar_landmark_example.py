"""The example against its committed recording: the guard that says the run has not moved.

`tests/visualization/fixtures/astar_landmark_rbf_chain_8x8_k3.json` is a snapshot, in the sense
CONTRIBUTING § "Behavioural baseline" means: it is the exact document the example writes for that
cell, byte for byte, and it is regenerated **deliberately** —

    uv run python examples/astar_landmark_walkthrough.py \
        --output-dir tests/visualization/fixtures --cell rbf_chain_8x8_k3
    mv tests/visualization/fixtures/rbf_chain_8x8_k3.json \
       tests/visualization/fixtures/astar_landmark_rbf_chain_8x8_k3.json

— in a PR that says why the frames are *supposed* to have changed. A diff here means the search
expanded different states, priced them differently, or captioned them differently; none of those is
allowed to happen quietly.

The SPECTF cell is not committed: it is the same code path on a larger kernel, and running it in the
unit suite would buy a second copy of this test's guarantee at several seconds a run.
"""

import importlib.util
from pathlib import Path

import pytest

from mllib.visualization.render import main as render_main

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "astar_landmark_walkthrough.py"

# One megabyte is the pre-commit `check-added-large-files` limit and the D-19 rule for committed
# data; a recording that outgrows it is a recording of the wrong cell.
MAXIMUM_FIXTURE_BYTES = 1_000_000


@pytest.fixture(scope="module")
def example_module():
    spec = importlib.util.spec_from_file_location(
        "astar_landmark_walkthrough_example", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_writes_a_recording_and_a_page_for_the_cell(example_module, tmp_path):
    recording_path, page_path = example_module.write_cell("rbf_chain_8x8_k3", tmp_path)

    assert recording_path == tmp_path / "rbf_chain_8x8_k3.json"
    assert page_path == tmp_path / "rbf_chain_8x8_k3.html"
    assert page_path.read_text().startswith("<!doctype html>")


def test_the_recorded_run_is_the_one_the_committed_fixture_holds(
    example_module, tmp_path, fixture_path
):
    recording_path, _ = example_module.write_cell("rbf_chain_8x8_k3", tmp_path)

    assert recording_path.read_text() == fixture_path.read_text()


def test_the_committed_fixture_stays_small_enough_to_commit(fixture_path):
    text = fixture_path.read_text()

    assert len(text.encode()) < MAXIMUM_FIXTURE_BYTES
    assert text.endswith("\n")


def test_the_last_frame_is_the_goal_pop_the_result_returned(fixture_recording):
    last = fixture_recording.frames[-1]

    assert last["goal"] is True
    assert last["children"] == []
    assert last["expanded_state"] == fixture_recording.result["state"]
    assert last["expansions"] == fixture_recording.result["nodes_expanded"]


def test_the_cli_re_renders_the_fixture_to_the_page_the_example_writes(example_module, tmp_path):
    _, page_path = example_module.write_cell("rbf_chain_8x8_k3", tmp_path / "example")
    rendered = render_main(
        [
            "--recording",
            str(tmp_path / "example" / "rbf_chain_8x8_k3.json"),
            "--output",
            str(tmp_path / "cli.html"),
        ]
    )

    assert rendered.read_text() == page_path.read_text()


def test_an_unknown_cell_is_refused_by_name(example_module):
    with pytest.raises(ValueError, match="two_hot_span"):
        example_module.kernel_for("two_hot_span")
