"""The example against its committed recording: the guard that says the run has not moved.

`tests/visualization/fixtures/astar_landmark_rbf_chain_8x8_k3.json` is a snapshot, in the sense
CONTRIBUTING § "Behavioural baseline" means: it is the document the example writes for that cell —
the same frames, states, captions and counts, floats up to the last-digit noise of the host's BLAS
(`conftest.assert_same_recording`) — and it is regenerated **deliberately** —

    uv run python examples/astar_landmark_walkthrough.py \
        --output-dir tests/visualization/fixtures --cell rbf_chain_8x8_k3
    mv tests/visualization/fixtures/rbf_chain_8x8_k3.json \
       tests/visualization/fixtures/astar_landmark_rbf_chain_8x8_k3.json

— in a PR that says why the frames are *supposed* to have changed. A diff here means the search
expanded different states, priced them differently, or captioned them differently; none of those is
allowed to happen quietly.

One difference is not a move. The cell's two optima, {1, 3, 6} and {1, 4, 6}, are the chain's
reflections of each other with the same residual trace, and which one pops first is the fifteenth
digit of a bound — Accelerate on the Mac that wrote the fixture puts it one way, OpenBLAS on the CI
runner the other. The guard reads that tie off the fixture's own goal frame and accepts either
reflection as the goal; everything else, frame by frame, has to be the fixture's.

The SPECTF cell is not committed: it is the same code path on a larger kernel, and running it in the
unit suite would buy a second copy of this test's guarantee at several seconds a run.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from mllib.visualization.render import main as render_main
from tests.visualization.conftest import assert_same_recording

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


def _states_tied_with_the_goal(document: dict) -> list[list[int]]:
    """The frontier states the goal frame shows at the goal's cost to four decimals."""
    goal_frame = document["frames"][-1]
    cost = f"{goal_frame['bound']:.4f}"
    return [state for bound, state in goal_frame["frontier"] if f"{bound:.4f}" == cost]


def _with_the_tied_goal(document: dict, goal: list[int]) -> dict:
    """The fixture as it reads when the tied state popped instead of the goal: the two exchange
    places on the goal frame and in the result, and nothing else in the document moves."""
    swapped = json.loads(json.dumps(document))
    previous = swapped["result"]["state"]
    frame = swapped["frames"][-1]
    frame["frontier"] = [[b, previous if s == goal else s] for b, s in frame["frontier"]]
    frame["expanded_state"] = goal
    frame["caption"] = frame["caption"].replace(f"Goal {tuple(previous)}", f"Goal {tuple(goal)}")
    swapped["result"]["state"] = goal
    return swapped


def assert_same_run_up_to_the_goal_tie(actual_text: str, expected_text: str) -> None:
    """The recording is the fixture's run, allowing the popped goal to be one it ties with."""
    actual, expected = json.loads(actual_text), json.loads(expected_text)
    goal = actual["result"]["state"]
    if goal == expected["result"]["state"]:
        assert_same_recording(actual_text, expected_text)
        return
    tied = _states_tied_with_the_goal(expected)
    assert goal in tied, (
        f"goal {goal} is neither the fixture's {expected['result']['state']} nor tied with it "
        f"at display precision ({tied})"
    )
    assert_same_recording(actual_text, json.dumps(_with_the_tied_goal(expected, goal)))


def test_the_recorded_run_is_the_one_the_committed_fixture_holds(
    example_module, tmp_path, fixture_path
):
    recording_path, _ = example_module.write_cell("rbf_chain_8x8_k3", tmp_path)

    assert_same_run_up_to_the_goal_tie(recording_path.read_text(), fixture_path.read_text())


def test_the_guard_accepts_the_reflected_optimum_and_nothing_else(fixture_path):
    # The run the other BLAS writes is the fixture with {1, 3, 6} and {1, 4, 6} exchanged on the
    # goal frame; a goal nothing ties with, or a frame that moved, is still a failure.
    fixture = json.loads(fixture_path.read_text())
    assert _states_tied_with_the_goal(fixture) == [[1, 4, 6]]
    reflected = _with_the_tied_goal(fixture, [1, 4, 6])

    assert_same_run_up_to_the_goal_tie(json.dumps(reflected), fixture_path.read_text())

    untied = _with_the_tied_goal(fixture, [0, 6])
    with pytest.raises(AssertionError, match="neither the fixture's"):
        assert_same_run_up_to_the_goal_tie(json.dumps(untied), fixture_path.read_text())

    moved = json.loads(json.dumps(reflected))
    moved["frames"][3]["frontier_size"] += 1
    with pytest.raises(AssertionError, match=r"frames\[3\]\.frontier_size"):
        assert_same_run_up_to_the_goal_tie(json.dumps(moved), fixture_path.read_text())


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
