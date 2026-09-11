"""The A* landmark recorder against the real search, on the fixtures the unit tests certify.

Two known-answer kernels from `tests/math/fixtures`, run through exact, pruned and anytime A* with
an `AStarRecorder` injected. Every claim here is about what the frames say and about the search
being unchanged by being watched; the numbers themselves are the baselines' business
(`tests/ml/test_nystrom_search_baseline.py`).
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mllib.math.algorithms.a_star_search import AStarSearch, SearchResult
from mllib.math.algorithms.anytime_a_star_search import AnytimeAStarSearch
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.visualization.recorders.astar_landmark import AStarFrame, AStarRecorder

FIXTURES = Path(__file__).resolve().parents[2] / "math" / "fixtures"

# The two cells every test below runs: an RBF chain with real gaps and the identity, whose every
# subset ties. Both are small enough that a frame per expansion is a handful of frames.
CELLS = (("rbf_chain_8x8.csv", 3), ("identity_8x8.csv", 2))
ENGINES = (AStarSearch, PrunedAStarSearch, AnytimeAStarSearch)

# The one cell that makes the pruned engine set a popped goal aside. Under a tie tolerance a goal in
# the same cell as the incumbent can pop before it, and the engine then returns the incumbent it
# already holds (D-29). Here the gap between the two is 86% of the popped goal's cost, so the branch
# is taken for a structural reason and not by how a platform's BLAS rounds a tie.
SUPERSEDING_CELL = ("rbf_chain_8x8.csv", 5)
SUPERSEDING_TOLERANCE_PER_TRACE = 0.1


def _kernel(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",")


def _run(name: str, landmark_count: int, engine=AStarSearch, **knobs):
    """One search over one fixture cell, with a recorder attached; returns both."""
    kernel = _kernel(name)
    problem = NystromLandmarkProblem(kernel, landmark_count)
    recorder = AStarRecorder()
    search = engine(problem, NystromCssCostFunction(problem), recorder=recorder, **knobs)
    return search.run(), recorder, search


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_every_expansion_and_the_goal_pop_leave_a_frame(name: str, landmark_count: int):
    # Two call sites, one per moment the loop has: after a state's children are priced and pushed,
    # and on the goal pop the search returns on. A run that closes on a goal therefore records
    # exactly as many frames as it expanded states.
    result, recorder, _ = _run(name, landmark_count)

    assert len(recorder.frames) == result.nodes_expanded
    assert [frame.expansions for frame in recorder.frames] == list(
        range(1, result.nodes_expanded + 1)
    )
    assert [frame.goal for frame in recorder.frames] == [False] * (result.nodes_expanded - 1) + [
        True
    ]


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_the_last_frame_is_the_goal_the_search_returned(name: str, landmark_count: int):
    # A walkthrough must end at the answer, so the run's last frame is the goal pop itself: its
    # state and cost are the result's, and it priced no children because a goal is never expanded.
    result, recorder, _ = _run(name, landmark_count)
    last = recorder.frames[-1]

    assert last.goal
    assert last.expanded_state == tuple(int(index) for index in result.state)
    assert last.bound == result.cost
    assert last.children == ()
    assert last.caption.startswith(f"Goal {last.expanded_state} popped at cost ")
    assert last.caption.endswith(f"{last.frontier_size} states left on the frontier.")
    assert not any(frame.goal for frame in recorder.frames[:-1])


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_the_returned_state_was_pushed_by_one_of_the_recorded_expansions(
    name: str, landmark_count: int
):
    # The goal is popped, never expanded, so it is not any frame's ``expanded_state``. It reached
    # the frontier as some earlier expansion's pushed child, which is what the frames must show.
    result, recorder, _ = _run(name, landmark_count)
    pushed = {
        state for frame in recorder.frames for state, _, was_pushed in frame.children if was_pushed
    }

    assert tuple(int(index) for index in result.state) in pushed


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_every_frame_holds_its_whole_frontier_sorted_by_bound_then_state(
    name: str, landmark_count: int
):
    _, recorder, _ = _run(name, landmark_count)

    for frame in recorder.frames:
        assert list(frame.frontier) == sorted(frame.frontier)
        assert frame.frontier_size == len(frame.frontier)


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_every_caption_is_a_sentence_naming_the_expanded_state(name: str, landmark_count: int):
    _, recorder, _ = _run(name, landmark_count)

    for frame in recorder.frames:
        assert frame.caption.endswith(".")
        if frame.goal:
            continue
        assert frame.caption.startswith(f"Expanded {frame.expanded_state}")
        assert f"{frame.expansions} expansions so far" in frame.caption


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
@pytest.mark.parametrize("engine", ENGINES)
def test_watching_the_search_changes_nothing_about_its_result(
    name: str, landmark_count: int, engine
):
    # The whole ``SearchResult``, not a chosen few fields — and over every engine, because the
    # pruned one branches on ``recorder.enabled`` inside ``_push_children``.
    watched, _, watched_search = _run(name, landmark_count, engine=engine)
    kernel = _kernel(name)
    problem = NystromLandmarkProblem(kernel, landmark_count)
    unwatched_search = engine(problem, NystromCssCostFunction(problem))
    unwatched = unwatched_search.run()

    assert watched == unwatched
    assert watched_search.configuration == unwatched_search.configuration
    assert getattr(watched_search, "frontier_peak", None) == getattr(
        unwatched_search, "frontier_peak", None
    )


def test_the_astar_recorder_is_on():
    assert AStarRecorder().enabled is True


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_the_pruned_engine_records_the_children_it_declined_to_store(
    name: str, landmark_count: int
):
    # The goal-sibling filter drops every goal child but the cheapest, so a run over these cells
    # prunes something; each pruned entry names its state, its bound and the mechanism.
    _, recorder, _ = _run(name, landmark_count, engine=PrunedAStarSearch)
    pruned = [entry for frame in recorder.frames for entry in frame.extras.get("pruned", ())]

    expansions = [frame for frame in recorder.frames if not frame.goal]
    assert all("pruned" in frame.extras for frame in expansions)
    # A goal frame priced nothing, so it carries no pruned list from the expansion before it.
    assert "pruned" not in recorder.frames[-1].extras
    assert pruned, "the pruned engine stored every child on these cells"
    assert {reason for _, _, reason in pruned} <= {"goal_sibling", "above_incumbent"}
    for frame in expansions:
        dropped = {tuple(state) for state, _, _ in frame.extras["pruned"]}
        assert dropped == {state for state, _, was_pushed in frame.children if not was_pushed}


def test_a_cap_stop_records_no_goal_frame():
    # The cap bites inside ``_push_children``, so the run never reaches a goal pop. Nothing may
    # claim it did: a stopped run's frames end on an expansion, and the result is an incumbent.
    result, recorder, _ = _run("identity_8x8.csv", 2, engine=AnytimeAStarSearch, max_expansions=5)

    assert not result.optimal
    assert not any(frame.goal for frame in recorder.frames)
    assert len(recorder.frames) == result.nodes_expanded - 1


def test_the_anytime_engine_records_the_incumbent_it_holds_when_a_cap_stops_it():
    # The identity kernel at k = 2 prices a goal on its second expansion, so an expansion cap of
    # five stops the search with an incumbent it has been carrying for most of the run.
    result, recorder, search = _run(
        "identity_8x8.csv", 2, engine=AnytimeAStarSearch, max_expansions=5
    )

    assert not result.optimal
    assert all("incumbent" in frame.extras for frame in recorder.frames)
    last = recorder.frames[-1]
    assert last.extras["incumbent"] == pytest.approx(result.cost)
    assert last.extras["incumbent_state"] == [int(index) for index in result.state]
    assert f"incumbent {tuple(last.extras['incumbent_state'])}" in last.caption
    # ``certified_gap`` is computed at the stop, not tracked mid-run, so no frame claims one.
    assert "certified_gap" not in last.extras
    assert search.certified_gap == pytest.approx(max(result.cost - search.frontier_min, 0.0))


def test_a_superseded_goal_is_explained_by_a_second_goal_frame():
    # Under a tie tolerance the pruned engine can pop one goal and return another. Without a second
    # goal frame the run's last frame would name a state that never came back.
    name, landmark_count = SUPERSEDING_CELL
    tolerance = SUPERSEDING_TOLERANCE_PER_TRACE * float(np.trace(_kernel(name)))
    result, recorder, _ = _run(
        name, landmark_count, engine=PrunedAStarSearch, tie_tolerance=tolerance
    )
    goal_frames = [frame for frame in recorder.frames if frame.goal]

    assert len(goal_frames) == 2, "the tie-tolerance substitution branch was not taken"
    popped, returned = goal_frames
    assert recorder.frames[-1] is returned
    assert returned.expanded_state == tuple(int(index) for index in result.state)
    assert returned.bound == result.cost
    assert returned.extras["superseded_goal"] == list(popped.expanded_state)
    assert returned.extras["superseded_cost"] == popped.bound
    assert popped.expanded_state != returned.expanded_state
    assert returned.bound < popped.bound
    assert returned.caption == (
        f"Returned the held incumbent {returned.expanded_state} at cost {result.cost:.4f}; "
        f"the popped goal {popped.expanded_state} at cost {popped.bound:.4f} was within the "
        f"tie tolerance and set aside."
    )


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_the_last_goal_frame_always_names_the_state_that_came_back(name: str, landmark_count: int):
    # The invariant slice 2 renders on, whether or not a substitution happened.
    for engine in (AStarSearch, PrunedAStarSearch):
        result, recorder, _ = _run(name, landmark_count, engine=engine)
        last = recorder.frames[-1]
        assert last.goal
        assert last.expanded_state == tuple(int(index) for index in result.state)
        assert last.bound == result.cost


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
@pytest.mark.parametrize("engine", ENGINES)
def test_every_frame_serialises_as_json_without_a_default(name: str, landmark_count: int, engine):
    # No numpy scalar may reach a frame: ``json.dumps`` without ``default`` is the assertion.
    _, recorder, _ = _run(name, landmark_count, engine=engine)

    for frame, as_dict in zip(recorder.frames, recorder.frame_dicts(), strict=True):
        assert isinstance(frame, AStarFrame)
        assert json.loads(json.dumps(as_dict)) == as_dict


@pytest.mark.parametrize(("name", "landmark_count"), CELLS)
def test_the_described_result_serialises_as_json_without_a_default(name: str, landmark_count: int):
    result, recorder, search = _run(name, landmark_count)
    # The engines leave ``engine_configuration`` unset; a harness fills it in, so describe it too.
    described = recorder.describe_result(replace(result, engine_configuration=search.configuration))

    assert json.loads(json.dumps(described)) == described
    assert described["state"] == [int(index) for index in result.state]
    assert described["nodes_expanded"] == result.nodes_expanded
    assert described["engine_configuration"]["engine"] == "AStarSearch"


def test_a_numpy_value_on_the_configuration_is_described_as_plain_python():
    # A caller scales a tolerance by ``np.trace``, so ``configuration`` carries a ``np.float64``.
    # It must not reach a recording: ``json.dumps`` with no ``default`` is the assertion.
    kernel = _kernel("rbf_chain_8x8.csv")
    problem = NystromLandmarkProblem(kernel, 3)
    tolerance = np.float64(1e-12) * np.trace(kernel)
    search = AStarSearch(problem, NystromCssCostFunction(problem), tie_tolerance=tolerance)
    result = search.run()

    described = AStarRecorder().describe_result(
        replace(result, engine_configuration=search.configuration)
    )

    assert type(search.configuration["tie_tolerance"]) is not float
    assert type(described["engine_configuration"]["tie_tolerance"]) is float
    assert json.loads(json.dumps(described)) == described


def test_a_frame_survives_being_described_without_an_engine_configuration():
    result = SearchResult(state=(1, 2), cost=0.5, optimal=True, nodes_expanded=3)

    assert AStarRecorder().describe_result(result) == {
        "state": [1, 2],
        "cost": 0.5,
        "optimal": True,
        "nodes_expanded": 3,
        "frontier_peak": None,
        "engine_configuration": None,
        "certified_gap": None,
    }
