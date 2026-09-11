"""What the walkthrough tests share: one real recorded run, and the committed fixture recording."""

from pathlib import Path

import numpy as np
import pytest

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.anytime_a_star_search import AnytimeAStarSearch
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.visualization.recorders.astar_landmark import AStarRecorder
from mllib.visualization.recording import Recording

# The smallest committed kernel with real gaps: a handful of frames, and every bound distinct.
SMALL_KERNEL = "rbf_chain_4x4.csv"
SMALL_LANDMARK_COUNT = 2

# The cell that makes the pruned engine set a popped goal aside, exactly as
# `tests/visualization/recorders/test_astar_landmark.py` builds it: under this tie tolerance a goal
# in the incumbent's cell pops first and the engine returns the incumbent it already holds (D-29),
# which is the one run whose recording carries two goal frames.
SUPERSEDING_KERNEL = "rbf_chain_8x8.csv"
SUPERSEDING_LANDMARK_COUNT = 5
SUPERSEDING_TOLERANCE_PER_TRACE = 0.1

# An expansion cap the identity kernel reaches while still holding an incumbent it priced early.
CAPPED_EXPANSIONS = 5

FIXTURE_NAME = "astar_landmark_rbf_chain_8x8_k3.json"
TWO_HOT_FIXTURE_NAME = "two_hot_span_roach_g5_30steps.json"


def _kernel(repository_root: Path, name: str) -> np.ndarray:
    return np.loadtxt(repository_root / "tests" / "math" / "fixtures" / name, delimiter=",")


def _recording(kernel: np.ndarray, landmark_count: int, cell: str, engine=AStarSearch, **knobs):
    """One watched search over one kernel, assembled into a recording the way the example does."""
    problem = NystromLandmarkProblem(kernel, landmark_count)
    recorder = AStarRecorder()
    search = engine(problem, NystromCssCostFunction(problem), recorder=recorder, **knobs)
    result = search.run()
    return Recording.from_recorder(
        recorder,
        problem={
            "kind": "astar_landmark",
            "cell": cell,
            "n": int(kernel.shape[0]),
            "landmark_count": landmark_count,
        },
        configuration=dict(search.configuration),
        result=result,
    )


@pytest.fixture
def recorded_run(repository_root: Path):
    """A real A* landmark search over a small committed kernel, watched; returns both halves."""
    kernel = _kernel(repository_root, SMALL_KERNEL)
    problem = NystromLandmarkProblem(kernel, SMALL_LANDMARK_COUNT)
    recorder = AStarRecorder()
    search = AStarSearch(problem, NystromCssCostFunction(problem), recorder=recorder)
    return search.run(), recorder, search


@pytest.fixture
def small_recording(recorded_run) -> Recording:
    """That run assembled into a recording, the way the example assembles one."""
    result, recorder, search = recorded_run
    return Recording.from_recorder(
        recorder,
        problem={"kind": "astar_landmark", "cell": "rbf_chain_4x4_k2", "n": 4, "landmark_count": 2},
        configuration=dict(search.configuration),
        result=result,
    )


@pytest.fixture
def superseded_recording(repository_root: Path) -> Recording:
    """A run whose recording holds two goal frames: one popped, and the incumbent that came back."""
    kernel = _kernel(repository_root, SUPERSEDING_KERNEL)
    return _recording(
        kernel,
        SUPERSEDING_LANDMARK_COUNT,
        "rbf_chain_8x8_k5_tied",
        engine=PrunedAStarSearch,
        tie_tolerance=SUPERSEDING_TOLERANCE_PER_TRACE * float(np.trace(kernel)),
    )


@pytest.fixture
def capped_recording(repository_root: Path) -> Recording:
    """A run stopped by a cap, so every frame carries the incumbent the engine was holding."""
    return _recording(
        _kernel(repository_root, "identity_8x8.csv"),
        2,
        "identity_8x8_k2_capped",
        engine=AnytimeAStarSearch,
        max_expansions=CAPPED_EXPANSIONS,
    )


@pytest.fixture
def fixture_path() -> Path:
    """The committed recording every render test draws, regenerated only deliberately."""
    return Path(__file__).resolve().parent / "fixtures" / FIXTURE_NAME


@pytest.fixture
def fixture_recording(fixture_path: Path) -> Recording:
    return Recording.load(fixture_path)


@pytest.fixture
def two_hot_fixture_path() -> Path:
    """The committed two-hot span recording, regenerated only deliberately (see its test)."""
    return Path(__file__).resolve().parent / "fixtures" / TWO_HOT_FIXTURE_NAME


@pytest.fixture
def two_hot_recording(two_hot_fixture_path: Path) -> Recording:
    """That recording, loaded. Loading rather than re-running is what keeps the view's tests free
    of torch: a view is a function of the document, and the document is already on disk."""
    return Recording.load(two_hot_fixture_path)
