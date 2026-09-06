"""Behavioural baseline for the certified Nyström search, taken before the bound is rebuilt.

`docs/plans/2026-09-nystrom-downdate.md` replaces the O(n³)-per-child lower bound with a rank-one
downdate of the parent spectrum (BL-27) and later truncates the spectrum (BL-29). Both are changes
to the cost of the search, not to the search: at the default tolerance the engine must expand the
same states and return the same landmarks. This snapshot is what makes that claim checkable.

Cells: the six known-answer fixtures, SPECTF at n = 40, k = 3 across the harness's three bandwidth
scales, and the larger rows `examples/nystrom_batched_bounds.py` times. Per cell the snapshot holds
the expansion count, the chosen landmarks, the residual trace and the certificate flag.

Ties are real and platform-dependent: the RBF-chain fixtures are palindromes whose mirror-image
subsets tie to the last bit, and on the rank-one all-ones kernel every subset ties at zero, so which
one comes back, and in what order the frontier is worked, depends on how the platform's BLAS rounds
a cancellation (the port's review; CI runs OpenBLAS, this machine Accelerate). A fixture cell
therefore accepts any member of its brute-force optimum set, and a cell where *every* subset is
optimal does not pin its expansion count either, since that is rounding order too. The UCI cells
have real gaps and pin subset and count exactly.

Regenerate deliberately with ``BASELINE_UPDATE=1`` in a PR that says why the numbers are meant to
change.

    uv run pytest tests/ml/test_nystrom_search_baseline.py -q
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.nystrom_landmark_selectors import GreedyResidualTraceLandmarkSelector
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.ml.projects.nystrom_uci_data import (
    DEFAULT_SMALL_UCI_SPECS,
    DEFAULT_UCI_DATA_DIR,
    build_rbf_kernel,
    downsample_rows,
    load_feature_matrix,
    standardize_columns,
)
from mllib.ml.projects.nystrom_uci_harness import DEFAULT_BANDWIDTH_SCALES

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[0] / "math" / "fixtures"
SNAPSHOT_PATH = HERE / "nystrom_search_snapshot.json"

COST_TOLERANCE = 1e-9  # relative; the residual trace is a deterministic function of the subset
SAMPLE_SEED = 7  # the harness's default row-sampling seed

# Known-answer fixtures at the landmark counts the unit tests certify them at.
FIXTURE_CELLS = (
    ("identity_8x8.csv", 3),
    ("all_ones_8x8.csv", 3),
    ("rbf_chain_4x4.csv", 2),
    ("rbf_chain_6x6.csv", 3),
    ("rbf_chain_8x8.csv", 4),
    ("rbf_chain_10x10.csv", 5),
)
TIE_TOLERANCE = 1e-9  # absolute, on the residual trace: the fixtures' traces are O(1)
# SPECTF (the first small UCI spec) at the harness's cell and across its bandwidth scales, plus the
# rows the batched-bounds example times at the default scale.
SPECTF_SPEC = DEFAULT_SMALL_UCI_SPECS[0]
UCI_CELLS = tuple((40, 3, scale) for scale in DEFAULT_BANDWIDTH_SCALES) + tuple(
    (n, k, 1.0) for n, k in ((60, 3), (80, 3), (40, 4), (60, 4))
)


def _search(kernel: np.ndarray, landmark_count: int) -> dict:
    problem = NystromLandmarkProblem(kernel, landmark_count)
    result = AStarSearch(problem, NystromCssCostFunction(problem)).run()
    return {
        "nodes_expanded": int(result.nodes_expanded),
        "state": [int(index) for index in result.state],
        "residual_trace": float(result.cost),
        "optimal": bool(result.optimal),
    }


def _optimum_set(kernel: np.ndarray, landmark_count: int) -> list[list[int]]:
    """Every subset within rounding of the best residual trace, by brute force (n = 8 here)."""
    problem = NystromLandmarkProblem(kernel, landmark_count)
    cost_function = NystromCssCostFunction(problem)
    scored = [
        (cost_function.goal_cost(subset), subset)
        for subset in itertools.combinations(range(kernel.shape[0]), landmark_count)
    ]
    best = min(cost for cost, _ in scored)
    return [list(subset) for cost, subset in scored if cost <= best + TIE_TOLERANCE]


def _fixture_kernel(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",")


def _uci_kernel(features: np.ndarray, n: int, scale: float) -> np.ndarray:
    sampled = standardize_columns(downsample_rows(features, max_rows=n, seed=SAMPLE_SEED))
    kernel, _ = build_rbf_kernel(sampled, gamma_scale=scale)
    return kernel


def _fixture_cell(name: str, landmark_count: int) -> tuple[str, dict]:
    kernel = _fixture_kernel(name)
    record = _search(kernel, landmark_count)
    record["accepted_states"] = _optimum_set(kernel, landmark_count)
    assert record["state"] in record["accepted_states"], (name, "A* missed the optimum set")
    # When every subset is optimal, the frontier order is decided by rounding, not by the engine.
    every_subset_ties = len(record["accepted_states"]) == math.comb(kernel.shape[0], landmark_count)
    record["expansions_pinned"] = not every_subset_ties
    return f"{name}:k={landmark_count}", record


def _uci_cell(features: np.ndarray, n: int, k: int, scale: float) -> tuple[str, dict]:
    record = _search(_uci_kernel(features, n, scale), k)
    record["accepted_states"] = [record["state"]]
    record["expansions_pinned"] = True
    return f"{SPECTF_SPEC.name}:n={n},k={k},scale={scale}", record


@pytest.fixture(scope="module")
def features() -> np.ndarray:
    return load_feature_matrix(SPECTF_SPEC, DEFAULT_UCI_DATA_DIR)


@pytest.fixture(scope="module")
def results(features) -> dict[str, dict]:
    cells = dict(_fixture_cell(name, k) for name, k in FIXTURE_CELLS)
    cells.update(_uci_cell(features, n, k, scale) for n, k, scale in UCI_CELLS)
    return cells


@pytest.fixture(scope="module")
def kernels(features) -> dict[str, tuple[np.ndarray, int]]:
    """The same cells as ``results``, as kernels, for the searches measured against it."""
    cells = {f"{name}:k={k}": (_fixture_kernel(name), k) for name, k in FIXTURE_CELLS}
    cells.update(
        {
            f"{SPECTF_SPEC.name}:n={n},k={k},scale={scale}": (_uci_kernel(features, n, scale), k)
            for n, k, scale in UCI_CELLS
        }
    )
    return cells


def test_every_cell_is_certified(results):
    assert all(record["optimal"] for record in results.values())


def test_every_cell_expands_at_least_one_state_per_landmark(results):
    for key, record in results.items():
        landmark_count = int(key.split("k=")[1].split(",")[0])
        assert record["nodes_expanded"] >= landmark_count, key


def test_snapshot_matches(results):
    if os.environ.get("BASELINE_UPDATE") == "1" or not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(json.dumps(results, indent=2) + "\n")
        pytest.skip(f"search baseline written to {SNAPSHOT_PATH.name}; re-run to compare")

    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert set(expected) == set(results), "cell set changed — update the snapshot deliberately"
    # Every mismatch is collected before failing, so one run reports the whole picture.
    mismatches = []
    for key, exp in expected.items():
        got = results[key]
        if exp["expansions_pinned"] and got["nodes_expanded"] != exp["nodes_expanded"]:
            mismatches.append(f"{key}: expanded {got['nodes_expanded']} != {exp['nodes_expanded']}")
        if got["state"] not in exp["accepted_states"]:
            mismatches.append(f"{key}: landmarks {got['state']} not in {exp['accepted_states']}")
        if got["residual_trace"] != pytest.approx(
            exp["residual_trace"], rel=COST_TOLERANCE, abs=1e-12
        ):
            mismatches.append(
                f"{key}: residual trace {got['residual_trace']} != {exp['residual_trace']}"
            )
        if got["optimal"] != exp["optimal"]:
            mismatches.append(f"{key}: optimal {got['optimal']} != {exp['optimal']}")
    assert not mismatches, "search baseline moved:\n" + "\n".join(mismatches)


# --- the pruned search on the baseline (plan §10, slice G) --------------------------------------


# The absolute rounding allowance stated beside a greedy seed: rounding on a residual trace scales
# with the kernel trace, not with the optimum (see the graph tests for the measured gaps).
INCUMBENT_SLACK_PER_TRACE = 1e-12


def _pruned_search(kernel: np.ndarray, k: int, incumbent_seed: float | None = None):
    problem = NystromLandmarkProblem(kernel, k)
    cost_function = NystromCssCostFunction(problem)
    search = PrunedAStarSearch(
        problem,
        cost_function,
        incumbent_seed=incumbent_seed,
        incumbent_slack=INCUMBENT_SLACK_PER_TRACE * float(np.trace(kernel)),
    )
    return search.run(), search.frontier_peak


def _moved(key: str, result, exact: dict) -> list[str]:
    """How a search differs from the baseline's record of a cell, where the record pins it."""
    moved = []
    if [int(i) for i in result.state] not in exact["accepted_states"]:
        moved.append(f"{key}: landmarks {list(result.state)} not in the optimum set")
    if exact["expansions_pinned"] and result.nodes_expanded != exact["nodes_expanded"]:
        moved.append(f"{key}: expanded {result.nodes_expanded} != {exact['nodes_expanded']}")
    if result.cost != pytest.approx(exact["residual_trace"], rel=COST_TOLERANCE, abs=1e-12):
        moved.append(f"{key}: residual trace {result.cost} != {exact['residual_trace']}")
    return moved


def test_the_pruned_search_matches_exact_a_star_on_every_cell(kernels, results):
    moved = []
    for key, (kernel, k) in kernels.items():
        result, _ = _pruned_search(kernel, k)
        moved.extend(_moved(key, result, results[key]))
    assert not moved, "pruned search moved the baseline:\n" + "\n".join(moved)


def test_the_greedy_incumbent_seed_changes_no_cell(kernels, results):
    """The greedy residual trace is an upper bound on the optimum, so seeding from it prunes only
    entries the search would never have popped: same expansions, same landmarks, on every cell
    whose order is the engine's rather than rounding's."""
    moved = []
    for key, (kernel, k) in kernels.items():
        problem = NystromLandmarkProblem(kernel, k)
        greedy = GreedyResidualTraceLandmarkSelector().select(
            problem, NystromCssCostFunction(problem)
        )
        result, seeded_peak = _pruned_search(kernel, k, incumbent_seed=greedy.cost)
        _, unseeded_peak = _pruned_search(kernel, k)
        moved.extend(_moved(key, result, results[key]))
        if seeded_peak > unseeded_peak:
            moved.append(f"{key}: seeded peak {seeded_peak} > unseeded {unseeded_peak}")
    assert not moved, "greedy incumbent seed moved the baseline:\n" + "\n".join(moved)


def test_frontier_peak_on_spectf_40_3_at_scale_1_is_within_the_analytic_bound(kernels):
    # 40 depth-1 states plus C(40, 2) = 780 depth-2 parents, each holding at most one goal
    # sibling on the frontier, against up to 9,880 goal children without the filter.
    kernel, k = kernels["SPECTF:n=40,k=3,scale=1.0"]
    _, frontier_peak = _pruned_search(kernel, k)
    assert frontier_peak <= 820
