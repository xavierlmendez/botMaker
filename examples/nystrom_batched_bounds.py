"""How much sharing a parent's work buys the Nyström search, measured with project code.

Three ways to price a parent's children, on the same kernels: the contract's per-child default (the
oracle, one decomposition per child), batched at goal depth only (D-24 as first shipped) and the
full downdate (BL-27: one decomposition per parent at every depth, which is what
``NystromCssCostFunction`` now does). Then the full UCI harness with the oracle and the downdate.
Everything is deterministic; only the timings
vary between machines, so ``--no-timings`` prints the deterministic columns alone and the result is
committed as ``nystrom_batched_bounds.example_output.txt``, the record every engine change is
diffed against (CONTRIBUTING § Behavioural baseline; `docs/plans/2026-09-nystrom-downdate.md`).

    uv run python examples/nystrom_batched_bounds.py
    uv run python examples/nystrom_batched_bounds.py --no-timings \
        > examples/nystrom_batched_bounds.example_output.txt
"""

from __future__ import annotations

import argparse
import time

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.math.search_cost_function import SearchCostFunction
from mllib.ml.projects.nystrom_uci_data import (
    DEFAULT_SMALL_UCI_SPECS,
    build_rbf_kernel,
    downsample_rows,
    load_feature_matrix,
    standardize_columns,
)
from mllib.ml.projects.nystrom_uci_harness import run_small_uci_suite

BANDWIDTH_SCALES = (0.25, 1.0, 4.0)
# A finer sweep for the expansion-count record: the harness's three scales plus the two between.
EXPANSION_COUNT_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
SEARCH_ROWS = ((40, 3), (60, 3), (80, 3), (40, 4), (60, 4))


class PerChildCost(NystromCssCostFunction):
    """The same objective scored through the contract's default: one call per child."""

    lower_bounds = SearchCostFunction.lower_bounds


class GoalDepthBatchedCost(NystromCssCostFunction):
    """Batched at goal depth only, as D-24 first shipped: children above it go to the oracle."""

    def _downdated_bounds(self, parent, added_columns):
        successors = [(int(column), (*parent, int(column))) for column in added_columns]
        return SearchCostFunction.lower_bounds(self, parent, successors)


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def search_with(cost_class, problem):
    return AStarSearch(problem, cost_class(problem)).run()


def full_suite():
    return run_small_uci_suite(gamma_scales=BANDWIDTH_SCALES)


def spectf_kernel(features_full, n: int, scale: float = 1.0):
    features = standardize_columns(downsample_rows(features_full, max_rows=n, seed=7))
    kernel, _ = build_rbf_kernel(features, gamma_scale=scale)
    return kernel


def search_comparison(features_full, timings: bool) -> None:
    print("A* alone, SPECTF, per-child vs goal-depth-batched vs downdated bounds")
    timing_header = (
        f"{'per-child':>10} {'batched':>9} {'downdated':>10} {'speedup':>8} " if timings else ""
    )
    print(f"{'n':>4} {'k':>2} {timing_header}{'expanded':>9}  same  landmarks")
    for n, k in SEARCH_ROWS:
        problem = NystromLandmarkProblem(spectf_kernel(features_full, n), k)

        per_child, per_child_seconds = timed(search_with, PerChildCost, problem)
        batched, batched_seconds = timed(search_with, GoalDepthBatchedCost, problem)
        downdated, downdated_seconds = timed(search_with, NystromCssCostFunction, problem)

        same_search = all(
            other.state == downdated.state and other.nodes_expanded == downdated.nodes_expanded
            for other in (per_child, batched)
        )
        timing_columns = (
            f"{per_child_seconds:>9.2f}s {batched_seconds:>8.2f}s {downdated_seconds:>9.2f}s "
            f"{per_child_seconds / downdated_seconds:>7.1f}x "
            if timings
            else ""
        )
        print(
            f"{n:>4} {k:>2} {timing_columns}{downdated.nodes_expanded:>9}  {same_search!s:5}"
            f" {list(downdated.state)}"
        )


def expansion_counts_by_bandwidth(features_full) -> None:
    print()
    print("A* alone, SPECTF n=40, k=3, expansions by bandwidth scale (downdated bounds)")
    print(f"{'scale':>6} {'expanded':>9}  landmarks")
    for scale in EXPANSION_COUNT_SCALES:
        problem = NystromLandmarkProblem(spectf_kernel(features_full, 40, scale), 3)
        result = search_with(NystromCssCostFunction, problem)
        print(f"{scale:>6} {result.nodes_expanded:>9}  {list(result.state)}")


def harness_comparison(timings: bool) -> None:
    print()
    print("Full harness (3 datasets x 3 bandwidths, n=40, k=3, 50 seeds)")
    runs, batched_seconds = timed(full_suite)
    print(f"{'dataset':>16} {'scale':>6} {'expanded':>9}  landmarks")
    for run in runs:
        optimum = run.certified_optimum
        print(
            f"{run.dataset_name:>16} {run.gamma_scale:>6} {optimum.nodes_expanded:>9}"
            f"  {list(optimum.state)}"
        )
    if not timings:
        return
    print(f"  downdated      : {batched_seconds:6.2f}s")

    # Swap the contract's default in for one harness run, then restore the override.
    override = NystromCssCostFunction.lower_bounds
    NystromCssCostFunction.lower_bounds = SearchCostFunction.lower_bounds
    try:
        _, per_child_seconds = timed(full_suite)
    finally:
        NystromCssCostFunction.lower_bounds = override
    print(f"  per-child      : {per_child_seconds:6.2f}s")
    print(f"  speedup        : {per_child_seconds / batched_seconds:6.1f}x")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-timings",
        action="store_true",
        help="print only the deterministic columns, so the committed record diffs byte-for-byte",
    )
    arguments = parser.parse_args(argv)
    timings = not arguments.no_timings

    features_full = load_feature_matrix(DEFAULT_SMALL_UCI_SPECS[0])
    search_comparison(features_full, timings)
    expansion_counts_by_bandwidth(features_full)
    harness_comparison(timings)


if __name__ == "__main__":
    main()
