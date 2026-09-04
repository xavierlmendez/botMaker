"""How much batched sibling scoring (D-24) buys the Nyström search, measured with project code.

Runs the certified A* selection with the batched bounds and with the contract's per-child default on
the same kernels, then the full UCI harness both ways. Everything is deterministic; only the timings
vary between machines.

    uv run python examples/nystrom_batched_bounds.py
"""

from __future__ import annotations

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


class PerChildCost(NystromCssCostFunction):
    """The same objective scored through the contract's default: one call per child."""

    lower_bounds = SearchCostFunction.lower_bounds


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def search_with(cost_class, problem):
    return AStarSearch(problem, cost_class(problem)).run()


def full_suite():
    return run_small_uci_suite(gamma_scales=BANDWIDTH_SCALES)


def search_comparison() -> None:
    print("A* alone, SPECTF, batched vs per-child bounds")
    print(
        f"{'n':>4} {'k':>2} {'per-child':>10} {'batched':>9} {'speedup':>8} {'expanded':>9}  same"
    )
    features_full = load_feature_matrix(DEFAULT_SMALL_UCI_SPECS[0])
    for n, k in ((40, 3), (60, 3), (80, 3), (40, 4), (60, 4)):
        features = standardize_columns(downsample_rows(features_full, max_rows=n, seed=7))
        kernel, _ = build_rbf_kernel(features)
        problem = NystromLandmarkProblem(kernel, k)

        per_child, per_child_seconds = timed(search_with, PerChildCost, problem)
        batched, batched_seconds = timed(search_with, NystromCssCostFunction, problem)

        same_search = (
            per_child.state == batched.state and per_child.nodes_expanded == batched.nodes_expanded
        )
        print(
            f"{n:>4} {k:>2} {per_child_seconds:>9.2f}s {batched_seconds:>8.2f}s "
            f"{per_child_seconds / batched_seconds:>7.1f}x "
            f"{batched.nodes_expanded:>9}  {same_search}"
        )


def harness_comparison() -> None:
    print()
    print("Full harness (3 datasets x 3 bandwidths, n=40, k=3, 50 seeds)")
    _, batched_seconds = timed(full_suite)
    print(f"  batched bounds : {batched_seconds:6.2f}s")

    # Swap the contract's default in for one harness run, then restore the override.
    override = NystromCssCostFunction.lower_bounds
    NystromCssCostFunction.lower_bounds = SearchCostFunction.lower_bounds
    try:
        _, per_child_seconds = timed(full_suite)
    finally:
        NystromCssCostFunction.lower_bounds = override
    print(f"  per-child      : {per_child_seconds:6.2f}s")
    print(f"  speedup        : {per_child_seconds / batched_seconds:6.1f}x")


if __name__ == "__main__":
    search_comparison()
    harness_comparison()
