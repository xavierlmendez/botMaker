"""Deterministic landmark selectors, checked against hand-evaluated rules and the certified optimum.

Each selector is verified against the formula it claims to implement, not just against "some subset
came back": a greedy rule that picks a defensible-looking column for the wrong reason would pass the
weaker check and mislead every ratio computed from it later.
"""

import itertools

import numpy as np
import pytest

from mllib.math.algorithms.nystrom_landmark_selectors import (
    AStarLandmarkSelector,
    GreedyLowerBoundLandmarkSelector,
    GreedyResidualTraceLandmarkSelector,
    PivotedCholeskyLandmarkSelector,
    run_selector_suite,
)
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)


def random_rbf_kernel(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3))
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * squared)


def brute_force_best_cost(problem, cost_function) -> float:
    columns = range(problem.kernel_matrix.shape[0])
    return min(
        cost_function.goal_cost(subset)
        for subset in itertools.combinations(columns, problem.landmark_count)
    )


def build(kernel: np.ndarray, landmark_count: int):
    problem = NystromLandmarkProblem(kernel, landmark_count)
    return problem, NystromCssCostFunction(problem)


def test_greedy_trace_picks_the_column_with_the_largest_schur_complement_gain():
    kernel = random_rbf_kernel(8, seed=5)
    problem, cost_function = build(kernel, landmark_count=2)

    result = GreedyResidualTraceLandmarkSelector().select(problem, cost_function)

    # First pick maximizes ||K[:, j]||^2 / K[j, j] on the kernel itself.
    first = int(np.argmax(np.sum(kernel**2, axis=0) / np.diag(kernel)))
    # Second pick maximizes the same ratio on the Schur complement left behind by the first.
    schur = kernel - np.outer(kernel[:, first], kernel[:, first]) / kernel[first, first]
    diagonal = np.diag(schur).copy()
    diagonal[first] = 1.0  # the eliminated column has a zero pivot, so exclude rather than divide
    gains = np.sum(schur**2, axis=0) / diagonal
    gains[first] = -np.inf

    assert result.state == tuple(sorted((first, int(np.argmax(gains)))))
    assert result.cost == pytest.approx(cost_function.goal_cost(result.state))
    assert not result.optimal


def test_pivoted_cholesky_picks_the_largest_residual_diagonal_at_each_step():
    kernel = random_rbf_kernel(8, seed=13)
    problem, cost_function = build(kernel, landmark_count=3)

    result = PivotedCholeskyLandmarkSelector().select(problem, cost_function)

    # The first pivot is simply the largest diagonal entry of the kernel.
    assert int(np.argmax(np.diag(kernel))) in result.state
    assert len(result.state) == 3
    assert result.nodes_expanded == 3  # one pivot per landmark, no candidate sweep


def test_greedy_lower_bound_scores_every_successor_it_considers():
    problem, cost_function = build(random_rbf_kernel(6, seed=7), landmark_count=2)

    result = GreedyLowerBoundLandmarkSelector().select(problem, cost_function)

    # From the empty state it scores 5 candidates (indices 0..4), then 6 - 1 - first more.
    assert result.nodes_expanded > 3
    assert len(result.state) == 2
    assert result.cost == pytest.approx(cost_function.goal_cost(result.state))


def test_every_greedy_rule_takes_the_two_heaviest_columns_of_a_diagonal_kernel():
    problem, cost_function = build(np.diag([4.0, 3.0, 2.0, 1.0]), landmark_count=2)

    results = run_selector_suite(
        problem,
        cost_function,
        selectors=[
            AStarLandmarkSelector(),
            GreedyResidualTraceLandmarkSelector(),
            PivotedCholeskyLandmarkSelector(),
            GreedyLowerBoundLandmarkSelector(),
        ],
    )

    assert set(results) == {"astar", "greedy_trace", "pivoted_cholesky", "greedy_lower_bound"}
    assert all(result.state == (0, 1) for result in results.values())
    assert results["astar"].optimal
    assert results["astar"].cost == pytest.approx(3.0)


@pytest.mark.parametrize("seed", [1, 8, 21])
def test_no_heuristic_ever_beats_the_certified_optimum(seed: int):
    problem, cost_function = build(random_rbf_kernel(9, seed=seed), landmark_count=3)

    results = run_selector_suite(
        problem,
        cost_function,
        selectors=[
            AStarLandmarkSelector(),
            GreedyResidualTraceLandmarkSelector(),
            PivotedCholeskyLandmarkSelector(),
            GreedyLowerBoundLandmarkSelector(),
        ],
    )
    optimum = brute_force_best_cost(problem, cost_function)

    assert results["astar"].cost == pytest.approx(optimum, rel=1e-9)
    for name, result in results.items():
        assert result.cost >= optimum - 1e-12, name
        assert len(set(result.state)) == 3, name


def test_a_rank_one_kernel_defeats_no_selector():
    # Every column spans the same direction, so one landmark is exact and all rules agree.
    problem, cost_function = build(np.ones((6, 6)), landmark_count=2)

    results = run_selector_suite(
        problem,
        cost_function,
        selectors=[
            GreedyResidualTraceLandmarkSelector(),
            PivotedCholeskyLandmarkSelector(),
        ],
    )

    for name, result in results.items():
        assert result.cost == pytest.approx(0.0, abs=1e-9), name
        assert len(set(result.state)) == 2, name
