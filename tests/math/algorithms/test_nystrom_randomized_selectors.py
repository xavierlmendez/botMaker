"""Randomized landmark selectors: seeded, summarized over seeds, and never better than optimal.

The tests pin the two properties that make a randomized number reportable. Determinism given a seed,
so a run can be reproduced, and honest accounting of work done, so best-of-N cannot be quoted beside
a single draw as though the two cost the same.
"""

import itertools

import numpy as np
import pytest

from mllib.math.algorithms.nystrom_landmark_selectors import run_selector_suite
from mllib.math.algorithms.nystrom_randomized_selectors import (
    BestOfRandomSamplingLandmarkSelector,
    RandomlyPivotedCholeskyLandmarkSelector,
    RandomSamplingLandmarkSelector,
    summarize_randomized_selector,
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


def build(kernel: np.ndarray, landmark_count: int):
    problem = NystromLandmarkProblem(kernel, landmark_count)
    return problem, NystromCssCostFunction(problem)


def brute_force_best_cost(problem, cost_function) -> float:
    columns = range(problem.kernel_matrix.shape[0])
    return min(
        cost_function.goal_cost(subset)
        for subset in itertools.combinations(columns, problem.landmark_count)
    )


def test_the_same_seed_gives_the_same_selection_and_a_different_seed_may_not():
    problem, cost_function = build(random_rbf_kernel(12, seed=4), landmark_count=3)

    first = RandomSamplingLandmarkSelector(seed=2).select(problem, cost_function)
    repeat = RandomSamplingLandmarkSelector(seed=2).select(problem, cost_function)
    other_seeds = {
        RandomSamplingLandmarkSelector(seed=seed).select(problem, cost_function).state
        for seed in range(10)
    }

    assert first.state == repeat.state
    assert first.cost == pytest.approx(repeat.cost)
    assert len(other_seeds) > 1  # the draw really is random across seeds, not a fixed subset


def test_rpcholesky_returns_distinct_valid_landmarks_on_every_seed():
    problem, cost_function = build(random_rbf_kernel(10, seed=4), landmark_count=3)
    optimum = brute_force_best_cost(problem, cost_function)

    for seed in range(5):
        result = RandomlyPivotedCholeskyLandmarkSelector(seed=seed).select(problem, cost_function)

        assert len(set(result.state)) == 3
        assert result.cost >= optimum - 1e-12
        assert not result.optimal


def test_rpcholesky_falls_back_to_a_uniform_draw_once_nothing_is_unexplained():
    # A rank-one kernel is exact after one landmark, so the residual diagonal is all zeros after it.
    problem, cost_function = build(np.ones((6, 6)), landmark_count=3)

    result = RandomlyPivotedCholeskyLandmarkSelector(seed=1).select(problem, cost_function)

    assert len(set(result.state)) == 3
    assert result.cost == pytest.approx(0.0, abs=1e-9)


def test_a_single_draw_costs_one_evaluation_and_best_of_n_costs_n():
    problem, cost_function = build(random_rbf_kernel(10, seed=1), landmark_count=3)

    single = RandomSamplingLandmarkSelector(seed=2).select(problem, cost_function)
    best_of = BestOfRandomSamplingLandmarkSelector(sample_count=16, seed=2).select(
        problem, cost_function
    )

    assert single.nodes_expanded == 1
    assert best_of.nodes_expanded == 16
    # Both start from the same seed, so the single draw is the first of the sixteen.
    assert best_of.cost <= single.cost


def test_best_of_n_carries_its_sample_count_in_its_name():
    assert BestOfRandomSamplingLandmarkSelector(sample_count=32).name == "best_of_32_random"
    assert BestOfRandomSamplingLandmarkSelector(sample_count=4).name == "best_of_4_random"


def test_best_of_n_rejects_a_budget_of_no_draws():
    problem, cost_function = build(np.eye(4), landmark_count=2)

    with pytest.raises(ValueError, match="sample_count must be positive"):
        BestOfRandomSamplingLandmarkSelector(sample_count=0, seed=1).select(problem, cost_function)


def test_more_draws_never_makes_best_of_n_worse():
    problem, cost_function = build(random_rbf_kernel(11, seed=6), landmark_count=3)

    few = BestOfRandomSamplingLandmarkSelector(sample_count=4, seed=3).select(
        problem, cost_function
    )
    many = BestOfRandomSamplingLandmarkSelector(sample_count=64, seed=3).select(
        problem, cost_function
    )

    # This is why it is an optimizer rather than a baseline: quality is bought with evaluations.
    assert many.cost <= few.cost + 1e-12


def test_summarizing_over_seeds_reports_a_spread_that_brackets_the_optimum():
    problem, cost_function = build(random_rbf_kernel(10, seed=8), landmark_count=3)
    optimum = brute_force_best_cost(problem, cost_function)

    summary = summarize_randomized_selector(
        RandomSamplingLandmarkSelector,
        problem,
        cost_function,
        trials=20,
        base_seed=100,
    )

    assert summary.name == "random_single_draw"
    assert summary.trials == 20
    assert summary.min_cost <= summary.median_cost <= summary.max_cost
    assert summary.min_cost <= summary.mean_cost <= summary.max_cost
    assert summary.min_cost >= optimum - 1e-12


def test_summarizing_rejects_a_run_of_no_trials():
    problem, cost_function = build(np.eye(4), landmark_count=2)

    with pytest.raises(ValueError, match="trials must be positive"):
        summarize_randomized_selector(
            RandomSamplingLandmarkSelector, problem, cost_function, trials=0
        )


def test_randomized_selectors_join_the_same_suite_as_the_deterministic_ones():
    problem, cost_function = build(np.diag([4.0, 3.0, 2.0, 1.0]), landmark_count=2)

    results = run_selector_suite(
        problem,
        cost_function,
        selectors=[
            RandomlyPivotedCholeskyLandmarkSelector(seed=11),
            RandomSamplingLandmarkSelector(seed=11),
            BestOfRandomSamplingLandmarkSelector(sample_count=6, seed=11),
        ],
    )

    assert set(results) == {"rpcholesky", "random_single_draw", "best_of_6_random"}
    for name, result in results.items():
        assert result.cost >= 3.0 - 1e-12, name  # 3.0 is the optimum on this kernel
