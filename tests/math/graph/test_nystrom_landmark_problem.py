"""The Nyström subset problem and its spectral bound, checked against brute force and by hand.

The kernels in `tests/math/fixtures/` are known-answer cases: an identity kernel where every column
is worth the same, an all-ones kernel of rank one where a single landmark is already exact, and RBF
chains of four to ten points where neighbouring columns overlap.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.math.search_cost_function import SearchCostFunction

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_kernel(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",")


def rbf_kernel_from_points(points: np.ndarray) -> np.ndarray:
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * squared)


def random_rbf_kernel(n: int, seed: int, dimensions: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rbf_kernel_from_points(rng.standard_normal((n, dimensions)))


def brute_force_best(problem, cost_function) -> tuple[float, tuple[int, ...]]:
    columns = range(problem.kernel_matrix.shape[0])
    return min(
        (cost_function.goal_cost(subset), subset)
        for subset in itertools.combinations(columns, problem.landmark_count)
    )


def brute_force_optima(problem, cost_function, tolerance: float = 1e-9) -> set[tuple[int, ...]]:
    """Every subset within rounding of the minimum: a symmetric kernel has several."""
    columns = range(problem.kernel_matrix.shape[0])
    scored = [
        (cost_function.goal_cost(subset), subset)
        for subset in itertools.combinations(columns, problem.landmark_count)
    ]
    best = min(cost for cost, _ in scored)
    return {subset for cost, subset in scored if cost <= best + tolerance}


def test_problem_rejects_kernels_that_are_not_square_symmetric_matrices():
    with pytest.raises(ValueError, match="2D"):
        NystromLandmarkProblem(np.array([1.0, 2.0]), landmark_count=1)
    with pytest.raises(ValueError, match="square"):
        NystromLandmarkProblem(np.ones((2, 3)), landmark_count=1)
    with pytest.raises(ValueError, match="symmetric"):
        NystromLandmarkProblem(np.array([[1.0, 0.5], [0.2, 1.0]]), landmark_count=1)


def test_problem_rejects_landmark_counts_outside_one_to_n():
    with pytest.raises(ValueError, match="positive"):
        NystromLandmarkProblem(np.eye(3), landmark_count=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        NystromLandmarkProblem(np.eye(3), landmark_count=4)


def test_search_starts_empty_and_ends_when_the_landmark_budget_is_spent():
    problem = NystromLandmarkProblem(np.eye(4), landmark_count=2)

    assert problem.initial_state() == ()
    assert not problem.is_goal(())
    assert not problem.is_goal((0,))
    assert problem.is_goal((0, 1))


def test_successors_append_one_higher_index_so_each_subset_is_one_state():
    problem = NystromLandmarkProblem(np.eye(4), landmark_count=2)

    assert list(problem.successors((1,))) == [(2, (1, 2)), (3, (1, 3))]


def test_successors_stop_where_too_few_columns_remain_to_finish():
    problem = NystromLandmarkProblem(np.eye(4), landmark_count=3)

    # From (0,) two more picks are needed, so index 3 alone cannot start a completion.
    assert list(problem.successors((0,))) == [(1, (0, 1)), (2, (0, 2))]
    assert list(problem.successors((0, 1, 2))) == []


def test_states_must_be_unique_ascending_and_in_range():
    problem = NystromLandmarkProblem(np.eye(4), landmark_count=2)

    with pytest.raises(ValueError, match="out-of-range"):
        problem.is_goal((0, 9))
    with pytest.raises(ValueError, match="ascending"):
        problem.is_goal((1, 0))
    with pytest.raises(ValueError, match="ascending"):
        problem.is_goal((1, 1))
    with pytest.raises(ValueError, match="more landmarks"):
        problem.is_goal((0, 1, 2))


def test_goal_cost_is_the_trace_of_the_kernel_the_landmarks_do_not_explain():
    kernel = np.diag([4.0, 3.0, 2.0, 1.0])
    problem = NystromLandmarkProblem(kernel, landmark_count=2)
    cost_function = NystromCssCostFunction(problem)

    # Diagonal kernel: a landmark explains its own entry and nothing else.
    assert cost_function.goal_cost((0, 1)) == pytest.approx(3.0)
    assert cost_function.goal_cost((2, 3)) == pytest.approx(7.0)


def test_goal_cost_matches_the_nystrom_formula_evaluated_directly():
    kernel = random_rbf_kernel(7, seed=11)
    problem = NystromLandmarkProblem(kernel, landmark_count=3)
    cost_function = NystromCssCostFunction(problem)
    state = (1, 4, 5)

    block = kernel[:, list(state)]
    expected = np.trace(
        kernel - block @ np.linalg.pinv(kernel[np.ix_(list(state), list(state))]) @ block.T
    )

    assert cost_function.goal_cost(state) == pytest.approx(expected)


def test_goal_cost_requires_a_complete_selection():
    problem = NystromLandmarkProblem(np.eye(4), landmark_count=2)

    with pytest.raises(ValueError, match="goal state"):
        NystromCssCostFunction(problem).goal_cost((0,))


def test_root_lower_bound_is_the_rank_k_eigenvalue_tail():
    kernel = random_rbf_kernel(8, seed=2)
    problem = NystromLandmarkProblem(kernel, landmark_count=3)
    cost_function = NystromCssCostFunction(problem)

    eigenvalues = np.sort(np.clip(np.linalg.eigvalsh(kernel), 0.0, None))[::-1]

    # Nothing chosen yet: the best conceivable rank-3 subspace is the top three eigenvectors, so the
    # bound is the tail. No subset of columns can do better, which is why the ratio between them is
    # the second number the Nystrom question needs.
    assert cost_function.lower_bound(()) == pytest.approx(float(np.sum(eigenvalues[3:])))


def test_lower_bound_equals_goal_cost_once_the_selection_is_complete():
    kernel = random_rbf_kernel(6, seed=5)
    problem = NystromLandmarkProblem(kernel, landmark_count=2)
    cost_function = NystromCssCostFunction(problem)

    for subset in itertools.combinations(range(6), 2):
        assert cost_function.lower_bound(subset) == pytest.approx(cost_function.goal_cost(subset))


def test_a_star_finds_and_certifies_the_best_landmarks_on_a_diagonal_kernel():
    problem = NystromLandmarkProblem(np.diag([4.0, 3.0, 2.0, 1.0]), landmark_count=2)

    result = AStarSearch(problem, NystromCssCostFunction(problem)).run()

    assert result.state == (0, 1)
    assert result.cost == pytest.approx(3.0)
    assert result.optimal


@pytest.mark.parametrize("seed", [3, 17, 29])
def test_a_star_matches_brute_force_on_random_rbf_kernels(seed: int):
    problem = NystromLandmarkProblem(random_rbf_kernel(9, seed=seed), landmark_count=3)
    cost_function = NystromCssCostFunction(problem)

    result = AStarSearch(problem, cost_function).run()
    best_cost, best_subset = brute_force_best(problem, cost_function)

    assert result.state == best_subset
    assert result.cost == pytest.approx(best_cost, rel=1e-9)
    assert result.nodes_expanded <= 84  # C(9, 3): the search must beat plain enumeration


def test_a_rank_one_kernel_is_explained_exactly_by_a_single_landmark():
    kernel = load_kernel("all_ones_8x8.csv")
    problem = NystromLandmarkProblem(kernel, landmark_count=3)

    result = AStarSearch(problem, NystromCssCostFunction(problem)).run()

    assert result.cost == pytest.approx(0.0, abs=1e-9)


def test_an_identity_kernel_leaves_one_unit_of_error_per_unchosen_column():
    kernel = load_kernel("identity_8x8.csv")
    problem = NystromLandmarkProblem(kernel, landmark_count=3)

    result = AStarSearch(problem, NystromCssCostFunction(problem)).run()

    # Every column is equally useless to every other, so any three leave the other five behind.
    assert result.cost == pytest.approx(5.0)
    assert result.optimal


@pytest.mark.parametrize(
    ("fixture_name", "landmark_count"),
    [
        ("rbf_chain_4x4.csv", 2),
        ("rbf_chain_6x6.csv", 3),
        ("rbf_chain_8x8.csv", 4),
        ("rbf_chain_10x10.csv", 5),
    ],
)
def test_a_star_certifies_the_optimum_on_the_bundled_rbf_chains(
    fixture_name: str, landmark_count: int
):
    problem = NystromLandmarkProblem(load_kernel(fixture_name), landmark_count)
    cost_function = NystromCssCostFunction(problem)

    result = AStarSearch(problem, cost_function).run()
    best_cost, _ = brute_force_best(problem, cost_function)

    # A chain kernel is a palindrome, so mirror-image subsets tie to the last bit; any of them is
    # the optimum, and which one comes back depends on rounding in the last place.
    assert result.optimal
    assert result.state in brute_force_optima(problem, cost_function)
    assert result.cost == pytest.approx(best_cost, rel=1e-9)


@settings(max_examples=30, deadline=None)
@given(
    coordinates=st.lists(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False), min_size=3, max_size=6
    ),
    landmark_count=st.integers(min_value=1, max_value=3),
    chosen_count=st.integers(min_value=0, max_value=2),
)
def test_the_bound_never_exceeds_any_completion_it_bounds(
    coordinates: list[float], landmark_count: int, chosen_count: int
):
    """Admissibility is the property the optimality proof rests on, so it is checked directly."""
    points = np.array(coordinates, dtype=float).reshape(-1, 1)
    landmark_count = min(landmark_count, len(points))
    chosen_count = min(chosen_count, landmark_count - 1)

    problem = NystromLandmarkProblem(rbf_kernel_from_points(points), landmark_count)
    cost_function = NystromCssCostFunction(problem)
    state = tuple(range(chosen_count))
    bound = cost_function.lower_bound(state)

    remaining = range(chosen_count, len(points))
    for completion in itertools.combinations(remaining, landmark_count - chosen_count):
        assert bound <= cost_function.goal_cost((*state, *completion)) + 1e-9


def rank_deficient_kernel(n: int, rank: int, scale: float, seed: int) -> np.ndarray:
    """A badly scaled kernel of deficient rank, where the residual cancels into rounding noise."""
    base = np.random.default_rng(seed).standard_normal((n, rank))
    return scale * (base @ base.T)


class CostWhoseRemainingSpectrumOvershoots(NystromCssCostFunction):
    """The real bound with its spectrum nudged so the top-r sum exceeds the energy by rounding.

    This is what a badly scaled kernel does on some BLAS implementations and not others; a test
    that asserts the sign of real rounding noise is not deterministic across platforms, so the
    overshoot is injected through the class's own seam instead.
    """

    def _remaining_spectrum(self, remaining_sqrt: np.ndarray) -> np.ndarray:
        spectrum = super()._remaining_spectrum(remaining_sqrt)
        spectrum[-1] += 1e-9 * max(abs(spectrum[-1]), 1.0)
        return spectrum


class CostWhoseResidualTraceUndershoots(NystromCssCostFunction):
    """The real residual kernel with its trace pushed just below zero, as cancellation can."""

    def _residual_kernel(self, state):
        residual_kernel = super()._residual_kernel(state)
        n = residual_kernel.shape[0]
        return residual_kernel - (float(np.trace(residual_kernel)) + 1e-9) / n * np.eye(n)


def test_the_lower_bound_is_clamped_when_cancellation_lands_below_zero():
    kernel = rank_deficient_kernel(7, rank=3, scale=1e8, seed=0)
    problem = NystromLandmarkProblem(kernel, landmark_count=4)
    honest = NystromCssCostFunction(problem)
    overshooting = CostWhoseRemainingSpectrumOvershoots(problem)

    # Rank 3 with one landmark chosen: the top three eigenvalues of what remains account for all of
    # it, so energy minus their sum is zero up to rounding, and the nudge tips it negative.
    assert honest.lower_bound((0,)) == pytest.approx(0.0, abs=1e-6 * 1e8)
    assert overshooting.lower_bound((0,)) == 0.0


def test_goal_cost_is_clamped_when_cancellation_lands_below_zero():
    kernel = rank_deficient_kernel(7, rank=2, scale=1e6, seed=2)
    problem = NystromLandmarkProblem(kernel, landmark_count=3)
    honest = NystromCssCostFunction(problem)
    undershooting = CostWhoseResidualTraceUndershoots(problem)

    # Rank 2 with three landmarks: the residual is explained exactly, so its trace is rounding
    # noise and the clamp decides its sign.
    assert honest.goal_cost((0, 1, 2)) == pytest.approx(0.0, abs=1e-6 * 1e6)
    assert undershooting.goal_cost((0, 1, 2)) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_real_cancellation_never_yields_a_negative_cost_or_bound(seed: int):
    """The platform-independent half: whichever side of zero rounding lands on, the clamp holds."""
    problem = NystromLandmarkProblem(
        rank_deficient_kernel(7, rank=2, scale=1e8, seed=seed), landmark_count=3
    )
    cost_function = NystromCssCostFunction(problem)

    for state in [(), (0,), (0, 1)]:
        assert cost_function.lower_bound(state) >= 0.0
    for goal in itertools.combinations(range(7), 3):
        assert cost_function.goal_cost(goal) >= 0.0


def test_a_badly_scaled_kernel_still_searches_to_a_certified_optimum():
    # The whole point of the clamps: a negative bound would empty the fringe and the search would
    # return the wrong subset rather than fail loudly.
    problem = NystromLandmarkProblem(
        rank_deficient_kernel(8, rank=2, scale=1e8, seed=1), landmark_count=3
    )
    cost_function = NystromCssCostFunction(problem)

    result = AStarSearch(problem, cost_function).run()
    best_cost, _ = brute_force_best(problem, cost_function)

    assert result.optimal
    assert result.cost == pytest.approx(best_cost, abs=1e-6)
    assert result.cost >= 0.0


def per_child_bounds(cost_function, successors) -> list[float]:
    return [cost_function.lower_bound(child) for _, child in successors]


@pytest.mark.parametrize("seed", [2, 9, 23])
def test_batched_goal_depth_bounds_equal_the_per_child_costs(seed: int):
    problem = NystromLandmarkProblem(random_rbf_kernel(12, seed=seed), landmark_count=3)
    cost_function = NystromCssCostFunction(problem)

    for parent in itertools.combinations(range(12), 2):
        successors = list(problem.successors(parent))
        if not successors:
            continue
        batched = cost_function.lower_bounds(parent, successors)

        expected = per_child_bounds(cost_function, successors)
        assert list(batched) == pytest.approx(expected, abs=1e-10)


def test_batched_bounds_fall_back_to_per_child_above_goal_depth():
    problem = NystromLandmarkProblem(random_rbf_kernel(9, seed=4), landmark_count=3)
    cost_function = NystromCssCostFunction(problem)

    for parent in [(), (1,), (4,)]:
        successors = list(problem.successors(parent))
        batched = cost_function.lower_bounds(parent, successors)

        assert list(batched) == pytest.approx(per_child_bounds(cost_function, successors))


def test_batched_bounds_fall_back_when_the_successors_do_not_extend_the_parent():
    problem = NystromLandmarkProblem(random_rbf_kernel(8, seed=6), landmark_count=2)
    cost_function = NystromCssCostFunction(problem)
    # Complete states, but not children of (0,): the override must not assume the prefix.
    unrelated = [(3, (2, 3)), (5, (4, 5))]

    batched = cost_function.lower_bounds((0,), unrelated)

    assert list(batched) == pytest.approx(per_child_bounds(cost_function, unrelated))
    assert cost_function.lower_bounds((0,), []) == []


def test_batched_bounds_handle_columns_already_in_the_span():
    # Rank one: after the first landmark every residual pivot is zero, so every child costs the
    # same and none can be preferred.
    problem = NystromLandmarkProblem(np.ones((6, 6)), landmark_count=2)
    cost_function = NystromCssCostFunction(problem)
    successors = list(problem.successors((0,)))

    batched = cost_function.lower_bounds((0,), successors)

    assert list(batched) == pytest.approx([0.0] * len(successors), abs=1e-9)
    assert list(batched) == pytest.approx(per_child_bounds(cost_function, successors), abs=1e-9)


def test_batched_bounds_stay_clamped_on_a_badly_scaled_kernel():
    problem = NystromLandmarkProblem(
        rank_deficient_kernel(7, rank=2, scale=1e6, seed=2), landmark_count=3
    )
    cost_function = NystromCssCostFunction(problem)
    successors = list(problem.successors((0, 1)))

    batched = cost_function.lower_bounds((0, 1), successors)

    assert all(bound >= 0.0 for bound in batched)


@pytest.mark.parametrize("seed", [3, 17, 29])
def test_a_star_expands_the_same_states_whether_bounds_are_batched_or_not(seed: int):
    """Batching must be a change of cost, not of search: same optimum, same expansion count."""

    class PerChildCost(NystromCssCostFunction):
        lower_bounds = SearchCostFunction.lower_bounds

    problem = NystromLandmarkProblem(random_rbf_kernel(11, seed=seed), landmark_count=3)

    batched = AStarSearch(problem, NystromCssCostFunction(problem)).run()
    per_child = AStarSearch(problem, PerChildCost(problem)).run()

    assert batched.state == per_child.state
    assert batched.cost == pytest.approx(per_child.cost, rel=1e-12)
    assert batched.nodes_expanded == per_child.nodes_expanded
