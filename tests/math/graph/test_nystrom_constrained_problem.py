"""Forced and forbidden columns: the conditional solves behind the necessity margins.

For every column j the margins ask two questions of the same search: the best subset that must
contain j, and the best that must not. Both are the Nyström problem on a modified ground set, the
bound unchanged, so brute force over the same subsets is the oracle, and the unconstrained optimum
is the floor every conditional cost sits on.
"""

import itertools
from collections import deque
from math import comb

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from tests.math.graph.test_nystrom_landmark_problem import (
    brute_force_best,
    brute_force_optima,
    load_kernel,
    random_rbf_kernel,
)

TOLERANCE = 1e-9


def _solve(kernel, k, **constraints):
    problem = NystromLandmarkProblem(kernel, k, **constraints)
    cost_function = NystromCssCostFunction(problem)
    return AStarSearch(problem, cost_function).run(), problem, cost_function


def _brute_force_conditional(kernel, k, *, forced=(), forbidden=frozenset()):
    """The best admissible subset by enumeration, respecting the same constraints."""
    problem = NystromLandmarkProblem(kernel, k)
    cost_function = NystromCssCostFunction(problem)
    return min(
        (cost_function.goal_cost(subset), subset)
        for subset in itertools.combinations(range(kernel.shape[0]), k)
        if set(forced) <= set(subset) and not (set(subset) & set(forbidden))
    )


def test_forcing_the_optimum_returns_it_after_one_expansion():
    # Seed test 1: the initial state is already a goal, popped at once.
    kernel = random_rbf_kernel(9, seed=1)
    unconstrained, _, _ = _solve(kernel, 3)

    result, problem, _ = _solve(kernel, 3, forced=unconstrained.state)

    assert problem.initial_state() == unconstrained.state
    assert result.state == unconstrained.state
    assert result.cost == pytest.approx(unconstrained.cost)
    assert result.optimal and result.nodes_expanded == 1


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_forbidding_a_column_of_the_optimum_costs_strictly_more_and_matches_brute_force(seed):
    # Seed test 2, on cells small enough to enumerate. A column of every optimum is the only kind
    # whose removal must raise the cost; a column of *some* optimum need not (a tie), so the strict
    # claim is made for columns in every optimum and brute force is the oracle for all of them.
    kernel = random_rbf_kernel(8, seed=seed)
    k = 3
    problem = NystromLandmarkProblem(kernel, k)
    optima = brute_force_optima(problem, NystromCssCostFunction(problem))
    optimum_cost, _ = brute_force_best(problem, NystromCssCostFunction(problem))
    in_every = set.intersection(*(set(state) for state in optima))
    assert in_every, "the random kernel should have at least one column in every optimum"

    for column in in_every:
        result, _, _ = _solve(kernel, k, forbidden=frozenset({column}))
        expected_cost, _ = _brute_force_conditional(kernel, k, forbidden={column})

        assert result.optimal
        assert column not in result.state
        assert result.cost > optimum_cost + TOLERANCE
        assert result.cost == pytest.approx(expected_cost, rel=TOLERANCE, abs=1e-12)


def test_forbidding_a_column_outside_every_optimum_leaves_the_cost_unchanged():
    # Seed test 3.
    kernel = random_rbf_kernel(8, seed=5)
    k = 3
    problem = NystromLandmarkProblem(kernel, k)
    cost_function = NystromCssCostFunction(problem)
    optima = brute_force_optima(problem, cost_function)
    optimum_cost, _ = brute_force_best(problem, cost_function)
    outside = set(range(8)) - set.union(*(set(state) for state in optima))
    assert outside

    column = min(outside)
    result, _, _ = _solve(kernel, k, forbidden=frozenset({column}))

    assert result.optimal
    assert result.cost == pytest.approx(optimum_cost, rel=TOLERANCE, abs=1e-12)
    assert tuple(result.state) in optima


def test_overlapping_or_too_many_or_out_of_range_constraints_are_rejected():
    # Seed test 4.
    kernel = load_kernel("rbf_chain_6x6.csv")

    with pytest.raises(ValueError, match="both forced and forbidden"):
        NystromLandmarkProblem(kernel, 3, forced=(1,), forbidden=frozenset({1}))
    with pytest.raises(ValueError, match="more forced columns"):
        NystromLandmarkProblem(kernel, 2, forced=(0, 1, 2))
    with pytest.raises(ValueError, match="in range"):
        NystromLandmarkProblem(kernel, 2, forced=(6,))
    with pytest.raises(ValueError, match="in range"):
        NystromLandmarkProblem(kernel, 2, forbidden=frozenset({-1}))
    with pytest.raises(ValueError, match="distinct"):
        NystromLandmarkProblem(kernel, 3, forced=(2, 2))
    with pytest.raises(ValueError, match="too few columns remain"):
        NystromLandmarkProblem(kernel, 3, forbidden=frozenset({0, 1, 2, 3}))


def test_a_forced_initial_state_is_sorted_and_successors_never_re_add_a_forced_column():
    # Seed test 5: canonicality. Every reachable state is an ascending tuple containing the forced
    # columns and no forbidden one, each admissible subset is generated exactly once, and the
    # number of goals is the count of admissible subsets.
    kernel = random_rbf_kernel(9, seed=7)
    k = 4
    forced, forbidden = (5, 2), frozenset({0, 7})
    problem = NystromLandmarkProblem(kernel, k, forced=forced, forbidden=forbidden)

    assert problem.initial_state() == (2, 5)
    assert problem.constraints == {"forced": [2, 5], "forbidden": [0, 7]}

    seen: set[tuple[int, ...]] = set()
    goals = 0
    queue = deque([problem.initial_state()])
    while queue:
        state = queue.popleft()
        assert state not in seen, "a subset was generated twice"
        seen.add(state)
        assert state == tuple(sorted(state)) and len(set(state)) == len(state)
        assert {2, 5} <= set(state) and not (set(state) & forbidden)
        if problem.is_goal(state):
            goals += 1
            assert list(problem.successors(state)) == []
            continue
        for action, child in problem.successors(state):
            assert action not in forced and action not in state and action not in forbidden
            assert child == tuple(sorted((*state, action)))
            queue.append(child)
    # Free columns: 9 - 2 forced - 2 forbidden = 5; two more picks.
    assert goals == comb(5, k - len(forced))


def test_an_unconstrained_problem_reports_no_constraints_and_the_old_successor_rule():
    kernel = load_kernel("rbf_chain_6x6.csv")
    problem = NystromLandmarkProblem(kernel, 3)

    assert problem.constraints == {}
    assert problem.initial_state() == ()
    # Index 5 cannot be followed by another column, so the old dead-end rule stops at 4.
    assert list(problem.successors((1,))) == [(2, (1, 2)), (3, (1, 3)), (4, (1, 4))]


def test_batched_bounds_equal_the_oracle_when_a_forced_column_sorts_above_the_added_one():
    # The batch prices children from the parent's decomposition; with a forced column at index 7
    # the added column sorts *inside* the child tuple, which the old prefix check would have sent
    # to the oracle one child at a time. Same values either way.
    kernel = random_rbf_kernel(9, seed=8)
    problem = NystromLandmarkProblem(kernel, 4, forced=(7,))
    cost_function = NystromCssCostFunction(problem)

    for parent in ((7,), (1, 7), (2, 5, 7)):
        successors = list(problem.successors(parent))
        assert successors
        batched = cost_function.lower_bounds(parent, successors)
        oracle = [cost_function.lower_bound(child) for _, child in successors]
        assert list(batched) == pytest.approx(oracle, rel=1e-9, abs=1e-12)


def test_the_pruned_engine_with_a_feasible_seed_solves_the_conditional_problem_too():
    kernel = random_rbf_kernel(9, seed=9)
    k = 3
    unconstrained, _, cost_function = _solve(kernel, k)
    column = unconstrained.state[1]
    expected_cost, expected_state = _brute_force_conditional(kernel, k, forced=(column,))
    seed_state = unconstrained.state
    problem = NystromLandmarkProblem(kernel, k, forced=(column,))

    result = PrunedAStarSearch(
        problem,
        NystromCssCostFunction(problem),
        incumbent_seed=cost_function.goal_cost(seed_state),
        incumbent_seed_state=seed_state,
        incumbent_slack=1e-12 * float(np.trace(kernel)),
    ).run()

    assert result.optimal
    assert result.cost == pytest.approx(expected_cost, rel=TOLERANCE, abs=1e-12)
    assert column in result.state


@settings(max_examples=12, deadline=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=10_000), n=st.integers(min_value=5, max_value=8))
def test_every_conditional_cost_sits_on_the_unconstrained_optimum_and_some_forced_one_meets_it(
    seed: int, n: int
):
    # Seed test 6.
    kernel = random_rbf_kernel(n, seed=seed)
    k = 2
    unconstrained, _, _ = _solve(kernel, k)
    optimum = unconstrained.cost

    forced_costs = []
    for column in range(n):
        forced, _, _ = _solve(kernel, k, forced=(column,))
        forbidden, _, _ = _solve(kernel, k, forbidden=frozenset({column}))
        assert forced.optimal and forbidden.optimal
        assert forced.cost >= optimum - TOLERANCE
        assert forbidden.cost >= optimum - TOLERANCE
        assert column in forced.state and column not in forbidden.state
        forced_costs.append(forced.cost)

    assert min(forced_costs) == pytest.approx(optimum, rel=TOLERANCE, abs=1e-12)
