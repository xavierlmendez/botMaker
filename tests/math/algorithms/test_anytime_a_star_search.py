"""The anytime variant on the keep-heaviest toy: exact when nothing stops it, honest at a cap.

Uncapped, `AnytimeAStarSearch` must be `AStarSearch` — same state, cost, expansions — because it
changes what happens at a cap and nothing else. Capped, it returns the incumbent with a gap that is
never smaller than the truth: the brute-force optimum of the toy sits inside `[frontier_min, cost]`
on every instance. The real cells are in `tests/ml/test_nystrom_search_baseline.py`, beside the
pruned engine's oracle checks.
"""

import math
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.anytime_a_star_search import AnytimeAStarSearch
from mllib.math.algorithms.pruned_a_star_search import IncumbentBelowOptimum, PrunedAStarSearch
from tests.math.algorithms.test_a_star_search import (
    WEIGHTS,
    KeepHeaviestProblem,
    LeftOutWeightCost,
    UninformedLeftOutWeightCost,
    UnorderedKeepHeaviestProblem,
    brute_force_best,
)
from tests.math.algorithms.test_pruned_a_star_search import assert_same_search


def _toy(weights=WEIGHTS, keep_count=2, cost_class=LeftOutWeightCost):
    problem = KeepHeaviestProblem(weights, keep_count)
    return problem, cost_class(problem)


def test_uncapped_it_is_exact_a_star_on_the_toy_and_its_gap_is_zero():
    for problem_class in (KeepHeaviestProblem, UnorderedKeepHeaviestProblem):
        for keep_count in (1, 2, 3, 4):
            problem = problem_class(WEIGHTS, keep_count)
            cost_function = LeftOutWeightCost(problem)
            anytime = AnytimeAStarSearch(problem, cost_function)

            result = anytime.run()

            assert result.optimal
            assert_same_search(result, AStarSearch(problem, cost_function).run())
            assert anytime.certified_gap == 0.0
            assert anytime.frontier_min == result.cost


def test_uncapped_with_a_seed_at_a_feasible_cost_it_is_still_exact():
    problem, cost_function = _toy(keep_count=3)
    seed_state = (0, 1, 2)
    seed = cost_function.goal_cost(seed_state)

    anytime = AnytimeAStarSearch(
        problem, cost_function, incumbent_seed=seed, incumbent_seed_state=seed_state
    )
    result = anytime.run()

    assert_same_search(result, AStarSearch(problem, cost_function).run())
    assert result.optimal
    assert anytime.certified_gap == 0.0


def test_a_frontier_cap_of_one_returns_the_seed_at_once_with_the_gap_to_the_root_bound():
    # Seed spec, test 3. The root is popped (one expansion), its children priced, and the cap bites
    # on the second push; on this toy the best child ties the root, so the frontier minimum at the
    # stop is the root bound and the gap is the seed's distance from it.
    problem, cost_function = _toy(keep_count=2)
    seed_state = (1, 3)  # leaves out 5 + 4 + 3 = 12
    seed = cost_function.goal_cost(seed_state)
    root_bound = cost_function.lower_bound(problem.initial_state())
    anytime = AnytimeAStarSearch(
        problem, cost_function, incumbent_seed=seed, incumbent_seed_state=seed_state, max_frontier=1
    )

    result = anytime.run()

    assert result.state == seed_state
    assert result.cost == seed
    assert not result.optimal
    assert result.nodes_expanded == 1
    assert anytime.frontier_peak == 1
    assert anytime.frontier_min == root_bound
    assert anytime.certified_gap == pytest.approx(seed - root_bound)
    assert anytime.certified_gap == pytest.approx(6.0)


def test_a_frontier_cap_returns_instead_of_raising_and_the_gap_brackets_the_optimum():
    problem, cost_function = _toy(keep_count=3, cost_class=UninformedLeftOutWeightCost)
    with pytest.raises(Exception, match="no certificate"):
        PrunedAStarSearch(problem, cost_function, max_frontier=2).run()
    best_cost, _ = brute_force_best(WEIGHTS, 3)

    anytime = AnytimeAStarSearch(problem, cost_function, max_frontier=2)
    result = anytime.run()

    assert not result.optimal
    assert anytime.frontier_peak == 2
    assert anytime.frontier_min <= best_cost <= result.cost
    assert anytime.certified_gap >= result.cost - best_cost


def test_an_expansion_cap_below_the_exact_count_returns_a_bounded_result():
    # Seed spec, test 4, on the toy: uninformed bound so the search is long enough to cut.
    problem, cost_function = _toy(keep_count=3, cost_class=UninformedLeftOutWeightCost)
    exact = AStarSearch(problem, cost_function).run()
    best_cost, _ = brute_force_best(WEIGHTS, 3)
    assert exact.nodes_expanded > 4

    anytime = AnytimeAStarSearch(problem, cost_function, max_expansions=4)
    result = anytime.run()

    assert not result.optimal
    assert result.nodes_expanded == 4
    assert result.cost >= best_cost
    assert anytime.frontier_min <= best_cost
    assert anytime.certified_gap >= result.cost - best_cost
    assert anytime.certified_gap == pytest.approx(result.cost - anytime.frontier_min)


def test_the_gap_never_widens_as_the_expansion_cap_grows():
    # Seed spec, test 5: the consistency finding as a test. The incumbent only tightens and, with a
    # monotone bound, the frontier minimum only rises, so the gap at 1, 2, 4, ... expansions is
    # non-increasing. Infinite gaps (no incumbent yet) count as the largest value.
    for cost_class in (LeftOutWeightCost, UninformedLeftOutWeightCost):
        problem, cost_function = _toy(keep_count=3, cost_class=cost_class)
        exact = AStarSearch(problem, cost_function).run()
        gaps = []
        cap = 1
        while cap < exact.nodes_expanded:
            anytime = AnytimeAStarSearch(problem, cost_function, max_expansions=cap)
            anytime.run()
            gaps.append(anytime.certified_gap)
            cap *= 2
        anytime = AnytimeAStarSearch(problem, cost_function, max_expansions=cap)
        assert anytime.run().optimal
        gaps.append(anytime.certified_gap)

        assert gaps[-1] == 0.0
        assert all(later <= earlier for earlier, later in pairwise(gaps)), gaps


def test_a_stop_before_any_goal_is_priced_and_without_a_seed_reports_an_infinite_gap():
    problem, cost_function = _toy(keep_count=3, cost_class=UninformedLeftOutWeightCost)

    anytime = AnytimeAStarSearch(problem, cost_function, max_expansions=1)
    result = anytime.run()

    assert result.state is None
    assert result.cost == math.inf
    assert not result.optimal
    assert result.nodes_expanded == 1
    assert anytime.certified_gap == math.inf
    assert anytime.frontier_min == 0.0  # the uninformed bound


def test_a_seed_without_its_state_reports_no_state_until_a_goal_tightens_it():
    problem, cost_function = _toy(keep_count=2)
    seed = cost_function.goal_cost((1, 3))

    early = AnytimeAStarSearch(problem, cost_function, incumbent_seed=seed, max_expansions=1)
    early_result = early.run()
    assert early_result.state is None
    assert early_result.cost == seed
    assert early.certified_gap == pytest.approx(seed - early.frontier_min)

    # Two expansions: the root and (0,), whose goal children include the optimum (0, 2) at 6.0.
    later = AnytimeAStarSearch(problem, cost_function, incumbent_seed=seed, max_expansions=2)
    later_result = later.run()
    assert later_result.state == (0, 2)
    assert later_result.cost == pytest.approx(6.0)
    assert not later_result.optimal
    assert later.certified_gap == pytest.approx(6.0 - later.frontier_min)


def test_the_incumbent_state_follows_the_goal_that_tightened_it_not_the_last_goal_seen():
    # (0, 2) at 6.0 is priced from parent (0,); later parents price worse goals, which must not
    # displace it.
    problem, cost_function = _toy(keep_count=2, cost_class=UninformedLeftOutWeightCost)
    exact = AStarSearch(problem, cost_function).run()

    for cap in range(2, exact.nodes_expanded):
        anytime = AnytimeAStarSearch(problem, cost_function, max_expansions=cap)
        result = anytime.run()
        if result.optimal:
            break
        assert result.cost == pytest.approx(cost_function.goal_cost(result.state))
        assert result.cost <= 6.0 + 1e-12 or result.state is None


def test_an_empty_frontier_at_the_cap_lets_the_base_loop_conclude():
    # With more picks than items no goal exists; the cap must not mask that diagnosis.
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=6)
    with pytest.raises(ValueError, match="no goal state is reachable"):
        AnytimeAStarSearch(problem, LeftOutWeightCost(problem), max_expansions=1).run()

    problem, cost_function = _toy(keep_count=2)
    best_cost, _ = brute_force_best(WEIGHTS, 2)
    with pytest.raises(IncumbentBelowOptimum):
        AnytimeAStarSearch(
            problem, cost_function, incumbent_seed=best_cost - 1.0, max_expansions=10_000
        ).run()


def test_configuration_states_the_expansion_cap_beside_the_inherited_knobs():
    problem, cost_function = _toy()

    search = AnytimeAStarSearch(
        problem, cost_function, incumbent_seed=9.0, incumbent_slack=1e-12, max_expansions=50
    )

    assert search.configuration == {
        "engine": "AnytimeAStarSearch",
        "tie_break": "fifo",
        "tie_tolerance": 0.0,
        "count_bound_drops": False,
        "bound_drop_slack": 0.0,
        "incumbent_seed": 9.0,
        "incumbent_slack": 1e-12,
        "max_frontier": None,
        "max_expansions": 50,
    }
    assert search.certified_gap is None
    assert search.frontier_min is None


def test_invalid_caps_and_seed_states_are_rejected_on_construction():
    problem, cost_function = _toy(keep_count=2)

    with pytest.raises(ValueError, match="max_expansions"):
        AnytimeAStarSearch(problem, cost_function, max_expansions=0)
    with pytest.raises(ValueError, match="incumbent_seed_state names"):
        AnytimeAStarSearch(problem, cost_function, incumbent_seed_state=(0, 1))
    with pytest.raises(ValueError, match="complete solution"):
        AnytimeAStarSearch(problem, cost_function, incumbent_seed=1.0, incumbent_seed_state=(0,))


@settings(max_examples=60, deadline=None, derandomize=True)
@given(
    weights=st.lists(st.floats(min_value=0.5, max_value=50.0), min_size=2, max_size=7),
    keep_count=st.integers(min_value=1, max_value=4),
    max_expansions=st.integers(min_value=1, max_value=40),
    seeded=st.booleans(),
)
def test_a_capped_result_brackets_the_brute_force_optimum(
    weights: list[float], keep_count: int, max_expansions: int, seeded: bool
):
    # Seed spec, test 7. The uninformed bound makes the search long, so caps actually bite.
    keep_count = min(keep_count, len(weights))
    problem = KeepHeaviestProblem(tuple(weights), keep_count)
    cost_function = UninformedLeftOutWeightCost(problem)
    best_cost, _ = brute_force_best(tuple(weights), keep_count)
    seed_state = tuple(range(keep_count))
    seed_kwargs = (
        {"incumbent_seed": cost_function.goal_cost(seed_state), "incumbent_seed_state": seed_state}
        if seeded
        else {}
    )

    anytime = AnytimeAStarSearch(
        problem, cost_function, max_expansions=max_expansions, **seed_kwargs
    )
    result = anytime.run()

    assert anytime.frontier_min <= best_cost + 1e-9
    assert result.cost >= best_cost - 1e-9
    if result.optimal:
        assert anytime.certified_gap == 0.0
        assert result.cost == pytest.approx(best_cost, abs=1e-9)
    else:
        assert result.nodes_expanded == max_expansions
        assert anytime.certified_gap >= result.cost - best_cost - 1e-9
    if result.state is not None:
        assert result.cost == pytest.approx(cost_function.goal_cost(result.state), abs=1e-9)
