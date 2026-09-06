"""The pruned variant of A*, measured against exact A* on the keep-heaviest toy.

`PrunedAStarSearch` stores less of the frontier — one goal child per parent, nothing above the
incumbent, nothing past a cap — and promises that none of it changes which states are expanded or
which goal comes back. Exact `AStarSearch` is the oracle for that promise; the toy problem, its
costs and its brute force are the ones `test_a_star_search.py` defines.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import AStarSearch, SearchResult
from mllib.math.algorithms.pruned_a_star_search import (
    FrontierLimitExceeded,
    IncumbentBelowOptimum,
    PrunedAStarSearch,
)
from tests.math.algorithms.test_a_star_search import (
    WEIGHTS,
    KeepHeaviestProblem,
    LeftOutWeightCost,
    Subset,
    UninformedLeftOutWeightCost,
    UnorderedKeepHeaviestProblem,
    brute_force_best,
)


class PeakCountingAStarSearch(AStarSearch):
    """Exact A* that also reports its frontier peak, through the hook the variant overrides."""

    frontier_peak = 1

    def _push_children(self, queue, children, insertion_index):
        insertion_index = super()._push_children(queue, children, insertion_index)
        self.frontier_peak = max(self.frontier_peak, len(queue))
        return insertion_index


def assert_same_search(result: SearchResult[Subset], exact: SearchResult[Subset]) -> None:
    """The mechanisms change what the frontier stores, never what the search expands or returns."""
    assert result.state == exact.state
    assert result.cost == pytest.approx(exact.cost, abs=1e-12)
    assert result.nodes_expanded == exact.nodes_expanded


@settings(max_examples=40, deadline=None)
@given(
    weights=st.lists(st.floats(min_value=0.5, max_value=50.0), min_size=2, max_size=7),
    keep_count=st.integers(min_value=1, max_value=4),
)
def test_the_goal_sibling_filter_returns_the_brute_force_optimum_with_many_goal_children(
    weights: list[float], keep_count: int
):
    # Every depth-(k-1) state here has up to n goal children; only one per parent is stored.
    keep_count = min(keep_count, len(weights))
    problem = KeepHeaviestProblem(tuple(weights), keep_count)
    cost_function = LeftOutWeightCost(problem)

    result = PrunedAStarSearch(problem, cost_function).run()

    best_cost, _ = brute_force_best(tuple(weights), keep_count)
    assert result.cost == pytest.approx(best_cost, abs=1e-9)
    assert_same_search(result, AStarSearch(problem, cost_function).run())


def test_the_frontier_holds_one_goal_child_per_parent_when_every_goal_ties():
    # Equal weights make every bound 2.0, so nothing is pruned (ties are kept) and the search
    # works the graph first-in-first-out: root, three one-pick states, six two-pick parents. Each
    # parent's goal children tie too, so exactly one per parent is stored: 6 entries at the peak,
    # against the 3 + 2 + 1 + 2 + 1 + 1 = 10 goal children exact A* stores.
    problem = KeepHeaviestProblem((1.0, 1.0, 1.0, 1.0, 1.0), keep_count=3)
    cost_function = LeftOutWeightCost(problem)
    pruned = PrunedAStarSearch(problem, cost_function)
    exact = PeakCountingAStarSearch(problem, cost_function)

    result = pruned.run()

    assert_same_search(result, exact.run())
    assert result.nodes_expanded == 11
    assert pruned.frontier_peak == 6
    assert exact.frontier_peak == 10


@settings(max_examples=40, deadline=None)
@given(
    weights=st.lists(st.floats(min_value=0.5, max_value=50.0), min_size=2, max_size=6),
    keep_count=st.integers(min_value=1, max_value=3),
)
def test_incumbent_pruning_changes_no_expansion_on_ordered_or_unordered_problems(
    weights: list[float], keep_count: int
):
    keep_count = min(keep_count, len(weights))
    for problem_class in (KeepHeaviestProblem, UnorderedKeepHeaviestProblem):
        problem = problem_class(tuple(weights), keep_count)
        cost_function = LeftOutWeightCost(problem)

        result = PrunedAStarSearch(problem, cost_function).run()

        assert_same_search(result, AStarSearch(problem, cost_function).run())


def test_an_incumbent_seed_at_the_optimum_keeps_the_tie_and_changes_nothing():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    best_cost, _ = brute_force_best(WEIGHTS, 2)

    seeded = PrunedAStarSearch(problem, cost_function, incumbent_seed=best_cost).run()

    assert_same_search(seeded, AStarSearch(problem, cost_function).run())


def test_an_incumbent_seed_above_the_optimum_changes_nothing():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    # The cost of any complete solution is a valid seed; the first two items leave out 9.0.
    seed = cost_function.goal_cost((0, 1))

    seeded = PrunedAStarSearch(problem, cost_function, incumbent_seed=seed).run()

    assert seed > seeded.cost
    assert_same_search(seeded, AStarSearch(problem, cost_function).run())


def test_an_incumbent_seed_below_the_optimum_raises_when_the_frontier_empties():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    best_cost, _ = brute_force_best(WEIGHTS, 2)
    search = PrunedAStarSearch(problem, cost_function, incumbent_seed=best_cost - 1.0)

    with pytest.raises(IncumbentBelowOptimum, match="not an upper bound") as raised:
        search.run()

    assert raised.value.incumbent_seed == pytest.approx(best_cost - 1.0)
    assert isinstance(search.frontier_peak, int)


def test_a_seed_below_the_optimum_by_less_than_the_absolute_slack_still_finds_it():
    # The slack is the caller's statement of how far a seed may sit below the truth by rounding.
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    best_cost, _ = brute_force_best(WEIGHTS, 2)

    seeded = PrunedAStarSearch(
        problem, cost_function, incumbent_seed=best_cost - 0.5, incumbent_slack=1.0
    ).run()

    assert_same_search(seeded, AStarSearch(problem, cost_function).run())


def test_a_negative_absolute_slack_is_rejected():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    with pytest.raises(ValueError, match="incumbent_slack"):
        PrunedAStarSearch(problem, LeftOutWeightCost(problem), incumbent_slack=-1e-9)


def test_the_search_does_not_seed_the_incumbent_itself():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    assert PrunedAStarSearch(problem, LeftOutWeightCost(problem)).incumbent_seed is None


def test_a_tiny_frontier_cap_raises_and_carries_the_peak():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=3)
    cost_function = UninformedLeftOutWeightCost(problem)
    search = PrunedAStarSearch(problem, cost_function, max_frontier=2)

    with pytest.raises(FrontierLimitExceeded, match="no certificate") as raised:
        search.run()

    # The cap bites at the first push that would make three entries, so the peak is the cap.
    assert raised.value.max_frontier == 2
    assert raised.value.frontier_peak == 2
    assert search.frontier_peak == 2


def test_no_frontier_cap_and_a_generous_cap_run_the_same_search():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=3)
    cost_function = UninformedLeftOutWeightCost(problem)
    uncapped = PrunedAStarSearch(problem, cost_function)
    capped = PrunedAStarSearch(problem, cost_function, max_frontier=10_000)

    assert uncapped.run() == capped.run()
    assert uncapped.frontier_peak == capped.frontier_peak
    assert_same_search(uncapped.run(), AStarSearch(problem, cost_function).run())


def test_frontier_peak_is_none_before_run_and_an_integer_after():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    search = PrunedAStarSearch(problem, LeftOutWeightCost(problem))

    assert search.frontier_peak is None
    search.run()
    assert isinstance(search.frontier_peak, int)
    assert search.frontier_peak >= 1


def test_a_frontier_cap_below_one_is_rejected():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    with pytest.raises(ValueError, match="max_frontier"):
        PrunedAStarSearch(problem, LeftOutWeightCost(problem), max_frontier=0)


def test_exact_a_star_still_raises_a_plain_error_when_no_goal_is_reachable():
    # The seeded diagnosis belongs to the variant; a dead end under the base class is unchanged.
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=6)  # more picks than items: no goal

    with pytest.raises(ValueError, match="no goal state is reachable"):
        AStarSearch(problem, LeftOutWeightCost(problem)).run()
    with pytest.raises(ValueError, match="no goal state is reachable"):
        PrunedAStarSearch(problem, LeftOutWeightCost(problem)).run()
