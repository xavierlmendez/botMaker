"""A* over an implicit graph problem, tested on a toy that has a known answer.

The toy is "keep the k heaviest of n weighted items, pay for what is left out". Its optimum is
obvious by inspection, which is what makes it a usable oracle for the search itself. Two admissible
bounds are supplied, an exact one and one carrying no information, so the tests can show that
admissibility buys optimality and tightness buys speed.
"""

from collections.abc import Iterable
from itertools import combinations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import AStarSearch, SearchResult
from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.search_cost_function import SearchCostFunction

Subset = tuple[int, ...]


class KeepHeaviestProblem(AbstractGraphProblem[Subset, int]):
    """Choose ``keep_count`` of ``weights``; states are ascending index tuples."""

    def __init__(self, weights: tuple[float, ...], keep_count: int):
        self.weights = weights
        self.keep_count = keep_count

    def initial_state(self) -> Subset:
        return ()

    def is_goal(self, state: Subset) -> bool:
        return len(state) == self.keep_count

    def successors(self, state: Subset) -> Iterable[tuple[int, Subset]]:
        still_needed = self.keep_count - len(state)
        first = 0 if not state else state[-1] + 1
        last = len(self.weights) - still_needed
        return [(index, (*state, index)) for index in range(first, last + 1)]


class UnorderedKeepHeaviestProblem(KeepHeaviestProblem):
    """The same problem reached by any order of picks, so one subset has many routes to it."""

    def successors(self, state: Subset) -> Iterable[tuple[int, Subset]]:
        if len(state) >= self.keep_count:
            return []
        return [
            (index, tuple(sorted((*state, index))))
            for index in range(len(self.weights))
            if index not in state
        ]


class LeftOutWeightCost(SearchCostFunction[Subset, int]):
    """Cost of a subset is the weight it leaves out; the bound credits the best possible finish."""

    def __init__(self, problem: KeepHeaviestProblem):
        self.problem = problem
        self.total = float(sum(problem.weights))

    def goal_cost(self, state: Subset) -> float:
        return self.total - sum(self.problem.weights[index] for index in state)

    def _reachable_weights(self, state: Subset) -> list[float]:
        first = 0 if not state else state[-1] + 1
        return sorted(self.problem.weights[first:], reverse=True)

    def lower_bound(self, state: Subset) -> float:
        still_needed = self.problem.keep_count - len(state)
        selected = sum(self.problem.weights[index] for index in state)
        best_finish = sum(self._reachable_weights(state)[:still_needed])
        return self.total - selected - best_finish


class UninformedLeftOutWeightCost(LeftOutWeightCost):
    """Same objective, no information: zero is admissible because every cost is non-negative."""

    def lower_bound(self, state: Subset) -> float:
        if self.problem.is_goal(state):
            return self.goal_cost(state)
        return 0.0


class DeadEndProblem(AbstractGraphProblem[Subset, int]):
    """A problem whose only state is not a goal and has no successors."""

    def initial_state(self) -> Subset:
        return ()

    def is_goal(self, state: Subset) -> bool:
        return False

    def successors(self, state: Subset) -> Iterable[tuple[int, Subset]]:
        return []


class ZeroCost(SearchCostFunction[Subset, int]):
    """Zero everywhere: admissible for any non-negative objective, and useless as a guide."""

    def lower_bound(self, state: Subset) -> float:
        return 0.0

    def goal_cost(self, state: Subset) -> float:
        return 0.0


WEIGHTS = (5.0, 1.0, 4.0, 2.0, 3.0)


def brute_force_best(weights: tuple[float, ...], keep_count: int) -> tuple[float, Subset]:
    total = sum(weights)
    return min(
        (total - sum(weights[index] for index in subset), subset)
        for subset in combinations(range(len(weights)), keep_count)
    )


def test_a_star_returns_the_best_subset_and_reports_it_as_optimal():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    result = AStarSearch(problem, LeftOutWeightCost(problem)).run()

    assert result.state == (0, 2)  # weights 5.0 and 4.0
    assert result.cost == pytest.approx(6.0)
    assert result.optimal


def test_a_star_runs_through_the_template_method_without_a_search_context():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    seen: list[SearchResult[Subset]] = []

    def record(outcome, graph, context):
        seen.append(outcome)

    result = AStarSearch(problem, LeftOutWeightCost(problem), evaluator=record).run()

    assert seen == [result]
    assert seen[0].nodes_expanded >= 1


def test_an_uninformative_bound_makes_a_star_enumerate_the_whole_graph():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=3)

    informed = AStarSearch(problem, LeftOutWeightCost(problem)).run()
    uninformed = AStarSearch(problem, UninformedLeftOutWeightCost(problem)).run()

    # Both are optimal: admissibility is what A* needs, tightness is what it charges for.
    assert informed.cost == pytest.approx(uninformed.cost)
    assert informed.nodes_expanded == 4  # one state per pick, plus the empty state
    # Every non-goal state is bounded at zero, so all ten of them precede the first goal:
    # the empty state, 3 one-pick prefixes and 6 two-pick prefixes that can still reach size 3.
    assert uninformed.nodes_expanded == 11


def test_a_star_expands_a_repeated_state_only_once():
    ordered = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    unordered = UnorderedKeepHeaviestProblem(WEIGHTS, keep_count=2)

    from_ordered = AStarSearch(ordered, LeftOutWeightCost(ordered)).run()
    from_unordered = AStarSearch(unordered, LeftOutWeightCost(unordered)).run()

    assert from_unordered.state == from_ordered.state
    # Every subset is reachable by k! routes here; without deduplication the count would exceed
    # the number of distinct subsets, which is 11 including the empty state.
    assert from_unordered.nodes_expanded <= 11


def test_a_star_raises_when_no_goal_state_is_reachable():
    problem = DeadEndProblem()

    with pytest.raises(ValueError, match="no goal state is reachable"):
        AStarSearch(problem, ZeroCost()).run()


def test_lower_bound_equals_goal_cost_at_a_goal_state():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    for subset in combinations(range(len(WEIGHTS)), 2):
        assert cost_function.lower_bound(subset) == pytest.approx(cost_function.goal_cost(subset))


@settings(max_examples=40, deadline=None)
@given(
    weights=st.lists(st.floats(min_value=0.5, max_value=50.0), min_size=2, max_size=7),
    keep_count=st.integers(min_value=1, max_value=4),
)
def test_a_star_matches_brute_force_on_random_instances(weights: list[float], keep_count: int):
    keep_count = min(keep_count, len(weights))
    problem = KeepHeaviestProblem(tuple(weights), keep_count)

    result = AStarSearch(problem, LeftOutWeightCost(problem)).run()
    best_cost, _ = brute_force_best(tuple(weights), keep_count)

    assert result.cost == pytest.approx(best_cost, abs=1e-9)


def test_the_default_batch_bound_is_the_per_child_bound_in_order():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    successors = list(problem.successors((0,)))

    batched = cost_function.lower_bounds((0,), successors)

    assert list(batched) == [cost_function.lower_bound(state) for _, state in successors]
    assert cost_function.lower_bounds((0,), []) == []


class RecordingCost(LeftOutWeightCost):
    """Counts how the search asks for bounds, to pin that it asks per parent, not per child."""

    def __init__(self, problem: KeepHeaviestProblem):
        super().__init__(problem)
        self.batch_parents: list[Subset] = []
        self.single_calls = 0

    def lower_bound(self, state: Subset) -> float:
        self.single_calls += 1
        return super().lower_bound(state)

    def lower_bounds(self, parent, successors):
        self.batch_parents.append(parent)
        return [super().lower_bound(state) for _, state in successors]


def test_a_star_scores_each_expanded_parents_successors_in_one_batch():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = RecordingCost(problem)

    result = AStarSearch(problem, cost_function).run()

    # One batch per non-goal state expanded; the only single call is the root's own bound.
    assert result.state == (0, 2)
    assert len(cost_function.batch_parents) == result.nodes_expanded - 1
    assert cost_function.batch_parents[0] == ()
    assert cost_function.single_calls == 1
