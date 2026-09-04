"""A* search over an implicit graph problem with an injected cost function."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from mllib.math.algorithms.abstract_graph_algorithm import (
    AbstractGraphAlgorithm,
    SearchContext,
)
from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.search_cost_function import SearchCostFunction


@dataclass(frozen=True, slots=True)
class SearchResult[State]:
    """What a search returns: the solution, its cost, and what the search paid to find it.

    ``optimal`` is True only when the algorithm proves it. A* proves it with an admissible bound;
    a heuristic selector returning this same type sets it False.
    """

    state: State
    cost: float
    optimal: bool
    nodes_expanded: int


class AStarSearch[State, Action](AbstractGraphAlgorithm):
    """Best-first search that expands the state with the smallest admissible bound.

    With a terminal-only objective there is no accumulated path cost, so the priority ``f`` is the
    lower bound alone. The first goal state popped is optimal: every state still queued has a bound
    no smaller than its cost, and no completion can cost less than its own bound.

    Ties are broken first-in-first-out by insertion order, which keeps the search deterministic
    without claiming that any tie-break is better than another.
    """

    def __init__(
        self,
        problem: AbstractGraphProblem[State, Action],
        cost_function: SearchCostFunction[State, Action],
        evaluator: Any | None = None,
    ):
        super().__init__(problem, evaluator)
        self.cost_function = cost_function

    @property
    def problem(self) -> AbstractGraphProblem[State, Action]:
        """The implicit problem being searched; the base class stores it as ``graph``."""
        return self.graph

    def _search(self, context: SearchContext | None) -> SearchResult[State]:
        """Expand states in order of their lower bound until a goal is popped.

        ``context`` is unused: an implicit problem carries its own start state and goal test.
        """
        initial_state = self.problem.initial_state()
        # Queue entries are (bound, insertion order, state): the heap pops the smallest bound, and
        # the insertion order breaks ties first-in-first-out.
        queue: list[tuple[float, int, State]] = [
            (self.cost_function.lower_bound(initial_state), 0, initial_state)
        ]
        insertion_index = 0
        explored: set[State] = set()
        nodes_expanded = 0

        while queue:
            _, _, state = heapq.heappop(queue)
            if state in explored:
                continue
            explored.add(state)
            nodes_expanded += 1

            if self.problem.is_goal(state):
                return SearchResult(
                    state=state,
                    cost=self.cost_function.goal_cost(state),
                    optimal=True,
                    nodes_expanded=nodes_expanded,
                )

            # Score all of a parent's successors in one call so a cost function can share the work
            # they have in common (D-24); the order is preserved so tie-breaking is unchanged.
            successors = [
                (action, successor)
                for action, successor in self.problem.successors(state)
                if successor not in explored
            ]
            bounds = self.cost_function.lower_bounds(state, successors)
            for (_, successor), bound in zip(successors, bounds, strict=True):
                insertion_index += 1
                heapq.heappush(queue, (bound, insertion_index, successor))

        raise ValueError("A* search failed: no goal state is reachable from the initial state.")
