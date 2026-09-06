"""A* search over an implicit graph problem with an injected cost function."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

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

    ``nodes_expanded`` is what the search paid in time, ``frontier_peak`` what it paid in memory:
    the largest number of entries the frontier held, for an engine that counts them. It is
    ``None`` when nobody counted — exact ``AStarSearch`` and every heuristic selector — so the
    field tells "not measured" apart from a measurement instead of carrying a fabricated zero.

    ``engine_configuration`` is the settings the search actually ran under, read off the engine
    rather than promised by the caller, and ``None`` for a selector that runs no search. A cost
    means nothing without it: a pruned run and an uncapped one are different measurements even on
    the same cell (D-26's rule, applied to the engine instead of to δ).

    The type carries these three and no more: what the search *paid* (``nodes_expanded``,
    ``frontier_peak``) and how it was *set up* (``engine_configuration``). A further measure of
    cost joins the first, a further knob is named inside the second — nothing else is added here
    (D-28).
    """

    state: State
    cost: float
    optimal: bool
    nodes_expanded: int
    frontier_peak: int | None = None
    engine_configuration: Mapping[str, object] | None = None


class AStarSearch[State, Action](AbstractGraphAlgorithm):
    """Best-first search that expands the state with the smallest admissible bound.

    With a terminal-only objective there is no accumulated path cost, so the priority ``f`` is the
    lower bound alone. The first goal state popped is optimal: every state still queued has a bound
    no smaller than its cost, and no completion can cost less than its own bound.

    Ties are broken first-in-first-out by insertion order, which keeps the search deterministic
    without claiming that any tie-break is better than another.

    This class stores every child it prices; it is the exact algorithm and the reference every
    variant is measured against. The loop is split into three steps a variant can override on its
    own: ``_price_children`` (generate and bound a parent's children), ``_push_children`` (decide
    what enters the frontier) and ``_no_goal_reachable`` (what an empty frontier means).
    ``PrunedAStarSearch`` overrides the last two.
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

    @property
    def configuration(self) -> dict[str, object]:
        """The settings this search ran under, as JSON-serialisable values.

        Read off the instance, so it reports what the engine *is* rather than what a caller said
        it would be; ``mllib.describe.describe`` is the complement, naming the knobs a class has
        rather than the values one instance was given. A variant overrides this to add its own
        knobs, and one that forgets shows only its class name — visibly incomplete rather than
        silently wrong. Exact A* has nothing to state but which engine ran.
        """
        return {"engine": type(self).__name__}

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
        expanded: set[State] = set()
        nodes_expanded = 0

        while queue:
            _, _, state = heapq.heappop(queue)
            if state in expanded:
                continue
            expanded.add(state)
            nodes_expanded += 1

            if self.problem.is_goal(state):
                return SearchResult(
                    state=state,
                    cost=self.cost_function.goal_cost(state),
                    optimal=True,
                    nodes_expanded=nodes_expanded,
                )

            children = self._price_children(state, expanded)
            insertion_index = self._push_children(queue, children, insertion_index)

        self._no_goal_reachable()

    def _price_children(self, parent: State, expanded: set[State]) -> list[tuple[State, float]]:
        """A parent's unexpanded children with their bounds, in the order the problem generates them.

        All successors are scored in one call so a cost function can share the work they have in
        common (D-24); the order is preserved so tie-breaking is unchanged.
        """
        successors = [
            (action, successor)
            for action, successor in self.problem.successors(parent)
            if successor not in expanded
        ]
        bounds = self.cost_function.lower_bounds(parent, successors)
        return [
            (successor, bound) for (_, successor), bound in zip(successors, bounds, strict=True)
        ]

    def _push_children(
        self,
        queue: list[tuple[float, int, State]],
        children: list[tuple[State, float]],
        insertion_index: int,
    ) -> int:
        """Place a parent's priced children on the frontier; return the last insertion index used.

        Exact A* keeps every child. A variant that keeps fewer must still return the index as if it
        had pushed them all, so that what it does push pops in the same order.
        """
        for state, bound in children:
            insertion_index += 1
            heapq.heappush(queue, (bound, insertion_index, state))
        return insertion_index

    def _no_goal_reachable(self) -> NoReturn:
        """The frontier emptied without a goal being popped."""
        raise ValueError("A* search failed: no goal state is reachable from the initial state.")
