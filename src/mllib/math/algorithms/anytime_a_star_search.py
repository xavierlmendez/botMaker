"""Anytime A*: when a cap stops the search, the incumbent comes back with a certified gap."""

from __future__ import annotations

import math
from typing import Any

from mllib.math.algorithms.a_star_search import SearchResult
from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.algorithms.pruned_a_star_search import FrontierLimitExceeded, PrunedAStarSearch
from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.search_cost_function import SearchCostFunction


# Private control flow, not an error a caller sees; N818's suffix is waived as the module's siblings do.
class _SearchStopped(Exception):  # noqa: N818
    """Raised from ``_push_children`` when a cap bites, carrying the frontier minimum at that moment."""

    def __init__(self, frontier_min: float):
        self.frontier_min = frontier_min
        super().__init__(frontier_min)


class AnytimeAStarSearch[State, Action](PrunedAStarSearch[State, Action]):
    """Pruned A* that returns its incumbent with a certified gap when a cap stops it, not an error.

    ``PrunedAStarSearch`` raises ``FrontierLimitExceeded`` at its memory cap and has no notion of a
    time cap, so a run that hits either ends with nothing to show. This variant adds
    ``max_expansions`` beside the inherited ``max_frontier`` and turns both stops into a result: the
    best complete solution seen so far (the incumbent), ``optimal=False``, and the two numbers that
    make the answer honest — ``frontier_min``, the smallest bound the frontier held when the search
    stopped, and ``certified_gap = max(incumbent - frontier_min, 0)``. No completion of any
    unexpanded state can cost less than ``frontier_min``, so the optimum lies in
    ``[frontier_min, incumbent]`` and the gap is an additive certificate on the incumbent. The gap
    rides out on ``SearchResult.certified_gap`` (what the search proved, D-28 amended 2026-09-07)
    and both numbers are left on the instance. When the search pops a goal before any cap bites it
    returns exactly what the base class returns, ``certified_gap == 0.0`` included, and
    ``frontier_min`` is the cost.

    The frontier minimum is read at the moment of the stop. At an expansion cap it is the heap's
    top: every unexpanded state that could still lead below the incumbent is on the heap (pruned
    children were above the incumbent, dropped goal siblings at or above a kept goal's cost). At a
    frontier cap the push of one batch was interrupted, so the minimum is taken over the heap and
    every child of that batch, pushed or not. Neither reading assumes the bound is monotone along
    the tree; monotonicity is what makes the gap non-increasing over a run, which is the finding
    the consistency measurement (BL-33) established and the test pins.

    ``incumbent_seed`` is a cost from another arithmetic path whose state the engine cannot know;
    ``incumbent_seed_state`` lets the caller supply it, so a run stopped before any goal is priced
    still reports the seed's state. Without it the result carries ``state=None`` and ``cost=seed``
    until a goal tightens the incumbent. A run stopped with no incumbent at all — no seed and no
    goal priced yet — reports ``cost=math.inf`` and ``certified_gap=math.inf``: nothing was proved,
    and the result says so instead of raising, since the caller who wants a finite answer supplies
    a seed. A stop never sets ``optimal``, even when the gap clamps to zero: the flag means a goal
    was popped as the frontier minimum, and only that.

    ``configuration`` adds ``max_expansions``; the seed state changes only what a stop reports, not
    what the search does, so it is not a setting (D-28). ``certified_gap`` and ``frontier_min`` are
    ``None`` before ``run``.
    """

    def __init__(
        self,
        problem: AbstractGraphProblem[State, Action],
        cost_function: SearchCostFunction[State, Action],
        evaluator: Any | None = None,
        *,
        max_expansions: int | None = None,
        incumbent_seed: float | None = None,
        incumbent_seed_state: State | None = None,
        incumbent_slack: float = 0.0,
        max_frontier: int | None = None,
        count_bound_drops: bool = False,
        bound_drop_slack: float = 0.0,
    ):
        super().__init__(
            problem,
            cost_function,
            evaluator,
            incumbent_seed=incumbent_seed,
            incumbent_slack=incumbent_slack,
            max_frontier=max_frontier,
            count_bound_drops=count_bound_drops,
            bound_drop_slack=bound_drop_slack,
        )
        if max_expansions is not None and max_expansions < 1:
            raise ValueError("max_expansions must be at least 1: the initial state is expanded.")
        if incumbent_seed_state is not None:
            if incumbent_seed is None:
                raise ValueError("incumbent_seed_state names the solution incumbent_seed costs.")
            if not problem.is_goal(incumbent_seed_state):
                raise ValueError("incumbent_seed_state must be a complete solution (a goal state).")
        self.max_expansions = max_expansions
        self.incumbent_seed_state = incumbent_seed_state
        self.certified_gap: float | None = None
        self.frontier_min: float | None = None
        self._incumbent_state: State | None = None
        self._expansions = 0

    @property
    def configuration(self) -> dict[str, object]:
        """The inherited knobs plus the expansion cap; a stop is a different measurement (D-28)."""
        return {**super().configuration, "max_expansions": self.max_expansions}

    def _search(self, context: SearchContext | None) -> SearchResult[State]:
        self._incumbent_state = self.incumbent_seed_state
        self._expansions = 0
        self.certified_gap = self.frontier_min = None
        try:
            result = super()._search(context)
        except _SearchStopped as stopped:
            return self._bounded_result(stopped.frontier_min)
        self.certified_gap, self.frontier_min = 0.0, result.cost
        return result

    def _push_children(
        self,
        queue: list[tuple[float, int, State]],
        children: list[tuple[State, float]],
        insertion_index: int,
    ) -> int:
        # The base class keeps the goal child with the smallest bound, first on ties, and lets its
        # cost tighten the incumbent; the same rule here remembers which state that was.
        goals = [
            (bound, position, state)
            for position, (state, bound) in enumerate(children)
            if self.problem.is_goal(state)
        ]
        if goals:
            bound, _, state = min(goals)
            if self._incumbent is None or bound < self._incumbent:
                self._incumbent_state = state
        # The parent was expanded before its children reached here: count it whatever happens next.
        self._expansions += 1
        try:
            insertion_index = super()._push_children(queue, children, insertion_index)
        except FrontierLimitExceeded:
            # The cap interrupted this batch: the frontier is the heap plus every child not pushed,
            # so the minimum is taken over both (a pushed child is on the heap already).
            raise _SearchStopped(min(queue[0][0], min(bound for _, bound in children))) from None
        # An empty heap after the last push means the base loop is about to conclude on its own.
        if self.max_expansions is not None and self._expansions >= self.max_expansions and queue:
            raise _SearchStopped(queue[0][0])
        return insertion_index

    def _bounded_result(self, frontier_min: float) -> SearchResult[State]:
        self.frontier_min = frontier_min
        incumbent = self._incumbent
        cost = math.inf if incumbent is None else incumbent
        self.certified_gap = math.inf if incumbent is None else max(incumbent - frontier_min, 0.0)
        return SearchResult(
            state=self._incumbent_state,
            cost=cost,
            optimal=False,
            nodes_expanded=self._expansions,
            certified_gap=self.certified_gap,
        )
