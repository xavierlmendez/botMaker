"""A* search over an implicit graph problem with an injected cost function."""

from __future__ import annotations

import heapq
from array import array
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, NoReturn

from mllib.math.algorithms.abstract_graph_algorithm import (
    AbstractGraphAlgorithm,
    SearchContext,
)
from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.search_cost_function import SearchCostFunction

# A parent's bound and its children's come from different arithmetic paths (each batch is priced from
# its own parent's decomposition, D-24), so two bounds that are equal in exact arithmetic agree only to
# rounding. A child below its parent by less than this fraction of the parent, plus the absolute
# ``rounding_slack`` the caller states, is that rounding and not a property of the bound; the counter
# reports both counts so a reader can tell the two apart. The relative part is the same order as
# ``INCUMBENT_RELATIVE_SLACK`` in the pruned variant; the absolute part exists for the same reason
# ``incumbent_slack`` does — rounding on a residual trace scales with the trace, not with a bound
# that may itself be rounding noise, and only the caller knows the trace.
BOUND_DROP_ROUNDING_TOLERANCE = 1e-9


@dataclass(slots=True)
class BoundDropCounter:
    """How often a child's bound fell below its parent's in one search: the consistency measurement.

    A* proves optimality with an admissible bound. *Consistency* — a child's bound never below its
    parent's, so the bound is monotone along the tree — is a second property that a terminal-only
    objective does not need for correctness: with g ≡ 0 the priority is path-independent and
    duplicates are detected by state (D-23). It decides instead how the frontier minimum behaves
    over a run, which is the anytime gap, and whether a tight root bound predicts pruning. That is
    a question about a particular bound on particular data, so it is measured, not argued: this
    counter records the answer and changes nothing about the search.

    ``children_priced`` is every child bounded during the run. ``drops`` counts the children whose
    bound was strictly below their parent's; ``drops_beyond_rounding`` those below by more than
    ``BOUND_DROP_ROUNDING_TOLERANCE`` of the parent's bound plus ``rounding_slack``, the absolute
    allowance the caller states for the objective's scale. ``drops_by_depth`` keys the strict drops
    by the child's depth (the initial state is depth 0). The ``worst_*`` fields describe the largest
    drop beyond rounding, relative to the parent, ``(parent - child) / |parent|``, which is ``inf``
    for a drop below a parent bounded at exactly zero. A run whose every drop is within rounding
    leaves the worst at zero and its depth at ``None``: the strict count says there were drops, the
    worst says none of them was the bound's doing.
    """

    rounding_slack: float = 0.0
    children_priced: int = 0
    drops: int = 0
    drops_beyond_rounding: int = 0
    drops_by_depth: dict[int, int] = field(default_factory=dict)
    worst_relative_drop: float = 0.0
    worst_drop_depth: int | None = None
    worst_drop_parent_bound: float | None = None
    worst_drop_child_bound: float | None = None

    def record(self, parent_bound: float, child_bounds: Iterable[float], child_depth: int) -> None:
        """Compare one parent's bound with each of its children's; children are at ``child_depth``."""
        child_bounds = list(child_bounds)
        self.children_priced += len(child_bounds)
        if not child_bounds or min(child_bounds) >= parent_bound:
            return
        scale = abs(parent_bound)
        rounding = BOUND_DROP_ROUNDING_TOLERANCE * scale + self.rounding_slack
        for child_bound in child_bounds:
            if child_bound >= parent_bound:
                continue
            self.drops += 1
            self.drops_by_depth[child_depth] = self.drops_by_depth.get(child_depth, 0) + 1
            drop = parent_bound - child_bound
            if drop <= rounding:
                continue
            self.drops_beyond_rounding += 1
            relative = drop / scale if scale > 0.0 else float("inf")
            if relative > self.worst_relative_drop:
                self.worst_relative_drop = relative
                self.worst_drop_depth = child_depth
                self.worst_drop_parent_bound = parent_bound
                self.worst_drop_child_bound = child_bound


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

    ``count_bound_drops`` turns on the one measurement the engine can make that no cost function
    can: whether a child's bound ever falls below its parent's as the search actually priced them
    (``BoundDropCounter``). It is off by default, changes no expansion and no result, and its
    answer is left on the instance as ``bound_drops`` after ``run`` — a property of the bound on
    the data, not something the search paid or was set up with, so it does not ride on
    ``SearchResult`` (D-28). ``bound_drop_slack`` is the absolute amount by which a child may fall
    below its parent and still be rounding rather than a drop, stated by the caller because only
    the caller knows the objective's scale (a Nyström caller passes a small multiple of the kernel
    trace, as it does for ``incumbent_slack``). Both are stated by ``configuration``, so a harness
    row that ran with the counter on says so.
    """

    def __init__(
        self,
        problem: AbstractGraphProblem[State, Action],
        cost_function: SearchCostFunction[State, Action],
        evaluator: Any | None = None,
        *,
        count_bound_drops: bool = False,
        bound_drop_slack: float = 0.0,
    ):
        super().__init__(problem, evaluator)
        if not bound_drop_slack >= 0.0:
            raise ValueError(
                "bound_drop_slack is an absolute rounding allowance and cannot be negative."
            )
        self.cost_function = cost_function
        self.count_bound_drops = count_bound_drops
        self.bound_drop_slack = bound_drop_slack
        self.bound_drops: BoundDropCounter | None = None

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
        silently wrong. Exact A* states which engine ran and whether it counted bound drops.
        """
        return {
            "engine": type(self).__name__,
            "count_bound_drops": self.count_bound_drops,
            "bound_drop_slack": self.bound_drop_slack,
        }

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

        # Bookkeeping for the bound-drop counter only. The children of one expansion take one
        # contiguous run of insertion indices, pushed or not (the ``_push_children`` contract), so
        # the depth of a popped entry is the depth of the run its index falls in: two arrays with
        # one entry per expansion, and nothing stored per state.
        counter = BoundDropCounter(self.bound_drop_slack) if self.count_bound_drops else None
        self.bound_drops = counter
        run_ends = array("q")
        run_depths = array("q")

        while queue:
            bound, index, state = heapq.heappop(queue)
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
            if counter is not None:
                depth = 0 if index == 0 else run_depths[bisect_left(run_ends, index)]
                counter.record(bound, (child_bound for _, child_bound in children), depth + 1)
                last_index = insertion_index
            insertion_index = self._push_children(queue, children, insertion_index)
            if counter is not None and insertion_index > last_index:
                run_ends.append(insertion_index)
                run_depths.append(depth + 1)

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
