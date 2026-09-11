"""A* that stores less of its frontier: goal-sibling filter, incumbent pruning, a frontier cap."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from typing import Any, NoReturn

from mllib.math.algorithms.a_star_search import AStarSearch, HeapEntry, SearchResult, TieBreak
from mllib.math.algorithms.abstract_graph_algorithm import SearchContext
from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.recorder import AbstractSearchRecorder
from mllib.math.search_cost_function import SearchCostFunction

# Relative slack on the incumbent before a child is pruned. An incumbent seeded from another
# arithmetic path (the greedy selector prices its subset through ``goal_cost``; the search prices
# the same subset through the goal-depth batch) agrees with the search's own goal bounds only to
# rounding, and on a cell where the seed *is* the optimum a strict comparison pruned it (chain-6).
# Keeping a child is always safe and pruning one is what can be wrong, so the threshold leans this
# far toward keeping; the entries it keeps are the ones within rounding of the incumbent.
# A relative slack is nothing when the incumbent itself is rounding noise (an optimum of zero on a
# rank-deficient or all-ones kernel), and the search has no notion of the objective's scale; the
# caller does, and states it through ``incumbent_slack``.
INCUMBENT_RELATIVE_SLACK = 1e-9


# The two exceptions carry the names the plan fixed (§10, P-16, Q19); N818's suffix is waived here.
class FrontierLimitExceeded(RuntimeError):  # noqa: N818
    """The frontier would have grown past ``max_frontier``: a memory cap, so no certificate.

    ``frontier_peak`` is the largest number of entries the frontier held before the cap bit; the
    same number is left on the search instance.
    """

    def __init__(self, max_frontier: int, frontier_peak: int):
        self.max_frontier = max_frontier
        self.frontier_peak = frontier_peak
        super().__init__(
            f"A* frontier would exceed max_frontier={max_frontier} "
            f"(frontier_peak={frontier_peak}); the search has no certificate."
        )


class IncumbentBelowOptimum(ValueError):  # noqa: N818
    """The frontier emptied without a goal while an incumbent seed was set.

    Every state on the way to the optimum has a bound no larger than the optimum, so it survives
    pruning against any true upper bound. Running out of states therefore means the seed was
    below the optimum: it was not an upper bound, and the caller's premise was wrong.
    """

    def __init__(self, incumbent_seed: float):
        self.incumbent_seed = incumbent_seed
        super().__init__(
            f"A* pruned every state against incumbent_seed={incumbent_seed!r} and found no goal: "
            "the seed is below the optimum, so it was not an upper bound."
        )


def _prune_threshold(incumbent: float, incumbent_slack: float) -> float:
    return incumbent + INCUMBENT_RELATIVE_SLACK * abs(incumbent) + incumbent_slack


class PrunedAStarSearch[State, Action](AStarSearch[State, Action]):
    """A* that declines to store what it can prove it will never pop.

    Three mechanisms keep the frontier small without changing which states are expanded or which
    goal is returned. All three rest on the cost contract that ``lower_bound == goal_cost`` at a
    goal state (D-23), so a goal's bound is its exact objective.

    - **Goal-sibling filter.** Of the goal states priced from one parent, only the one with the
      smallest bound (first generated on ties) enters the frontier: a goal sibling with a larger
      bound costs at least as much and can never be the first goal popped.
    - **Incumbent pruning.** The smallest goal cost seen so far is an upper bound on the optimum;
      a child whose bound is above it by more than ``INCUMBENT_RELATIVE_SLACK`` (relative) plus
      ``incumbent_slack`` (absolute, default 0) is not pushed, since the incumbent goal is popped
      first. Ties, and anything within rounding of one, are kept, so the optimum set is unchanged.
      ``incumbent_seed`` lets a caller start the incumbent from any complete solution's cost; the
      search never seeds it itself. A seed is computed on another arithmetic path, so the caller
      also states its rounding allowance: ``incumbent_slack`` is the absolute amount by which a
      seed may fall below the search's own goal bounds and still be trusted (a Nyström caller
      passes a small multiple of the kernel trace, the quantity that rounding scales with, since an
      optimum of zero leaves the relative slack with nothing to act on). A seed below the optimum
      by more than that prunes every path and raises ``IncumbentBelowOptimum`` (D-27: an incumbent
      is always an exact objective, never a truncated bound).
    - **Frontier cap.** ``max_frontier`` counts frontier entries; a push that would exceed it
      raises ``FrontierLimitExceeded`` rather than returning an uncertified solution.

    Why a variant and not the algorithm: exact ``AStarSearch`` compares nothing it did not compute
    itself. A seeded incumbent is a number from another arithmetic path, and the slack that keeps
    it from pruning the optimum by rounding is a margin, not a proof. The base class stays the
    reference every "unchanged" test measures this one against.

    ``frontier_peak`` is ``None`` before ``run`` and afterwards the largest number of entries the
    frontier held, whether the search returned or raised: the memory cost beside ``nodes_expanded``
    as the time cost.

    The incumbent's *state* is kept beside its cost (``incumbent_seed_state`` names the seed's,
    when the caller knows it), for two readers: the anytime variant, which returns the incumbent at
    a stop, and the tie tolerance (D-29). Under a tolerance a goal in the same cell as the incumbent
    can pop before it, so a popped goal may be worse than a solution the search already holds; this
    engine then returns the incumbent instead, with the gap recomputed against the same frontier
    minimum, and certifies it only if the incumbent came from a priced goal and the gap is zero. At
    tolerance zero the first goal popped is the frontier minimum and this never arises, so the
    default path is unchanged; under a tolerance incumbent pruning also drops same-cell children
    the base engine would expand, so the expansion count may differ from the base's there.
    """

    def __init__(
        self,
        problem: AbstractGraphProblem[State, Action],
        cost_function: SearchCostFunction[State, Action],
        evaluator: Any | None = None,
        *,
        incumbent_seed: float | None = None,
        incumbent_seed_state: State | None = None,
        incumbent_slack: float = 0.0,
        max_frontier: int | None = None,
        tie_break: TieBreak = "fifo",
        tie_tolerance: float = 0.0,
        count_bound_drops: bool = False,
        bound_drop_slack: float = 0.0,
        recorder: AbstractSearchRecorder | None = None,
    ):
        super().__init__(
            problem,
            cost_function,
            evaluator,
            tie_break=tie_break,
            tie_tolerance=tie_tolerance,
            count_bound_drops=count_bound_drops,
            bound_drop_slack=bound_drop_slack,
            recorder=recorder,
        )
        if max_frontier is not None and max_frontier < 1:
            raise ValueError("max_frontier must be at least 1: the frontier starts with one state.")
        if not incumbent_slack >= 0.0:
            raise ValueError(
                "incumbent_slack is an absolute rounding allowance and cannot be negative."
            )
        if incumbent_seed_state is not None:
            if incumbent_seed is None:
                raise ValueError("incumbent_seed_state names the solution incumbent_seed costs.")
            if not problem.is_goal(incumbent_seed_state):
                raise ValueError("incumbent_seed_state must be a complete solution (a goal state).")
        self.incumbent_seed = incumbent_seed
        self.incumbent_seed_state = incumbent_seed_state
        self.incumbent_slack = incumbent_slack
        self.max_frontier = max_frontier
        self.frontier_peak: int | None = None
        self._incumbent: float | None = None
        self._incumbent_state: State | None = None
        self._incumbent_from_search = False
        self._prune_above = math.inf
        # What the last expansion declined to store; only ever replaced while a recorder is
        # enabled, so an unwatched run keeps this one empty tuple from start to finish.
        self._pruned_children: Sequence[tuple[State, float, str]] = ()

    @property
    def configuration(self) -> dict[str, object]:
        """The three knobs that change what this search stores, beside the engine name.

        All three change the measurement and none change the answer, so a cost recorded without
        them is not comparable to one recorded with different ones (D-28).
        """
        return {
            **super().configuration,
            "incumbent_seed": self.incumbent_seed,
            "incumbent_slack": self.incumbent_slack,
            "max_frontier": self.max_frontier,
        }

    def _search(self, context: SearchContext | None) -> SearchResult[State]:
        self._incumbent = self.incumbent_seed
        self._prune_above = (
            math.inf
            if self._incumbent is None
            else _prune_threshold(self._incumbent, self.incumbent_slack)
        )
        self._incumbent_state = self.incumbent_seed_state
        self._incumbent_from_search = False
        self.frontier_peak = 1  # the initial state
        result = super()._search(context)
        # Only a tie tolerance can pop a goal above a held incumbent (same cell, earlier or deeper);
        # the incumbent is the better answer and the frontier minimum at the pop bounds it too.
        if (
            self.tie_tolerance > 0.0
            and self._incumbent is not None
            and self._incumbent < result.cost
            and self._incumbent_state != result.state
        ):
            frontier_min = result.cost - result.certified_gap
            gap = max(self._incumbent - frontier_min, 0.0)
            # The base loop already recorded the popped goal, and this engine is about to return a
            # different solution: without a second goal frame the run's last frame would name a
            # state that never came back. The frontier is the one at the pop, which is also what
            # the gap is measured against.
            if self.recorder.enabled:
                goal_frontier, expanded_set_size = self._recorded_goal_frontier
                self.recorder.record_goal(
                    self._incumbent_state,
                    self._incumbent,
                    result.nodes_expanded,
                    goal_frontier,
                    expanded_set_size,
                    superseded_goal=result.state,
                    superseded_cost=result.cost,
                    **self._recorder_extras(),
                )
            return SearchResult(
                state=self._incumbent_state,
                cost=self._incumbent,
                optimal=gap == 0.0 and self._incumbent_from_search,
                nodes_expanded=result.nodes_expanded,
                certified_gap=gap,
            )
        return result

    def _push_children(
        self,
        queue: list[HeapEntry[State]],
        children: list[tuple[State, float]],
        insertion_index: int,
    ) -> int:
        is_goal = [self.problem.is_goal(state) for state, _ in children]

        # Goal-sibling filter: keep the goal child with the smallest bound, first on ties, and let
        # its cost tighten the incumbent before any sibling is pushed.
        kept_goal = min(
            (position for position, goal in enumerate(is_goal) if goal),
            key=lambda position: children[position][1],
            default=None,
        )
        if kept_goal is not None:
            goal_cost = children[kept_goal][1]
            if self._incumbent is None or goal_cost < self._incumbent:
                self._incumbent = goal_cost
                self._incumbent_state = children[kept_goal][0]
                self._incumbent_from_search = True
                self._prune_above = _prune_threshold(goal_cost, self.incumbent_slack)

        prune_above = self._prune_above
        max_frontier = self.max_frontier
        # Off by default: with no recorder watching, nothing below is allocated and nothing is
        # appended — ``_pruned_children`` keeps the empty tuple ``__init__`` gave it.
        recording = self.recorder.enabled
        pruned: list[tuple[State, float, str]] = []
        if recording:
            self._pruned_children = pruned
        for position, (state, bound) in enumerate(children):
            # Every child takes an insertion index, pushed or not, so the entries that are pushed
            # pop in exactly the order exact A* would pop them.
            insertion_index += 1
            if is_goal[position] and position != kept_goal:
                if recording:
                    pruned.append((state, bound, "goal_sibling"))
                continue
            if bound > prune_above:
                if recording:
                    pruned.append((state, bound, "above_incumbent"))
                continue
            if max_frontier is not None and len(queue) >= max_frontier:
                self.frontier_peak = max(self.frontier_peak, len(queue))
                raise FrontierLimitExceeded(max_frontier, self.frontier_peak)
            heapq.heappush(queue, self._heap_entry(bound, insertion_index, state))
        # Pops happen once per expansion before any push, so the frontier is largest here.
        self.frontier_peak = max(self.frontier_peak, len(queue))
        return insertion_index

    def _recorder_extras(self) -> dict[str, object]:
        """The two things this variant tracks that exact A* does not: prunings and the incumbent.

        ``pruned`` is one entry per child left off the frontier this expansion — its state, its
        bound and which mechanism dropped it — and the incumbent is the best complete solution the
        search holds, with the state it belongs to. Nothing here is measured for the recorder's
        sake; all of it is state the engine keeps anyway to decide what to store.
        """
        return {
            **super()._recorder_extras(),
            "pruned": tuple(self._pruned_children),
            "incumbent": self._incumbent,
            "incumbent_state": self._incumbent_state,
        }

    def _no_goal_reachable(self) -> NoReturn:
        if self.incumbent_seed is not None:
            raise IncumbentBelowOptimum(self.incumbent_seed)
        super()._no_goal_reachable()
