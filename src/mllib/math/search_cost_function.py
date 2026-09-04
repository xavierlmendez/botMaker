"""Cost contract injected into search algorithms over implicit graph problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class SearchCostFunction[State, Action](ABC):
    """Defines the objective a search minimizes and the bound that makes A* admissible.

    The contract is deliberately *terminal-objective only*: cost is a property of a completed
    solution, not of the path taken to it. Column subset selection is the motivating case — the
    error of a subset does not depend on the order its members were chosen in. A problem whose cost
    accumulates along edges needs an edge-cost contract; this is not it (D-23).

    ``lower_bounds`` is the extension point for problems where the successors of one parent share
    work. A search scores a parent's successors together through it; the default scores them one at
    a time, so a cost function that has nothing to share implements only ``lower_bound`` (D-24).
    """

    @abstractmethod
    def lower_bound(self, state: State) -> float:
        """Return an admissible bound on the best goal cost reachable from ``state``.

        Admissible means: never greater than the true cost of the best completion. At a goal state
        it must equal ``goal_cost``, since the only completion is the state itself.
        """
        raise NotImplementedError

    @abstractmethod
    def goal_cost(self, state: State) -> float:
        """Return the exact objective value of a completed solution."""
        raise NotImplementedError

    def lower_bounds(
        self, parent: State, successors: Sequence[tuple[Action, State]]
    ) -> Sequence[float]:
        """Return the bound of every successor of ``parent``, in the order given.

        Override when siblings share work: a decomposition of the parent that each child then
        updates cheaply. Each returned value must equal what ``lower_bound`` would return for that
        successor, up to rounding, so the two paths are interchangeable for correctness.
        """
        return [self.lower_bound(state) for _, state in successors]
