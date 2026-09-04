"""Contract for graph problems whose states and edges are generated during search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable


class AbstractGraphProblem[State: Hashable, Action](ABC):
    """Defines an implicit directed graph for a search algorithm.

    Unlike ``Graph``, this contract does not require every node and edge to be materialized before
    search begins: states are produced on demand from the state the search is currently at. That is
    what makes subset selection searchable, since its graph has one node per subset (D-23).

    States must be hashable and must identify a position uniquely, so that two routes to the same
    position produce equal states and the search can recognize them as one node.
    """

    @abstractmethod
    def initial_state(self) -> State:
        """Return the state from which search begins."""
        raise NotImplementedError

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        """Return whether ``state`` is a completed solution."""
        raise NotImplementedError

    @abstractmethod
    def successors(self, state: State) -> Iterable[tuple[Action, State]]:
        """Yield the legal actions out of ``state`` with the state each one leads to."""
        raise NotImplementedError
