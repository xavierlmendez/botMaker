"""The recorder for a graph traversal: one frame per visit, one for the moment it stops.

Breadth-first and depth-first search differ in one thing only — the container they keep their
pending nodes in — so they share one recorder rather than having one each. What the reader is meant
to see is exactly that difference: the same graph, the same start, the same frames, and a pending
list that empties from the front in one run and from the back in the other. Two recorders would
have made that comparison a comparison of two documents that happen to look alike.

The engine hands over the raw material it already holds and this module does the rest: the copy to
plain Python, the caption, and the decision about which of a traversal's numbers a reader actually
wants (how deep, how many are waiting, how far it has got).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mllib.math.recorder import AbstractTraversalRecorder, Frame


@dataclass(frozen=True, slots=True)
class TraversalFrame(Frame):
    """One visit of a graph traversal — or the moment it ended — in plain Python.

    ``node_id`` is the node just added to the traversal order and ``depth`` how far from the start
    it was reached; both are ``None`` on the end frame, which observes no node. ``pending`` is the
    queue or stack as the algorithm holds it, as ``(node_id, depth)`` pairs **in the container's
    own order**: a queue's first pair is the one that comes out next, a stack's last pair is.
    ``visited`` is the algorithm's visited set, sorted — which for breadth-first search is every
    node discovered (walked or merely queued) and for depth-first search every node already walked,
    because that is where each of them puts a node in the set. ``traversal_order`` is the order so
    far, and its length is what the caption calls "visited so far".

    ``end`` marks the run's last frame and ``found`` says why it is the last: the target matched,
    or the traversal ran out of nodes. Only a frame recorded after the final visit can tell those
    apart, which is why the end is a frame of its own and not a field on the last visit. On that
    frame ``pending`` and ``visited`` are both empty: ``record_traversal_end`` is handed neither
    container, because by then the question is about the run and not about what it was holding. A
    reader who wants the finished walk reads ``traversal_order``, which is complete on that frame.
    """

    node_id: int | None
    depth: int | None
    pending: list[tuple[int, int]] = field(default_factory=list)
    visited: list[int] = field(default_factory=list)
    traversal_order: list[int] = field(default_factory=list)
    end: bool = False
    found: bool = False

    def to_dict(self) -> dict[str, object]:
        """The frame as JSON-serialisable plain Python, pairs flattened to two-element lists.

        ``Frame.to_dict`` is named rather than reached through ``super()``: ``slots=True`` rebuilds
        the class after the method's ``__class__`` cell is bound, so a zero-argument ``super()``
        here resolves against the discarded class and raises.
        """
        return {
            **Frame.to_dict(self),
            "node_id": self.node_id,
            "depth": self.depth,
            "pending": [[node_id, depth] for node_id, depth in self.pending],
            "visited": list(self.visited),
            "traversal_order": list(self.traversal_order),
            "end": self.end,
            "found": self.found,
        }


class TraversalRecorder(AbstractTraversalRecorder):
    """Builds a ``TraversalFrame`` per visit and describes the traversal a run returned.

    Both breadth-first and depth-first search speak to this class through the same two calls, and it
    never asks which of them it is watching: everything that separates them is already in the
    ``pending`` snapshot it is handed. The recording says which algorithm produced it, in its
    ``problem``; the frames are one shape either way.
    """

    def record_visit(
        self,
        node_id: int,
        depth: int,
        queue_or_stack: Sequence[tuple[int, int]],
        visited: Sequence[int],
        traversal_order: Sequence[int],
        **extras: object,
    ) -> None:
        pending = [
            (int(pending_id), int(pending_depth)) for pending_id, pending_depth in queue_or_stack
        ]
        order = [int(visited_id) for visited_id in traversal_order]
        self.record(
            TraversalFrame(
                index=len(self.frames),
                caption=self._visit_caption(int(node_id), int(depth), len(pending), len(order)),
                node_id=int(node_id),
                depth=int(depth),
                pending=pending,
                visited=[int(seen) for seen in visited],
                traversal_order=order,
            )
        )

    def record_traversal_end(
        self,
        traversal_order: Sequence[int],
        found: bool,
        **extras: object,
    ) -> None:
        order = [int(visited_id) for visited_id in traversal_order]
        self.record(
            TraversalFrame(
                index=len(self.frames),
                caption=self._end_caption(len(order), bool(found)),
                node_id=None,
                depth=None,
                pending=[],
                visited=[],
                traversal_order=order,
                end=True,
                found=bool(found),
            )
        )

    def describe_result(self, result: Sequence[int]) -> dict[str, object]:
        """A traversal's result is the order it walked; the count is stored so nobody counts it."""
        order = [int(node_id) for node_id in result]
        return {"traversal_order": order, "visit_count": len(order)}

    @staticmethod
    def _visit_caption(node_id: int, depth: int, pending: int, visits: int) -> str:
        """One sentence: which node, how deep, how much is waiting, how far the run has got."""
        return (
            f"Visited node {node_id} at depth {depth}; "
            f"{pending} node{'' if pending == 1 else 's'} pending; "
            f"{visits} visited so far."
        )

    @staticmethod
    def _end_caption(visits: int, found: bool) -> str:
        """The last sentence: how long the traversal was and whether it stopped on its target."""
        outcome = "target found" if found else "target not found"
        return f"Traversal ended after {visits} visit{'' if visits == 1 else 's'}; {outcome}."
