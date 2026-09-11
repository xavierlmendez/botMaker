from collections import deque
from typing import Any

from mllib.math.algorithms.abstract_graph_algorithm import (
    AbstractGraphAlgorithm,
    SearchContext,
)
from mllib.math.graph.graph_structures import Graph
from mllib.math.recorder import AbstractTraversalRecorder, NullRecorder


class BreadthFirstSearch(AbstractGraphAlgorithm):
    """Graph traversal/search using breadth-first order."""

    def __init__(self, graph: Graph, *, recorder: AbstractTraversalRecorder | None = None):
        """``recorder`` watches the traversal; the default is a fresh ``NullRecorder`` (D-32).

        Keyword-only because a traversal's positional argument is the graph and nothing else, and a
        fresh instance rather than a shared default because a recorder accumulates frames: one
        shared between two searches would hold both runs.
        """
        super().__init__(graph)
        self.recorder: AbstractTraversalRecorder = recorder or NullRecorder()

    def _search(self, context: SearchContext) -> Any:
        """
        BFS traverses nodes level by level from a starting node.
        Returns the traversal order of node IDs; traversal stops when the target is found.
        `allowRevisiting=False` (default) keeps a visited set so cycles terminate;
        `max_depth` stops expanding nodes deeper than that level (start node is depth 0).
        """
        if context.start_node_id not in self.graph.nodes:
            raise KeyError(f"Unknown node: {context.start_node_id}")

        if context.target_node_criteria is None:
            raise KeyError("BFS must have target criteria")

        queue: deque[tuple[int, int]] = deque([(context.start_node_id, 0)])
        visited = {context.start_node_id}
        traversal_order: list[int] = []

        while queue:
            node_id, depth = queue.popleft()
            traversal_order.append(node_id)

            # The moment the traversal advances: this node has joined the order, and what is still
            # pending is the queue as it stands — in the order it will come out of it.
            if self.recorder.enabled:
                self.recorder.record_visit(
                    node_id,
                    depth,
                    list(queue),
                    sorted(visited),
                    list(traversal_order),
                )

            node = self.graph.nodes[node_id]

            if context.target_node_criteria(node):
                if self.recorder.enabled:
                    self.recorder.record_traversal_end(list(traversal_order), True)
                return traversal_order

            if context.max_depth is not None and depth >= context.max_depth:
                continue

            for neighbor_id in sorted(node.neighbors):
                if context.allow_revisiting or neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        if self.recorder.enabled:
            self.recorder.record_traversal_end(list(traversal_order), False)
        return traversal_order
