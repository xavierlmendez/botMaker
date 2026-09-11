from typing import Any

from mllib.math.algorithms.abstract_graph_algorithm import (
    AbstractGraphAlgorithm,
    SearchContext,
)
from mllib.math.graph.graph_structures import Graph
from mllib.math.recorder import AbstractTraversalRecorder, NullRecorder


class DepthFirstSearch(AbstractGraphAlgorithm):
    """Graph traversal/search using depth-first order."""

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
        DFS goes as deep as possible along each branch before backtracking (iterative, with an
        explicit stack). Returns the traversal order of node IDs; stops when the target is found.
        `allowRevisiting=False` (default) keeps a visited set so cycles terminate;
        `max_depth` stops expanding nodes deeper than that level (start node is depth 0).
        Neighbours are pushed in reverse sorted order so the smallest id is explored first.
        """
        if context.start_node_id not in self.graph.nodes:
            raise KeyError(f"Unknown node: {context.start_node_id}")

        if context.target_node_criteria is None:
            raise KeyError("DFS must have target criteria")

        stack: list[tuple[int, int]] = [(context.start_node_id, 0)]
        visited: set[int] = set()
        traversal_order: list[int] = []

        while stack:
            node_id, depth = stack.pop()
            if not context.allow_revisiting:
                if node_id in visited:
                    continue
                visited.add(node_id)

            traversal_order.append(node_id)

            # The moment the traversal advances, recorded exactly as breadth-first search records
            # it. The pending container is passed in its own order — the stack bottom first, so its
            # *last* pair is the one that pops next, where a queue's *first* pair is. That
            # difference is the whole difference between the two algorithms, and recording the raw
            # order is what keeps it visible in the walkthrough instead of normalising it away.
            if self.recorder.enabled:
                self.recorder.record_visit(
                    node_id,
                    depth,
                    list(stack),
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

            for neighbor_id in sorted(node.neighbors, reverse=True):
                if context.allow_revisiting or neighbor_id not in visited:
                    stack.append((neighbor_id, depth + 1))

        if self.recorder.enabled:
            self.recorder.record_traversal_end(list(traversal_order), False)
        return traversal_order
