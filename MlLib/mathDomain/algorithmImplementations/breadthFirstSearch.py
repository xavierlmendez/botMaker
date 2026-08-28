from collections import deque
from typing import Any

from MlLib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import (
    AbstractGraphAlgorithm,
    SearchContext,
)
from MlLib.mathDomain.graphBased.graphStructures import Graph


class BreadthFirstSearch(AbstractGraphAlgorithm):
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.metadata = {
            "name": "Breadth First Search",
            "description": "Graph traversal/search using breadth-first order.",
        }

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
            node = self.graph.nodes[node_id]

            if context.target_node_criteria(node):
                return traversal_order

            if context.max_depth is not None and depth >= context.max_depth:
                continue

            for neighbor_id in sorted(node.neighbors):
                if context.allowRevisiting or neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return traversal_order
