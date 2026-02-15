from collections import deque
from typing import Any

from botMaker.MlLib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import AbstractGraphAlgorithm, SearchContext
from botMaker.MlLib.mathDomain.graphBased.graphStructures import Graph


class BreadthFirstSearch(AbstractGraphAlgorithm):
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.metadata = {
            "name": "Breadth First Search",
            "description": "Graph traversal/search using breadth-first order."
        }

    def _search(self, context: SearchContext) -> Any:
        """
        BFS traverses nodes level by level from a starting node.
        Returns the traversal order of node IDs, traversal stops when target is found.
        """
        if context.start_node_id not in self.graph.nodes:
            raise KeyError(f"Unknown node: {context.start_node_id}")

        if context.target_node_criteria is None:
            raise KeyError(f"BFS must have target criteria")

        if context.allowRevisiting is not None and not context.allowRevisiting:
            return self.search_without_revisiting(context)

        if context.max_depth is not None:
            return self.search_with_max_depth(context)

        queue = deque([context.start_node_id])
        traversal_order: list[int] = []

        while queue:
            node_id = queue.popleft()

            traversal_order.append(node_id)
            node = self.graph.nodes[node_id]

            if context.target_node_criteria(node):
                return traversal_order

            for neighbor_id in sorted(node.neighbors):
                queue.append(neighbor_id)

        return traversal_order

    def search_without_revisting(self, context: SearchContext) -> Any:
        pass

    def search_with_max_depth(self, context: SearchContext) -> Any:
        pass