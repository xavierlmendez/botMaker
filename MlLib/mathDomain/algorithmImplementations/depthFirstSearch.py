from collections import deque
from typing import Any

from botMaker.MlLib.mathDomain.algorithmImplementations.abstractGraphAlgorithm import AbstractGraphAlgorithm, SearchContext
from botMaker.MlLib.mathDomain.graphBased.graphStructures import Graph


class DepthFirstSearch(AbstractGraphAlgorithm):
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.metadata = {
            "name": "Depth First Search",
            "description": "Graph traversal/search using depth-first order."
        }

    def _search(self, context: SearchContext) -> Any:
        """
        Depth‑first search (tree‑only, no revisiting):
        - Traverses as deep as possible along a branch before backtracking.
        - Assumes the input is a tree (no cycles), so no visited set is used.
        - Returns traversal order of node IDs; stops early when target is found.
        """
        if context.start_node_id not in self.graph.nodes:
            raise KeyError(f"Unknown node: {context.start_node_id}")

        if context.target_node_criteria is None:
            raise KeyError(f"DFS must have target criteria")

        queue = deque([context.start_node_id])
        traversal_order: list[int] = []


        return traversal_order
