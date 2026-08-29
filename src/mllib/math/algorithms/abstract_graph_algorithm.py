from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mllib.math.graph.graph_structures import Graph


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Explicit inputs for graph search algorithms."""

    start_node_id: int
    target_node_criteria: Callable[[Any], bool] | None = None
    max_depth: int | None = None
    allow_revisiting: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractGraphAlgorithm(ABC):
    """Base class for graph search algorithms with evaluator orchestration."""

    def __init__(self, graph: Graph, evaluator: Any | None = None):
        self.graph = graph
        self.evaluator = evaluator

    def run(self, context: SearchContext) -> Any:
        """Execute the algorithm and optionally evaluate results."""
        result = self._search(context)
        self._notify_evaluator(result, context)
        return result

    @abstractmethod
    def _search(self, context: SearchContext) -> Any:
        """Algorithm-specific search implementation."""
        raise NotImplementedError

    def _notify_evaluator(self, result: Any, context: SearchContext) -> None:
        """
        If an evaluator is provided, pass it the result.

        Supported evaluator forms:
        - Object with an `evaluate(result, graph, context)` method
        - Callable: `evaluator(result, graph, context)`
        """
        if self.evaluator is None:
            return

        evaluate = getattr(self.evaluator, "evaluate", None)
        if callable(evaluate):
            evaluate(result, self.graph, context)
            return

        if callable(self.evaluator):
            self.evaluator(result, self.graph, context)
