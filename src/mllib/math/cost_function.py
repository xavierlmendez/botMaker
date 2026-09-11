"""The cost contract: a scalar of the parameters being optimized, the term a training loss starts from.

An abstract base under D-35 (2). A training loss is a cost plus its penalties
(`regularization_function.py`); the two-hot span cost is the first implementation
(`math/graph/two_hot_span_problem.py`), and a cost built from a per-sample loss for the descent
models is BL-49's. The base names no array library (D-35 (6)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mllib.math.task_kind import TaskKind


class AbstractCostFunction(ABC):
    """The scalar an optimizer descends before any penalty is added, as a function of its parameters.

    ``compute_cost(parameters)`` returns that scalar in the implementation's arithmetic,
    differentiable where the optimizer needs a gradient. ``task_kind`` is ``None`` unless the cost
    only makes sense for one kind of task; a subclass may narrow it.
    """

    __slots__ = ()

    task_kind: TaskKind | None = None

    @abstractmethod
    def compute_cost(self, parameters: Any) -> Any:
        """The cost at ``parameters``."""
        raise NotImplementedError
