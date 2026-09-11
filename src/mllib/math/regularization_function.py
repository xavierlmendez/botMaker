"""The regularization contract: a penalty added to a training loss, as a scalar of the parameters.

An abstract base under D-35 (2); the concrete penalties live beside the optimizer that sums them
(`math/algorithms/two_hot_span/penalties.py`). The base names no array library (D-35 (6)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mllib.math.task_kind import TaskKind


class AbstractRegularizationFunction(ABC):
    """A term a training loss adds to its cost, weighted by a knob the term owns.

    ``term(parameters)`` is the unweighted quantity the term measures, the number a hand check
    reads; ``compute_penalty(parameters)`` is the signed, weighted scalar the loss adds: positive for
    a penalty, negative for a term that is rewarded, so a loss is always a plain sum of its cost and
    its terms. ``task_kind`` is ``None`` because a term on the parameters applies to regression and
    classification alike; a subclass may narrow it.
    """

    __slots__ = ()

    task_kind: TaskKind | None = None

    @abstractmethod
    def term(self, parameters: Any) -> Any:
        """The unweighted quantity this term measures at ``parameters``."""
        raise NotImplementedError

    @abstractmethod
    def compute_penalty(self, parameters: Any) -> Any:
        """The signed, weighted scalar added to the training loss at ``parameters``."""
        raise NotImplementedError
