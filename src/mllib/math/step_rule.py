"""The step-rule contract: the update an optimizer applies to its parameters at each step.

A step rule is injected into an optimizer (D-35 (3)) and carries state across the steps of one
run, which is what earns it a class (D-35 (1)): Adam's moment estimates live on the rule, and
one rule serves one run. Its learning rate and schedule are its knobs. The base names no array
library (D-35 (6)); the torch member is `math/algorithms/two_hot_span/step_rules.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractStepRule(ABC):
    """One run's update rule, bound once to the parameters it moves.

    The optimizer calls ``bind(parameters, step_count)`` once before its loop, then each step
    ``zero_gradient()``, reads ``learning_rate_in_force()`` for the record, fills the gradient
    by its own backward pass, and calls ``step()``. Binding twice is a mistake the rule refuses:
    a rebuilt rule would reset its state and change the trajectory, not just the step size.
    """

    __slots__ = ()

    @abstractmethod
    def bind(self, parameters: Any, step_count: int) -> None:
        """Attach to the parameters of one run of ``step_count`` steps; refuse a second binding."""
        raise NotImplementedError

    @abstractmethod
    def learning_rate_in_force(self) -> float:
        """The learning rate the next ``step()`` will use, the schedule applied."""
        raise NotImplementedError

    @abstractmethod
    def zero_gradient(self) -> None:
        """Clear the parameters' gradient before the step's backward pass."""
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        """Move the parameters by the gradient they hold and advance the schedule."""
        raise NotImplementedError
