"""Dataset-level cost built on a per-sample loss."""

from __future__ import annotations

from mllib.math.loss_function import LossFunction
from mllib.math.task_kind import TaskKind


class CostFunction:
    """Dataset-level cost: aggregates per-sample loss over one pass through the data. Template for
    cost function classes. Its task kind is the injected loss's.
    """

    def __init__(self, loss_function: LossFunction):
        self.loss_function = loss_function

    @property
    def task_kind(self) -> TaskKind | None:
        return self.loss_function.task_kind

    def compute_cost(self, hypothesis_function, data_values, data_targets):
        raise NotImplementedError
