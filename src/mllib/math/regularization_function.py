"""Regularization penalties added to a cost."""

from __future__ import annotations

from mllib.math.task_kind import TaskKind


class RegularizationFunction:
    """Template for regularization penalties added to the cost. ``task_kind`` is ``None`` because a
    penalty on the weights applies to regression and classification alike; a subclass may narrow it.
    """

    task_kind: TaskKind | None = None

    def compute_penalty(self, weights):
        raise NotImplementedError
