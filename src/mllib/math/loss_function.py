"""Per-sample loss functions, each carrying its own gradient, injected into models."""

from __future__ import annotations

import numpy as np

from mllib.math.task_kind import TaskKind


class LossFunction:
    """Template for loss functions: loss between predicted and actual values, plus its gradient.

    Subclasses declare ``task_kind`` so a model (or a reader) can tell what the loss is meant for.
    """

    task_kind: TaskKind | None = None

    def compute_loss(self, actual, predicted):
        raise NotImplementedError

    def compute_gradient(self, actual, predicted):
        raise NotImplementedError

    def supports(self, kind: TaskKind) -> bool:
        """True when this loss is meant for ``kind`` (a loss with no declared kind supports both)."""
        return self.task_kind is None or self.task_kind is kind


class MSE(LossFunction):
    """Loss function computing mean squared error and its gradient."""

    task_kind = TaskKind.REGRESSION

    def compute_loss(self, actual, predicted):
        return np.mean((actual - predicted) ** 2)

    def compute_gradient(self, actual, predicted):
        n = actual.shape[0]
        return (2.0 / n) * (predicted - actual)


class MAE(LossFunction):
    """Loss function computing mean absolute error and its gradient."""

    task_kind = TaskKind.REGRESSION

    def compute_loss(self, actual, predicted):
        return np.mean(abs(actual - predicted))

    def compute_gradient(self, actual, predicted):
        return np.sign(predicted - actual)


class PerceptronLoss(LossFunction):
    """Perceptron loss with sub-gradient and bias updates."""

    task_kind = TaskKind.CLASSIFICATION

    def compute_loss(self, actual: np.ndarray, predicted: np.ndarray):
        return np.maximum(0.0, -actual * predicted)

    def compute_gradient(self, actual, predicted, data_values):
        # this is actually a sub gradient but reusing the function name for consistency
        zero_if_classified_correctly = actual * predicted <= 0
        return data_values.T @ (zero_if_classified_correctly * actual)

    def compute_bias(self, actual, predicted):
        zero_if_classified_correctly = actual * predicted <= 0
        return (zero_if_classified_correctly * actual).sum()


class HingeLoss(LossFunction):
    """Hinge loss for margin-based classifiers with sub-gradient updates."""

    task_kind = TaskKind.CLASSIFICATION

    def compute_loss(self, actual, predicted):
        return np.maximum(0.0, 1.0 - actual * predicted)

    def compute_gradient(self, actual, predicted, data_values):
        # this is actually a sub gradient but reusing the function name for consistency
        zero_if_classified_correctly = actual * predicted <= 0
        return data_values.T @ (zero_if_classified_correctly * -actual)

    def compute_bias(self, actual, predicted):
        zero_if_classified_correctly = actual * predicted <= 0
        return (zero_if_classified_correctly * actual).sum()
