"""Loss/cost/regularization components declare the task they are meant for (BL-10)."""

import numpy as np

from mllib.describe import describe
from mllib.math.cost_function import AbstractCostFunction
from mllib.math.graph.two_hot_span_problem import SpanCost
from mllib.math.loss_function import MAE, MSE, HingeLoss, LossFunction, PerceptronLoss
from mllib.math.projector import ExactProjector
from mllib.math.regularization_function import AbstractRegularizationFunction
from mllib.math.task_kind import TaskKind


def test_losses_declare_their_task_kind():
    assert MSE.task_kind is TaskKind.REGRESSION
    assert MAE.task_kind is TaskKind.REGRESSION
    assert PerceptronLoss.task_kind is TaskKind.CLASSIFICATION
    assert HingeLoss.task_kind is TaskKind.CLASSIFICATION


def test_supports_is_permissive_only_when_undeclared():
    assert MSE().supports(TaskKind.REGRESSION)
    assert not MSE().supports(TaskKind.CLASSIFICATION)
    assert LossFunction().supports(TaskKind.CLASSIFICATION)


def test_a_cost_applies_to_both_kinds_unless_narrowed():
    assert AbstractCostFunction.task_kind is None
    assert SpanCost(ExactProjector(), np.eye(3)).task_kind is None


def test_regularization_applies_to_both_by_default():
    assert AbstractRegularizationFunction.task_kind is None


def test_describe_reports_task_kind():
    assert describe(MSE)["task_kind"] == "regression"
    assert describe(AbstractRegularizationFunction)["task_kind"] is None
    assert describe(AbstractCostFunction)["task_kind"] is None
    assert describe(dict)["task_kind"] is None
