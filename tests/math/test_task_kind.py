"""Loss/cost/regularization components declare the task they are meant for (BL-10)."""

from mllib.describe import describe
from mllib.math.cost_function import CostFunction
from mllib.math.loss_function import MAE, MSE, HingeLoss, LossFunction, PerceptronLoss
from mllib.math.regularization_function import RegularizationFunction
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


def test_cost_function_takes_its_kind_from_the_loss():
    assert CostFunction(HingeLoss()).task_kind is TaskKind.CLASSIFICATION


def test_regularization_applies_to_both_by_default():
    assert RegularizationFunction.task_kind is None


def test_describe_reports_task_kind():
    assert describe(MSE)["task_kind"] == "regression"
    assert describe(RegularizationFunction)["task_kind"] is None
    assert describe(dict)["task_kind"] is None
