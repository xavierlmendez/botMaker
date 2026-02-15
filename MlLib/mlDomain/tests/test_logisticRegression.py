import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from MlLib.mathDomain.hypothesis import HypothesisFunction
from MlLib.mathDomain.hypothesisExpander import PolynomialRegressionExpander
from MlLib.mathDomain.lossFunction import MSE
from MlLib.mlDomain.logisticRegression import MyLogisticRegression
from MlLib.mlDomain.modelEvaluators.genericEvaluator import LogisticRegressionModelEvaluator


def _build_model() -> MyLogisticRegression:
    model = MyLogisticRegression.__new__(MyLogisticRegression)
    model.numWeights = 2
    model.learningRate = 0.1
    model.epochs = 1
    model.lossFunction = MSE()
    model.evaluator = LogisticRegressionModelEvaluator()

    expander = PolynomialRegressionExpander(degree=1)
    model.learningModel = HypothesisFunction(
        np.array([1.0, -1.0]),
        0.0,
        degree=1,
        hypothesisExpander=expander,
    )
    return model


def test_predict_values_classification_signs():
    model = _build_model()

    data_values = np.array(
        [
            [2.0, 1.0],  # dot = 1
            [1.0, 2.0],  # dot = -1
        ]
    )

    predictions = model.predictValues(data_values)

    assert np.allclose(predictions, np.array([1.0, -1.0]))


def test_calculate_gradient_descent_updates_weights():
    model = _build_model()

    data_values = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    data_targets = np.array([1.0, 1.0])

    new_weights, new_bias = model.calculateGradientDescent(data_values, data_targets)
    model.updateWeights(new_weights, new_bias)

    assert np.allclose(model.learningModel.getWeights(), np.array([1.0, -0.8]))
    assert np.isclose(model.learningModel.getBias(), 0.2)
