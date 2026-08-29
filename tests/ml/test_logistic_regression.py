import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MSE
from mllib.ml.evaluators.generic_evaluator import LogisticRegressionModelEvaluator
from mllib.ml.logistic_regression import MyLogisticRegression


def _build_model() -> MyLogisticRegression:
    model = MyLogisticRegression.__new__(MyLogisticRegression)
    model.num_weights = 2
    model.learning_rate = 0.1
    model.epochs = 1
    model.loss_function = MSE()
    model.evaluator = LogisticRegressionModelEvaluator()

    expander = PolynomialRegressionExpander(degree=1)
    model.learning_model = HypothesisFunction(
        np.array([1.0, -1.0]),
        0.0,
        degree=1,
        hypothesis_expander=expander,
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

    predictions = model.predict_values(data_values)

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

    new_weights, new_bias = model.calculate_gradient_descent(data_values, data_targets)
    model.update_weights(new_weights, new_bias)

    assert np.allclose(model.learning_model.get_weights(), np.array([1.0, -0.8]))
    assert np.isclose(model.learning_model.get_bias(), 0.2)


def test_learns_a_linearly_separable_problem_from_zero_one_labels():
    # BL-23: before encode_targets, {0, 1} labels against sign outputs stalled near 0.4 accuracy.
    rng = np.random.default_rng(0)
    data_values = rng.normal(size=(400, 2))
    labels = (data_values[:, 0] + 0.5 * data_values[:, 1] > 0.2).astype(int)

    model = MyLogisticRegression(learning_rate=0.01, epochs=500, num_weights=2)
    model.fit(data_values, labels)
    predictions = np.where(model.predict_values(data_values) == -1, 0, 1)

    assert (predictions == labels).mean() >= 0.98
