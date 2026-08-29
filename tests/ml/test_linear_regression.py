import numpy as np

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MSE
from mllib.ml.linear_regression import MyLinearRegression


def test_calculate_gradient_descent_and_update_weights():
    initial_weights = np.array([1.0, 1.0])
    initial_bias = 0.0
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(
        initial_weights, initial_bias, degree=1, hypothesis_expander=expander
    )

    model = MyLinearRegression(hypothesis, MSE(), learning_rate=0.1, epochs=1)

    data_values = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    data_targets = np.array([1.0, 2.0])

    new_weights, new_bias = model.calculate_gradient_descent(data_values, data_targets)
    model.update_weights(new_weights, new_bias)

    assert np.allclose(model.learning_model.get_weights(), np.array([1.0, 1.1]))
    assert np.isclose(model.learning_model.get_bias(), 0.1)


def test_predict_values_returns_expected_shape():
    initial_weights = np.array([2.0, -1.0])
    initial_bias = 0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(
        initial_weights, initial_bias, degree=1, hypothesis_expander=expander
    )

    model = MyLinearRegression(hypothesis, MSE(), learning_rate=0.01, epochs=1)

    data_values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    predictions = model.predict_values(data_values)

    assert predictions.shape == (2,)
    assert np.allclose(
        predictions, np.array([2.0 * 1.0 + -1.0 * 2.0 + 0.5, 2.0 * 3.0 + -1.0 * 4.0 + 0.5])
    )
