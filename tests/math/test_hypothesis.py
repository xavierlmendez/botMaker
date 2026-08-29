import numpy as np

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import PolynomialRegressionExpander


def test_init_expands_weights_with_polynomial_expander():
    initial_weights = np.array([2.0, 3.0])
    expander = PolynomialRegressionExpander(degree=2)

    hypothesis = HypothesisFunction(
        initial_weights,
        initial_bias=1.5,
        degree=2,
        hypothesis_expander=expander,
    )

    assert hypothesis.hypothesis_expander.degree == 2
    assert np.allclose(hypothesis.get_hypothesis(), np.array([2.0, 4.0, 3.0, 9.0]))


def test_compute_prediction_degree_one_linear():
    weights = np.array([2.0, -1.0])
    bias = 0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesis_expander=expander)

    data = np.array([3.0, 4.0])
    predicted = hypothesis.compute_prediction(data)

    assert predicted == weights[0] * data[0] + weights[1] * data[1] + bias


def test_compute_classification_degree_one_sign():
    weights = np.array([1.0, -2.0])
    bias = -0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesis_expander=expander)

    assert hypothesis.compute_classification(np.array([1.0, 1.0])) == -1.0
    assert hypothesis.compute_classification(np.array([3.0, 0.0])) == 1.0


def test_update_and_getters():
    weights = np.array([0.5, 1.5])
    bias = -2.0
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesis_expander=expander)

    new_weights = np.array([2.0, 4.0])
    hypothesis.update_weights(new_weights)
    hypothesis.update_bias(3.0)

    assert np.allclose(hypothesis.get_weights(), new_weights)
    assert hypothesis.get_bias() == 3.0
