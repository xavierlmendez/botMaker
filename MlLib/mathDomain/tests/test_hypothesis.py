import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from MlLib.mathDomain.hypothesis import HypothesisFunction
from MlLib.mathDomain.hypothesisExpander import PolynomialRegressionExpander


def test_init_expands_weights_with_polynomial_expander():
    initial_weights = np.array([2.0, 3.0])
    expander = PolynomialRegressionExpander(degree=2)

    hypothesis = HypothesisFunction(
        initial_weights,
        initialBias=1.5,
        degree=2,
        hypothesisExpander=expander,
    )

    assert hypothesis.hypothesisExpander.degree == 2
    assert np.allclose(hypothesis.getHypothesis(), np.array([2.0, 4.0, 3.0, 9.0]))


def test_compute_prediction_degree_one_linear():
    weights = np.array([2.0, -1.0])
    bias = 0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesisExpander=expander)

    data = np.array([3.0, 4.0])
    predicted = hypothesis.computePrediction(data)

    assert predicted == weights[0] * data[0] + weights[1] * data[1] + bias


def test_compute_classification_degree_one_sign():
    weights = np.array([1.0, -2.0])
    bias = -0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesisExpander=expander)

    assert hypothesis.computeClassification(np.array([1.0, 1.0])) == -1.0
    assert hypothesis.computeClassification(np.array([3.0, 0.0])) == 1.0


def test_update_and_getters():
    weights = np.array([0.5, 1.5])
    bias = -2.0
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesisExpander=expander)

    new_weights = np.array([2.0, 4.0])
    hypothesis.updateWeights(new_weights)
    hypothesis.updateBias(3.0)

    assert np.allclose(hypothesis.getWeights(), new_weights)
    assert hypothesis.getBias() == 3.0
