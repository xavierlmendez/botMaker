import numpy as np

from mllib.mathDomain.hypothesis import HypothesisFunction
from mllib.mathDomain.hypothesisExpander import PolynomialRegressionExpander
from mllib.mathDomain.lossFunction import HingeLoss, PerceptronLoss
from mllib.mlDomain.perceptron import MyPerceptron
from mllib.mlDomain.probabilisticKNN import ProbabilisticKNN
from mllib.mlDomain.SVM import MySVM


def build_hypothesis():
    weights = np.array([1.0, -1.0])
    bias = 0.0
    expander = PolynomialRegressionExpander(degree=1)
    return HypothesisFunction(weights, bias, degree=1, hypothesisExpander=expander)


def test_perceptron_predict_values():
    model = MyPerceptron(build_hypothesis(), PerceptronLoss(), learningRate=0.1, epochs=1)

    data_values = np.array(
        [
            [2.0, 1.0],  # dot = 1
            [1.0, 2.0],  # dot = -1
        ]
    )

    predictions = model.predictValues(data_values)

    assert np.allclose(predictions, np.array([1.0, -1.0]))


def test_svm_predict_values():
    model = MySVM(build_hypothesis(), HingeLoss(), learningRate=0.1, epochs=1)

    data_values = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    predictions = model.predictValues(data_values)

    assert np.allclose(predictions, np.array([1.0, -1.0]))


def test_probabilistic_knn_stores_prior():
    prior_obj = object()
    model = ProbabilisticKNN(prior=prior_obj)

    assert model.prior is prior_obj
