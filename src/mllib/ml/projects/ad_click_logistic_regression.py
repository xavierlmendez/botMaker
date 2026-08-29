import numpy as np

from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MAE, MSE
from mllib.ml.evaluators.generic_evaluator import LogisticRegressionModelEvaluator
from mllib.ml.logistic_regression import MyLogisticRegression


class LogisticRegression(MyLogisticRegression):
    """Project-specific logistic regression configuration for ad click prediction."""

    def __init__(self):
        super().__init__(num_weights=19)  # F4: base sets learningModel/lossFunction/epochs
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameter_grid_options = np.array(
            [
                {
                    "modelName": ["LogisticRegression"],
                    "learningRate": [
                        0.008,
                        0.01,
                        0.012,
                        0.015,
                        0.02,
                        0.03,
                        0.05,
                        0.07,
                        0.12,
                        0.15,
                    ],  # [0.001, 0.004, 0.008, 0.01, 0.015, 0.1],
                    "epoch": [10, 25, 50, 200, 300, 500],  # [6, 8, 10, 15, 20],
                    "lossFunction": [MSE(), MAE()],
                    "HypothesisExpander": [PolynomialRegressionExpander()],
                    "polynomialDegree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # [1,2,3,4,5],
                    "weightRandSeed": [1, 2, 27],
                    "initialBias": [0.5, 1, 2, 5],
                }
            ]
        )
        self.exporter = (
            None  # TODO(BL-11): persist fitted weights, hypothesis config and evaluator record
        )


class LogisticRegressionWithAgeBinning(MyLogisticRegression):
    """Project-specific logistic regression with age binning for ad click prediction."""

    def __init__(self):
        super().__init__(num_weights=26)  # F4: base sets learningModel/lossFunction/epochs
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameter_grid_options = np.array(
            [
                {
                    "modelName": ["LogisticRegressionModelWithAgeBinning"],
                    "learningRate": [
                        0.008,
                        0.01,
                        0.012,
                        0.015,
                        0.02,
                        0.03,
                        0.05,
                        0.07,
                        0.12,
                        0.15,
                    ],  # [0.001, 0.004, 0.008, 0.01, 0.015, 0.1],
                    "epoch": [10, 25, 50, 200, 300, 500],  # [6, 8, 10, 15, 20],
                    "lossFunction": [MSE(), MAE()],
                    "HypothesisExpander": [PolynomialRegressionExpander()],
                    "polynomialDegree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # [1,2,3,4,5],
                    "weightRandSeed": [1, 2, 27],
                    "initialBias": [0.5, 1, 2, 5],
                }
            ]
        )
        self.exporter = (
            None  # TODO(BL-11): persist fitted weights, hypothesis config and evaluator record
        )
