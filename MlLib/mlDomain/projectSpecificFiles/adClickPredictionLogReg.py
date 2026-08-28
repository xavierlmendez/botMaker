import numpy as np

from MlLib.mathDomain.hypothesisExpander import PolynomialRegressionExpander
from MlLib.mathDomain.lossFunction import MAE, MSE
from MlLib.mlDomain.logisticRegression import MyLogisticRegression
from MlLib.mlDomain.modelEvaluators.genericEvaluator import LogisticRegressionModelEvaluator


class LogisticRegression(MyLogisticRegression):
    def __init__(self):
        super().__init__(numWeights=19)  # F4: base sets learningModel/lossFunction/epochs
        self.metadata = {
            "name": "Ad Click Logistic Regression",
            "description": "Project-specific logistic regression configuration for ad click prediction.",
        }
        # TODO: review metadata (auto-generated)
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = np.array(
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
        self.exporter = None  # TODO implement exporter


class LogisticRegressionWithAgeBinning(MyLogisticRegression):
    def __init__(self):
        super().__init__(numWeights=26)  # F4: base sets learningModel/lossFunction/epochs
        self.metadata = {
            "name": "Ad Click Logistic Regression With Age Binning",
            "description": "Project-specific logistic regression with age binning for ad click prediction.",
        }
        # TODO: review metadata (auto-generated)
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = np.array(
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
        self.exporter = None  # TODO implement exporter
