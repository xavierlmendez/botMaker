import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mlDomain.logisticRegression import MyLogisticRegression
from mathDomain.lossFunction import MSE, MAE
from mathDomain.hypothesisExpander import PolynomialRegressionExpander
from mlDomain.modelEvaluators.genericEvaluator import LogisticRegressionModelEvaluator

class LogisticRegression(MyLogisticRegression):
    def __init__(self):
        self.numWeights = 19 # todo refactor to more descriptive name like num features or numfeature weights
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = np.array([
            {
                'modelName': ['LogisticRegression'],
                'learningRate': [0.001, 0.004, 0.008],#[0.001, 0.004, 0.008, 0.01, 0.015, 0.1],
                'epoch':  [100, 200],#[6, 8, 10, 15, 20],
                'lossFunction': [MAE()],# [MSE(), MAE()],
                'HypothesisExpander': [PolynomialRegressionExpander()],
                'polynomialDegree':  [3, 4],#[1,2,3,4,5],
                'weightRandSeed': [2], #[1,2,27],
                'initialBias': [1], #[.5, 1, 2, 5]
            }
        ])
        self.exporter = None # todo implement exporter

class LogisticRegressionWithAgeBinning(MyLogisticRegression):
    def __init__(self):
        self.numWeights = 26 # todo refactor to more descriptive name like num features or numfeature weights
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = np.array([
            {
                'modelName': ['LogisticRegressionModelWithAgeBinning'],
                'learningRate': [0.001, 0.004, 0.008],#[0.001, 0.004, 0.008, 0.01, 0.015, 0.1],
                'epoch':  [100, 200],#[6, 8, 10, 15, 20],
                'lossFunction': [MAE()],# [MSE(), MAE()],
                'HypothesisExpander': [PolynomialRegressionExpander()],
                'polynomialDegree':  [3, 4],#[1,2,3,4,5],
                'weightRandSeed': [2], #[1,2,27],
                'initialBias': [1], #[.5, 1, 2, 5]
            }
        ])
        self.exporter = None # todo implement exporter