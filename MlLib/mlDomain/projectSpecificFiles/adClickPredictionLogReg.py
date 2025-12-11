import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mlDomain.logisticRegression import MyLogisticRegression
from mathDomain.hypothesis import HypothesisFunction
from mathDomain.lossFunction import LossFunction, MSE, MAE
from mlDomain.modelEvaluators.genericEvaluator import LogisticRegressionModelEvaluator

class LogisticRegression(MyLogisticRegression):
    def __init__(self, learningRate = 0.001, epochs = 10):
        seededRand = np.random.default_rng(10) # seeting a seed for random initial weights,
        self.numWeights = 19
        initialWeights = seededRand.random(self.numWeights) # hardcoded for now but this should be influenced by one hot encoding results and hypothesis space later
        initialBias = 0
        self.learningModel = HypothesisFunction(initialWeights, initialBias)
        self.lossFunction = MSE()
        self.learningRate = learningRate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = np.array([
            {
                'modelName': ['LogisticRegressionModel'],
                'learningRate': [0.0005, 0.001, 0.004, 0.008, 0.01, 0.015, 0.1],
                'epoch': [4, 5, 6, 7, 8, 10, 15],
                'lossFunction': [MSE(), MAE()],
                'weightRandSeed': [1,2,27],
                'initialBias': [-100, 1, 100]
            }
        ])

class LogisticRegressionWithAgeBinning(MyLogisticRegression):
    def __init__(self, learningRate = 0.001, epochs = 10):
        seededRand = np.random.default_rng(10) # seeting a seed for random initial weights,
        self.numWeights = 26
        initialWeights = seededRand.random(self.numWeights) # hardcoded for now but this should be influenced by one hot encoding results and hypothesis space later
        initialBias = 0
        self.learningModel = HypothesisFunction(initialWeights, initialBias)
        self.lossFunction = MSE()
        self.learningRate = learningRate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = np.array([
            {
                'modelName': ['LogisticRegressionModelWithAgeBinning'],
                'learningRate': [0.0005, 0.001, 0.004, 0.008, 0.01, 0.015, 0.1],
                'epoch': [4, 5, 6, 7, 8, 10, 15],
                'lossFunction': [MSE(), MAE()],
                'weightRandSeed': [1,2,27],
                'initialBias': [-100, 1, 100]
            }
        ])