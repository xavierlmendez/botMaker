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
        initialWeights = seededRand.random(19) # hardcoded for now but this should be influenced by one hot encoding results and hypothesis space later
        initialBias = 0
        self.learningModel = HypothesisFunction(initialWeights, initialBias)
        self.lossFunction = MSE()
        self.learningRate = learningRate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()

class LogisticRegressionWithAgeBinning(MyLogisticRegression):
    def __init__(self, learningRate = 0.001, epochs = 10):
        seededRand = np.random.default_rng(10) # seeting a seed for random initial weights,
        initialWeights = seededRand.random(26) # hardcoded for now but this should be influenced by one hot encoding results and hypothesis space later
        initialBias = 0
        self.learningModel = HypothesisFunction(initialWeights, initialBias)
        self.lossFunction = MSE()
        self.learningRate = learningRate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()