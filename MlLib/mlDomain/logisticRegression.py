import os
import sys

import numpy as np
import pandas as pd
from numpy import array, mean, sum

# Linear and logistic are very similar however have a major difference in that logistic is for classification
# Todo look into seeing if I can reuse the linear class without so much code duplication 
#  as the main difference is the compute prediction function

# add parent folder (/mlLib) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from the sibling package mathDomain
from mathDomain.hypothesis import HypothesisFunction
from mathDomain.lossFunction import LossFunction, MSE
from mlDomain.modelEvaluators.genericEvaluator import LogisticRegressionModelEvaluator
class MyLogisticRegression: # prefixing with my for the comparison script, rename later when cleaning up files
    # choosing 0.001 for default learning rate bc thats what adam uses
    def __init__(self, learningRate = 0.001, epochs = 10):
        seededRand = np.random.default_rng(10) # seeting a seed for random initial weights,
        initialWeights = seededRand.random(19) # hardcoded for now but this should be influenced by one hot encoding results and hypothesis space later
        initialBias = 0
        self.learningModel = HypothesisFunction(initialWeights, initialBias)
        self.lossFunction = MSE()
        self.learningRate = learningRate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()

    def fit(self, dataValues, dataTargets):
        # for n epochs
        # calculate the cost function - in the gradientDescent function
        # compute the gradient
        # update the weights
        # repeat
        for epoch in range(self.epochs):
            newWeights, newBias = self.calculateGradientDescent(dataValues, dataTargets)
            self.updateWeights(newWeights, newBias)
            cost = self.calculateCostFunction(dataValues, dataTargets)
            # use evaluator class here to aggregate data on performance during training
        return self

    def predict(self, data):
        return self.learningModel.computeClassification(data)

    def predictValues(self, dataValues, isDataframe=False):
        predictedValues = []
        for data in dataValues:
            predictedValues.append(self.predict(data))
        return array(predictedValues)

    def evaluate(self, dataValues, dataTargets):
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        dataValues, dataTargets = self.dataFrameCrossCapatibility(dataValues, dataTargets)
        predictedValues = self.predictValues(dataValues, dataTargets)
        self.evaluator.updateTestingPredictionData(dataValues, dataTargets, predictedValues)

    def calculateGradientDescent(self, dataValues, dataTargets):
        # standardize Inputs for compatibility with pandas dataframes as parameters
        dataValues, dataTargets = self.dataFrameCrossCapatibility(dataValues, dataTargets)

        # calculate the gradient
        predicted = self.predictValues(dataValues)
        gradientDescentAdjustedDataTargets = self.lossFunction.computeGradient(dataTargets, predicted)

        gradientDescentAdjustedWeights = dataValues.T @ gradientDescentAdjustedDataTargets
        adjustedBias = sum(gradientDescentAdjustedDataTargets)
        return gradientDescentAdjustedWeights, adjustedBias

    def updateWeights(self, gradientDescentAdjustedWeights, gradientDescentAdjustedBias):
        # update the weights and bias
        newWeights = self.learningModel.getWeights() - self.learningRate * gradientDescentAdjustedWeights
        newBias = self.learningModel.getBias() - self.learningRate * gradientDescentAdjustedBias
        self.learningModel.updateWeights(newWeights)
        self.learningModel.updateBias(newBias)

    def calculateCostFunction(self, dataValues, dataTargets):
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        dataValues, dataTargets = self.dataFrameCrossCapatibility(dataValues, dataTargets)
        
        # here were putting together the cost function as a set of linear equations
        # doing it this way to leverage linear algebra packages
        predicted = self.predictValues(dataValues)
        lossAcrossData = self.lossFunction.computeLoss(dataTargets, predicted)
        return mean(lossAcrossData)

    def dataFrameCrossCapatibility(self, dataValues, dataTargets):
        if isinstance(dataValues, pd.DataFrame):
            dataValues = dataValues.to_numpy()
            dataTargets = np.asarray(dataTargets).ravel()
        return dataValues, dataTargets