from numpy import array, mean, sum

from MlLib.mathDomain.hypothesis import HypothesisFunction
from MlLib.mathDomain.lossFunction import LossFunction
class MyLinearRegression: # prefixing with my for the comparison script, rename later when cleaning up files
    # choosing 0.001 for default learning rate bc thats what adam uses
    def __init__(self, hypothesisFunction : HypothesisFunction, lossFunction : LossFunction, learningRate = 0.001, epochs = 10):
        self.metadata = {
            "name": "Linear Regression Base Class",
            "description": "Core linear regression implementation with gradient descent and prediction helpers."
        }
        # TODO: review metadata (auto-generated)
        self.learningModel = hypothesisFunction
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.epochs = epochs

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
        return self

    def predict(self, data):
        return self.learningModel.computePrediction(data)

    def predictValues(self, dataValues):
        predictedValues = []
        for data in dataValues:
            predictedValues.append(self.predict(data))
        return array(predictedValues)

    def evaluate(self, dataValues, dataTargets):
        # used same evaluate function as boston data set demo
        pass

    def calculateGradientDescent(self, dataValues, dataTargets):
        # calculate the gradient
        predicted = self.predictValues(dataValues)
        gradientDescentAdjusteddataTargets = self.lossFunction.computeGradient(dataTargets, predicted)

        gradientDescentAdjustedWeights = dataValues.T @ gradientDescentAdjusteddataTargets
        adjustedBias = sum(gradientDescentAdjusteddataTargets)
        return gradientDescentAdjustedWeights, adjustedBias

    def updateWeights(self, gradientDescentAdjustedWeights, gradientDescentAdjustedBias):
        # update the weights and bias
        newWeights = self.learningModel.getWeights() - self.learningRate * gradientDescentAdjustedWeights
        newBias = self.learningModel.getBias() - self.learningRate * gradientDescentAdjustedBias
        self.learningModel.updateWeights(newWeights)
        self.learningModel.updateBias(newBias)

    def calculateCostFunction(self, dataValues, dataTargets):
        # here were putting together the cost function as a set of linear equations
        # doing it this way to leverage linear algebra packages
        predicted = self.predictValues(dataValues)
        lossAcrossData = self.lossFunction.computeLoss(dataTargets, predicted)
        return mean(lossAcrossData)
