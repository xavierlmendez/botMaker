import numpy as np

from MlLib.mathDomain.hypothesis import HypothesisFunction
from MlLib.mathDomain.lossFunction import LossFunction


class MySVM: # prefixing with my for the comparison script, rename later when cleaning up files
    def __init__(self, hypothesisFunction : HypothesisFunction, lossFunction : LossFunction, learningRate = 0.001, epochs = 10):
        self.metadata = {
            "name": "SVM Base Class",
            "description": "Support Vector Machine implementation using hinge-style sub-gradient updates."
        }
        # TODO: review metadata (auto-generated)
        self.learningModel = hypothesisFunction
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.epochs = epochs

    def fit(self, dataValues, dataTargets):
        for epoch in range(self.epochs):
            subGradientDirectionMatrix, subGradientDiasDirectionMatrix = self.calculateSubGradientDescent(dataValues, dataTargets)
            self.updateWeights(subGradientDirectionMatrix, subGradientDiasDirectionMatrix)
            error = self.calculateError(dataValues, dataTargets)
            print(f"Epoch: {epoch}, Error: {error}")
        return self

    def predict(self, data):
        return np.sign(self.learningModel.computeClassification(data))

    def predictValues(self, dataValues):
        predictedValues = []
        for data in dataValues:
            predictedValues.append(self.predict(data))
        return np.array(predictedValues)

    def evaluate(self, dataValues, dataTargets):
        pass

    def calculateSubGradientDescent(self, dataValues, dataTargets):
        # calculate the gradient
        predicted = self.predictValues(dataValues)
        subGradientDirection = self.lossFunction.computeGradient(dataTargets, predicted, dataValues)
        subGradientBiasDirection = self.lossFunction.computeBias(dataTargets, predicted)
        return subGradientDirection, subGradientBiasDirection

    def updateWeights(self, subGradientDirection, subGradientBiasDirection):
        # update the weights and bias
        print('sub gradient:', subGradientDirection)
        newWeights = self.learningModel.getWeights() - self.learningRate * subGradientDirection
        print('updated weights:', newWeights)
        newBias = self.learningModel.getBias() - self.learningRate * subGradientBiasDirection
        self.learningModel.updateWeights(newWeights)
        self.learningModel.updateBias(newBias)

    def calculateError(self, dataValues, dataTargets):
        predicted = self.predictValues(dataValues)
        misclassified = predicted != dataTargets
        return np.mean(misclassified)
