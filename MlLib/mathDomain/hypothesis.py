import numpy as np


class HypothesisFunction:
    def __init__(self, initialWeights, initialBias):
        # the X in the normal hypothesis function will be passed into the compute prediction function instead of a part of instantiation
        self.hypothesis = initialWeights
        self.bias = initialBias
        self.metadata = {
            "name": "hypothesis function parent class",
            "description": "A library class serving as a template for hypothesis function classes used to compute a prediction and expand/contract the hypothesis space the hypothesis in this context is a nparray containing the weight and degree of the hypothesis space"
        }

    def setHypothesis(self, hypothesis):
        self.hypothesis = hypothesis

    def getHypothesis(self):
        return self.hypothesis

    def updateBias(self, bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def printHypothesis(self):
        print(self.hypothesis)

    def computePrediction(self, data):
        # multiplying the weights by the data and adding the bias
        data = data.reshape(-1)
        return self.hypothesis @ data + self.bias

    def computeClassification(self, data):
        # for hw one keeping it simple and non polynomial
        # multiplying the weights by the data and adding the bias
        data = data.reshape(-1)
        if np.sign(self.hypothesis @ data + self.bias) == 'nan':
            ahh = 1
        return np.sign(self.hypothesis @ data + self.bias)

    def expandHypothesis(self, degree):
        # if the hypothesis is [[x1], [x2]] and degree=3 then we will return [[1, x1, x1^2, x1^3], [1, x2, x2^2, x2^3]]
        # not using for HW1
        pass

    def getWeights(self):
        return self.hypothesis

    def updateWeights(self, newWeights):
        self.hypothesis = newWeights