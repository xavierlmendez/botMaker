import numpy as np
from MlLib.mathDomain.hypothesisExpander import HypothesisExpander


# numpy documentation ref for linear algebra functions https://numpy.org/devdocs/reference/routines.linalg.html
class HypothesisFunction:
    def __init__(self, initialWeights, initialBias, degree, hypothesisExpander = None):
        # the X in the normal hypothesis function will be passed into the compute prediction function instead of a part of instantiation
        self.initialHypothesis = initialWeights
        self.hypothesis = initialWeights
        self.bias = initialBias
        self.degree = degree
        self.hypothesisExpander = hypothesisExpander or HypothesisExpander(self.degree)
        self.hypothesisExpander.degree = self.degree
        self.hypothesis = self.hypothesisExpander.expandHypothesis(self.hypothesis)
        self.metadata = {
            "name": "hypothesis function parent class",
            "description": "A library class serving as a template for hypothesis function classes used to compute a prediction, hypothesis in this context is a nparray containing the weight and degree of the hypothesis space"
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

    def computePrediction(self, data: np.ndarray):
        # multiplying the weights by the data and adding the bias
        data = self.hypothesisExpander.fitDataToHypothesis(data)
        return self.hypothesis @ data + self.bias

    def computeClassification(self, data: np.ndarray):
        # multiplying the weights by the data and adding the bias
        #todo call hypothesisExpander to shape data if needed
        data = self.hypothesisExpander.fitDataToHypothesis(data, True)
        if self.hypothesis.shape[0] != data.shape[0]:
            ahh = 1  # common issue when building so leaving this to break point on

        return np.sign(self.hypothesis @ data + self.bias)

    def expandHypothesis(self):
        # if the hypothesis is [[x1], [x2]] and degree=3 then we will return [[1, x1, x1^2, x1^3], [1, x2, x2^2, x2^3]]
        # in this application the data's features are a basis vector of the dimensional space 
        self.hypothesis = self.hypothesisExpander.expand(self.hypothesis, self.degree)

    def getWeights(self):
        return self.hypothesis

    def updateWeights(self, newWeights):
        self.hypothesis = newWeights
