import numpy as np


class HypothesisExpander:
    def __init__(self, degree: int = 0):
        self.degree = degree
        self.expanderType = 'Generic Parent Class'
        self.metadata = {
            "name": "hypothesis expander parent class",
            "description": "A library class serving as a template for hypothesis expander classes used to expand/contract the hypothesis space"
        }

    def expandHypothesis(self, initialHypothesis: np.ndarray):
        pass

    def fitDataToHypothesis(self, data: np.ndarray):
        pass


class PolynomialRegressionExpander(HypothesisExpander):
    def __init__(self, degree: int = 1):
        self.degree = degree
        self.expanderType = 'Polynomial Regression'
        self.metadata = {
            "name": "Polynomial Regression hypothesis expander",
            "description": "Extends hypothesis according to polynomial regression. Main characteristic is viewing feature independently in higher dim spaces than linear regression alone"
        }

    def expand(self, valueArray):
        expandedWeightsArray = np.asarray(valueArray, dtype=float)
        exponents = np.arange(1, self.degree + 1, dtype=int)

        if expandedWeightsArray.ndim == 1:
            expandedWeightsArray = expandedWeightsArray.reshape(1, -1)

        featureSpaceExpandedToDegree = expandedWeightsArray[:, :, None] ** exponents[None, None, :] # AKA Phi to represent feature brought to a higher dimensionality
        # the [:, :, None] function adds a new axis to the array with the : retaining the previous two axis 
        # this results in a new axis mapped that holds the degrees that would need to be applied to the second axis
        featureSpaceExpandedToDegree = featureSpaceExpandedToDegree.reshape(featureSpaceExpandedToDegree.shape[0], featureSpaceExpandedToDegree.shape[1] * self.degree)
        # above were getting rid of the thrid axis used to map the degrees and reshaping the ndarray based on the number of features shape[1] * number of degrees the array holds 
        return featureSpaceExpandedToDegree

    def expandHypothesis(self, initialArray: np.ndarray):  # can be used to shape both hypothesis function and features 
        return self.expand(initialArray).reshape(-1)

    def fitDataToHypothesis(self, data: np.ndarray, needsReshape = False):  # This expander implementation uses the expandHypothesis function for both
        if self.degree == 1:
            return data
        else:
            return self.expand(data).reshape(-1) if needsReshape else self.expand(data)

    