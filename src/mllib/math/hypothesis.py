import numpy as np

from mllib.math.hypothesis_expander import HypothesisExpander


# numpy documentation ref for linear algebra functions https://numpy.org/devdocs/reference/routines.linalg.html
class HypothesisFunction:
    def __init__(self, initial_weights, initial_bias, degree, hypothesis_expander=None):
        # the X in the normal hypothesis function will be passed into the compute prediction function instead of a part of instantiation
        self.initial_hypothesis = initial_weights
        self.hypothesis = initial_weights
        self.bias = initial_bias
        self.degree = degree
        self.hypothesis_expander = hypothesis_expander or HypothesisExpander(self.degree)
        self.hypothesis_expander.degree = self.degree
        self.hypothesis = self.hypothesis_expander.expand_hypothesis(self.hypothesis)
        self.metadata = {
            "name": "hypothesis function parent class",
            "description": "A library class serving as a template for hypothesis function classes used to compute a prediction, hypothesis in this context is a nparray containing the weight and degree of the hypothesis space",
        }

    def set_hypothesis(self, hypothesis):
        self.hypothesis = hypothesis

    def get_hypothesis(self):
        return self.hypothesis

    def update_bias(self, bias):
        self.bias = bias

    def get_bias(self):
        return self.bias

    def print_hypothesis(self):
        print(self.hypothesis)

    def compute_prediction(self, data: np.ndarray):
        # multiplying the weights by the data and adding the bias
        data = self.hypothesis_expander.fit_data_to_hypothesis(data)
        return self.hypothesis @ data + self.bias

    def compute_classification(self, data: np.ndarray):
        # multiplying the weights by the data and adding the bias
        # TODO(BL-12): expander reshapes data in the shared descent base
        data = self.hypothesis_expander.fit_data_to_hypothesis(data, True)
        if self.hypothesis.shape[0] != data.shape[0]:
            ahh = 1  # common issue when building so leaving this to break point on

        return np.sign(self.hypothesis @ data + self.bias)

    def expand_hypothesis(self):
        # if the hypothesis is [[x1], [x2]] and degree=3 then we will return [[1, x1, x1^2, x1^3], [1, x2, x2^2, x2^3]]
        # in this application the data's features are a basis vector of the dimensional space
        self.hypothesis = self.hypothesis_expander.expand(self.hypothesis, self.degree)

    def get_weights(self):
        return self.hypothesis

    def update_weights(self, new_weights):
        self.hypothesis = new_weights
