import numpy as np

from mllib.math.hypothesis_expander import HypothesisExpander


# numpy documentation ref for linear algebra functions https://numpy.org/devdocs/reference/routines.linalg.html
class HypothesisFunction:
    """A library class serving as a template for hypothesis function classes used to compute a
    prediction, hypothesis in this context is a nparray containing the weight and degree of the
    hypothesis space
    """

    def __init__(self, initial_weights, initial_bias, degree=1, hypothesis_expander=None):
        # the X in the normal hypothesis function will be passed into the compute prediction function instead of a part of instantiation
        self.initial_hypothesis = initial_weights
        self.hypothesis = initial_weights
        self.bias = initial_bias
        self.degree = degree
        self.hypothesis_expander = hypothesis_expander or HypothesisExpander(self.degree)
        self.hypothesis_expander.degree = self.degree
        self.hypothesis = self.hypothesis_expander.expand_hypothesis(self.hypothesis)

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
        # sign of the affine score; the expander reshapes a single row into the hypothesis space
        data = self.hypothesis_expander.fit_data_to_hypothesis(data, True)
        return np.sign(self.hypothesis @ data + self.bias)

    def expand_hypothesis(self):
        """Recompute the expanded hypothesis from the initial weights through the expander.

        The same call the constructor makes, so the result is the constructor's: for the polynomial
        expander at degree d, each initial weight becomes its powers 1..d. It expands the initial
        weights, never the already-expanded hypothesis, or a second call would expand twice. Fixed
        in BL-48 slice 5: it used to call an ``expand(hypothesis, degree)`` no expander defines.
        """
        self.hypothesis = self.hypothesis_expander.expand_hypothesis(self.initial_hypothesis)

    def get_weights(self):
        return self.hypothesis

    def update_weights(self, new_weights):
        self.hypothesis = new_weights
