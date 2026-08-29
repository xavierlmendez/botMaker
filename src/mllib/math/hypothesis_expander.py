import numpy as np


class HypothesisExpander:
    """A library class serving as a template for hypothesis expander classes used to
    expand/contract the hypothesis space
    """

    def __init__(self, degree: int = 0):
        self.degree = degree
        self.expander_type = "Generic Parent Class"

    def expand_hypothesis(self, initial_hypothesis: np.ndarray):
        pass

    def fit_data_to_hypothesis(self, data: np.ndarray):
        pass


class PolynomialRegressionExpander(HypothesisExpander):
    """Extends hypothesis according to polynomial regression. Main characteristic is viewing
    feature independently in higher dim spaces than linear regression alone
    """

    def __init__(self, degree: int = 1):
        self.degree = degree
        self.expander_type = "Polynomial Regression"

    def expand(self, value_array):
        expanded_weights_array = np.asarray(value_array, dtype=float)
        exponents = np.arange(1, self.degree + 1, dtype=int)

        if expanded_weights_array.ndim == 1:
            expanded_weights_array = expanded_weights_array.reshape(1, -1)

        feature_space_expanded_to_degree = (
            expanded_weights_array[:, :, None] ** exponents[None, None, :]
        )  # AKA Phi to represent feature brought to a higher dimensionality
        # the [:, :, None] function adds a new axis to the array with the : retaining the previous two axis
        # this results in a new axis mapped that holds the degrees that would need to be applied to the second axis
        feature_space_expanded_to_degree = feature_space_expanded_to_degree.reshape(
            feature_space_expanded_to_degree.shape[0],
            feature_space_expanded_to_degree.shape[1] * self.degree,
        )
        # above were getting rid of the third axis used to map the degrees and reshaping the ndarray based on the number of features shape[1] * number of degrees the array holds
        return feature_space_expanded_to_degree

    def expand_hypothesis(
        self, initial_array: np.ndarray
    ):  # can be used to shape both hypothesis function and features
        return self.expand(initial_array).reshape(-1)

    def fit_data_to_hypothesis(
        self, data: np.ndarray, needs_reshape=False
    ):  # This expander implementation uses the expandHypothesis function for both
        if self.degree == 1:
            return data
        else:
            return self.expand(data).reshape(-1) if needs_reshape else self.expand(data)
