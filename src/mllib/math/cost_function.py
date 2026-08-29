from mllib.math.loss_function import LossFunction


class CostFunction:
    """Dataset-level cost: aggregates per-sample loss over one pass through the data. Template for
    cost function classes.
    """

    def __init__(self, loss_function: LossFunction):
        self.loss_function = loss_function
        # the hypothesisFunction I am expecting to be a data object representing the vector notation used for hypothesisspaces so I can exapand this to polynomial
        # later I want the hypothesis function to have an enum so that I can dictate if it is meant to be for regression or classification

    def compute_cost(self, hypothesis_function, data_values, data_targets):
        pass
