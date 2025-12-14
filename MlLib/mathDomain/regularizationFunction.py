class RegularizationFunction:
    def __init__(self):
        # the hypothesisFunction I am expecting to be a data object representing the vector notation used for hypothesisspaces so I can exapand this to polynomial
        # later I want the hypothesis function to have an enum so that I can dictate if it is meant to be for regression or classification
        self.metadata = {
            "name": "loss function parent class",
            "description": "A library class serving as a template for loss function classes that compute loss between a single predicted and actual value"
        }

    def computePenalty(self, actual, predicted):
        pass
    def compute(self, actual, predicted):
        pass
