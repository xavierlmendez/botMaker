class CostFunction():
    def __init__(self, lossFunction : LossFunction):
        self.lossFunction = lossFunction
        # the hypothesisFunction I am expecting to be a data object representing the vector notation used for hypothesisspaces so I can exapand this to polynomial
        # later I want the hypothesis function to have an enum so that I can dictate if it is meant to be for regression or classification
        self.metadata = {
            "name": "loss function parent class",
            "description": "A library class serving as a template for cost function classes that compute cost for a run through of a data set"
        }

    def computeCost(self, hypothesisFunction, dataValues, dataTargets):
        pass