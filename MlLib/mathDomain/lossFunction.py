# defining the loss functions ill be plugging into the linear regression model, I want this to be abstract for reuse with the project if I decide to make a library
# https://docs.python.org/3/tutorial/classes.html
import numpy as np


class LossFunction:
    def __init__(self):
      # the hypothesisFunction I am expecting to be a data object representing the vector notation used for hypothesisspaces so I can exapand this to polynomial
      # later I want the hypothesis function to have an enum so that I can dictate if it is meant to be for regression or classification
        self.metadata = {
            "name": "loss function parent class",
            "description": "A library class serving as a template for loss function classes that compute loss between a single predicted and actual value"
        }

    def computeLoss(self, actual, predicted):
        pass
    def computeGradient(self, actual, predicted):
        pass

class MSE(LossFunction):
    metadata = {
        "name": "Mean Squared Error",
        "description": "Loss function computing mean squared error and its gradient."
    }
    # TODO: review metadata (auto-generated)
    def computeLoss(self, actual, predicted):
        return np.mean((actual - predicted)**2)

    def computeGradient(self, actual, predicted):
        n = actual.shape[0]
        return (2.0/ n) * (predicted - actual)

class MAE(LossFunction):
    metadata = {
        "name": "Mean Absolute Error",
        "description": "Loss function computing mean absolute error and its gradient."
    }
    # TODO: review metadata (auto-generated)
    def computeLoss(self, actual, predicted):
        return np.mean(abs(actual - predicted))

    def computeGradient(self, actual, predicted):
        return np.sign(predicted - actual)

class PerceptronLoss(LossFunction):
    metadata = {
        "name": "Perceptron Loss",
        "description": "Perceptron loss with sub-gradient and bias updates."
    }
    # TODO: review metadata (auto-generated)
    def computeLoss(self, actual: np.ndarray, predicted: np.ndarray):
        return np.maximum(0.0, -actual * predicted)

    def computeGradient(self, actual, predicted, dataValues):
        # this is actually a sub gradient but reusing the function name for consistency
        zeroIfClassifiedCorrectly = (actual * predicted <= 0)
        return dataValues.T @ (zeroIfClassifiedCorrectly * actual)

    def computeBias(self, actual, predicted):
        zeroIfClassifiedCorrectly = (actual * predicted <= 0)
        return (zeroIfClassifiedCorrectly * actual).sum()

class HingeLoss(LossFunction):
    metadata = {
        "name": "Hinge Loss",
        "description": "Hinge loss for margin-based classifiers with sub-gradient updates."
    }
    # TODO: review metadata (auto-generated)
    def computeLoss(self, actual, predicted):
        return np.maximum(0.0, 1.0 - actual * predicted)

    def computeGradient(self, actual, predicted, dataValues):
        # this is actually a sub gradient but reusing the function name for consistency
        zeroIfClassifiedCorrectly = (actual * predicted <= 0)
        return dataValues.T @ (zeroIfClassifiedCorrectly * -actual)

    def computeBias(self, actual, predicted):
        zeroIfClassifiedCorrectly = (actual * predicted <= 0)
        return (zeroIfClassifiedCorrectly * actual).sum()
