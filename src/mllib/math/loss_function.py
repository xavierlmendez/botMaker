# defining the loss functions ill be plugging into the linear regression model, I want this to be abstract for reuse with the project if I decide to make a library
# https://docs.python.org/3/tutorial/classes.html
import numpy as np


class LossFunction:
    """A library class serving as a template for loss function classes that compute loss between a
    single predicted and actual value
    """

    def compute_loss(self, actual, predicted):
        pass

    def compute_gradient(self, actual, predicted):
        pass


class MSE(LossFunction):
    """Loss function computing mean squared error and its gradient."""

    def compute_loss(self, actual, predicted):
        return np.mean((actual - predicted) ** 2)

    def compute_gradient(self, actual, predicted):
        n = actual.shape[0]
        return (2.0 / n) * (predicted - actual)


class MAE(LossFunction):
    """Loss function computing mean absolute error and its gradient."""

    def compute_loss(self, actual, predicted):
        return np.mean(abs(actual - predicted))

    def compute_gradient(self, actual, predicted):
        return np.sign(predicted - actual)


class PerceptronLoss(LossFunction):
    """Perceptron loss with sub-gradient and bias updates."""

    def compute_loss(self, actual: np.ndarray, predicted: np.ndarray):
        return np.maximum(0.0, -actual * predicted)

    def compute_gradient(self, actual, predicted, data_values):
        # this is actually a sub gradient but reusing the function name for consistency
        zero_if_classified_correctly = actual * predicted <= 0
        return data_values.T @ (zero_if_classified_correctly * actual)

    def compute_bias(self, actual, predicted):
        zero_if_classified_correctly = actual * predicted <= 0
        return (zero_if_classified_correctly * actual).sum()


class HingeLoss(LossFunction):
    """Hinge loss for margin-based classifiers with sub-gradient updates."""

    def compute_loss(self, actual, predicted):
        return np.maximum(0.0, 1.0 - actual * predicted)

    def compute_gradient(self, actual, predicted, data_values):
        # this is actually a sub gradient but reusing the function name for consistency
        zero_if_classified_correctly = actual * predicted <= 0
        return data_values.T @ (zero_if_classified_correctly * -actual)

    def compute_bias(self, actual, predicted):
        zero_if_classified_correctly = actual * predicted <= 0
        return (zero_if_classified_correctly * actual).sum()
