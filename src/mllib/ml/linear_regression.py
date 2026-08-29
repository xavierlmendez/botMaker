from numpy import array, mean, sum

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.loss_function import LossFunction


class MyLinearRegression:  # prefixing with my for the comparison script, rename later when cleaning up files
    # choosing 0.001 for default learning rate bc thats what adam uses
    """Core linear regression implementation with gradient descent and prediction helpers."""

    def __init__(
        self,
        hypothesis_function: HypothesisFunction,
        loss_function: LossFunction,
        learning_rate=0.001,
        epochs=10,
    ):
        self.learning_model = hypothesis_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, data_values, data_targets):
        # for n epochs
        # calculate the cost function - in the gradientDescent function
        # compute the gradient
        # update the weights
        # repeat
        for epoch in range(self.epochs):
            new_weights, new_bias = self.calculate_gradient_descent(data_values, data_targets)
            self.update_weights(new_weights, new_bias)
            cost = self.calculate_cost_function(data_values, data_targets)
        return self

    def predict(self, data):
        return self.learning_model.compute_prediction(data)

    def predict_values(self, data_values):
        predicted_values = []
        for data in data_values:
            predicted_values.append(self.predict(data))
        return array(predicted_values)

    def evaluate(self, data_values, data_targets):
        # used same evaluate function as boston data set demo
        pass

    def calculate_gradient_descent(self, data_values, data_targets):
        # calculate the gradient
        predicted = self.predict_values(data_values)
        gradient_descent_adjusteddata_targets = self.loss_function.compute_gradient(
            data_targets, predicted
        )

        gradient_descent_adjusted_weights = data_values.T @ gradient_descent_adjusteddata_targets
        adjusted_bias = sum(gradient_descent_adjusteddata_targets)
        return gradient_descent_adjusted_weights, adjusted_bias

    def update_weights(self, gradient_descent_adjusted_weights, gradient_descent_adjusted_bias):
        # update the weights and bias
        new_weights = (
            self.learning_model.get_weights()
            - self.learning_rate * gradient_descent_adjusted_weights
        )
        new_bias = (
            self.learning_model.get_bias() - self.learning_rate * gradient_descent_adjusted_bias
        )
        self.learning_model.update_weights(new_weights)
        self.learning_model.update_bias(new_bias)

    def calculate_cost_function(self, data_values, data_targets):
        # here were putting together the cost function as a set of linear equations
        # doing it this way to leverage linear algebra packages
        predicted = self.predict_values(data_values)
        loss_across_data = self.loss_function.compute_loss(data_targets, predicted)
        return mean(loss_across_data)
