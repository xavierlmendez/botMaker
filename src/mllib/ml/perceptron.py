import numpy as np

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.loss_function import LossFunction


class MyPerceptron:  # prefixing with my for the comparison script, rename later when cleaning up files
    """Perceptron implementation using sub-gradient updates for classification."""

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
        for epoch in range(self.epochs):
            sub_gradient_direction_matrix, sub_gradient_dias_direction_matrix = (
                self.calculate_sub_gradient_descent(data_values, data_targets)
            )
            self.update_weights(sub_gradient_direction_matrix, sub_gradient_dias_direction_matrix)
            error = self.calculate_error(data_values, data_targets)
            print(f"Epoch: {epoch}, Error: {error}")
        return self

    def predict(self, data):
        return np.sign(self.learning_model.compute_classification(data))

    def predict_values(self, data_values):
        predicted_values = []
        for data in data_values:
            predicted_values.append(self.predict(data))
        return np.array(predicted_values)

    def evaluate(self, data_values, data_targets):
        pass

    def calculate_sub_gradient_descent(self, data_values, data_targets):
        # calculate the gradient
        predicted = self.predict_values(data_values)
        sub_gradient_direction = self.loss_function.compute_gradient(
            data_targets, predicted, data_values
        )
        sub_gradient_bias_direction = self.loss_function.compute_bias(data_targets, predicted)
        return sub_gradient_direction, sub_gradient_bias_direction

    def update_weights(self, sub_gradient_direction, sub_gradient_bias_direction):
        # update the weights and bias
        print("sub gradient:", sub_gradient_direction)
        new_weights = (
            self.learning_model.get_weights() + self.learning_rate * sub_gradient_direction
        )
        print("updated weights:", new_weights)
        new_bias = self.learning_model.get_bias() + self.learning_rate * sub_gradient_bias_direction
        self.learning_model.update_weights(new_weights)
        self.learning_model.update_bias(new_bias)

    def calculate_error(self, data_values, data_targets):
        predicted = self.predict_values(data_values)
        misclassified = predicted != data_targets
        return np.mean(misclassified)
