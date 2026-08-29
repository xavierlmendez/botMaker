"""Batch gradient descent shared by the regression and classification models (R2, BL-20)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.loss_function import LossFunction


class GradientDescentModel:
    """Host for an injected hypothesis and loss, trained by batch gradient descent.

    The loop is written once. A subclass chooses how the hypothesis turns a row into an output by
    setting ``predict_method``: ``"compute_prediction"`` for regression, ``"compute_classification"``
    for a sign classifier. That attribute is the whole difference between linear and logistic
    regression in this library.
    """

    predict_method = "compute_prediction"

    def __init__(
        self,
        hypothesis_function: HypothesisFunction,
        loss_function: LossFunction,
        learning_rate: float = 0.001,
        epochs: int = 10,
    ):
        self.learning_model = hypothesis_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, data_values, data_targets):
        for _ in range(self.epochs):
            weight_gradient, bias_gradient = self.calculate_gradient_descent(
                data_values, data_targets
            )
            self.update_weights(weight_gradient, bias_gradient)
        return self

    def predict(self, data):
        return getattr(self.learning_model, self.predict_method)(data)

    def predict_values(self, data_values):
        data_values, _ = self.as_arrays(data_values, None)
        return np.array([self.predict(row) for row in data_values])

    def encode_targets(self, data_targets):
        """Map targets into the space the hypothesis predicts in. Identity for regression; a sign
        classifier overrides this so the loss compares like with like (BL-23).
        """
        return data_targets

    def calculate_gradient_descent(self, data_values, data_targets):
        data_values, data_targets = self.as_arrays(data_values, data_targets)
        data_targets = self.encode_targets(data_targets)
        predicted = self.predict_values(data_values)
        loss_gradient = self.loss_function.compute_gradient(data_targets, predicted)
        design = self.learning_model.hypothesis_expander.fit_data_to_hypothesis(data_values)
        return design.T @ loss_gradient, np.sum(loss_gradient)

    def update_weights(self, weight_gradient, bias_gradient):
        model = self.learning_model
        model.update_weights(model.get_weights() - self.learning_rate * weight_gradient)
        model.update_bias(model.get_bias() - self.learning_rate * bias_gradient)

    def calculate_cost_function(self, data_values, data_targets):
        data_values, data_targets = self.as_arrays(data_values, data_targets)
        data_targets = self.encode_targets(data_targets)
        predicted = self.predict_values(data_values)
        return np.mean(self.loss_function.compute_loss(data_targets, predicted))

    @staticmethod
    def as_arrays(data_values, data_targets):
        """Accept DataFrames/Series as well as arrays; models compute on arrays."""
        if isinstance(data_values, pd.DataFrame):
            data_values = data_values.to_numpy()
        if data_targets is not None and not isinstance(data_targets, np.ndarray):
            data_targets = np.asarray(data_targets).ravel()
        return data_values, data_targets
