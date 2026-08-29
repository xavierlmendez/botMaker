import time

import numpy as np
import pandas as pd
from numpy import array, mean, sum
from sklearn.model_selection import ParameterGrid

# Linear and logistic are very similar however have a major difference in that logistic is for classification
# TODO(BL-20): share a gradient-descent base with linear regression
#  as the main difference is the compute prediction function
from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MSE
from mllib.ml.evaluators.generic_evaluator import LogisticRegressionModelEvaluator


class MyLogisticRegression:  # prefixing with my for the comparison script, rename later when cleaning up files
    # choosing 0.001 for default learning rate bc thats what adam uses
    """Core logistic regression implementation with training, prediction, and evaluation helpers."""

    def __init__(self, learning_rate=0.001, epochs=10, num_weights=1):
        seeded_rand = np.random.default_rng(10)  # seeting a seed for random initial weights,
        self.num_weights = num_weights  # number of feature weights; project subclasses pass theirs
        initial_weights = seeded_rand.random(self.num_weights)
        initial_bias = 0
        # Same construction gridFit uses per permutation; a plain linear hypothesis by default.
        self.learning_model = HypothesisFunction(
            initial_weights,
            initial_bias,
            degree=1,
            hypothesis_expander=PolynomialRegressionExpander(1),
        )
        self.loss_function = MSE()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameter_grid_options = None

    def grid_fit(
        self, train_values, test_values, train_targets, test_targets
    ):  # TODO(BL-20): move the train/test split into gridFit as a grid parameter
        hyperparameter_combinations = list(ParameterGrid(self.hyperparameter_grid_options))
        model_implementation_name = self.hyperparameter_grid_options[0]["modelName"][0]
        print(
            f" Starting model training for {type(self).__name__} implementation: {model_implementation_name}\n"
        )

        count_models = hyperparameter_combinations.__len__()
        print(f" Total model permutations: {count_models}")

        model_number = 0
        start_time = time.perf_counter()
        for parameter_setting in hyperparameter_combinations:
            model_number += 1

            self.loss_function = parameter_setting["lossFunction"]
            self.epochs = parameter_setting["epoch"]
            self.learning_rate = parameter_setting["learningRate"]
            seeded_rand = np.random.default_rng(parameter_setting["weightRandSeed"])
            initial_weights = seeded_rand.random(self.num_weights)
            initial_bias = parameter_setting["initialBias"]
            self.learning_model = HypothesisFunction(
                initial_weights,
                initial_bias,
                parameter_setting["polynomialDegree"],
                parameter_setting["HypothesisExpander"],
            )
            hypothesis_space_adjusted_weights = (
                self.learning_model.hypothesis_expander.expand_hypothesis(initial_weights)
            )
            self.update_weights(hypothesis_space_adjusted_weights, initial_bias)

            for epoch in range(self.epochs):
                new_weights, new_bias = self.calculate_gradient_descent(train_values, train_targets)
                self.update_weights(new_weights, new_bias)
                cost = self.calculate_cost_function(train_values, train_targets)

            self.evaluate(test_values, test_targets, parameter_setting)

            if (
                model_number % 1 == 0
            ):  # modify depending on number of permutations i.e. 300+ then probably modulo 40 or 80
                print(f"\tModel Number {model_number}/{count_models} complete")

        end_time = time.perf_counter()
        time_elapsed = end_time - start_time
        time_per_model = time_elapsed / count_models
        print(f" Training Time Elapsed: {time_elapsed}, time per model: {time_per_model}")

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
            # use evaluator class here to aggregate data on performance during training
        return self

    def predict(self, data):
        return self.learning_model.compute_classification(data)

    def predict_values(self, data_values, is_dataframe=False):
        predicted_values = []
        for data in data_values:
            predicted_values.append(self.predict(data))
        return array(predicted_values)

    def evaluate(self, data_values, data_targets, evaluation_meta_data):
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        data_values, data_targets = self.data_frame_cross_capatibility(data_values, data_targets)
        predicted_values = self.predict_values(data_values, data_targets)
        self.evaluator.update_testing_prediction_data(
            data_values, data_targets, predicted_values, evaluation_meta_data
        )

    def calculate_gradient_descent(self, data_values, data_targets):
        # standardize Inputs for compatibility with pandas dataframes as parameters
        data_values, data_targets = self.data_frame_cross_capatibility(data_values, data_targets)

        # calculate the gradient
        predicted = self.predict_values(data_values)
        gradient_descent_adjusted_data_targets = self.loss_function.compute_gradient(
            data_targets, predicted
        )
        data_values = self.learning_model.hypothesis_expander.fit_data_to_hypothesis(data_values)
        gradient_descent_adjusted_weights = data_values.T @ gradient_descent_adjusted_data_targets
        adjusted_bias = sum(gradient_descent_adjusted_data_targets)
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
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        data_values, data_targets = self.data_frame_cross_capatibility(data_values, data_targets)

        # here were putting together the cost function as a set of linear equations
        # doing it this way to leverage linear algebra packages
        predicted = self.predict_values(data_values)
        loss_across_data = self.loss_function.compute_loss(data_targets, predicted)
        return mean(loss_across_data)

    def data_frame_cross_capatibility(self, data_values, data_targets):
        if isinstance(data_values, pd.DataFrame):
            data_values = data_values.to_numpy()
            data_targets = np.asarray(data_targets).ravel()
        return data_values, data_targets
