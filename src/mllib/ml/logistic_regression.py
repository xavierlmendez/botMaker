"""Logistic (sign) classification on the shared gradient-descent loop, plus grid search."""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MSE
from mllib.ml.evaluators.generic_evaluator import LogisticRegressionModelEvaluator
from mllib.ml.gradient_descent import GradientDescentModel


class MyLogisticRegression(GradientDescentModel):
    """Classifier: the shared descent loop with the hypothesis's sign output, an evaluator, and a
    hyper-parameter grid search. Project subclasses set ``num_weights`` and the grid.
    """

    predict_method = "compute_classification"

    def encode_targets(self, data_targets):
        """The hypothesis emits ``sign(w·x + b)`` in {-1, +1}; train against targets in the same
        space. {0, 1} (or {-1, 1}) labels both map to ±1. Without this the gradient compared -1
        predictions with 0 targets and the classifier could not learn even separable data (BL-23).
        """
        return np.where(data_targets > 0, 1.0, -1.0)

    def __init__(self, learning_rate=0.001, epochs=10, num_weights=1):
        self.num_weights = num_weights  # number of feature weights; project subclasses pass theirs
        initial_weights = np.random.default_rng(10).random(num_weights)
        super().__init__(
            HypothesisFunction(
                initial_weights, 0, degree=1, hypothesis_expander=PolynomialRegressionExpander(1)
            ),
            MSE(),
            learning_rate,
            epochs,
        )
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameter_grid_options = None

    def grid_fit(self, data_values, data_targets, test_size=0.2, random_state=None):
        """Train one model per grid permutation on a train split and evaluate each on the held-out
        split. The split is made here (BL-20) with the same defaults ``DataOrchestrator`` used.
        """
        train_values, test_values, train_targets, test_targets = train_test_split(
            data_values, data_targets, test_size=test_size, random_state=random_state
        )
        hyperparameter_combinations = list(ParameterGrid(self.hyperparameter_grid_options))
        model_implementation_name = self.hyperparameter_grid_options[0]["modelName"][0]
        print(
            f" Starting model training for {type(self).__name__} implementation: {model_implementation_name}\n"
        )
        count_models = len(hyperparameter_combinations)
        print(f" Total model permutations: {count_models}")

        start_time = time.perf_counter()
        for model_number, parameter_setting in enumerate(hyperparameter_combinations, start=1):
            self.loss_function = parameter_setting["lossFunction"]
            self.epochs = parameter_setting["epoch"]
            self.learning_rate = parameter_setting["learningRate"]
            initial_weights = np.random.default_rng(parameter_setting["weightRandSeed"]).random(
                self.num_weights
            )
            initial_bias = parameter_setting["initialBias"]
            self.learning_model = HypothesisFunction(
                initial_weights,
                initial_bias,
                parameter_setting["polynomialDegree"],
                parameter_setting["HypothesisExpander"],
            )
            self.fit(train_values, train_targets)
            self.evaluate(test_values, test_targets, parameter_setting)
            print(f"\tModel Number {model_number}/{count_models} complete")

        time_elapsed = time.perf_counter() - start_time
        print(
            f" Training Time Elapsed: {time_elapsed}, time per model: {time_elapsed / count_models}"
        )

    def evaluate(self, data_values, data_targets, evaluation_meta_data):
        data_values, data_targets = self.as_arrays(data_values, data_targets)
        predicted_values = self.predict_values(data_values)
        self.evaluator.update_testing_prediction_data(
            data_values, data_targets, predicted_values, evaluation_meta_data
        )
