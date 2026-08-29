"""The shared descent base (R2): one loop, two models, differing only in the predict method."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MSE
from mllib.ml.gradient_descent import GradientDescentModel
from mllib.ml.linear_regression import MyLinearRegression
from mllib.ml.logistic_regression import MyLogisticRegression
from mllib.ml.projects.ad_click_logistic_regression import LogisticRegression


def _line_hypothesis():
    return HypothesisFunction(
        np.zeros(1), 0.0, degree=1, hypothesis_expander=PolynomialRegressionExpander(1)
    )


@given(slope=st.floats(-3, 3), intercept=st.floats(-3, 3))
@settings(max_examples=25, deadline=None)
def test_linear_regression_recovers_a_line(slope, intercept):
    x = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = slope * x[:, 0] + intercept

    model = MyLinearRegression(_line_hypothesis(), MSE(), learning_rate=0.1, epochs=300).fit(x, y)

    assert np.allclose(model.predict_values(x), y, atol=0.05)


def test_models_share_the_loop_and_differ_only_in_predict_method():
    for method in (
        "fit",
        "calculate_gradient_descent",
        "update_weights",
        "calculate_cost_function",
    ):
        assert getattr(MyLinearRegression, method) is getattr(GradientDescentModel, method)
        assert getattr(MyLogisticRegression, method) is getattr(GradientDescentModel, method)
    assert MyLinearRegression.predict_method == "compute_prediction"
    assert MyLogisticRegression.predict_method == "compute_classification"


def test_dataframes_are_accepted_everywhere():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    targets = pd.Series([2.0, 4.0, 6.0])
    model = MyLinearRegression(_line_hypothesis(), MSE(), learning_rate=0.1, epochs=200).fit(
        frame, targets
    )
    assert model.predict_values(frame).shape == (3,)
    assert model.calculate_cost_function(frame, targets) < 0.05


def test_grid_fit_splits_internally_and_evaluates_every_permutation():
    rng = np.random.default_rng(0)
    data_values = rng.random((10, 19))
    data_targets = rng.integers(0, 2, size=10)
    model = LogisticRegression()
    model.hyperparameter_grid_options = np.array(
        [
            {
                "modelName": ["t"],
                "learningRate": [0.01],
                "epoch": [2],
                "lossFunction": [MSE()],
                "HypothesisExpander": [PolynomialRegressionExpander()],
                "polynomialDegree": [1, 2],
                "weightRandSeed": [1],
                "initialBias": [1],
            }
        ]
    )

    model.grid_fit(data_values, data_targets, test_size=0.2, random_state=0)

    records = model.evaluator.evaluation_record
    assert sorted(records) == [1, 2]
    for record in records.values():
        total = sum(
            record[k]
            for k in ("truePositives", "falsePositives", "trueNegatives", "falseNegatives")
        )
        assert total == 2  # 20 % of 10 rows held out
