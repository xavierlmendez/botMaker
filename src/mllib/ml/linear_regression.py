"""Linear regression: gradient descent on h(x) = w·x + b with a regression loss."""

from mllib.ml.gradient_descent import GradientDescentModel


class MyLinearRegression(GradientDescentModel):
    """Linear regression. Inject a ``PolynomialRegressionExpander`` into the hypothesis for
    polynomial regression — the model does not change.
    """

    predict_method = "compute_prediction"
