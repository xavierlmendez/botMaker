import numpy as np
import pytest

from mllib.math.hypothesis import HypothesisFunction
from mllib.math.hypothesis_expander import HypothesisExpander, PolynomialRegressionExpander


def test_init_expands_weights_with_polynomial_expander():
    initial_weights = np.array([2.0, 3.0])
    expander = PolynomialRegressionExpander(degree=2)

    hypothesis = HypothesisFunction(
        initial_weights,
        initial_bias=1.5,
        degree=2,
        hypothesis_expander=expander,
    )

    assert hypothesis.hypothesis_expander.degree == 2
    assert np.allclose(hypothesis.get_hypothesis(), np.array([2.0, 4.0, 3.0, 9.0]))


def test_compute_prediction_degree_one_linear():
    weights = np.array([2.0, -1.0])
    bias = 0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesis_expander=expander)

    data = np.array([3.0, 4.0])
    predicted = hypothesis.compute_prediction(data)

    assert predicted == weights[0] * data[0] + weights[1] * data[1] + bias


def test_compute_classification_degree_one_sign():
    weights = np.array([1.0, -2.0])
    bias = -0.5
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesis_expander=expander)

    assert hypothesis.compute_classification(np.array([1.0, 1.0])) == -1.0
    assert hypothesis.compute_classification(np.array([3.0, 0.0])) == 1.0


def test_update_and_getters():
    weights = np.array([0.5, 1.5])
    bias = -2.0
    expander = PolynomialRegressionExpander(degree=1)
    hypothesis = HypothesisFunction(weights, bias, degree=1, hypothesis_expander=expander)

    new_weights = np.array([2.0, 4.0])
    hypothesis.update_weights(new_weights)
    hypothesis.update_bias(3.0)

    assert np.allclose(hypothesis.get_weights(), new_weights)
    assert hypothesis.get_bias() == 3.0


@pytest.mark.parametrize(
    ("expander", "degree"),
    [
        (HypothesisExpander(), 1),
        (PolynomialRegressionExpander(degree=1), 1),
        (PolynomialRegressionExpander(degree=3), 3),
    ],
    ids=["identity", "polynomial_degree_1", "polynomial_degree_3"],
)
def test_expand_hypothesis_recomputes_the_constructors_expansion_of_the_initial_weights(
    expander, degree
):
    """BL-48 slice 5: the method used to call an `expand(hypothesis, degree)` no expander defines.

    Fails on `main` before the fix with an `AttributeError` or a `TypeError`; passes after.
    """
    initial_weights = np.array([2.0, -3.0])
    hypothesis = HypothesisFunction(
        initial_weights, 0.0, degree=degree, hypothesis_expander=expander
    )
    expanded_by_the_constructor = hypothesis.get_hypothesis().copy()
    hypothesis.update_weights(np.zeros_like(expanded_by_the_constructor))

    hypothesis.expand_hypothesis()

    assert np.array_equal(hypothesis.get_hypothesis(), expanded_by_the_constructor)
    assert np.array_equal(hypothesis.get_hypothesis(), expander.expand_hypothesis(initial_weights))


def test_expanding_twice_expands_the_initial_weights_once():
    """A second call is not a second expansion: degree 3 on two weights stays six entries."""
    hypothesis = HypothesisFunction(
        np.array([2.0, -3.0]), 0.0, degree=3, hypothesis_expander=PolynomialRegressionExpander(3)
    )

    hypothesis.expand_hypothesis()
    hypothesis.expand_hypothesis()

    assert hypothesis.get_hypothesis().shape == (6,)
