"""Smoke tests for modules that had no test (BL-14): each must import, construct, and do the one
thing it claims. Placeholders (BL-07, BL-10) are pinned at "constructs" until implemented.

`math/graph/visualizer.py` was one of these until BL-43 slice 4 deleted it: the traversal it
animated is now a recording plus a walkthrough page, so its smoke test (and the headless matplotlib
backend that test alone needed) went with it. The traversals themselves are covered by
`tests/math/algorithms/`, which includes the fingerprint the deleted animation's example left
behind."""

import numpy as np

from mllib.math.linear_algebra_helpers import QuadraticFormHelper
from mllib.math.probability.bayes_rule import BayesRule
from mllib.math.probability.gaussian_prior import GaussianPrior
from mllib.math.regularization_function import AbstractRegularizationFunction


def test_quadratic_form_helper_builds_diagonal_of_squares():
    Q = QuadraticFormHelper().compute_q(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(Q, np.diag([1.0, 4.0, 9.0]))


def test_gaussian_prior_stores_mean_and_variance():
    prior = GaussianPrior(mean=1.5, variance=0.25)
    assert (prior.mean, prior.variance) == (1.5, 0.25)


def test_probability_and_regularization_placeholders_construct():
    # BL-07: the probability bodies are placeholders until the CS 6344 pairing; constructing must
    # not fail. The regularization base is an ABC since BL-48 slice 1; only its class attribute
    # is read here.
    assert BayesRule().probability_event == 0
    assert AbstractRegularizationFunction.task_kind is None
