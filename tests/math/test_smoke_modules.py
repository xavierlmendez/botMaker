"""Smoke tests for modules that had no test (BL-14): each must import, construct, and do the one
thing it claims. Placeholders (BL-07, BL-10) are pinned at "constructs" until implemented."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: visualizer must not need a display

from mllib.math.graph.graph_structures import Graph
from mllib.math.graph.visualizer import Visualizer
from mllib.math.linear_algebra_helpers import QuadraticFormHelper
from mllib.math.probability.bayes_rule import BayesRule
from mllib.math.probability.gaussian_prior import GaussianPrior
from mllib.math.regularization_function import RegularizationFunction


def test_quadratic_form_helper_builds_diagonal_of_squares():
    Q = QuadraticFormHelper().compute_q(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(Q, np.diag([1.0, 4.0, 9.0]))


def test_visualizer_show_without_traversal_draws_nothing_and_does_not_block():
    graph = Graph()
    graph.name = "smoke"
    graph.add_node(data="A")
    Visualizer().show(graph)  # traversalOrder=None: figure + title only


def test_gaussian_prior_stores_mean_and_variance():
    prior = GaussianPrior(mean=1.5, variance=0.25)
    assert (prior.mean, prior.variance) == (1.5, 0.25)


def test_probability_and_regularization_placeholders_construct():
    # BL-07 / BL-10: bodies are placeholders until the CS 6344 pairing; constructing must not fail.
    assert BayesRule().probability_event == 0
    assert RegularizationFunction.task_kind is None
