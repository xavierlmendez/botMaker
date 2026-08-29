"""Smoke tests for modules that had no test (BL-14): each must import, construct, and do the one
thing it claims. Placeholders (BL-07, BL-10) are pinned at "constructs" until implemented."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: visualizer must not need a display

from mllib.mathDomain.graphBased.graphStructures import Graph
from mllib.mathDomain.graphBased.visualizer import Visualizer
from mllib.mathDomain.linearAlgebraHelpers import QuadraticFormHelper
from mllib.mathDomain.probabilityBased.bayesRule import BayesRule
from mllib.mathDomain.probabilityBased.gaussianPrior import GaussianPrior
from mllib.mathDomain.regularizationFunction import RegularizationFunction


def test_quadratic_form_helper_builds_diagonal_of_squares():
    Q = QuadraticFormHelper().computeQ(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(Q, np.diag([1.0, 4.0, 9.0]))


def test_visualizer_show_without_traversal_draws_nothing_and_does_not_block():
    graph = Graph()
    graph.name = "smoke"
    graph.addNode(data="A")
    Visualizer().show(graph)  # traversalOrder=None: figure + title only


def test_gaussian_prior_stores_mean_and_variance():
    prior = GaussianPrior(mean=1.5, variance=0.25)
    assert (prior.mean, prior.variance) == (1.5, 0.25)


def test_probability_and_regularization_placeholders_construct():
    # BL-07 / BL-10: bodies are placeholders until the CS 6344 pairing; constructing must not fail.
    assert BayesRule().probabilityEvent == 0
    assert RegularizationFunction().computePenalty(1, 1) is None
