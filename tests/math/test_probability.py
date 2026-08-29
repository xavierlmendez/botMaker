from mllib.math.probability.prior import prior
from mllib.math.probability.product_rule import productRule
from mllib.math.probability.sum_rule import SumRule


def test_prior_constructs():
    instance = prior()
    assert hasattr(instance, "metadata")


def test_sum_rule_constructs():
    instance = SumRule()
    assert hasattr(instance, "metadata")


def test_product_rule_constructs():
    instance = productRule()
    assert hasattr(instance, "metadata")
