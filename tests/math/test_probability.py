from mllib.math.probability.prior import Prior
from mllib.math.probability.product_rule import ProductRule
from mllib.math.probability.sum_rule import SumRule


def test_prior_constructs():
    instance = Prior()
    assert hasattr(instance, "metadata")


def test_sum_rule_constructs():
    instance = SumRule()
    assert hasattr(instance, "metadata")


def test_product_rule_constructs():
    instance = ProductRule()
    assert hasattr(instance, "metadata")
