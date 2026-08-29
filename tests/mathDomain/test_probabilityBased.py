from mllib.mathDomain.probabilityBased.prior import prior
from mllib.mathDomain.probabilityBased.productRule import productRule
from mllib.mathDomain.probabilityBased.sumRule import SumRule


def test_prior_constructs():
    instance = prior()
    assert hasattr(instance, "metadata")


def test_sum_rule_constructs():
    instance = SumRule()
    assert hasattr(instance, "metadata")


def test_product_rule_constructs():
    instance = productRule()
    assert hasattr(instance, "metadata")
