from MlLib.mathDomain.probabilityBased.prior import prior
from MlLib.mathDomain.probabilityBased.sumRule import SumRule
from MlLib.mathDomain.probabilityBased.productRule import productRule


def test_prior_constructs():
    instance = prior()
    assert hasattr(instance, "metadata")


def test_sum_rule_constructs():
    instance = SumRule()
    assert hasattr(instance, "metadata")


def test_product_rule_constructs():
    instance = productRule()
    assert hasattr(instance, "metadata")
