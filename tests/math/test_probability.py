from mllib.describe import describe
from mllib.math.probability.prior import Prior
from mllib.math.probability.product_rule import ProductRule
from mllib.math.probability.sum_rule import SumRule


def test_prior_constructs():
    instance = Prior()
    assert describe(instance)["name"] == "Prior"  # BL-07 placeholder: constructs and self-describes


def test_sum_rule_constructs():
    instance = SumRule()
    assert (
        describe(instance)["name"] == "SumRule"
    )  # BL-07 placeholder: constructs and self-describes


def test_product_rule_constructs():
    instance = ProductRule()
    assert (
        describe(instance)["name"] == "ProductRule"
    )  # BL-07 placeholder: constructs and self-describes
