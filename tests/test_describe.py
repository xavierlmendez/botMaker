"""describe() derives a component descriptor from the class itself (BL-16)."""

import json

from mllib.describe import describe
from mllib.math.loss_function import MSE
from mllib.ml.logistic_regression import MyLogisticRegression


def test_describe_uses_class_name_docstring_and_domain():
    d = describe(MSE())
    assert d["name"] == "MSE"
    assert d["module"] == "mllib.math.loss_function"
    assert d["kind"] == "math"
    assert d["doc"].startswith("Loss function computing mean squared error")


def test_describe_lists_constructor_parameters():
    d = describe(MyLogisticRegression)
    assert d["kind"] == "ml"
    assert d["params"] == ["learning_rate", "epochs", "num_weights"]
    assert "learning_rate=0.001" in d["signature"]


def test_describe_is_the_same_for_class_and_instance_and_serialises():
    assert describe(MSE) == describe(MSE())
    json.dumps(describe(MyLogisticRegression()))


def test_describe_outside_the_package_falls_back_to_module():
    d = describe(dict)
    assert d["name"] == "dict"
    assert d["kind"] == "builtins"
