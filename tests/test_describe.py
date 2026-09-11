"""describe() derives a component descriptor from the class itself (BL-16)."""

import json

import numpy as np
import pytest

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


def test_configuration_of_records_a_penaltys_knob_and_omits_its_tensor():
    pytest.importorskip("torch")
    from mllib.describe import configuration_of
    from mllib.math.algorithms.two_hot_span.penalties import LaplacianAdjacencyPenalty

    record = configuration_of(LaplacianAdjacencyPenalty(np.eye(3), weight=0.5))

    assert record == {"name": "LaplacianAdjacencyPenalty", "weight": 0.5}
    json.dumps(record)


def test_configuration_of_recurses_into_an_injected_mllib_object():
    from mllib.describe import configuration_of
    from mllib.math.graph.two_hot_span_problem import SpanCost
    from mllib.math.projector import ExactProjector

    record = configuration_of(SpanCost(ExactProjector(), np.eye(2)))

    assert record == {"name": "SpanCost", "projector": {"name": "ExactProjector"}}


def test_configuration_of_writes_an_enum_by_its_value_and_keeps_none_and_strings():
    from mllib.describe import configuration_of
    from mllib.math.task_kind import TaskKind

    class Labelled:
        def __init__(self, kind=TaskKind.REGRESSION, label="x", nothing=None, flag=True):
            self.kind = kind
            self.label = label
            self.nothing = nothing
            self.flag = flag

    assert configuration_of(Labelled()) == {
        "name": "Labelled",
        "kind": "regression",
        "label": "x",
        "nothing": None,
        "flag": True,
    }


def test_configuration_of_a_class_with_no_constructor_parameters_is_just_its_name():
    from mllib.describe import configuration_of

    class Bare:
        pass

    assert configuration_of(Bare()) == {"name": "Bare"}


def test_configuration_of_a_dataclass_records_only_its_scalars_and_nested_mllib_objects():
    from dataclasses import dataclass, field

    from mllib.describe import configuration_of
    from mllib.math.projector import ExactProjector

    @dataclass(frozen=True)
    class Holder:
        weight: float
        projector: ExactProjector
        data: np.ndarray = field(compare=False)
        pairs: tuple[int, ...] = (1, 2)

    record = configuration_of(Holder(0.5, ExactProjector(), np.eye(2)))

    assert record == {"name": "Holder", "weight": 0.5, "projector": {"name": "ExactProjector"}}
    json.dumps(record)
