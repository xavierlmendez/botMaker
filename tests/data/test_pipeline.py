"""Declarative pipelines: config -> transformers by name -> frames identical to the legacy code."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mllib.data.data_orchestrator import DataOrchestrator
from mllib.data.pipeline import ProjectTransformations, TransformerPipeline
from mllib.data.transformers import DropColumns, OneHotEncode

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ad_click_dataset.csv"
CONFIG_PATH = DATA_PATH.parent / "configs" / "ad_click_transformations.json"


def test_from_config_resolves_transformers_by_name_in_order():
    pipeline = TransformerPipeline.from_config(
        [
            {"transformer": "DropColumns", "args": {"columns": ["id"]}},
            {"transformer": "OneHotEncode", "args": {"columns": ["color"]}},
        ]
    )
    assert [type(s) for s in pipeline.steps] == [DropColumns, OneHotEncode]
    out = pipeline.fit_transform(pd.DataFrame({"id": [1, 2], "color": ["a", "b"]}))
    assert list(out.columns) == ["color_b", "color_nan"]


def test_unknown_or_non_transformer_names_are_rejected():
    with pytest.raises(ValueError, match="unknown transformer"):
        TransformerPipeline.from_config([{"transformer": "NoSuchThing"}])
    with pytest.raises(ValueError, match="unknown transformer"):
        TransformerPipeline.from_config([{"transformer": "Transformer"}])  # the ABC itself
    with pytest.raises(TypeError):
        TransformerPipeline([object()])


def test_fit_then_transform_applies_learned_state_to_new_data():
    pipeline = TransformerPipeline([OneHotEncode(["color"])]).fit(
        pd.DataFrame({"color": ["a", "b", "c"]})
    )
    out = pipeline.transform(pd.DataFrame({"color": ["b"]}))
    assert list(out.columns) == ["color_b", "color_c", "color_nan"]
    assert out.iloc[0].tolist() == [1.0, 0.0, 0.0]


def test_project_config_declares_target_and_four_frames():
    project = ProjectTransformations.from_file(CONFIG_PATH)
    assert project.target == "click"
    assert project.frame_names() == [
        "logisticReg",
        "logisticRegWithAgeBinning",
        "decisionTree",
        "neuralNetwork",
    ]


@pytest.mark.skipif(not DATA_PATH.is_file(), reason="ad-click dataset not present")
def test_orchestrator_frames_from_config_match_the_legacy_transformer():
    orchestrator = DataOrchestrator(str(DATA_PATH), "csv", str(CONFIG_PATH))
    legacy = orchestrator.data_transformer
    assert_frame_equal(orchestrator.frames["logisticReg"], legacy.logistic_model_data_frame)
    assert_frame_equal(
        orchestrator.frames["logisticRegWithAgeBinning"],
        legacy.logistic_model_with_age_binning_data_frame,
    )
    assert_frame_equal(orchestrator.frames["decisionTree"], legacy.decision_tree_data_frame)
    X, y = orchestrator.get_transformed_data("logisticReg")
    assert "click" not in X.columns and y.name == "click"
    with pytest.raises(KeyError, match="unknown frame"):
        orchestrator.get_transformed_data("nope")
