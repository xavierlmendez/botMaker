"""Declarative pipelines: config -> transformers by name -> frames identical to the legacy code."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mllib.data.data_orchestrator import DataOrchestrator
from mllib.data.pipeline import ProjectTransformations, TransformerPipeline
from mllib.data.transformers import DropColumns, OneHotEncode

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ad_click_dataset.csv"
CONFIG_PATH = DATA_PATH.parent / "configs" / "ad_click_transformations.json"
FINGERPRINTS = json.loads((Path(__file__).parent / "frame_fingerprints.json").read_text())


def _fingerprint(frame):
    return {
        "shape": list(frame.shape),
        "columns": [str(c) for c in frame.columns],
        "sha256": hashlib.sha256(frame.astype(str).to_csv(index=False).encode()).hexdigest(),
    }


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
def test_orchestrator_frames_from_config_match_the_recorded_legacy_frames():
    orchestrator = DataOrchestrator(str(DATA_PATH), "csv", str(CONFIG_PATH))
    for name, expected in FINGERPRINTS.items():
        assert _fingerprint(orchestrator.frames[name]) == expected, name
    X, y = orchestrator.get_transformed_data("logisticReg")
    assert "click" not in X.columns and y.name == "click"
    with pytest.raises(KeyError, match="unknown frame"):
        orchestrator.get_transformed_data("nope")
