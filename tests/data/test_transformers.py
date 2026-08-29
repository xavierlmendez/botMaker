"""Transformer contract (fit/transform, no mutation) and exact equivalence with the legacy
``DataOrchestrator`` frames the training baseline depends on (R1, BL-09)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mllib.data.data_orchestrator import DataOrchestrator
from mllib.data.transformers import (
    BinByStdRanges,
    DropColumns,
    FillNaNWithMean,
    OneHotEncode,
    ReplaceNaNWithString,
    StandardizeNumeric,
    Transformer,
)
from mllib.describe import describe

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ad_click_dataset.csv"
CONFIG_PATH = DATA_PATH.parent / "configs" / "ad_click_transformations.json"


def _frame():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [20.0, np.nan, 40.0, 60.0],
            "color": ["red", "blue", None, "red"],
        }
    )


def test_transformers_never_mutate_their_input():
    frame = _frame()
    original = frame.copy()
    for transformer in (
        DropColumns(["id"]),
        OneHotEncode(["color"]),
        StandardizeNumeric(["age"]),
        FillNaNWithMean(["age"]),
        ReplaceNaNWithString(["color"]),
        BinByStdRanges("age"),
    ):
        transformer.fit_transform(frame)
        assert_frame_equal(frame, original)


def test_drop_and_replace_nan_with_string():
    out = ReplaceNaNWithString(["color"]).transform(DropColumns(["id"]).transform(_frame()))
    assert list(out.columns) == ["age", "color"]
    assert out["color"].tolist() == ["red", "blue", "nan", "red"]


def test_standardize_then_fill_learns_mean_and_scale():
    out = FillNaNWithMean(["age"]).fit_transform(
        StandardizeNumeric(["age"]).fit_transform(_frame())
    )
    assert out["age"].isna().sum() == 0
    assert out["age"].mean() == pytest.approx(0.0)


def test_one_hot_encode_encodes_new_data_against_fitted_levels():
    encoder = OneHotEncode(["color"]).fit(_frame())
    assert encoder.categories_ == {"color": ["blue", "red"]}
    fitted_columns = list(encoder.transform(_frame()).columns)

    new = pd.DataFrame({"id": [9, 10], "age": [30.0, 31.0], "color": ["blue", "green"]})
    out = encoder.transform(new)

    assert list(out.columns) == fitted_columns  # same columns, same dropped first level
    assert out["color_red"].tolist() == [0.0, 0.0]
    assert out["color_nan"].tolist() == [0.0, 1.0]  # unseen level lands in the missing column


def test_bin_by_std_ranges_edges_are_increasing_and_bounded():
    binner = BinByStdRanges("age").fit(_frame())  # tight column: mean - 2*std < min
    assert binner.edges_ == sorted(set(binner.edges_))
    assert binner.edges_[0] == 20.0 and binner.edges_[-1] == 60.0
    out = binner.transform(_frame())
    assert str(out["age"].dtype) == "category"


def test_bin_by_std_ranges_keeps_all_nine_edges_on_a_wide_column():
    wide = pd.DataFrame({"age": np.concatenate([np.full(50, 40.0), [0.0, 80.0]])})
    assert len(BinByStdRanges("age").fit(wide).edges_) == 9


def test_transformers_are_self_describing():
    assert describe(OneHotEncode)["kind"] == "data"
    assert issubclass(OneHotEncode, Transformer)


@pytest.mark.skipif(not DATA_PATH.is_file(), reason="ad-click dataset not present")
def test_pipeline_of_transformers_reproduces_legacy_frames_exactly():
    orchestrator = DataOrchestrator(str(DATA_PATH), "csv", str(CONFIG_PATH))
    raw = pd.read_csv(DATA_PATH, header=0)
    categorical = ["gender", "device_type", "ad_position", "browsing_history", "time_of_day"]

    logistic = raw
    for t in (
        DropColumns(["id", "full_name"]),
        OneHotEncode(categorical),
        StandardizeNumeric(["age"]),
        FillNaNWithMean(["age"]),
    ):
        logistic = t.fit_transform(logistic)
    assert_frame_equal(logistic, orchestrator.data_transformer.logistic_model_data_frame)

    binned = raw
    for t in (
        DropColumns(["id", "full_name"]),
        StandardizeNumeric(["age"]),
        FillNaNWithMean(["age"]),
        BinByStdRanges("age"),
        OneHotEncode(["age", *categorical]),
    ):
        binned = t.fit_transform(binned)
    assert_frame_equal(
        binned, orchestrator.data_transformer.logistic_model_with_age_binning_data_frame
    )
