"""Pre-migration behavioural baseline for the model-training feature.

Exercises the same path as ``examples/ad_click_model_comparison.py``
(real CSV -> DataOrchestrator -> project LogisticRegression classes -> gridFit ->
LogisticRegressionModelEvaluator) but with a tiny hyper-parameter grid so it runs
in seconds instead of hours, and with the global RNG seeded so the
train/test split inside ``grid_fit`` is reproducible.

Two layers of assertion:

1. Invariants that must hold on any environment: the pipeline runs end to end,
   every grid permutation is evaluated, metrics are in range, and accuracy is
   at least 0.5 (today's baseline equals the 0.65 majority rate — see BL-23).
2. A metrics snapshot (``baseline_snapshot.json`` next to this file). The first
   run writes it; later runs compare against it within a small tolerance. This
   is what catches silent behavioural drift during the refactor. Regenerate it
   deliberately with ``BASELINE_UPDATE=1`` when a change is *meant* to alter
   numbers, and say so in the PR.

Run (from the repo root):

    uv run pytest tests/ml/test_training_baseline.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mllib.data.data_orchestrator import DataOrchestrator
from mllib.math.hypothesis_expander import PolynomialRegressionExpander
from mllib.math.loss_function import MAE, MSE
from mllib.ml.projects.ad_click_logistic_regression import (
    LogisticRegression,
    LogisticRegressionWithAgeBinning,
)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "data" / "ad_click_dataset.csv"
CONFIG_PATH = HERE.parents[1] / "data" / "configs" / "ad_click_transformations.json"
SNAPSHOT_PATH = HERE / "baseline_snapshot.json"

SPLIT_SEED = 20260828
TOLERANCE = 1e-6  # metrics are deterministic given the seeds; drift means behaviour changed
MAJORITY_CLASS_RATE = 0.65  # `click` mean in ad_click_dataset.csv

# Small but representative slice of the production grid in adClickPredictionLogReg.py:
# both loss functions, a linear and a polynomial hypothesis, two weight seeds.
SMOKE_GRID = np.array(
    [
        {
            "modelName": ["baseline"],
            "learningRate": [0.01],
            "epoch": [10],
            "lossFunction": [MSE(), MAE()],
            "HypothesisExpander": [PolynomialRegressionExpander()],
            "polynomialDegree": [1, 2],
            "weightRandSeed": [1, 27],
            "initialBias": [1],
        }
    ]
)
EXPECTED_PERMUTATIONS = 2 * 2 * 2  # lossFunction x polynomialDegree x weightRandSeed

MODELS = {
    "logisticReg": LogisticRegression,
    "logisticRegWithAgeBinning": LogisticRegressionWithAgeBinning,
}


def _train(model_cls, split_key: str, orchestrator: DataOrchestrator):
    model = model_cls()
    model.hyperparameter_grid_options = SMOKE_GRID
    data_values, data_targets = orchestrator.get_transformed_data(split_key)
    np.random.seed(SPLIT_SEED)  # grid_fit splits with random_state=None -> global RNG
    model.grid_fit(data_values, data_targets)
    return model


def _metrics(model) -> list[dict]:
    """Per-permutation metrics in evaluation order, JSON-serialisable."""
    record = model.evaluator.evaluation_record
    out = []
    for iteration in sorted(record):
        entry = record[iteration]
        out.append(
            {
                "iteration": iteration,
                "accuracy": round(float(entry["accuracy"]), 6),
                "precision": round(float(entry["precision"]), 6),
                "recall": round(float(entry["recall"]), 6),
                "truePositives": int(entry["truePositives"]),
                "falsePositives": int(entry["falsePositives"]),
                "trueNegatives": int(entry["trueNegatives"]),
                "falseNegatives": int(entry["falseNegatives"]),
            }
        )
    return out


@pytest.fixture(scope="module")
def orchestrator() -> DataOrchestrator:
    assert DATA_PATH.is_file(), f"dataset missing: {DATA_PATH}"
    return DataOrchestrator(str(DATA_PATH), "csv", str(CONFIG_PATH))


@pytest.fixture(scope="module")
def results(orchestrator) -> dict[str, list[dict]]:
    return {key: _metrics(_train(cls, key, orchestrator)) for key, cls in MODELS.items()}


@pytest.mark.parametrize("key", list(MODELS))
def test_pipeline_runs_and_evaluates_every_permutation(results, key):
    metrics = results[key]
    assert len(metrics) == EXPECTED_PERMUTATIONS
    assert [m["iteration"] for m in metrics] == list(range(1, EXPECTED_PERMUTATIONS + 1))


@pytest.mark.parametrize("key", list(MODELS))
def test_metrics_are_well_formed(results, key):
    n_test = None
    for m in results[key]:
        for name in ("accuracy", "precision", "recall"):
            assert 0.0 <= m[name] <= 1.0, (name, m)
        total = m["truePositives"] + m["falsePositives"] + m["trueNegatives"] + m["falseNegatives"]
        n_test = n_test or total
        assert total == n_test, "confusion matrix should cover the whole test split every time"
        assert m["accuracy"] == pytest.approx(
            (m["truePositives"] + m["trueNegatives"]) / total, abs=1e-6
        )
    assert n_test == 2000, "10 000 rows with test_size=0.20"


@pytest.mark.parametrize("key", list(MODELS))
def test_best_model_beats_a_coin_flip(results, key):
    best = max(m["accuracy"] for m in results[key])
    # The ad-click features have no linear signal: sklearn logistic regression scores 0.650
    # (5-fold CV) against a 0.650 majority rate, so ~0.66 is the ceiling for this family (BL-23).
    assert best >= 0.5, f"{key}: best accuracy {best} — pipeline output is worse than the baseline"


def test_snapshot_matches(results):
    if os.environ.get("BASELINE_UPDATE") == "1" or not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(json.dumps(results, indent=2) + "\n")
        pytest.skip(f"baseline snapshot written to {SNAPSHOT_PATH.name}; re-run to compare")

    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert set(expected) == set(results), "model set changed — update the snapshot deliberately"
    for key in expected:
        assert len(expected[key]) == len(results[key]), key
        for exp, got in zip(expected[key], results[key], strict=True):
            for field, value in exp.items():
                if isinstance(value, float):
                    assert got[field] == pytest.approx(value, abs=TOLERANCE), (
                        key,
                        exp["iteration"],
                        field,
                    )
                else:
                    assert got[field] == value, (key, exp["iteration"], field)
