"""Confusion-matrix counts and derived metrics must agree with scikit-learn (BL-22)."""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from mllib.ml.evaluators.generic_evaluator import (
    DecisionTreeModelEvaluator,
    LogisticRegressionModelEvaluator,
)

TARGETS = np.array([1, 1, 1, 0, 0, 0, 1, 0])
PREDICTIONS = np.array([1, 0, 1, 1, 0, 0, 1, 1])  # 3 TP, 2 FP, 2 TN, 1 FN


@pytest.mark.parametrize(
    "evaluator_cls", [LogisticRegressionModelEvaluator, DecisionTreeModelEvaluator]
)
def test_counts_match_sklearn(evaluator_cls):
    evaluator = evaluator_cls()
    evaluator.updateTestingPredictionData(np.zeros((8, 1)), TARGETS, PREDICTIONS, {})

    tn, fp, fn, tp = confusion_matrix(TARGETS, PREDICTIONS).ravel()
    assert (evaluator.truePositives, evaluator.falsePositives) == (tp, fp) == (3, 2)
    assert (evaluator.trueNegatives, evaluator.falseNegatives) == (tn, fn) == (2, 1)
    assert evaluator.precision == pytest.approx(precision_score(TARGETS, PREDICTIONS))
    assert evaluator.recall == pytest.approx(recall_score(TARGETS, PREDICTIONS))


def test_minus_one_labels_count_as_negative():
    evaluator = LogisticRegressionModelEvaluator()
    targets = np.array([1, -1, 1, -1])
    predictions = np.array([1, 1, -1, -1])  # sign-classifier output

    evaluator.updateTestingPredictionData(np.zeros((4, 1)), targets, predictions, {})

    assert (evaluator.truePositives, evaluator.falsePositives) == (1, 1)
    assert (evaluator.trueNegatives, evaluator.falseNegatives) == (1, 1)
