"""Tests for the ModelEvaluator base class (the template subclasses inherit)."""

import json

import numpy as np

from mllib.ml.evaluators.generic_evaluator import ModelEvaluator


class CountingEvaluator(ModelEvaluator):
    """Minimal concrete evaluator that relies on the *base* persistEvaluationRecord."""

    def evaluate_model(self):
        self.correct_predictions = int((self.test_targets == self.predictions).sum())
        self.true_positives = int(((self.test_targets == 1) & (self.predictions == 1)).sum())
        self.false_positives = int(((self.test_targets == 0) & (self.predictions == 1)).sum())
        self.true_negatives = int(((self.test_targets == 0) & (self.predictions == 0)).sum())
        self.false_negatives = int(((self.test_targets == 1) & (self.predictions == 0)).sum())


def test_base_persist_writes_a_dict_record_with_counts():
    evaluator = CountingEvaluator()
    targets = np.array([1, 0, 1, 0])
    predictions = np.array([1, 1, 0, 0])
    metadata = {"learningRate": 0.01, "epoch": 3}  # unhashable: would break a set literal

    evaluator.update_testing_prediction_data(np.zeros((4, 1)), targets, predictions, metadata)

    record = evaluator.evaluation_record[1]
    assert isinstance(record, dict)
    assert record == {
        "modelData": metadata,
        "correctPredictions": 2,
        "truePositives": 1,
        "falsePositives": 1,
        "trueNegatives": 1,
        "falseNegatives": 1,
    }
    json.dumps(evaluator.evaluation_record)  # printable, like the subclass records


def test_base_persist_keeps_one_record_per_iteration():
    evaluator = CountingEvaluator()
    targets = np.array([1, 1])
    for _ in range(3):
        evaluator.update_testing_prediction_data(np.zeros((2, 1)), targets, targets, {})

    assert sorted(evaluator.evaluation_record) == [1, 2, 3]
