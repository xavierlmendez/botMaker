"""Tests for the ModelEvaluator base class (the template subclasses inherit)."""

import json

import numpy as np

from mllib.mlDomain.modelEvaluators.genericEvaluator import ModelEvaluator


class CountingEvaluator(ModelEvaluator):
    """Minimal concrete evaluator that relies on the *base* persistEvaluationRecord."""

    def evaluateModel(self):
        self.correctPredictions = int((self.testTargets == self.predictions).sum())
        self.truePositives = int(((self.testTargets == 1) & (self.predictions == 1)).sum())
        self.falsePositives = int(((self.testTargets == 0) & (self.predictions == 1)).sum())
        self.trueNegatives = int(((self.testTargets == 0) & (self.predictions == 0)).sum())
        self.falseNegatives = int(((self.testTargets == 1) & (self.predictions == 0)).sum())


def test_base_persist_writes_a_dict_record_with_counts():
    evaluator = CountingEvaluator()
    targets = np.array([1, 0, 1, 0])
    predictions = np.array([1, 1, 0, 0])
    metadata = {"learningRate": 0.01, "epoch": 3}  # unhashable: would break a set literal

    evaluator.updateTestingPredictionData(np.zeros((4, 1)), targets, predictions, metadata)

    record = evaluator.evaluationRecord[1]
    assert isinstance(record, dict)
    assert record == {
        "modelData": metadata,
        "correctPredictions": 2,
        "truePositives": 1,
        "falsePositives": 1,
        "trueNegatives": 1,
        "falseNegatives": 1,
    }
    json.dumps(evaluator.evaluationRecord)  # printable, like the subclass records


def test_base_persist_keeps_one_record_per_iteration():
    evaluator = CountingEvaluator()
    targets = np.array([1, 1])
    for _ in range(3):
        evaluator.updateTestingPredictionData(np.zeros((2, 1)), targets, targets, {})

    assert sorted(evaluator.evaluationRecord) == [1, 2, 3]
