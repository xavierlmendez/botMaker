"""Project logistic-regression subclasses must be usable without gridFit (F4)."""

import numpy as np
import pytest

from mllib.mlDomain.projectSpecificFiles.adClickPredictionLogReg import (
    LogisticRegression,
    LogisticRegressionWithAgeBinning,
)


@pytest.mark.parametrize(
    ("model_cls", "n_features"),
    [(LogisticRegression, 19), (LogisticRegressionWithAgeBinning, 26)],
)
def test_fit_and_predict_work_without_grid_fit(model_cls, n_features):
    rng = np.random.default_rng(0)
    X = rng.random((8, n_features))
    y = np.array([0, 1] * 4)

    model = model_cls()
    assert model.numWeights == n_features
    model.fit(X, y)  # AttributeError before F4: learningModel/lossFunction/epochs unset

    predictions = model.predictValues(X)
    assert predictions.shape == (8,)
    assert set(np.unique(predictions)).issubset({-1, 0, 1})  # computeClassification uses np.sign
