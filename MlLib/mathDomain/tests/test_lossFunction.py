import numpy as np

from MlLib.mathDomain.lossFunction import MSE, MAE, PerceptronLoss, HingeLoss


def test_mse_loss_and_gradient():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([2.0, 0.0, 4.0])

    loss_fn = MSE()

    assert loss_fn.computeLoss(actual, predicted) == 2.0
    assert np.allclose(
        loss_fn.computeGradient(actual, predicted),
        np.array([2.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0]),
    )


def test_mae_loss_and_gradient():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([2.0, 0.0, 4.0])

    loss_fn = MAE()

    assert loss_fn.computeLoss(actual, predicted) == 4.0 / 3.0
    assert np.allclose(loss_fn.computeGradient(actual, predicted), np.array([1.0, -1.0, 1.0]))


def test_perceptron_loss_gradient_and_bias():
    actual = np.array([1.0, -1.0, 1.0])
    predicted = np.array([0.5, 0.1, -0.2])
    data_values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    loss_fn = PerceptronLoss()

    assert np.allclose(loss_fn.computeLoss(actual, predicted), np.array([0.0, 0.1, 0.2]))
    assert np.allclose(loss_fn.computeGradient(actual, predicted, data_values), np.array([2.0, 2.0]))
    assert loss_fn.computeBias(actual, predicted) == 0.0


def test_hinge_loss_gradient_and_bias():
    actual = np.array([1.0, -1.0, 1.0])
    predicted = np.array([0.5, 0.1, -0.2])
    data_values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    loss_fn = HingeLoss()

    assert np.allclose(loss_fn.computeLoss(actual, predicted), np.array([0.5, 1.1, 1.2]))
    assert np.allclose(loss_fn.computeGradient(actual, predicted, data_values), np.array([-2.0, -2.0]))
    assert loss_fn.computeBias(actual, predicted) == 0.0
