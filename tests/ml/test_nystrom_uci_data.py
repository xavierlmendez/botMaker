"""Loading, scaling, subsampling and kernelizing the committed UCI files.

These steps decide what the landmark experiment is measuring, so each is checked on data whose
answer is known by construction, plus one check per real file that its declared shape is what the
loader actually produces.
"""

from pathlib import Path

import numpy as np
import pytest

from mllib.ml.projects.nystrom_uci_data import (
    DEFAULT_SMALL_UCI_SPECS,
    UciDatasetSpec,
    build_rbf_kernel,
    downsample_rows,
    load_feature_matrix,
    standardize_columns,
    svd_rank_k_residual,
)

POINTS = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])


def test_a_numeric_file_drops_its_label_column_from_either_end(tmp_path: Path):
    (tmp_path / "first.data").write_text("1,0.2,0.3\n0,0.5,0.6\n")
    (tmp_path / "last.data").write_text("0.2,0.3,1\n0.5,0.6,0\n")

    from_first = load_feature_matrix(
        UciDatasetSpec(name="f", file_name="first.data", label_position="first"), tmp_path
    )
    from_last = load_feature_matrix(
        UciDatasetSpec(name="l", file_name="last.data", label_position="last"), tmp_path
    )

    assert np.allclose(from_first, [[0.2, 0.3], [0.5, 0.6]])
    assert np.allclose(from_last, [[0.2, 0.3], [0.5, 0.6]])


def test_the_wdbc_loader_drops_both_the_patient_id_and_the_diagnosis(tmp_path: Path):
    (tmp_path / "wdbc.data").write_text("1001,M,1.0,2.0,3.0\n1002,B,2.0,3.0,4.0\n")

    features = load_feature_matrix(
        UciDatasetSpec(name="wdbc", file_name="wdbc.data", label_position=None, loader_kind="wdbc"),
        tmp_path,
    )

    # The id is five digits and the measurements are single digits; keeping it would set the kernel.
    assert features.shape == (2, 3)
    assert np.allclose(features[0], [1.0, 2.0, 3.0])


def test_a_file_with_no_label_column_keeps_every_column(tmp_path: Path):
    (tmp_path / "plain.data").write_text("1.0,2.0\n3.0,4.0\n")

    features = load_feature_matrix(
        UciDatasetSpec(name="p", file_name="plain.data", label_position=None), tmp_path
    )

    assert np.allclose(features, [[1.0, 2.0], [3.0, 4.0]])


def test_standardizing_centers_each_column_and_leaves_constant_ones_untouched():
    values = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])

    standardized = standardize_columns(values)

    assert np.allclose(standardized.mean(axis=0), [0.0, 0.0])
    assert np.allclose(standardized[:, 0].std(), 1.0)
    assert np.allclose(standardized[:, 1], [0.0, 0.0, 0.0])  # constant column, not a division by 0


def test_downsampling_is_seeded_keeps_row_order_and_is_a_no_op_when_small_enough():
    values = np.arange(40.0).reshape(20, 2)

    sampled = downsample_rows(values, max_rows=5, seed=3)
    repeat = downsample_rows(values, max_rows=5, seed=3)

    assert sampled.shape == (5, 2)
    assert np.array_equal(sampled, repeat)
    assert np.array_equal(sampled[:, 0], np.sort(sampled[:, 0]))  # original order preserved
    assert np.array_equal(downsample_rows(values, max_rows=50), values)


def test_downsampling_rejects_an_empty_budget():
    with pytest.raises(ValueError, match="max_rows must be positive"):
        downsample_rows(np.zeros((3, 2)), max_rows=0)


def test_the_rbf_kernel_is_symmetric_positive_semi_definite_with_a_unit_diagonal():
    kernel, gamma = build_rbf_kernel(POINTS)

    assert kernel.shape == (3, 3)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), np.ones(3))
    assert np.min(np.linalg.eigvalsh(kernel)) > -1e-9
    assert gamma > 0.0


def test_the_bandwidth_scale_multiplies_the_median_heuristic():
    _, base = build_rbf_kernel(POINTS)
    _, widened = build_rbf_kernel(POINTS, gamma_scale=4.0)

    assert widened == pytest.approx(4.0 * base)


def test_an_explicit_bandwidth_overrides_the_heuristic():
    kernel, gamma = build_rbf_kernel(POINTS, gamma=0.5)

    assert gamma == pytest.approx(0.5)
    assert kernel[0, 1] == pytest.approx(np.exp(-0.5 * 2.0))


def test_a_non_positive_bandwidth_scale_is_rejected():
    with pytest.raises(ValueError, match="gamma_scale must be positive"):
        build_rbf_kernel(POINTS, gamma_scale=0.0)


def test_the_rank_k_residual_is_the_tail_of_the_eigenvalue_spectrum():
    kernel = np.diag([4.0, 3.0, 2.0, 1.0])

    assert svd_rank_k_residual(kernel, 2) == pytest.approx(3.0)
    assert svd_rank_k_residual(kernel, 4) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("dataset_name", "expected_rows", "expected_features"),
    [("SPECTF", 187, 44), ("movement_libras", 360, 90), ("wdbc", 569, 30)],
)
def test_every_committed_dataset_loads_at_the_shape_its_readme_declares(
    uci_data_dir: Path, dataset_name: str, expected_rows: int, expected_features: int
):
    spec = next(s for s in DEFAULT_SMALL_UCI_SPECS if s.name == dataset_name)

    features = load_feature_matrix(spec, uci_data_dir)

    assert features.shape == (expected_rows, expected_features)
    assert np.isfinite(features).all()
