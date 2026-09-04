"""Turning a UCI file into a kernel the landmark search can be run on.

Everything between "a file on disk" and "a positive semi-definite matrix": which columns are
features, how they are scaled, how many rows are kept, and what bandwidth the kernel uses. It is
separated from the harness because these choices decide what the experiment is measuring, and they
should be readable and testable without running a search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# src/mllib/ml/projects/ -> repository root. The datasets live beside the code, not in the package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_UCI_DATA_DIR = REPOSITORY_ROOT / "data" / "uci"


@dataclass(frozen=True, slots=True)
class UciDatasetSpec:
    """Declares how to read one source file into a rows-by-features numeric matrix.

    ``label_position`` names the column to drop when the file is plain numeric text.
    ``loader_kind`` escapes to a named loader for files that need more than that; see `wdbc`, whose
    first two columns are an id and a non-numeric diagnosis.
    """

    name: str
    file_name: str
    delimiter: str = ","
    label_position: str | None = "last"
    loader_kind: str = "numeric"


DEFAULT_SMALL_UCI_SPECS = (
    UciDatasetSpec(name="SPECTF", file_name="SPECTF.test", label_position="first"),
    UciDatasetSpec(name="movement_libras", file_name="movement_libras.data", label_position="last"),
    UciDatasetSpec(
        name="wdbc",
        file_name="wdbc.data",
        label_position=None,
        loader_kind="wdbc",
    ),
)


def load_feature_matrix(spec: UciDatasetSpec, data_dir: Path | None = None) -> np.ndarray:
    """Read one dataset into a rows-by-features array of floats, dropping label and id columns."""
    path = (data_dir or DEFAULT_UCI_DATA_DIR) / spec.file_name

    if spec.loader_kind == "wdbc":
        # Column 0 is a patient id and column 1 is an M/B diagnosis. Keeping the id would not fail,
        # it would silently dominate the kernel, since it is far larger than any measurement.
        rows = [line.strip().split(",") for line in path.read_text().splitlines() if line.strip()]
        return np.asarray([[float(value) for value in row[2:]] for row in rows], dtype=float)

    matrix = np.loadtxt(path, delimiter=spec.delimiter)
    matrix = matrix if matrix.ndim == 2 else matrix[np.newaxis, :]
    if spec.label_position == "first":
        return np.asarray(matrix[:, 1:], dtype=float)
    if spec.label_position == "last":
        return np.asarray(matrix[:, :-1], dtype=float)
    return np.asarray(matrix, dtype=float)


def standardize_columns(features: np.ndarray) -> np.ndarray:
    """Center each feature and scale it to unit variance, leaving constant columns alone.

    Without this a feature measured in thousands sets the distances and therefore the kernel, and
    the landmark comparison becomes a comparison of measurement units.
    """
    means = features.mean(axis=0)
    standard_deviations = features.std(axis=0)
    standard_deviations = np.where(standard_deviations == 0.0, 1.0, standard_deviations)
    return (features - means) / standard_deviations


def downsample_rows(features: np.ndarray, max_rows: int, *, seed: int = 7) -> np.ndarray:
    """Take a seeded sample of at most ``max_rows`` rows, keeping their original order."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    row_count = features.shape[0]
    if row_count <= max_rows:
        return features

    rng = np.random.default_rng(seed)
    kept_rows = np.sort(rng.choice(row_count, size=max_rows, replace=False))
    return features[kept_rows]


def build_rbf_kernel(
    features: np.ndarray,
    gamma: float | None = None,
    *,
    gamma_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Build an RBF kernel ``exp(-gamma * ||x_i - x_j||^2)`` and report the bandwidth used.

    Without an explicit ``gamma`` the median heuristic sets it from the median squared distance
    between distinct points, so roughly half the pairs count as near. ``gamma_scale`` multiplies
    it, which is the knob that moves the kernel between a sharply decaying spectrum and a flat one.
    """
    if gamma_scale <= 0.0:
        raise ValueError("gamma_scale must be positive.")

    squared_norms = np.sum(features * features, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * (features @ features.T)
    # The expansion above can give small negative values on identical points; distances cannot.
    squared_distances = np.maximum(squared_distances, 0.0)

    if gamma is None:
        distinct_pairs = squared_distances[np.triu_indices(features.shape[0], k=1)]
        separated_pairs = distinct_pairs[distinct_pairs > 0.0]
        median_squared_distance = float(np.median(separated_pairs)) if separated_pairs.size else 1.0
        gamma = gamma_scale / (2.0 * median_squared_distance)

    kernel = np.exp(-gamma * squared_distances)
    np.fill_diagonal(kernel, 1.0)
    return kernel, float(gamma)


def svd_rank_k_residual(kernel: np.ndarray, landmark_count: int) -> float:
    """Trace of the error left by the best rank-k approximation of a psd kernel.

    This is what the eigenvectors achieve when they are free to be any direction rather than a
    column. No subset of landmarks can beat it, so it is the floor the certified optimum is measured
    against, and the gap between the two is the price of choosing landmarks at all.
    """
    eigenvalues = np.clip(np.linalg.eigvalsh(kernel), 0.0, None)[::-1]
    return float(np.sum(eigenvalues[landmark_count:]))
