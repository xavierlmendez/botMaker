"""Nyström landmark selection as an implicit subset-search problem.

The Nyström approximation of a positive semi-definite kernel ``K`` from a set of landmark columns
``S`` is ``K[:, S] K[S, S]^+ K[S, :]``, and the error it leaves is the trace of the residual. That
trace equals the squared Frobenius residual of column subset selection applied to ``K^{1/2}``, which
is what lets a column-subset-selection search choose landmarks without modification.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.search_cost_function import SearchCostFunction

LandmarkState = tuple[int, ...]
LandmarkAction = int

# Below this multiple of the largest singular value a direction counts as numerically absent.
RANK_TOLERANCE = 1e-12
# A residual diagonal entry at or below this is treated as fully explained already.
PIVOT_TOLERANCE = 1e-12


class NystromLandmarkProblem(AbstractGraphProblem[LandmarkState, LandmarkAction]):
    """The subset graph for choosing a fixed number of kernel landmarks.

    A state is the ascending tuple of chosen column indices, and a successor appends one index
    greater than the last. Canonical order is what makes each subset exactly one node: without it
    the same set of landmarks would be reached by every permutation of its members.
    """

    def __init__(self, kernel_matrix: np.ndarray, landmark_count: int):
        self.kernel_matrix = np.asarray(kernel_matrix, dtype=float)
        if self.kernel_matrix.ndim != 2:
            raise ValueError("kernel_matrix must be a 2D array.")
        if self.kernel_matrix.shape[0] != self.kernel_matrix.shape[1]:
            raise ValueError("kernel_matrix must be square.")
        if landmark_count <= 0:
            raise ValueError("landmark_count must be positive.")
        if landmark_count > self.kernel_matrix.shape[0]:
            raise ValueError("landmark_count cannot exceed the number of data points.")
        if not np.allclose(self.kernel_matrix, self.kernel_matrix.T, atol=1e-10):
            raise ValueError("kernel_matrix must be symmetric.")

        # K^{1/2} exists because K is psd; eigenvalues are clipped because a psd matrix built from
        # data can carry small negative values from rounding, and their square roots are not real.
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        self.kernel_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        self.landmark_count = landmark_count

    def initial_state(self) -> LandmarkState:
        return ()

    def is_goal(self, state: LandmarkState) -> bool:
        self._validate_state(state)
        return len(state) == self.landmark_count

    def successors(self, state: LandmarkState) -> Iterable[tuple[LandmarkAction, LandmarkState]]:
        self._validate_state(state)
        still_needed = self.landmark_count - len(state)
        if still_needed <= 0:
            return []

        column_count = self.kernel_matrix.shape[0]
        first_candidate = 0 if not state else state[-1] + 1
        # Stop where too few columns remain to finish the selection: those branches are dead ends.
        last_candidate = column_count - still_needed
        if first_candidate > last_candidate:
            return []

        return [(index, (*state, index)) for index in range(first_candidate, last_candidate + 1)]

    def _validate_state(self, state: LandmarkState) -> None:
        column_count = self.kernel_matrix.shape[0]
        if len(state) > self.landmark_count:
            raise ValueError("State contains more landmarks than landmark_count.")
        if any(index < 0 or index >= column_count for index in state):
            raise ValueError("State contains out-of-range landmark indices.")
        if tuple(sorted(state)) != state or len(set(state)) != len(state):
            raise ValueError("State indices must be unique and in ascending order.")


class NystromCssCostFunction(SearchCostFunction[LandmarkState, LandmarkAction]):
    """The Nyström residual trace, with the spectral bound that makes the search a proof.

    "Css" is column subset selection: choosing landmarks for ``K`` is selecting columns of
    ``K^{1/2}``, and this cost function is that selection's objective and bound.

    ``goal_cost`` is the error the chosen landmarks actually leave. ``lower_bound`` asks what the
    best possible remaining choices could achieve: after projecting the chosen columns out of
    ``K^{1/2}``, no set of ``r`` further columns can remove more energy than the top ``r``
    eigenvalues of what is left, because columns are a restricted choice of directions and the
    eigenvectors are the unrestricted best. Subtracting them therefore under-states the true
    remaining error, which is exactly what admissibility requires.
    """

    def __init__(self, problem: NystromLandmarkProblem):
        self.problem = problem

    def lower_bound(self, state: LandmarkState) -> float:
        still_needed = self.problem.landmark_count - len(state)
        if still_needed < 0:
            raise ValueError("State contains more landmarks than configured landmark_count.")
        if still_needed == 0:
            return self.goal_cost(state)

        # What is left of K^{1/2} once the chosen columns are projected out.
        remaining_sqrt = self._project_out(
            self._basis_of_selected_columns(state), self.problem.kernel_sqrt
        )
        remaining_energy = float(np.sum(remaining_sqrt * remaining_sqrt))
        remaining_spectrum = self._remaining_spectrum(remaining_sqrt)
        best_possible_completion = float(np.sum(remaining_spectrum[-still_needed:]))
        # Clamped because the subtraction cancels at scale: on a badly scaled kernel the difference
        # of two near-equal large numbers can land just below zero and prune the whole fringe.
        # Which side of zero it lands on is rounding, and differs between BLAS implementations.
        return max(remaining_energy - best_possible_completion, 0.0)

    def goal_cost(self, state: LandmarkState) -> float:
        if not self.problem.is_goal(state):
            raise ValueError("goal_cost requires a goal state with exactly landmark_count indices.")
        return max(float(np.trace(self._residual_kernel(state))), 0.0)

    def lower_bounds(
        self, parent: LandmarkState, successors: Sequence[tuple[LandmarkAction, LandmarkState]]
    ) -> Sequence[float]:
        """Score every child of ``parent`` together where the children are complete selections.

        Adding column ``j`` to a selection ``S`` reduces the residual trace by exactly
        ``||R[:, j]||² / R[j, j]``, where ``R`` is the residual kernel after ``S``. So one Schur
        complement of the parent prices all of its goal-depth children in a single vectorised pass,
        instead of one pseudo-inverse per child. This is the identity the greedy trace selector is
        built on, applied to the search.

        Children that are not complete fall back to the per-child bound: what they would share is
        the eigendecomposition of the residual, and sharing that is the rank-one downdate of BL-27.
        """
        if not successors:
            return []
        children_are_goals = len(parent) + 1 == self.problem.landmark_count
        children_extend_parent = all(child[:-1] == parent for _, child in successors)
        if not (children_are_goals and children_extend_parent):
            return super().lower_bounds(parent, successors)

        residual_kernel = self._residual_kernel(parent)
        added_columns = np.fromiter((child[-1] for _, child in successors), dtype=int)
        residual_diagonal = np.diag(residual_kernel)[added_columns]
        # A column already in the parent's span has nothing left unexplained and adds nothing.
        unexplained = residual_diagonal > PIVOT_TOLERANCE
        trace_reductions = np.zeros(added_columns.size)
        trace_reductions[unexplained] = (
            np.sum(residual_kernel[:, added_columns[unexplained]] ** 2, axis=0)
            / residual_diagonal[unexplained]
        )
        return np.maximum(float(np.trace(residual_kernel)) - trace_reductions, 0.0).tolist()

    @staticmethod
    def _remaining_spectrum(remaining_sqrt: np.ndarray) -> np.ndarray:
        """Ascending eigenvalues of the energy left in ``remaining_sqrt``, the bound's raw material.

        The top ``r`` of these are what the best conceivable ``r`` further landmarks could remove.
        """
        # TODO(BL-27): a full n x n eigendecomposition per child, O(n^3), which caps the search near
        # n = 40. The AAAI-15 method downdates the parent's spectrum by rank one instead, and this
        # is the method that downdate replaces.
        return np.linalg.eigvalsh(remaining_sqrt @ remaining_sqrt.T)

    def _residual_kernel(self, state: LandmarkState) -> np.ndarray:
        """The kernel left unexplained by ``state``: ``K - K[:, S] K[S, S]^+ K[S, :]``."""
        if not state:
            return self.problem.kernel_matrix
        landmarks = list(state)
        kernel = self.problem.kernel_matrix
        landmark_columns = kernel[:, landmarks]
        landmark_subkernel = kernel[np.ix_(landmarks, landmarks)]
        # Pseudo-inverse, not inverse: repeated or dependent landmarks make K[S, S] singular.
        subkernel_pseudo_inverse = np.linalg.pinv(landmark_subkernel, hermitian=True)
        return kernel - landmark_columns @ subkernel_pseudo_inverse @ landmark_columns.T

    @staticmethod
    def _project_out(basis: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Remove from ``matrix`` everything the orthonormal ``basis`` already explains."""
        if basis.size == 0:
            return matrix
        return matrix - basis @ (basis.T @ matrix)

    @staticmethod
    def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros((matrix.shape[0], 0))

        left_singular_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        threshold = singular_values[0] * RANK_TOLERANCE if singular_values.size else 0.0
        rank = int(np.sum(singular_values > threshold))
        return left_singular_vectors[:, :rank]

    def _basis_of_selected_columns(self, state: LandmarkState) -> np.ndarray:
        """An orthonormal basis for the span of the chosen columns of K^{1/2}."""
        if not state:
            return np.zeros((self.problem.kernel_sqrt.shape[0], 0))
        return self._orthonormal_basis(self.problem.kernel_sqrt[:, list(state)])
