"""Nyström landmark selection as an implicit subset-search problem.

The Nyström approximation of a positive semi-definite kernel ``K`` from a set of landmark columns
``S`` is ``K[:, S] K[S, S]^+ K[S, :]``, and the error it leaves is the trace of the residual. That
trace equals the squared Frobenius residual of column subset selection applied to ``K^{1/2}``, which
is what lets a column-subset-selection search choose landmarks without modification.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence

import numpy as np

from mllib.math.graph.abstract_graph_problem import AbstractGraphProblem
from mllib.math.search_cost_function import SearchCostFunction
from mllib.math.secular_equation import top_eigenvalues_of_rank_one_downdate

LandmarkState = tuple[int, ...]
LandmarkAction = int

# Below this multiple of the largest singular value a direction counts as numerically absent.
RANK_TOLERANCE = 1e-12
# Below this multiple of the largest eigenvalue a mode of the kernel is rounding: the numeric rank.
EIGENVALUE_TOLERANCE = 1e-12
# A residual diagonal entry at or below this is treated as fully explained already (goal depth).
PIVOT_TOLERANCE = 1e-12
# A column whose residual norm is at or below this multiple of sqrt(tr K) is an explained column:
# already in the span of the chosen landmarks. The goal-depth rule above is absolute; unifying the
# two is BL-30.
EXPLAINED_COLUMN_TOLERANCE = 1e-12


class NystromLandmarkProblem(AbstractGraphProblem[LandmarkState, LandmarkAction]):
    """The subset graph for choosing a fixed number of kernel landmarks.

    A state is the ascending tuple of chosen column indices, and a successor appends one index
    greater than the last. Canonical order is what makes each subset exactly one node: without it
    the same set of landmarks would be reached by every permutation of its members.

    The root decomposition ``K = V D Vᵀ`` is done once here and kept: ``eigenvalues`` (descending,
    clipped at zero), ``eigenvectors``, the ``retained_rank`` r and the ``reduced_coordinates``
    ``D_r^{1/2} V_rᵀ`` of shape (r, n), which are ``K^{1/2}`` in its own eigenbasis with every
    column inner product preserved. The bound works in those r dimensions instead of n.

    ``spectrum_mass_tolerance`` δ truncates the spectrum the bound sees: r is the smallest rank
    whose dropped mass is within δ·tr(K), never below one and never above the numeric rank. The
    bound computed on the truncated kernel stays admissible for the true objective (D-26); the
    objective itself, ``kernel_matrix`` and every goal cost, is never truncated. δ = 0 keeps every
    mode above rounding.

    ``forced`` and ``forbidden`` make the search conditional: every subset must contain the forced
    columns and none of the forbidden ones, with ``landmark_count`` unchanged. The search starts at
    the forced columns, ``initial_state()`` is their sorted tuple, and successors add one *free*
    column (neither forced nor forbidden) above the largest free column already chosen, so each
    admissible subset is still exactly one node and states stay canonical ascending tuples. The
    bound needs no change: removing candidates only raises the true optimum, and starting deeper
    is a subtree of the same tree, so what is admissible for the whole tree is admissible here. The
    2n conditional solves of the necessity margins (for every column, the best subset with it and
    the best without it) are this knob; ``constraints`` states it for the record, empty by default.
    The constraints shape the tree only: ``goal_cost`` and ``lower_bound`` stay defined on every
    subset, so a selector that does not read the tree can still price an unconstrained choice.
    """

    def __init__(
        self,
        kernel_matrix: np.ndarray,
        landmark_count: int,
        spectrum_mass_tolerance: float = 0.0,
        *,
        forced: tuple[int, ...] = (),
        forbidden: frozenset[int] = frozenset(),
    ):
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
        if not 0.0 <= spectrum_mass_tolerance < 1.0:
            raise ValueError("spectrum_mass_tolerance must lie in [0, 1).")
        self.landmark_count = landmark_count
        self.spectrum_mass_tolerance = spectrum_mass_tolerance

        column_count = self.kernel_matrix.shape[0]
        self.forced = tuple(sorted(int(index) for index in forced))
        self.forbidden = frozenset(int(index) for index in forbidden)
        if len(set(self.forced)) != len(self.forced):
            raise ValueError("forced columns must be distinct.")
        if any(index < 0 or index >= column_count for index in (*self.forced, *self.forbidden)):
            raise ValueError("forced and forbidden columns must be in range.")
        if self.forbidden & set(self.forced):
            raise ValueError("a column cannot be both forced and forbidden.")
        if len(self.forced) > landmark_count:
            raise ValueError("more forced columns than landmark_count.")
        # The free columns are the only ones a successor may add, in ascending order.
        self._free_columns = tuple(
            index
            for index in range(column_count)
            if index not in self.forbidden and index not in self.forced
        )
        if len(self._free_columns) < landmark_count - len(self.forced):
            raise ValueError("too few columns remain outside forbidden to reach landmark_count.")

        # K^{1/2} exists because K is psd; eigenvalues are clipped because a psd matrix built from
        # data can carry small negative values from rounding, and their square roots are not real.
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_matrix)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        self.eigenvectors = eigenvectors[:, order]
        self.retained_rank = self._retained_rank()
        self.dropped_mass = float(np.sum(self.eigenvalues[self.retained_rank :]))

        retained_sqrt = np.sqrt(self.eigenvalues[: self.retained_rank])
        retained_vectors = self.eigenvectors[:, : self.retained_rank]
        self.reduced_coordinates = retained_sqrt[:, None] * retained_vectors.T
        self.kernel_sqrt = retained_vectors @ (retained_sqrt[:, None] * retained_vectors.T)

    def _retained_rank(self) -> int:
        """The smaller of the numeric rank and the rank the dropped-mass tolerance allows."""
        largest = float(self.eigenvalues[0])
        numeric_rank = int(np.sum(self.eigenvalues > EIGENVALUE_TOLERANCE * largest))
        # tail[r] is the mass dropped by keeping r modes; the smallest admissible r wins.
        tail = np.concatenate([np.cumsum(self.eigenvalues[::-1])[::-1], [0.0]])
        allowed = self.spectrum_mass_tolerance * float(np.sum(self.eigenvalues))
        mass_rank = int(np.argmax(tail <= allowed))
        return max(1, min(numeric_rank, mass_rank))

    @property
    def constraints(self) -> dict[str, list[int]]:
        """The forced and forbidden columns, for the record; empty when the problem has neither."""
        if not self.forced and not self.forbidden:
            return {}
        return {"forced": list(self.forced), "forbidden": sorted(self.forbidden)}

    def initial_state(self) -> LandmarkState:
        return self.forced

    def is_goal(self, state: LandmarkState) -> bool:
        self._validate_state(state)
        return len(state) == self.landmark_count

    def successors(self, state: LandmarkState) -> Iterable[tuple[LandmarkAction, LandmarkState]]:
        self._validate_state(state)
        still_needed = self.landmark_count - len(state)
        if still_needed <= 0:
            return []

        free = self._free_columns
        # Candidates are the free columns above the largest free column already chosen; the forced
        # columns sit anywhere in the state and never move this cursor. Without constraints ``free``
        # is every column and this is ``range(state[-1] + 1, ...)`` exactly.
        chosen_free = [index for index in state if index not in self.forced]
        first = 0 if not chosen_free else bisect_right(free, chosen_free[-1])
        # Stop where too few free columns remain to finish the selection: dead ends.
        last = len(free) - still_needed
        if first > last:
            return []
        if not self.forced:
            return [(index, (*state, index)) for index in free[first : last + 1]]
        return [(index, tuple(sorted((*state, index)))) for index in free[first : last + 1]]

    def _validate_state(self, state: LandmarkState) -> None:
        column_count = self.kernel_matrix.shape[0]
        if len(state) > self.landmark_count:
            raise ValueError("State contains more landmarks than landmark_count.")
        if any(index < 0 or index >= column_count for index in state):
            raise ValueError("State contains out-of-range landmark indices.")
        if tuple(sorted(state)) != state or len(set(state)) != len(state):
            raise ValueError("State indices must be unique and in ascending order.")
        # The constraints are not checked here: they shape the tree (``initial_state`` and
        # ``successors`` never produce a state outside them), while the objective is defined on
        # every subset, and a selector that ignores the constraints may still price its choice.


class NystromCssCostFunction(SearchCostFunction[LandmarkState, LandmarkAction]):
    """The Nyström residual trace, with the spectral bound that makes the search a proof.

    "Css" is column subset selection: choosing landmarks for ``K`` is selecting columns of
    ``K^{1/2}``, and this cost function is that selection's objective and bound.

    ``goal_cost`` is the residual trace the chosen landmarks actually leave. ``lower_bound`` asks
    what the best possible remaining choices could achieve: after projecting the chosen columns out
    of ``K^{1/2}``, no set of ``r`` further columns can remove more energy than the top ``r``
    eigenvalues of what is left, because columns are a restricted choice of directions and the
    eigenvectors are the unrestricted best. Subtracting them therefore under-states the true
    remaining error, which is exactly what admissibility requires.

    ``lower_bound`` computes that one state at a time with a fresh decomposition: it is the oracle
    the fast path is tested against. ``lower_bounds`` prices all of a parent's children from the
    parent's own decomposition, which is where the search actually spends its time.
    """

    def __init__(self, problem: NystromLandmarkProblem):
        self.problem = problem

    def lower_bound(self, state: LandmarkState) -> float:
        """The bound for one state, by a full decomposition of what remains: the oracle."""
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
        # of two near-equal large numbers can land just below zero and empty the whole frontier.
        # Which side of zero it lands on is rounding, and differs between BLAS implementations.
        return max(remaining_energy - best_possible_completion, 0.0)

    def goal_cost(self, state: LandmarkState) -> float:
        if not self.problem.is_goal(state):
            raise ValueError("goal_cost requires a goal state with exactly landmark_count indices.")
        return max(float(np.trace(self._residual_kernel(state))), 0.0)

    def lower_bounds(
        self, parent: LandmarkState, successors: Sequence[tuple[LandmarkAction, LandmarkState]]
    ) -> Sequence[float]:
        """Score every child of ``parent`` together, from one decomposition of the parent.

        Goal-depth children share the parent's residual kernel: adding column ``j`` to ``S``
        reduces the residual trace by exactly ``||R[:, j]||² / R[j, j]``, so one Schur complement
        prices them all (D-24). Children above goal depth share the parent's spectrum: each child's
        spectrum is a rank-one downdate of it, solved through the secular equation, so one
        eigendecomposition per parent replaces one per child (BL-27).

        A child is the parent plus one column, wherever that column sorts (a forced column can sit
        above it). Successors that do not extend the parent fall back to the oracle, one at a time.
        """
        if not successors:
            return []
        added = self._added_columns(parent, successors)
        if added is None:
            return super().lower_bounds(parent, successors)
        added_columns = np.fromiter(added, dtype=int, count=len(successors))
        if len(parent) + 1 == self.problem.landmark_count:
            return self._goal_depth_bounds(parent, added_columns)
        return self._downdated_bounds(parent, added_columns)

    @staticmethod
    def _added_columns(
        parent: LandmarkState, successors: Sequence[tuple[LandmarkAction, LandmarkState]]
    ) -> list[int] | None:
        """The one column each child adds to ``parent``, or ``None`` if some child is not that."""
        added = []
        parent_set = set(parent)
        for _, child in successors:
            if len(child) != len(parent) + 1:
                return None
            extra = [index for index in child if index not in parent_set]
            if len(extra) != 1:
                return None
            added.append(extra[0])
        return added

    def _goal_depth_bounds(self, parent: LandmarkState, added_columns: np.ndarray) -> list[float]:
        """The exact residual trace of every complete child, from the parent's Schur complement."""
        residual_kernel = self._residual_kernel(parent)
        residual_diagonal = np.diag(residual_kernel)[added_columns]
        # A column already in the parent's span has nothing left unexplained and adds nothing.
        unexplained = residual_diagonal > PIVOT_TOLERANCE
        trace_reductions = np.zeros(added_columns.size)
        trace_reductions[unexplained] = (
            np.sum(residual_kernel[:, added_columns[unexplained]] ** 2, axis=0)
            / residual_diagonal[unexplained]
        )
        return np.maximum(float(np.trace(residual_kernel)) - trace_reductions, 0.0).tolist()

    def _downdated_bounds(self, parent: LandmarkState, added_columns: np.ndarray) -> list[float]:
        """Every child's bound from the parent spectrum: one ``eigh``, then one downdate per child.

        In the reduced coordinates ``Y_r`` the parent's residual Gram has the same spectrum as
        ``H = D - Z Zᵀ`` with ``Z = D^{1/2} Q`` and ``Q`` an orthonormal basis of the chosen
        columns.
        A child extends ``Q`` by the unit residual ``q_j`` of its column, so its spectrum is that of
        ``H - z_j z_jᵀ`` with ``z_j = D^{1/2} q_j``, a rank-one downdate: in ``H``'s eigenbasis,
        ``diag(λ) - w_j w_jᵀ`` with ``w_j = Uᵀ z_j``. The child's remaining energy is
        ``tr(H) - ||z_j||²`` and its best possible completion is the sum of the top eigenvalues of
        that downdate.
        """
        problem = self.problem
        coordinates = problem.reduced_coordinates
        eigenvalue_sqrt = np.sqrt(problem.eigenvalues[: problem.retained_rank])
        still_needed_after_child = problem.landmark_count - len(parent) - 1

        basis = self._orthonormal_basis(coordinates[:, list(parent)]) if parent else None
        if basis is None:
            parent_gram = np.diag(eigenvalue_sqrt**2)
            residuals = coordinates[:, added_columns]
        else:
            scaled_basis = eigenvalue_sqrt[:, None] * basis
            parent_gram = np.diag(eigenvalue_sqrt**2) - scaled_basis @ scaled_basis.T
            candidates = coordinates[:, added_columns]
            residuals = candidates - basis @ (basis.T @ candidates)
        parent_spectrum, parent_vectors = np.linalg.eigh(parent_gram)
        # eigh returns ascending; the secular solver wants descending, and rounding can dip below 0.
        parent_spectrum = np.clip(parent_spectrum[::-1], 0.0, None)
        parent_vectors = parent_vectors[:, ::-1]
        parent_trace = float(np.sum(parent_spectrum))

        residual_norms = np.linalg.norm(residuals, axis=0)
        explained = residual_norms <= EXPLAINED_COLUMN_TOLERANCE * np.sqrt(
            float(np.sum(eigenvalue_sqrt**2))
        )
        unit_residuals = np.where(
            explained[None, :], 0.0, residuals / np.where(explained, 1.0, residual_norms)[None, :]
        )
        downdates = eigenvalue_sqrt[:, None] * unit_residuals
        weights = parent_vectors.T @ downdates

        remaining_energy = parent_trace - np.sum(downdates * downdates, axis=0)
        # Beyond the retained rank every eigenvalue is zero, so asking for more adds nothing.
        top_count = min(still_needed_after_child, problem.retained_rank)
        best_possible_completion = np.sum(
            top_eigenvalues_of_rank_one_downdate(parent_spectrum, weights, top_count), axis=0
        )
        return np.maximum(remaining_energy - best_possible_completion, 0.0).tolist()

    @staticmethod
    def _remaining_spectrum(remaining_sqrt: np.ndarray) -> np.ndarray:
        """Ascending eigenvalues of the energy left in ``remaining_sqrt``: the oracle's material.

        The top ``r`` of these are what the best conceivable ``r`` further landmarks could remove.
        A full n x n eigendecomposition per call, O(n³): this is what ``_downdated_bounds`` replaces
        on the search path, and it is kept as the seam the clamp tests inject through.
        """
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
