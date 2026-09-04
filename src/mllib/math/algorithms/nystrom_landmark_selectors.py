"""Deterministic landmark selectors, and the contract every selector answers to.

Each selector returns a ``SearchResult`` whose ``cost`` is the Nyström residual trace of what it
chose, so results from different selectors are directly comparable to the A* optimum. Two kinds are
kept apart on purpose, because conflating them is how an optimality gap gets overstated (D-22):

* **Published baselines** whose distance from the optimum is the thing being measured — here the
  deterministic two, ``GreedyResidualTraceLandmarkSelector`` (greedy Nyström, Farahat et al.
  AISTATS 2011, which is the ``f = u`` greedy of Wan & Schweitzer IJCAI 2021) and
  ``PivotedCholeskyLandmarkSelector``. RPCholesky and a uniform draw are the other two, in
  ``nystrom_randomized_selectors``; the harness's ``PUBLISHED_BASELINES`` names all four.
* **Instrumented heuristics** that exist only to explain the search —
  ``GreedyLowerBoundLandmarkSelector`` greedily minimizes A*'s own bound, a lookahead rule with no
  published counterpart. It must never be quoted as "greedy".

Randomized selectors live in ``nystrom_randomized_selectors``, because a single draw of one is not a
number worth reporting and they arrive with the machinery to average over seeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from mllib.math.algorithms.a_star_search import AStarSearch, SearchResult
from mllib.math.graph.nystrom_landmark_problem import (
    PIVOT_TOLERANCE,
    LandmarkState,
    NystromCssCostFunction,
    NystromLandmarkProblem,
)


class AbstractNystromLandmarkSelector(ABC):
    """Contract for anything that chooses landmarks for the same Nyström problem.

    ``nodes_expanded`` on the returned result is the selector's own unit of work, so the price of a
    selection is visible next to its quality: candidate evaluations for a greedy rule, subsets drawn
    for a sampler, states expanded for a search.
    """

    name: str

    @abstractmethod
    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        """Choose ``problem.landmark_count`` landmarks and report the cost of the choice."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AStarLandmarkSelector(AbstractNystromLandmarkSelector):
    """The certified optimum: the selection every other selector is measured against."""

    name: str = "astar"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        return AStarSearch(problem, cost_function).run()


@dataclass(frozen=True, slots=True)
class GreedyResidualTraceLandmarkSelector(AbstractNystromLandmarkSelector):
    """Greedy Nyström: at each step add the column that removes the most residual trace.

    On the residual kernel ``R`` left by the current selection, adding column ``j`` reduces
    ``tr(R)`` by exactly ``||R[:, j]||² / R[j, j]``, so the rule needs no trial evaluations. This
    is the standard greedy baseline (Farahat, Ghodsi & Kamel, AISTATS 2011).
    """

    name: str = "greedy_trace"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        residual_kernel = problem.kernel_matrix.copy()
        column_count = residual_kernel.shape[0]
        selected: list[int] = []
        scored_candidates = 0

        for _ in range(problem.landmark_count):
            residual_diagonal = np.diag(residual_kernel)
            # A column is worth scoring if something of it is still unexplained and it is not
            # already chosen; a chosen column's pivot is eliminated, so the second test is a guard.
            eligible = residual_diagonal > PIVOT_TOLERANCE
            eligible[selected] = False
            scored_candidates += int(np.sum(eligible))

            if not np.any(eligible):
                # The kernel is already fully explained; any remaining column adds nothing, so take
                # the lowest unused index and keep the selection a valid set of the requested size.
                selected.append(next(j for j in range(column_count) if j not in selected))
                continue

            trace_reductions = np.full(column_count, -np.inf)
            trace_reductions[eligible] = (
                np.sum(residual_kernel[:, eligible] ** 2, axis=0) / residual_diagonal[eligible]
            )
            pivot = int(np.argmax(trace_reductions))
            selected.append(pivot)
            pivot_column = residual_kernel[:, pivot]
            residual_kernel = (
                residual_kernel - np.outer(pivot_column, pivot_column) / pivot_column[pivot]
            )

        state = tuple(sorted(selected))
        return SearchResult(
            state=state,
            cost=cost_function.goal_cost(state),
            optimal=False,
            nodes_expanded=scored_candidates,
        )


def pivoted_cholesky_step(
    kernel: np.ndarray,
    cholesky_factor: np.ndarray,
    residual_diagonal: np.ndarray,
    step: int,
    pivot: int,
) -> np.ndarray:
    """Fill column ``step`` of the Cholesky factor from ``pivot``; return the new residual diagonal.

    ``cholesky_factor`` is filled in place, one column per step. ``residual_diagonal[j]`` is how
    much of column ``j`` the factor built so far leaves unexplained. Shared by the deterministic and
    the randomized pivot rules: they differ only in how the pivot is chosen, never in what choosing
    it does.
    """
    pivot_value = max(residual_diagonal[pivot], 0.0)
    if pivot_value <= PIVOT_TOLERANCE:
        return residual_diagonal
    residual_column = kernel[:, pivot] - cholesky_factor[:, :step] @ cholesky_factor[pivot, :step]
    cholesky_factor[:, step] = residual_column / np.sqrt(pivot_value)
    return np.maximum(residual_diagonal - cholesky_factor[:, step] ** 2, 0.0)


@dataclass(frozen=True, slots=True)
class PivotedCholeskyLandmarkSelector(AbstractNystromLandmarkSelector):
    """Greedy pivoted Cholesky: take the largest residual *diagonal* entry each step.

    Cheaper than the trace rule because it reads one number per column rather than a norm, and
    correspondingly blinder: an outlier far from everything has a large diagonal and explains
    nothing else. That failure is what randomizing the pivot was designed to soften.
    """

    name: str = "pivoted_cholesky"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        kernel = problem.kernel_matrix
        column_count = kernel.shape[0]
        landmark_count = problem.landmark_count
        residual_diagonal = np.diag(kernel).copy()
        cholesky_factor = np.zeros((column_count, landmark_count))
        selected: list[int] = []
        already_selected = np.zeros(column_count, dtype=bool)

        for step in range(landmark_count):
            # A pivot always exists: the problem rejects landmark_count > n, so some column is
            # still unselected on every iteration.
            candidate_diagonal = np.where(already_selected, -np.inf, residual_diagonal)
            pivot = int(np.argmax(candidate_diagonal))
            selected.append(pivot)
            already_selected[pivot] = True
            residual_diagonal = pivoted_cholesky_step(
                kernel, cholesky_factor, residual_diagonal, step, pivot
            )

        state = tuple(sorted(selected))
        return SearchResult(
            state=state,
            cost=cost_function.goal_cost(state),
            optimal=False,
            nodes_expanded=landmark_count,
        )


@dataclass(frozen=True, slots=True)
class GreedyLowerBoundLandmarkSelector(AbstractNystromLandmarkSelector):
    """Instrumented, not published: forward selection that minimizes A*'s own lower bound.

    The bound already credits the best possible completion, so this is a lookahead rule that no
    Nyström paper proposes. It is kept because comparing it to the trace rule shows what the bound
    knows that a one-step gain does not. Report ``greedy_trace`` when a report says "greedy" (D-22).
    """

    name: str = "greedy_lower_bound"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        state = problem.initial_state()
        scored_candidates = 0
        tie_tolerance = 1e-12

        while not problem.is_goal(state):
            best_successor: LandmarkState | None = None
            best_lower_bound = float("inf")

            successors = list(problem.successors(state))
            bounds = cost_function.lower_bounds(state, successors)
            for (_, successor), lower_bound in zip(successors, bounds, strict=True):
                scored_candidates += 1
                improves = lower_bound < best_lower_bound - tie_tolerance
                ties_but_sorts_first = abs(lower_bound - best_lower_bound) <= tie_tolerance and (
                    best_successor is None or successor < best_successor
                )
                if improves or ties_but_sorts_first:
                    best_successor = successor
                    best_lower_bound = lower_bound

            if best_successor is None:
                raise ValueError("Greedy selection failed: no successor states were available.")
            state = best_successor

        return SearchResult(
            state=state,
            cost=cost_function.goal_cost(state),
            optimal=False,
            nodes_expanded=scored_candidates,
        )


def run_selector_suite(
    problem: NystromLandmarkProblem,
    cost_function: NystromCssCostFunction,
    selectors: list[AbstractNystromLandmarkSelector],
) -> dict[str, SearchResult[LandmarkState]]:
    """Run every selector on the same problem and return the results keyed by selector name."""
    return {selector.name: selector.select(problem, cost_function) for selector in selectors}
