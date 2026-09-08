"""Randomized landmark selectors, and the machinery that stops one lucky draw being quoted.

``RandomlyPivotedCholeskyLandmarkSelector`` and ``RandomSamplingLandmarkSelector`` are published
baselines in the same sense as the deterministic ones in ``nystrom_landmark_selectors``: their gap
to the optimum is the result. But a randomized selector run once produces a number that says as much
about its seed as about the method. Every selector here therefore takes an explicit seed, and
``summarize_randomized_selector`` is the intended way to report one: as a mean and median over
independent seeds, with any single draw labelled as such (D-22).

``BestOfRandomSamplingLandmarkSelector`` is here for the same reason but is not a baseline at all.
Drawing N subsets and keeping the best spends N evaluations of the objective, which makes it a crude
optimizer; on a problem with C(n, k) subsets it approaches the optimum as N grows. Its name carries
its N so that no report can present it as "random".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mllib.math.algorithms.a_star_search import SearchResult
from mllib.math.algorithms.nystrom_landmark_selectors import (
    PIVOT_TOLERANCE,
    AbstractNystromLandmarkSelector,
    pivoted_cholesky_step,
)
from mllib.math.graph.nystrom_landmark_problem import (
    LandmarkState,
    NystromCssCostFunction,
    NystromLandmarkProblem,
)


@dataclass(frozen=True, slots=True)
class RandomlyPivotedCholeskyLandmarkSelector(AbstractNystromLandmarkSelector):
    """RPCholesky: sample each pivot with probability proportional to the residual diagonal.

    The same update as deterministic pivoted Cholesky, with the pivot drawn rather than maximized.
    Weighting by the residual diagonal keeps the search near the columns that are still unexplained
    while leaving an outlier only a proportional chance of being taken, which is what softens the
    failure the deterministic rule walks into.
    """

    seed: int = 7
    name: str = "rpcholesky"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        rng = np.random.default_rng(self.seed)
        kernel = problem.kernel_matrix
        column_count = kernel.shape[0]
        landmark_count = problem.landmark_count
        residual_diagonal = np.diag(kernel).copy()
        cholesky_factor = np.zeros((column_count, landmark_count))
        selected: list[int] = []

        for step in range(landmark_count):
            pivot_weights = np.maximum(residual_diagonal, 0.0)
            pivot_weights[selected] = 0.0
            total_weight = float(pivot_weights.sum())
            if total_weight <= PIVOT_TOLERANCE:
                # Nothing is left unexplained; fall back to a uniform draw among unused columns so
                # the selection still has the requested size.
                unselected = [j for j in range(column_count) if j not in selected]
                pivot = int(rng.choice(unselected))
            else:
                pivot = int(rng.choice(column_count, p=pivot_weights / total_weight))
            selected.append(pivot)
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
class RandomSamplingLandmarkSelector(AbstractNystromLandmarkSelector):
    """One uniform draw of k columns: the baseline every informed selector has to beat."""

    seed: int = 7
    name: str = "random_single_draw"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        rng = np.random.default_rng(self.seed)
        column_count = problem.kernel_matrix.shape[0]
        drawn = rng.choice(column_count, size=problem.landmark_count, replace=False)
        state = tuple(sorted(drawn.tolist()))
        return SearchResult(
            state=state,
            cost=cost_function.goal_cost(state),
            optimal=False,
            nodes_expanded=1,
        )


@dataclass(frozen=True, slots=True)
class BestOfRandomSamplingLandmarkSelector(AbstractNystromLandmarkSelector):
    """The best of ``sample_count`` uniform draws: an optimizer, not a baseline.

    It spends ``sample_count`` objective evaluations, so its apparent quality is bought rather than
    earned. Reporting it as "random" is what made the first version of this harness overstate how
    far the published baselines sit from the optimum.
    """

    sample_count: int = 32
    seed: int = 7

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"best_of_{self.sample_count}_random"

    def select(
        self,
        problem: NystromLandmarkProblem,
        cost_function: NystromCssCostFunction,
    ) -> SearchResult[LandmarkState]:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        rng = np.random.default_rng(self.seed)
        column_count = problem.kernel_matrix.shape[0]
        best_state: LandmarkState | None = None
        best_cost = float("inf")

        for _ in range(self.sample_count):
            drawn = rng.choice(column_count, size=problem.landmark_count, replace=False)
            state = tuple(sorted(drawn.tolist()))
            cost = cost_function.goal_cost(state)
            if cost < best_cost:
                best_cost = cost
                best_state = state

        if best_state is None:
            raise ValueError("Random sampling failed to produce any landmark subset.")

        return SearchResult(
            state=best_state,
            cost=best_cost,
            optimal=False,
            nodes_expanded=self.sample_count,
        )


@dataclass(frozen=True, slots=True)
class RandomizedSelectorSummary:
    """What a randomized selector costs across seeds, rather than on the one that was run."""

    name: str
    trials: int
    mean_cost: float
    median_cost: float
    min_cost: float
    max_cost: float


def summarize_randomized_selector(
    selector_factory: Callable[[int], AbstractNystromLandmarkSelector],
    problem: NystromLandmarkProblem,
    cost_function: NystromCssCostFunction,
    *,
    trials: int,
    base_seed: int = 0,
) -> RandomizedSelectorSummary:
    """Run ``selector_factory(seed)`` over consecutive seeds and summarize the costs it produced.

    The spread matters as much as the mean: a selector whose best draw beats greedy and whose worst
    draw is twice the optimum is a different proposition from one that is consistently mediocre.
    """
    if trials <= 0:
        raise ValueError("trials must be positive.")

    costs = np.array(
        [
            selector_factory(base_seed + trial).select(problem, cost_function).cost
            for trial in range(trials)
        ]
    )
    return RandomizedSelectorSummary(
        name=selector_factory(base_seed).name,
        trials=trials,
        mean_cost=float(costs.mean()),
        median_cost=float(np.median(costs)),
        min_cost=float(costs.min()),
        max_cost=float(costs.max()),
    )
