"""Measuring how far landmark heuristics sit from the best possible landmarks, on real data.

The question this answers has two halves, and reporting only the first is what makes the answer
misleading (D-22):

* **algorithm versus best subset** — how much worse is a heuristic than the landmarks A*
  certifies?
* **best subset versus SVD** — how much does choosing *landmarks* cost, compared to being free to
  use any rank-k subspace? No selector can close this second gap, so a heuristic that looks bad
  against the optimum may simply be working on a kernel where no subset does well.

Alongside them the harness prints what the certificate cost: states A* expanded, next to the number
of subsets that exist. The bandwidth is the median heuristic times ``gamma_scale``, and the suite is
meant to be run across several scales, since one bandwidth on one subsample is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path

from mllib.math.algorithms.a_star_search import SearchResult
from mllib.math.algorithms.nystrom_landmark_selectors import (
    AStarLandmarkSelector,
    GreedyLowerBoundLandmarkSelector,
    GreedyResidualTraceLandmarkSelector,
    PivotedCholeskyLandmarkSelector,
    run_selector_suite,
)
from mllib.math.algorithms.nystrom_randomized_selectors import (
    BestOfRandomSamplingLandmarkSelector,
    RandomizedSelectorSummary,
    RandomlyPivotedCholeskyLandmarkSelector,
    RandomSamplingLandmarkSelector,
    summarize_randomized_selector,
)
from mllib.math.graph.nystrom_landmark_problem import (
    LandmarkState,
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from mllib.ml.projects.nystrom_uci_data import (
    DEFAULT_SMALL_UCI_SPECS,
    UciDatasetSpec,
    build_rbf_kernel,
    downsample_rows,
    load_feature_matrix,
    standardize_columns,
    svd_rank_k_residual,
)

# Selectors whose distance from the optimum is a result about the literature. Everything else in a
# run is instrumentation, and a report must not blur the two.
PUBLISHED_BASELINES = ("greedy_trace", "pivoted_cholesky", "rpcholesky", "random_single_draw")

# Randomized selectors whose per-run line is a single draw. Their ratio is a sample, not a
# property of the method, and the printed block has to say so or D-22 is violated in the one
# artefact a reader copies numbers out of.
SINGLE_DRAW_SELECTORS = ("rpcholesky", "random_single_draw")

DEFAULT_BANDWIDTH_SCALES = (0.25, 1.0, 4.0)


@dataclass(frozen=True, slots=True)
class UciHarnessResult:
    """One landmark-selection run: what was solved, by whom, and at what cost."""

    dataset_name: str
    row_count: int
    feature_count: int
    landmark_count: int
    gamma: float
    gamma_scale: float
    certified_optimum: SearchResult[LandmarkState]
    selector_results: dict[str, SearchResult[LandmarkState]]
    cost_ratios_to_optimal: dict[str, float]
    randomized_summaries: dict[str, RandomizedSelectorSummary]
    randomized_mean_ratios_to_optimal: dict[str, float]
    svd_rank_k_residual: float
    subset_to_svd_ratio: float
    subset_count: int


def cost_ratio(cost: float, optimal_cost: float) -> float:
    """Ratio of a cost to the optimum, defined when the optimum is zero."""
    if optimal_cost == 0.0:
        return 1.0 if cost == 0.0 else float("inf")
    return cost / optimal_cost


def run_nystrom_on_uci_dataset(
    spec: UciDatasetSpec,
    *,
    data_dir: Path | None = None,
    max_rows: int = 40,
    landmark_count: int = 3,
    sample_seed: int = 7,
    gamma: float | None = None,
    gamma_scale: float = 1.0,
    randomized_trials: int = 50,
) -> UciHarnessResult:
    """Load one dataset, build a kernel, and run every selector against the certified optimum."""
    features = load_feature_matrix(spec, data_dir)
    features = downsample_rows(features, max_rows=max_rows, seed=sample_seed)
    features = standardize_columns(features)

    if landmark_count >= features.shape[0]:
        raise ValueError("landmark_count must be smaller than the number of sampled rows.")

    kernel, gamma_used = build_rbf_kernel(features, gamma=gamma, gamma_scale=gamma_scale)
    problem = NystromLandmarkProblem(kernel, landmark_count=landmark_count)
    cost_function = NystromCssCostFunction(problem)

    selector_results = run_selector_suite(
        problem,
        cost_function,
        selectors=[
            AStarLandmarkSelector(),
            GreedyResidualTraceLandmarkSelector(),
            PivotedCholeskyLandmarkSelector(),
            RandomlyPivotedCholeskyLandmarkSelector(seed=sample_seed),
            RandomSamplingLandmarkSelector(seed=sample_seed),
            GreedyLowerBoundLandmarkSelector(),
            BestOfRandomSamplingLandmarkSelector(sample_count=32, seed=sample_seed),
        ],
    )
    certified_optimum = selector_results["astar"]
    optimal_cost = certified_optimum.cost

    # The single runs above are one draw each; these are what a report should quote.
    randomized_summaries = {
        summary.name: summary
        for summary in (
            summarize_randomized_selector(
                RandomlyPivotedCholeskyLandmarkSelector,
                problem,
                cost_function,
                trials=randomized_trials,
                base_seed=sample_seed,
            ),
            summarize_randomized_selector(
                RandomSamplingLandmarkSelector,
                problem,
                cost_function,
                trials=randomized_trials,
                base_seed=sample_seed,
            ),
        )
    }
    svd_residual = svd_rank_k_residual(kernel, landmark_count)

    return UciHarnessResult(
        dataset_name=spec.name,
        row_count=features.shape[0],
        feature_count=features.shape[1],
        landmark_count=landmark_count,
        gamma=gamma_used,
        gamma_scale=gamma_scale,
        certified_optimum=certified_optimum,
        selector_results=selector_results,
        cost_ratios_to_optimal={
            name: cost_ratio(result.cost, optimal_cost) for name, result in selector_results.items()
        },
        randomized_summaries=randomized_summaries,
        randomized_mean_ratios_to_optimal={
            name: cost_ratio(summary.mean_cost, optimal_cost)
            for name, summary in randomized_summaries.items()
        },
        svd_rank_k_residual=svd_residual,
        subset_to_svd_ratio=cost_ratio(optimal_cost, svd_residual),
        subset_count=comb(features.shape[0], landmark_count),
    )


def run_small_uci_suite(
    *,
    data_dir: Path | None = None,
    max_rows: int = 40,
    landmark_count: int = 3,
    sample_seed: int = 7,
    gamma_scales: tuple[float, ...] = (1.0,),
    randomized_trials: int = 50,
) -> list[UciHarnessResult]:
    """Run every bundled dataset once per bandwidth scale, dataset-major."""
    return [
        run_nystrom_on_uci_dataset(
            spec,
            data_dir=data_dir,
            max_rows=max_rows,
            landmark_count=landmark_count,
            sample_seed=sample_seed,
            gamma_scale=gamma_scale,
            randomized_trials=randomized_trials,
        )
        for spec in DEFAULT_SMALL_UCI_SPECS
        for gamma_scale in gamma_scales
    ]


def selector_kind(name: str) -> str:
    """Whether a selector's number is a result, a reference point, or instrumentation."""
    if name == "astar":
        return "optimum"
    if name in PUBLISHED_BASELINES:
        return "baseline"
    return "instrumented"


def selector_label(name: str) -> str:
    """The tag printed beside a selector, marking a single draw as the sample it is."""
    kind = selector_kind(name)
    if name in SINGLE_DRAW_SELECTORS:
        return f"{kind}, 1 seed"
    return kind


def format_run(run: UciHarnessResult) -> str:
    """Render one run as the block ``main`` prints, labelled so no number can be misread."""
    lines = [
        f"{run.dataset_name}: n={run.row_count} d={run.feature_count} k={run.landmark_count} "
        f"gamma={run.gamma:.6f} (scale {run.gamma_scale:g})  "
        f"A*={run.certified_optimum.cost:.4f} nodes={run.certified_optimum.nodes_expanded} "
        f"of C(n,k)={run.subset_count}  "
        f"SVD rank-k residual={run.svd_rank_k_residual:.4f}  "
        f"subset/SVD={run.subset_to_svd_ratio:.3f}"
    ]
    for name, result in run.selector_results.items():
        lines.append(
            f"  {name:>22} [{selector_label(name):>20}]: "
            f"ratio={run.cost_ratios_to_optimal[name]:.3f} cost={result.cost:.4f} "
            f"state={result.state} nodes={result.nodes_expanded}"
        )
    optimal_cost = run.certified_optimum.cost
    for name, summary in run.randomized_summaries.items():
        lines.append(
            f"  {name:>22} [{f'{summary.trials} seeds':>20}]: "
            f"mean ratio={run.randomized_mean_ratios_to_optimal[name]:.3f} "
            f"median={cost_ratio(summary.median_cost, optimal_cost):.3f} "
            f"min={cost_ratio(summary.min_cost, optimal_cost):.3f} "
            f"max={cost_ratio(summary.max_cost, optimal_cost):.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the bundled datasets across three bandwidths and print one block per run."""
    for run in run_small_uci_suite(gamma_scales=DEFAULT_BANDWIDTH_SCALES):
        print(format_run(run))


if __name__ == "__main__":
    main()
