"""The harness end to end on committed data, kept small enough to stay in the unit suite.

The assertions are mostly invariants rather than fixed numbers, because the numbers depend on the
subsample and the bandwidth. The invariants are the ones a wrong report would violate: nothing beats
the certified optimum, no subset beats the SVD floor, and each selector's reported work matches what
it actually spent.
"""

from pathlib import Path

import pytest

from mllib.ml.projects import nystrom_uci_harness
from mllib.ml.projects.nystrom_uci_data import UciDatasetSpec
from mllib.ml.projects.nystrom_uci_harness import (
    DEFAULT_BANDWIDTH_SCALES,
    PUBLISHED_BASELINES,
    cost_ratio,
    format_run,
    main,
    run_nystrom_on_uci_dataset,
    run_small_uci_suite,
    selector_kind,
    selector_label,
)

EXPECTED_SELECTORS = {
    "astar",
    "greedy_trace",
    "pivoted_cholesky",
    "rpcholesky",
    "random_single_draw",
    "greedy_lower_bound",
    "best_of_32_random",
}

SPECTF = UciDatasetSpec(name="SPECTF", file_name="SPECTF.test", label_position="first")


@pytest.fixture
def spectf_run(uci_data_dir: Path):
    return run_nystrom_on_uci_dataset(
        SPECTF,
        data_dir=uci_data_dir,
        max_rows=14,
        landmark_count=2,
        sample_seed=5,
        randomized_trials=8,
    )


def test_the_run_reports_the_dataset_it_actually_solved(spectf_run):
    assert spectf_run.dataset_name == "SPECTF"
    assert spectf_run.row_count == 14
    assert spectf_run.feature_count == 44
    assert spectf_run.landmark_count == 2
    assert spectf_run.gamma > 0.0


def test_the_search_returns_a_certified_selection_of_the_requested_size(spectf_run):
    assert spectf_run.certified_optimum.optimal
    assert len(set(spectf_run.certified_optimum.state)) == 2


def test_no_selector_beats_the_certified_optimum(spectf_run):
    assert set(spectf_run.selector_results) == EXPECTED_SELECTORS
    assert spectf_run.cost_ratios_to_optimal["astar"] == 1.0
    assert all(ratio >= 1.0 - 1e-12 for ratio in spectf_run.cost_ratios_to_optimal.values())


def test_the_certificate_costs_less_than_enumerating_every_subset(spectf_run):
    assert spectf_run.subset_count == 91  # C(14, 2)
    assert spectf_run.certified_optimum.nodes_expanded <= spectf_run.subset_count


def test_the_best_subset_cannot_beat_the_rank_k_svd_floor(spectf_run):
    # The second gap: how much choosing landmarks costs against being free to use any subspace.
    assert spectf_run.svd_rank_k_residual <= spectf_run.certified_optimum.cost + 1e-12
    assert spectf_run.subset_to_svd_ratio >= 1.0 - 1e-12


def test_randomized_selectors_are_reported_as_a_spread_over_seeds(spectf_run):
    assert set(spectf_run.randomized_summaries) == {"rpcholesky", "random_single_draw"}
    assert all(summary.trials == 8 for summary in spectf_run.randomized_summaries.values())
    assert all(
        ratio >= 1.0 - 1e-12 for ratio in spectf_run.randomized_mean_ratios_to_optimal.values()
    )
    for summary in spectf_run.randomized_summaries.values():
        assert summary.min_cost <= summary.median_cost <= summary.max_cost


def test_each_selector_reports_the_work_it_actually_spent(spectf_run):
    results = spectf_run.selector_results

    assert results["random_single_draw"].nodes_expanded == 1
    assert results["best_of_32_random"].nodes_expanded == 32
    assert results["pivoted_cholesky"].nodes_expanded == 2  # one pivot per landmark


def test_published_baselines_and_instrumented_heuristics_are_labelled_apart(spectf_run):
    assert set(PUBLISHED_BASELINES) <= set(spectf_run.selector_results)
    assert selector_kind("astar") == "optimum"
    assert selector_kind("greedy_trace") == "baseline"
    assert selector_kind("greedy_lower_bound") == "instrumented"
    assert selector_kind("best_of_32_random") == "instrumented"


def test_a_single_draw_is_labelled_as_one_seed_not_as_a_settled_baseline():
    # D-22: the printed block is the artefact numbers get copied out of, so it carries the caveat.
    assert selector_label("greedy_trace") == "baseline"
    assert selector_label("rpcholesky") == "baseline, 1 seed"
    assert selector_label("random_single_draw") == "baseline, 1 seed"
    assert selector_label("astar") == "optimum"


def test_the_printed_block_carries_both_gaps_and_the_selector_labels(spectf_run):
    rendered = format_run(spectf_run)

    assert "subset/SVD=" in rendered
    assert "SVD rank-k residual=" in rendered
    assert "of C(n,k)=91" in rendered
    assert "[            baseline]" in rendered
    assert "[        instrumented]" in rendered
    assert "[             optimum]" in rendered
    assert "[             8 seeds]" in rendered
    # A one-draw ratio must not sit in the same column as a settled baseline number (D-22).
    assert "[    baseline, 1 seed]" in rendered


def test_a_landmark_budget_that_exceeds_the_sample_is_rejected(uci_data_dir: Path):
    with pytest.raises(ValueError, match="smaller than the number of sampled rows"):
        run_nystrom_on_uci_dataset(
            SPECTF, data_dir=uci_data_dir, max_rows=3, landmark_count=3, randomized_trials=2
        )


def test_the_suite_runs_every_dataset_at_every_bandwidth_scale(uci_data_dir: Path):
    runs = run_small_uci_suite(
        data_dir=uci_data_dir,
        max_rows=12,
        landmark_count=2,
        sample_seed=3,
        gamma_scales=(0.5, 2.0),
        randomized_trials=4,
    )

    assert [(run.dataset_name, run.gamma_scale) for run in runs] == [
        ("SPECTF", 0.5),
        ("SPECTF", 2.0),
        ("movement_libras", 0.5),
        ("movement_libras", 2.0),
        ("wdbc", 0.5),
        ("wdbc", 2.0),
    ]
    assert all(run.certified_optimum.optimal for run in runs)
    assert all(run.cost_ratios_to_optimal["astar"] == 1.0 for run in runs)


def test_a_wider_bandwidth_changes_the_kernel_it_builds(uci_data_dir: Path):
    narrow, wide = run_small_uci_suite(
        data_dir=uci_data_dir,
        max_rows=12,
        landmark_count=2,
        sample_seed=3,
        gamma_scales=(0.5, 4.0),
        randomized_trials=2,
    )[:2]

    # Bandwidth is the knob that moves the spectrum between decaying and flat, which is where the
    # interesting behaviour of both A* and the heuristics is expected to live.
    assert wide.gamma == pytest.approx(8.0 * narrow.gamma)
    assert wide.certified_optimum.cost != pytest.approx(narrow.certified_optimum.cost)


def test_a_ratio_against_a_zero_optimum_is_defined():
    assert cost_ratio(0.0, 0.0) == 1.0
    assert cost_ratio(1.0, 0.0) == float("inf")
    assert cost_ratio(3.0, 1.5) == pytest.approx(2.0)


def test_main_sweeps_the_three_declared_bandwidth_scales(monkeypatch):
    # The sweep is the point of `main`, so the scales it passes are pinned rather than assumed.
    recorded: dict[str, tuple[float, ...]] = {}

    def fake_suite(**kwargs):
        recorded["gamma_scales"] = kwargs["gamma_scales"]
        return []

    monkeypatch.setattr(nystrom_uci_harness, "run_small_uci_suite", fake_suite)
    main()

    assert DEFAULT_BANDWIDTH_SCALES == (0.25, 1.0, 4.0)
    assert recorded["gamma_scales"] == DEFAULT_BANDWIDTH_SCALES
