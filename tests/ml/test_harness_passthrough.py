"""The engine is a parameter of the harness, and the default run is unchanged by it being one.

Three claims, in the order they matter. First, defaults: injecting the machinery must leave the
committed run field-for-field identical, which is what makes every number taken before this slice
still comparable to one taken after. Second, a variant engine runs under its own name beside the
certified reference, on the same cell, rather than replacing it. Third, the reference cannot be
dropped: every ratio in a result is taken against ``"astar"``, so a suite without it is refused
rather than silently measured against something else.

``frontier_peak`` rides out with the result, so the memory an engine paid is readable next to the
time it paid, and stays ``None`` for engines that do not count it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.nystrom_landmark_selectors import (
    AStarLandmarkSelector,
    GreedyResidualTraceLandmarkSelector,
)
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.ml.projects.nystrom_uci_data import UciDatasetSpec
from mllib.ml.projects.nystrom_uci_harness import (
    default_selectors,
    format_run,
    run_nystrom_on_uci_dataset,
    selector_kind,
)

SPECTF = UciDatasetSpec(name="SPECTF", file_name="SPECTF.test", label_position="first")

# The same cell as `test_nystrom_uci_harness.spectf_run`, so a difference here is the engine's.
CELL = {"max_rows": 14, "landmark_count": 2, "sample_seed": 5, "randomized_trials": 8}


def _run(uci_data_dir: Path, **overrides):
    return run_nystrom_on_uci_dataset(SPECTF, data_dir=uci_data_dir, **CELL, **overrides)


def test_passing_the_default_suite_explicitly_reproduces_the_default_run(uci_data_dir: Path):
    # (i) The defaults are a value, not a behaviour: naming them changes nothing about the run.
    implicit = _run(uci_data_dir)
    explicit = _run(uci_data_dir, selectors=default_selectors(CELL["sample_seed"]))

    assert list(explicit.selector_results) == list(implicit.selector_results)
    for name, result in implicit.selector_results.items():
        assert explicit.selector_results[name] == result
    assert explicit.certified_optimum == implicit.certified_optimum
    assert explicit.cost_ratios_to_optimal == implicit.cost_ratios_to_optimal
    assert explicit.frontier_peaks == implicit.frontier_peaks
    assert explicit.svd_rank_k_residual == implicit.svd_rank_k_residual
    assert explicit.subset_to_svd_ratio == implicit.subset_to_svd_ratio
    assert explicit.subset_count == implicit.subset_count


def test_the_default_suite_is_pinned_because_its_order_and_seeds_are_part_of_every_number(
    uci_data_dir: Path,
):
    # The list moved out of the runner into a function; nothing else pins it, and a reorder or a
    # reseeded sampler would change committed numbers without failing a ratio invariant.
    assert [selector.name for selector in default_selectors(5)] == [
        "astar",
        "greedy_trace",
        "pivoted_cholesky",
        "rpcholesky",
        "random_single_draw",
        "greedy_lower_bound",
        "best_of_32_random",
    ]
    seeded = {
        selector.name: selector.seed
        for selector in default_selectors(5)
        if hasattr(selector, "seed")
    }
    assert set(seeded.values()) == {5}


def test_the_a_star_selector_uses_the_engine_it_was_given_and_stays_immutable():
    built: list[str] = []

    def factory(problem, cost_function):
        built.append("called")
        return AStarSearch(problem, cost_function)

    selector = AStarLandmarkSelector(name="astar-echo", search_factory=factory)
    assert selector.search_factory is factory
    assert AStarLandmarkSelector().search_factory is AStarSearch
    with pytest.raises(AttributeError):
        selector.name = "astar"  # frozen: the reference's name cannot be taken by a variant


def test_a_certified_variant_is_still_labelled_instrumentation_until_certified_itself(
    uci_data_dir: Path,
):
    # D-22, D-28: `selector_kind` keys on the name, so a variant that proves its own optimum still
    # prints as instrumentation. Pinned so the day that changes is a decision, not a drift.
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-pruned",
            search_factory=lambda problem, cost: PrunedAStarSearch(problem, cost),
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    assert run.selector_results["astar-pruned"].optimal
    assert selector_kind("astar-pruned") == "instrumented"
    assert selector_kind("astar") == "optimum"


def test_the_default_run_records_no_frontier_peak_because_exact_a_star_counts_none(
    uci_data_dir: Path,
):
    run = _run(uci_data_dir)

    assert set(run.frontier_peaks) == set(run.selector_results)
    assert all(peak is None for peak in run.frontier_peaks.values())
    # The printed block is unchanged when nothing measured a peak.
    assert "frontier peak" not in format_run(run)


def test_a_second_engine_runs_under_its_own_name_beside_the_certified_reference(
    uci_data_dir: Path,
):
    # (ii) A trivially wrapped `AStarSearch` is the same search, so it must agree exactly.
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-echo",
            search_factory=lambda problem, cost: AStarSearch(problem, cost),
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    echoed = run.selector_results["astar-echo"]
    assert run.selector_results["astar"].state == echoed.state
    assert run.selector_results["astar"].cost == echoed.cost
    assert echoed.optimal
    assert run.cost_ratios_to_optimal["astar-echo"] == pytest.approx(1.0)
    # The reference is still the one named "astar", not whichever engine ran last.
    assert run.certified_optimum is run.selector_results["astar"]


def test_a_suite_without_the_certified_reference_is_refused(uci_data_dir: Path):
    # (iii) Silently ratioing against a variant would corrupt every number in the result.
    with pytest.raises(ValueError, match="must include one named 'astar'"):
        _run(uci_data_dir, selectors=[GreedyResidualTraceLandmarkSelector()])


def test_an_engine_that_counts_its_frontier_reports_the_peak_it_reached(uci_data_dir: Path):
    # (iv) `PrunedAStarSearch` counts frontier entries; plain A* does not, and says so with None.
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-pruned",
            search_factory=lambda problem, cost: PrunedAStarSearch(problem, cost),
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    peak = run.frontier_peaks["astar-pruned"]
    assert isinstance(peak, int)
    assert peak >= 1
    assert run.frontier_peaks["astar"] is None
    # Same optimum: the variant stores less, it does not search differently.
    assert run.selector_results["astar-pruned"].cost == run.certified_optimum.cost
    assert run.selector_results["astar-pruned"].state == run.certified_optimum.state
    # The line appears exactly once a peak was measured, naming only the engines that counted.
    rendered = format_run(run)
    assert f"frontier peak: astar-pruned={peak}" in rendered
