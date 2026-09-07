"""The engine is a parameter of the harness, and the default run is unchanged by it being one.

Three claims, in the order they matter. First, defaults: injecting the machinery must leave the
committed run field-for-field identical, which is what makes every number taken before this slice
still comparable to one taken after. Second, a variant engine runs under its own name beside the
certified reference, on the same cell, rather than replacing it. Third, the reference cannot be
dropped: every ratio in a result is taken against ``"astar"``, so a suite without it is refused
rather than silently measured against something else.

``frontier_peak`` rides out with the result, so the memory an engine paid is readable next to the
time it paid, and stays ``None`` for engines that do not count it. So does the engine's
``configuration``, read off the engine that ran rather than promised by whoever built it: two rows
on the same cell are only comparable when the settings behind them are on the record (D-28).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.anytime_a_star_search import AnytimeAStarSearch
from mllib.math.algorithms.nystrom_landmark_selectors import (
    AStarLandmarkSelector,
    GreedyResidualTraceLandmarkSelector,
)
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
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


def test_a_variant_that_certifies_its_own_optimum_is_labelled_certified_not_instrumented(
    uci_data_dir: Path,
):
    # The day D-28 anticipated: until the anytime engine, `selector_kind` keyed on the name and a
    # variant that proved its optimum printed as instrumentation. The label now derives from
    # `result.optimal`; the reference is still the one *named* "astar", and it alone is "optimum".
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-pruned",
            search_factory=lambda problem, cost: PrunedAStarSearch(problem, cost),
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    pruned = run.selector_results["astar-pruned"]
    assert pruned.optimal
    assert selector_kind("astar-pruned", pruned) == "certified"
    assert selector_kind("astar", run.selector_results["astar"]) == "optimum"
    assert "astar-pruned [           certified]" in format_run(run)


def _anytime_factory(**knobs):
    def factory(problem, cost):
        greedy = GreedyResidualTraceLandmarkSelector().select(problem, cost)
        return AnytimeAStarSearch(
            problem,
            cost,
            incumbent_seed=greedy.cost,
            incumbent_seed_state=greedy.state,
            incumbent_slack=1e-12 * float(np.trace(problem.kernel_matrix)),
            **knobs,
        )

    return factory


def test_an_anytime_row_that_stopped_early_prints_as_bounded_with_its_gap(uci_data_dir: Path):
    # Entry-ticket seed, test 6 and guard (ii): `bounded` derives from optimal=False with a finite
    # certified_gap; the row states its knobs (max_expansions among them) and its gap. The default
    # block is untouched, which `test_the_default_run_records_the_reference_engine_and_nothing_for_
    # the_rest` and the uci-harness tests pin.
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-anytime", search_factory=_anytime_factory(max_expansions=1)
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    anytime = run.selector_results["astar-anytime"]
    assert not anytime.optimal
    assert anytime.nodes_expanded == 1
    assert anytime.certified_gap is not None and 0.0 <= anytime.certified_gap < float("inf")
    assert anytime.cost >= run.certified_optimum.cost - 1e-9
    assert anytime.cost - run.certified_optimum.cost <= anytime.certified_gap + 1e-9
    assert run.certified_gaps["astar-anytime"] == anytime.certified_gap
    assert run.engine_configurations["astar-anytime"]["max_expansions"] == 1
    assert run.engine_configurations["astar-anytime"]["engine"] == "AnytimeAStarSearch"
    assert selector_kind("astar-anytime", anytime) == "bounded"

    rendered = format_run(run)
    assert "astar-anytime [             bounded]" in rendered
    assert f"gap={anytime.certified_gap:.4f}" in rendered
    assert "astar-anytime [              engine]" in rendered
    assert "max_expansions=1" in rendered
    assert "astar [              engine]" not in rendered


def test_an_uncapped_anytime_row_is_certified_and_states_a_zero_gap(uci_data_dir: Path):
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(name="astar-anytime", search_factory=_anytime_factory()),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    anytime = run.selector_results["astar-anytime"]
    assert anytime.optimal
    assert anytime.certified_gap == 0.0
    assert anytime.state == run.certified_optimum.state
    assert anytime.cost == run.certified_optimum.cost
    assert selector_kind("astar-anytime", anytime) == "certified"
    assert "gap=" not in format_run(run)


def test_a_reference_that_did_not_certify_its_optimum_is_refused(uci_data_dir: Path):
    # Guard (i): the name finds the reference, the certificate makes it one. An anytime engine
    # stopped at one expansion under the name "astar" would ratio every row against an incumbent.
    selectors = [
        AStarLandmarkSelector(name="astar", search_factory=_anytime_factory(max_expansions=1)),
        GreedyResidualTraceLandmarkSelector(),
    ]

    with pytest.raises(ValueError, match="must certify its optimum"):
        _run(uci_data_dir, selectors=selectors)


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


def test_the_default_run_records_the_reference_engine_and_nothing_for_the_rest(uci_data_dir: Path):
    run = _run(uci_data_dir)

    assert set(run.engine_configurations) == set(run.selector_results)
    assert run.engine_configurations["astar"] == {
        "engine": "AStarSearch",
        "count_bound_drops": False,
        "bound_drop_slack": 0.0,
    }
    # A selector that runs no search has no engine to describe, and says so rather than inventing.
    assert run.engine_configurations["greedy_trace"] is None
    assert run.engine_configurations["random_single_draw"] is None
    # Every row ran the same engine, so the block states it nowhere: today's output is unchanged.
    assert "[              engine]" not in format_run(run)


def test_a_variant_records_the_settings_it_actually_ran_under(uci_data_dir: Path):
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-pruned",
            search_factory=lambda problem, cost: PrunedAStarSearch(
                problem, cost, incumbent_slack=1e-12, max_frontier=5_000
            ),
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    assert run.engine_configurations["astar-pruned"] == {
        "engine": "PrunedAStarSearch",
        "count_bound_drops": False,
        "bound_drop_slack": 0.0,
        "incumbent_seed": None,
        "incumbent_slack": 1e-12,
        "max_frontier": 5_000,
    }
    # The variant's settings are stated in the block; the reference's are not, being the default.
    rendered = format_run(run)
    assert "astar-pruned [              engine]" in rendered
    assert "max_frontier=5000" in rendered
    assert "astar [              engine]" not in rendered


def test_the_record_follows_the_engine_that_ran_not_the_name_it_was_given(uci_data_dir: Path):
    # The point of reading the configuration off the instance: a name is a caller's word, and a
    # row whose settings came from the caller could claim a run that never happened.
    selectors = [
        *default_selectors(CELL["sample_seed"]),
        AStarLandmarkSelector(
            name="astar-claims-to-be-capped",
            search_factory=lambda problem, cost: PrunedAStarSearch(problem, cost),
        ),
    ]
    run = _run(uci_data_dir, selectors=selectors)

    assert run.engine_configurations["astar-claims-to-be-capped"]["max_frontier"] is None


def test_a_variant_that_states_no_knobs_is_visibly_incomplete_rather_than_silently_wrong():
    # The failure mode the design accepts: forgetting to override `configuration` loses the knobs
    # from the record, and shows the bare class name, which is readable as "this one never said".
    class UndeclaredVariant(AStarSearch):
        def __init__(self, problem, cost_function, *, budget: int = 3):
            super().__init__(problem, cost_function)
            self.budget = budget

    problem = NystromLandmarkProblem(np.eye(4), landmark_count=2)
    search = UndeclaredVariant(problem, NystromCssCostFunction(problem), budget=9)

    assert search.configuration == {
        "engine": "UndeclaredVariant",
        "count_bound_drops": False,
        "bound_drop_slack": 0.0,
    }
    assert "budget" not in search.configuration
