"""Tie-breaking on the frontier: deeper-first as an option, near-ties by a stated tolerance.

On a plateau where every state shares the optimum's bound, first-in-first-out works the tree level
by level and pops a goal only after the whole plateau above it is expanded; preferring the deeper
state walks one path down and certifies after about k expansions. Neither changes the answer at
tolerance zero, and the toy problems from `test_a_star_search.py` say so against brute force. A
tolerance makes near-ties equal, and then the popped goal is certified only when nothing smaller
remains; otherwise the result carries the honest additive gap, below the tolerance by construction.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import AStarSearch
from mllib.math.algorithms.anytime_a_star_search import AnytimeAStarSearch
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from tests.math.algorithms.test_a_star_search import (
    WEIGHTS,
    KeepHeaviestProblem,
    LeftOutWeightCost,
    UnorderedKeepHeaviestProblem,
    brute_force_best,
)
from tests.math.algorithms.test_pruned_a_star_search import assert_same_search

ENGINES = (AStarSearch, PrunedAStarSearch, AnytimeAStarSearch)


def _plateau(keep_count: int):
    # Equal weights: every state's bound is the optimum, so the whole tree above goal depth is one
    # plateau. n = 2k keeps the canonical tree wide enough to enumerate.
    problem = KeepHeaviestProblem((1.0,) * (2 * keep_count), keep_count)
    return problem, LeftOutWeightCost(problem)


def test_on_a_plateau_fifo_enumerates_and_deepest_first_walks_one_path():
    # Engine-slices seed, PR 2 test 3.
    for keep_count in (3, 4, 5):
        problem, cost_function = _plateau(keep_count)

        fifo = AStarSearch(problem, cost_function).run()
        deepest = AStarSearch(problem, cost_function, tie_break="deepest").run()

        assert fifo.optimal and deepest.optimal
        assert fifo.cost == deepest.cost == pytest.approx(float(keep_count))
        assert fifo.nodes_expanded >= 2**keep_count - 1
        assert deepest.nodes_expanded <= 2 * keep_count
        # Deeper-first on a plateau is the depth-first path: root plus one state per pick.
        assert deepest.nodes_expanded == keep_count + 1


def test_deepest_first_at_tolerance_zero_returns_the_same_optimum_as_fifo():
    for problem_class in (KeepHeaviestProblem, UnorderedKeepHeaviestProblem):
        for keep_count in (1, 2, 3, 4):
            problem = problem_class(WEIGHTS, keep_count)
            cost_function = LeftOutWeightCost(problem)
            best_cost, _ = brute_force_best(WEIGHTS, keep_count)

            fifo = AStarSearch(problem, cost_function).run()
            deepest = AStarSearch(problem, cost_function, tie_break="deepest").run()

            assert deepest.optimal and deepest.certified_gap == 0.0
            assert deepest.cost == pytest.approx(best_cost) == pytest.approx(fifo.cost)


def test_the_pruned_and_anytime_engines_inherit_the_knob_and_still_match_exact_a_star():
    # The pruning mechanisms drop only what would never pop, whatever the tie-break; goal siblings
    # share a depth, so the kept one is the same under either order.
    for keep_count in (2, 3, 4):
        for problem, cost_function in (
            _plateau(keep_count),
            (KeepHeaviestProblem(WEIGHTS, keep_count), None),
        ):
            cost_function = cost_function or LeftOutWeightCost(problem)
            exact = AStarSearch(problem, cost_function, tie_break="deepest").run()
            for engine in (PrunedAStarSearch, AnytimeAStarSearch):
                variant = engine(problem, cost_function, tie_break="deepest").run()
                assert_same_search(variant, exact)
                assert variant.optimal


# Two goals 0.005 apart and a shallower non-goal whose bound is the smaller of the two: under a
# tolerance of 0.1 they share a cell, the goal at 1.005 is deeper and pops first, and the search
# can certify only the 0.005 it did not look under. Brute force says the optimum is 1.0.
NEAR_TIE_WEIGHTS = (1.0, 1.005, 3.0)


def test_under_a_tolerance_a_goal_popped_above_a_smaller_remaining_bound_is_bounded_not_optimal():
    # Engine-slices seed, PR 2 test 4, the case where the certificate is not exact.
    problem = KeepHeaviestProblem(NEAR_TIE_WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    best_cost, best_state = brute_force_best(NEAR_TIE_WEIGHTS, 2)
    assert best_state == (1, 2) and best_cost == pytest.approx(1.0)

    for tie_break in ("fifo", "deepest"):
        result = AStarSearch(problem, cost_function, tie_break=tie_break, tie_tolerance=0.1).run()

        assert not result.optimal, tie_break
        assert result.state == (0, 2)
        assert result.cost == pytest.approx(1.005)
        assert result.certified_gap == pytest.approx(0.005)
        assert result.certified_gap < 0.1
        assert result.cost - result.certified_gap <= best_cost + 1e-12 <= result.cost


def test_engines_that_hold_an_incumbent_return_it_when_the_popped_goal_is_worse():
    # First-in-first-out expands (1,) before the 1.005 goal pops, so the pruned and anytime engines
    # have priced the 1.0 goal and hold it as the incumbent; it is the frontier minimum, and they
    # return it certified where the base engine returns the goal it popped. Deeper-first pops the
    # 1.005 goal before (1,) is expanded, so no better incumbent exists; the result is the base's.
    problem = KeepHeaviestProblem(NEAR_TIE_WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    for engine in (PrunedAStarSearch, AnytimeAStarSearch):
        fifo = engine(problem, cost_function, tie_break="fifo", tie_tolerance=0.1).run()
        assert fifo.state == (1, 2), engine.__name__
        assert fifo.cost == pytest.approx(1.0)
        assert fifo.optimal and fifo.certified_gap == 0.0

        deepest = engine(problem, cost_function, tie_break="deepest", tie_tolerance=0.1).run()
        assert deepest.state == (0, 2), engine.__name__
        assert not deepest.optimal
        assert deepest.certified_gap == pytest.approx(0.005)


def test_a_seed_below_the_popped_goal_is_returned_bounded_not_certified():
    # A seed is a cost from another arithmetic path; returned in place of a worse popped goal it
    # carries the honest gap to the frontier minimum and is never marked optimal.
    problem = KeepHeaviestProblem((1.0, 1.005, 1.006, 3.0), keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    seed_state = (1, 3)
    seed = cost_function.goal_cost(seed_state)  # leaves out 1.0 + 1.006 = 2.006
    best_cost, _ = brute_force_best((1.0, 1.005, 1.006, 3.0), 2)

    result = PrunedAStarSearch(
        problem,
        cost_function,
        incumbent_seed=seed,
        incumbent_seed_state=seed_state,
        tie_break="deepest",
        tie_tolerance=0.5,
    ).run()

    assert result.cost <= 2.006 + 1e-12
    assert result.cost - result.certified_gap <= best_cost + 1e-12 <= result.cost
    if result.state == seed_state:
        assert not result.optimal


def test_a_tolerance_too_small_to_merge_the_near_tie_leaves_the_certificate_exact():
    problem = KeepHeaviestProblem(NEAR_TIE_WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    for tie_break in ("fifo", "deepest"):
        result = AStarSearch(problem, cost_function, tie_break=tie_break, tie_tolerance=0.001).run()

        assert result.optimal and result.certified_gap == 0.0
        assert result.state == (1, 2)
        assert result.cost == pytest.approx(1.0)


def test_the_anytime_engine_reports_the_base_certificate_under_a_tolerance():
    problem = KeepHeaviestProblem(NEAR_TIE_WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    anytime = AnytimeAStarSearch(problem, cost_function, tie_break="deepest", tie_tolerance=0.1)
    result = anytime.run()

    assert not result.optimal
    assert anytime.certified_gap == result.certified_gap == pytest.approx(0.005)
    assert anytime.frontier_min == pytest.approx(1.0)  # the raw minimum, not the cell's floor


def test_a_capped_anytime_run_under_a_tolerance_reads_the_raw_frontier_minimum():
    # Under a tolerance the heap's top is only the smallest cell; the bracket must use the smallest
    # raw bound, which may sit deeper in the heap. Brute force is the oracle.
    problem = KeepHeaviestProblem(NEAR_TIE_WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    best_cost, _ = brute_force_best(NEAR_TIE_WEIGHTS, 2)

    anytime = AnytimeAStarSearch(
        problem, cost_function, tie_break="deepest", tie_tolerance=0.1, max_expansions=2
    )
    result = anytime.run()

    assert not result.optimal
    assert anytime.frontier_min <= best_cost + 1e-12
    assert result.cost >= best_cost - 1e-12
    assert anytime.certified_gap >= result.cost - best_cost - 1e-12


def test_configuration_states_both_knobs_on_every_engine():
    # Engine-slices seed, PR 2 test 5.
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    for engine in ENGINES:
        default = engine(problem, cost_function).configuration
        assert default["tie_break"] == "fifo"
        assert default["tie_tolerance"] == 0.0
        chosen = engine(problem, cost_function, tie_break="deepest", tie_tolerance=1e-9)
        assert chosen.configuration["tie_break"] == "deepest"
        assert chosen.configuration["tie_tolerance"] == 1e-9
        assert set(chosen.configuration) == set(default)


def test_an_unknown_tie_break_or_a_negative_tolerance_is_rejected():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    with pytest.raises(ValueError, match="tie_break"):
        AStarSearch(problem, cost_function, tie_break="shallowest")
    with pytest.raises(ValueError, match="tie_tolerance"):
        AStarSearch(problem, cost_function, tie_tolerance=-1e-9)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    weights=st.lists(st.floats(min_value=0.5, max_value=20.0), min_size=2, max_size=6),
    keep_count=st.integers(min_value=1, max_value=3),
    tie_break=st.sampled_from(["fifo", "deepest"]),
    tie_tolerance=st.sampled_from([0.0, 0.01, 0.1, 1.0]),
    engine=st.sampled_from(ENGINES),
)
def test_the_certificate_is_exact_or_the_gap_is_honest_and_below_the_tolerance(
    weights, keep_count, tie_break, tie_tolerance, engine
):
    keep_count = min(keep_count, len(weights))
    problem = KeepHeaviestProblem(tuple(weights), keep_count)
    cost_function = LeftOutWeightCost(problem)
    best_cost, _ = brute_force_best(tuple(weights), keep_count)

    result = engine(problem, cost_function, tie_break=tie_break, tie_tolerance=tie_tolerance).run()

    assert result.cost >= best_cost - 1e-9
    if result.optimal:
        assert result.certified_gap == 0.0
        assert result.cost == pytest.approx(best_cost, abs=1e-9)
    else:
        assert tie_tolerance > 0.0
        assert 0.0 < result.certified_gap < tie_tolerance
        assert result.cost - result.certified_gap <= best_cost + 1e-9
