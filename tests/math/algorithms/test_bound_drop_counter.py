"""The opt-in bound-drop counter: does a child's bound ever fall below its parent's?

Consistency is not something A* needs here (D-23), so the counter is a measurement and nothing
else: off by default, stated by ``configuration`` when on, and without effect on what the search
expands or returns. The keep-heaviest toy from `test_a_star_search.py` supplies both outcomes — its
exact bound is monotone along the tree, and two deliberately loosened bounds are admissible but not
monotone, with drops whose size and depth are known by inspection.
"""

import json
from dataclasses import asdict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.math.algorithms.a_star_search import (
    BOUND_DROP_ROUNDING_TOLERANCE,
    AStarSearch,
    BoundDropCounter,
)
from mllib.math.algorithms.pruned_a_star_search import PrunedAStarSearch
from mllib.math.graph.nystrom_landmark_problem import (
    NystromCssCostFunction,
    NystromLandmarkProblem,
)
from tests.math.algorithms.test_a_star_search import (
    WEIGHTS,
    KeepHeaviestProblem,
    LeftOutWeightCost,
    UnorderedKeepHeaviestProblem,
)
from tests.math.graph.test_nystrom_landmark_problem import load_kernel


class LooseBelowDepthCost(LeftOutWeightCost):
    """Exact above ``loose_from`` picks, zero from there to the goal, exact at the goal.

    Zero is admissible for a non-negative objective, so A* stays a proof; every child at depth
    ``loose_from`` is bounded at zero under an exactly bounded parent, which is a drop of the whole
    parent bound: relative size 1.0, at that depth and nowhere else.
    """

    def __init__(self, problem: KeepHeaviestProblem, loose_from: int):
        super().__init__(problem)
        self.loose_from = loose_from

    def lower_bound(self, state):
        if self.problem.is_goal(state) or len(state) < self.loose_from:
            return super().lower_bound(state)
        return 0.0


class RoundingBelowParentCost(LeftOutWeightCost):
    """The exact bound, shaved by a fixed fraction at every pick below the root.

    Only a child that ties its parent can be shaved below it, so on ``WEIGHTS`` with two picks the
    shave produces exactly one strict drop, of relative size ``shave``, at depth 1.
    """

    def __init__(self, problem: KeepHeaviestProblem, shave: float = 1e-12):
        super().__init__(problem)
        self.shave = shave

    def lower_bound(self, state):
        exact = super().lower_bound(state)
        if not state or self.problem.is_goal(state):
            return exact
        return exact * (1.0 - self.shave)


def counted_run(engine, problem, cost_function, slack: float = 0.0):
    search = engine(problem, cost_function, count_bound_drops=True, bound_drop_slack=slack)
    return search.run(), search


def test_the_counter_is_off_by_default_and_the_configuration_says_so():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    search = AStarSearch(problem, LeftOutWeightCost(problem))

    assert search.configuration == {
        "engine": "AStarSearch",
        "count_bound_drops": False,
        "bound_drop_slack": 0.0,
    }
    search.run()
    assert search.bound_drops is None


def test_the_configuration_states_that_the_counter_was_on_and_with_what_slack():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    cost_function = LeftOutWeightCost(problem)

    exact = AStarSearch(problem, cost_function, count_bound_drops=True, bound_drop_slack=1e-12)
    pruned = PrunedAStarSearch(
        problem, cost_function, count_bound_drops=True, bound_drop_slack=2e-12
    )

    assert exact.configuration["count_bound_drops"] is True
    assert exact.configuration["bound_drop_slack"] == 1e-12
    assert pruned.configuration["count_bound_drops"] is True
    assert pruned.configuration["bound_drop_slack"] == 2e-12
    assert pruned.configuration["engine"] == "PrunedAStarSearch"
    assert pruned.bound_drop_slack == 2e-12


def test_a_negative_slack_is_rejected():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    with pytest.raises(ValueError, match="bound_drop_slack"):
        AStarSearch(problem, LeftOutWeightCost(problem), bound_drop_slack=-1e-12)


def test_a_monotone_bound_records_no_drop():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    _, search = counted_run(AStarSearch, problem, LeftOutWeightCost(problem))

    counter = search.bound_drops
    # Root: four one-pick children; the pick of the heaviest item: four goal children.
    assert counter.children_priced == 8
    assert counter.drops == 0
    assert counter.drops_beyond_rounding == 0
    assert counter.drops_by_depth == {}
    assert counter.worst_relative_drop == 0.0
    assert counter.worst_drop_depth is None
    assert counter.worst_drop_parent_bound is None
    assert counter.worst_drop_child_bound is None


def test_a_bound_that_loosens_below_the_root_records_every_drop_with_its_depth():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    _, search = counted_run(AStarSearch, problem, LooseBelowDepthCost(problem, loose_from=1))

    counter = search.bound_drops
    # The root is bounded exactly at 6.0 (leave out 1, 2 and 3); its four children at zero.
    assert counter.drops == 4
    assert counter.drops_beyond_rounding == 4
    assert counter.drops_by_depth == {1: 4}
    assert counter.worst_relative_drop == 1.0
    assert counter.worst_drop_depth == 1
    assert counter.worst_drop_parent_bound == pytest.approx(6.0)
    assert counter.worst_drop_child_bound == 0.0
    # Goal children are exact and non-negative: no goal sits below a parent bounded at zero.
    assert counter.children_priced > counter.drops


@pytest.mark.parametrize("engine", [AStarSearch, PrunedAStarSearch])
def test_the_depth_recorded_is_the_depth_at_which_the_bound_loosened(engine):
    # The pruned engine skips pushes (goal siblings, children above the incumbent) while consuming
    # their insertion indices, which is the contract the depth bookkeeping rests on.
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=3)

    _, search = counted_run(engine, problem, LooseBelowDepthCost(problem, loose_from=2))

    counter = search.bound_drops
    assert counter.drops > 0
    assert set(counter.drops_by_depth) == {2}
    assert counter.worst_drop_depth == 2
    assert counter.worst_relative_drop == 1.0


def test_the_pruned_engine_attributes_the_same_depths_as_the_exact_one():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=3)
    cost_function = LooseBelowDepthCost(problem, loose_from=2)

    _, exact = counted_run(AStarSearch, problem, cost_function)
    _, pruned = counted_run(PrunedAStarSearch, problem, cost_function)

    # Both expand the same states in the same order, so the same parents price the same children.
    assert asdict(pruned.bound_drops) == asdict(exact.bound_drops)


def test_a_drop_within_rounding_is_counted_strictly_but_not_beyond_rounding():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)

    _, search = counted_run(AStarSearch, problem, RoundingBelowParentCost(problem))

    counter = search.bound_drops
    # Only a child that ties its parent can be shaved below it: the pick of the heaviest item has
    # the root's own bound, 6.0, and its three siblings are well above it.
    assert counter.drops == 1
    assert counter.drops_by_depth == {1: 1}
    assert BOUND_DROP_ROUNDING_TOLERANCE > 1e-12
    assert counter.drops_beyond_rounding == 0
    assert counter.worst_relative_drop == 0.0
    assert counter.worst_drop_depth is None


def test_a_drop_just_above_the_relative_tolerance_is_beyond_rounding():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    shave = 10 * BOUND_DROP_ROUNDING_TOLERANCE

    _, search = counted_run(AStarSearch, problem, RoundingBelowParentCost(problem, shave))

    counter = search.bound_drops
    assert counter.drops == 1
    assert counter.drops_beyond_rounding == 1
    assert counter.worst_relative_drop == pytest.approx(shave, rel=1e-6)
    assert counter.worst_drop_depth == 1


def test_the_relative_and_absolute_allowances_add():
    # A drop of 1e-8 relative on a parent of 6.0 is 6e-8 absolute: above the relative allowance
    # alone, and within it once an absolute slack of 6e-8 is added on top.
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    shave = 1e-8
    cost_function = RoundingBelowParentCost(problem, shave)

    _, relative_only = counted_run(AStarSearch, problem, cost_function)
    _, with_slack = counted_run(AStarSearch, problem, cost_function, slack=6.0 * shave)

    assert relative_only.bound_drops.drops_beyond_rounding == 1
    assert with_slack.bound_drops.drops == 1
    assert with_slack.bound_drops.drops_beyond_rounding == 0


def test_a_drop_within_the_absolute_slack_is_rounding_even_when_the_parent_is_tiny():
    # The slice-G lesson, read for bounds: a parent bounded at rounding level makes any drop below
    # it look large relative to the parent, and only an allowance on the objective's scale can say
    # that a drop of 1e-16 on a total of 2 is nothing.
    problem = KeepHeaviestProblem((1.0, 1.049137434526428), keep_count=2)
    cost_function = LeftOutWeightCost(problem)
    # After the first pick the bound is total - 1.0 - 1.049...: exactly zero in exact arithmetic,
    # a few ulps above it in floating point, while the goal child sums the two picks first and lands
    # on zero exactly.
    parent_bound = cost_function.lower_bound((0,))
    assert 0.0 < parent_bound < 1e-15

    _, unslacked = counted_run(AStarSearch, problem, cost_function)
    _, slacked = counted_run(AStarSearch, problem, cost_function, slack=1e-12 * cost_function.total)

    assert unslacked.bound_drops.drops == 1
    assert unslacked.bound_drops.drops_beyond_rounding == 1
    assert unslacked.bound_drops.worst_relative_drop == 1.0
    assert slacked.bound_drops.drops == 1
    assert slacked.bound_drops.drops_beyond_rounding == 0
    assert slacked.bound_drops.worst_relative_drop == 0.0


@pytest.mark.parametrize("engine", [AStarSearch, PrunedAStarSearch])
@pytest.mark.parametrize("problem_type", [KeepHeaviestProblem, UnorderedKeepHeaviestProblem])
@pytest.mark.parametrize("loose_from", [None, 1, 2])
def test_counting_changes_neither_the_result_nor_the_expansions(engine, problem_type, loose_from):
    problem = problem_type(WEIGHTS, keep_count=3)
    cost_function = (
        LeftOutWeightCost(problem)
        if loose_from is None
        else LooseBelowDepthCost(problem, loose_from)
    )

    plain = engine(problem, cost_function).run()
    counted, _ = counted_run(engine, problem, cost_function)

    assert counted == plain


def test_a_parent_bounded_at_zero_makes_a_drop_below_it_infinite():
    counter = BoundDropCounter()

    counter.record(0.0, [-1.0, 0.0, 2.0], child_depth=3)

    assert counter.drops == 1
    assert counter.worst_relative_drop == float("inf")
    assert counter.worst_drop_depth == 3


def test_the_counter_is_fresh_on_every_run():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    search = AStarSearch(
        problem, LooseBelowDepthCost(problem, loose_from=1), count_bound_drops=True
    )

    search.run()
    first = asdict(search.bound_drops)
    search.run()

    assert asdict(search.bound_drops) == first


def test_the_counter_serialises_to_json_for_a_result_record():
    problem = KeepHeaviestProblem(WEIGHTS, keep_count=2)
    _, search = counted_run(AStarSearch, problem, LooseBelowDepthCost(problem, loose_from=1))

    record = json.loads(json.dumps(asdict(search.bound_drops)))

    assert record["drops"] == 4
    assert record["drops_by_depth"] == {"1": 4}


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    weights=st.lists(st.floats(min_value=0.5, max_value=50.0), min_size=2, max_size=7),
    keep_count=st.integers(min_value=1, max_value=4),
)
def test_the_exact_keep_heaviest_bound_is_monotone_on_random_instances(weights, keep_count):
    # A child's best finish is one admissible choice of the parent's, so the child's bound is never
    # below the parent's in exact arithmetic: the toy's exact bound is consistent. In floating point
    # the parent and the child sum the same weights in a different order, so strict drops of a few
    # ulps do occur, and the counter must call them rounding once told the objective's scale.
    keep_count = min(keep_count, len(weights))
    problem = KeepHeaviestProblem(tuple(weights), keep_count)
    cost_function = LeftOutWeightCost(problem)

    _, search = counted_run(AStarSearch, problem, cost_function, slack=1e-12 * cost_function.total)

    assert search.bound_drops.drops_beyond_rounding == 0
    assert search.bound_drops.worst_drop_depth is None


def test_the_counter_runs_on_the_nystrom_bound_through_the_batched_path():
    # Integration only: the real bound's answer is measured on the S1 grid, not asserted here.
    problem = NystromLandmarkProblem(load_kernel("rbf_chain_6x6.csv"), landmark_count=3)
    cost_function = NystromCssCostFunction(problem)

    plain = PrunedAStarSearch(problem, cost_function).run()
    counted, search = counted_run(PrunedAStarSearch, problem, cost_function)

    counter = search.bound_drops
    assert counted.state == plain.state
    assert counted.nodes_expanded == plain.nodes_expanded
    assert counter.children_priced > 0
    assert 0 <= counter.drops <= counter.children_priced
    assert all(1 <= depth <= problem.landmark_count for depth in counter.drops_by_depth)
