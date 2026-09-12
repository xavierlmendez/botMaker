"""The two-hot optimizer's own contract on `AbstractOptimizer` (BL-48 slice 4).

What the optimizer adds to the base: the projection every forward pass applies, its settings, the
configuration it assembles from its injected objects (D-35 (4)), the stops it returns instead of
raising (D-35 (9)), and the one-run rule. The projection is pinned against the old
`_project_zero_sum`, pasted verbatim, to the last bit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torch")

import torch

from mllib.math.algorithms.abstract_optimizer import StopReason
from mllib.math.algorithms.two_hot_span.optimizer import (
    TwoHotSpanOptimizer,
    TwoHotSpanResult,
    TwoHotSpanSettings,
    project_orthogonal,
    training_cost,
)
from mllib.math.algorithms.two_hot_span.penalties import (
    CollisionPenalty,
    DiversityPenalty,
    LaplacianAdjacencyPenalty,
)
from mllib.math.algorithms.two_hot_span.projectors import RidgeProjector
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.graph.two_hot_span_problem import TwoHotSpanProblem, roach_graph
from mllib.math.learning_rate_schedule import CosineSchedule

from .two_hot_span_support import optimizer_for, run

NODE_TOTAL = 20
SPANNING_COUNT = 18


def _old_project_zero_sum(weights: torch.Tensor, normalize_columns: bool) -> torch.Tensor:
    """`_project_zero_sum` as the optimizer had it before the slice, verbatim."""
    node_total = weights.shape[0]
    ones = torch.ones(node_total, dtype=weights.dtype, device=weights.device)
    spanning_set_t = weights - torch.outer(ones, ones @ weights) / node_total
    if normalize_columns:
        spanning_set_t = spanning_set_t / torch.linalg.vector_norm(spanning_set_t, dim=0)
    return spanning_set_t


@pytest.fixture(scope="module")
def weights() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(NODE_TOTAL, SPANNING_COUNT, dtype=torch.float64, generator=generator)


@pytest.mark.parametrize("normalize_columns", [True, False])
def test_the_projection_onto_the_ones_complement_is_the_old_zero_sum_projection_to_the_last_bit(
    weights, normalize_columns
):
    ones = torch.ones(NODE_TOTAL, dtype=torch.float64)

    new = project_orthogonal(weights, ones, normalize_columns)
    old = _old_project_zero_sum(weights, normalize_columns)

    assert torch.equal(new, old)


def test_the_projection_leaves_every_column_orthogonal_to_the_constraint(weights):
    constraint = torch.tensor(np.arange(1.0, NODE_TOTAL + 1.0))
    projected = project_orthogonal(weights, constraint, False)
    # cᵀV is a sum of twenty products of order 20 x 2, so its rounding floor is ~1e-13, not 1e-16.
    residual = constraint @ projected
    assert torch.allclose(residual, torch.zeros(SPANNING_COUNT, dtype=torch.float64), atol=1e-11)
    assert float(residual.abs().max()) < 1e-11 * float((constraint @ constraint) ** 0.5)


def test_the_settings_defaults_are_the_loops_own_knobs():
    settings = TwoHotSpanSettings()
    assert (settings.step_count, settings.seed) == (200, 0)
    assert settings.normalize_columns is True
    assert settings.zero_sum_tolerance == 1e-12


@pytest.mark.parametrize(
    ("knobs", "message"),
    [({"step_count": -1}, "step_count"), ({"zero_sum_tolerance": -1e-12}, "zero_sum_tolerance")],
)
def test_the_settings_refuse_a_negative_knob(knobs, message):
    with pytest.raises(ValueError, match=message):
        TwoHotSpanSettings(**knobs)


def test_the_configuration_names_every_class_and_knob_and_omits_the_arrays():
    problem = TwoHotSpanProblem(roach_graph(5), 2, name="roach_g5")
    optimizer = TwoHotSpanOptimizer(
        problem,
        training_cost(problem, RidgeProjector(1e-3)),
        [CollisionPenalty(weight=10.0), DiversityPenalty(weight=2.0)],
        AdamStepRule(0.02, CosineSchedule(final_fraction=0.1)),
        settings=TwoHotSpanSettings(step_count=7, seed=4),
    )

    configuration = optimizer.configuration

    json.dumps(configuration)
    assert configuration["name"] == "TwoHotSpanOptimizer"
    assert (configuration["step_count"], configuration["seed"]) == (7, 4)
    assert configuration["problem"]["graph"] == "roach_g5"
    assert configuration["problem"]["cluster_count"] == 2
    assert configuration["cost"] == {
        "name": "SpanCost",
        "projector": {"name": "RidgeProjector", "epsilon": 1e-3},
    }
    assert configuration["penalties"] == [
        {"name": "CollisionPenalty", "weight": 10.0},
        {"name": "DiversityPenalty", "weight": 2.0},
    ]
    assert configuration["step_rule"] == {
        "name": "AdamStepRule",
        "learning_rate": 0.02,
        "schedule": {"name": "CosineSchedule", "final_fraction": 0.1},
    }
    assert configuration["initial_spanning_set"] == "seeded"
    assert "X" not in configuration["cost"]


def test_the_result_carries_the_configuration_it_ran_under():
    optimizer = optimizer_for(roach_graph(5), 2, step_count=3, collision_weight=1.0)
    result = optimizer.run()
    assert result.configuration == optimizer.configuration
    assert result.configuration["penalties"][0]["weight"] == 1.0


def test_an_initial_spanning_set_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="initial_spanning_set must be"):
        run(roach_graph(5), 2, step_count=1, initial=np.zeros((NODE_TOTAL, SPANNING_COUNT - 1)))


def test_a_zero_tolerance_stops_the_run_with_a_constraint_violation_and_still_reports():
    """The sum of a projected column is rounding noise, never exactly zero, so a tolerance of 0.0
    trips the check at initialisation or after the first step; the result still delivers."""
    result = run(roach_graph(5), 2, step_count=10, collision_weight=1.0, zero_sum_tolerance=0.0)

    assert result.stop_reason is StopReason.CONSTRAINT_VIOLATION
    assert "exceeds" in result.stop_detail
    assert result.steps_taken == 0
    assert np.isfinite(result.relaxed_objective)
    assert np.isfinite(result.rounded_cut)
    assert result.labels.shape == (NODE_TOTAL,)


def test_the_step_rule_is_bound_after_the_run_and_a_second_run_is_refused():
    optimizer = optimizer_for(roach_graph(5), 2, step_count=2)
    optimizer.run()

    assert optimizer.step_rule.is_bound
    with pytest.raises(RuntimeError, match="one run"):
        optimizer.run()


def test_the_spanning_set_is_the_parameters_under_the_domains_name():
    result = run(roach_graph(5), 2, step_count=1)
    assert isinstance(result, TwoHotSpanResult)
    assert result.spanning_set is result.parameters


# ------------------------------------------------------------------------------------------------
# Every stop site returns; the constraint seam; a penalty over the wrong graph.
# ------------------------------------------------------------------------------------------------


def constant_column_start() -> np.ndarray:
    """A start whose first column is constant: projected it is the zero column, and normalising
    that column divides 0 by 0."""
    start = np.random.default_rng(3).standard_normal((NODE_TOTAL, SPANNING_COUNT))
    start[:, 0] = 1.0
    return start


def test_a_constant_column_start_under_normalisation_stops_at_initialisation_with_nans():
    result = run(roach_graph(5), 2, step_count=5, initial=constant_column_start())

    assert result.stop_reason is StopReason.NON_FINITE
    assert "at initialisation" in result.stop_detail
    assert result.steps_taken == 0
    assert result.final_training_loss is None
    assert np.isnan(result.relaxed_objective) and np.isnan(result.rounded_cut)
    assert np.isnan(result.collision_measures).all()
    assert np.array_equal(result.labels, np.full(NODE_TOTAL, -1))


def test_the_same_start_without_normalisation_runs_its_budget():
    result = run(
        roach_graph(5), 2, step_count=5, initial=constant_column_start(), normalize_columns=False
    )

    assert result.stop_reason is StopReason.STEP_BUDGET
    assert result.steps_taken == 5
    assert np.isfinite(result.rounded_cut)


def test_a_graph_penalty_over_another_graphs_matrix_is_refused_at_construction():
    problem = TwoHotSpanProblem(roach_graph(5), 2)
    other = TwoHotSpanProblem(roach_graph(3), 2)

    with pytest.raises(ValueError, match=r"\(12, 12\), not the problem's \(20, 20\)"):
        TwoHotSpanOptimizer(
            problem,
            training_cost(problem),
            [LaplacianAdjacencyPenalty(other.laplacian, weight=0.5)],
            settings=TwoHotSpanSettings(step_count=1),
        )


CONSTRAINTS = {
    "ones": np.ones(NODE_TOTAL),
    "ramp": np.linspace(1.0, 2.0, NODE_TOTAL),
    "alternating": np.where(np.arange(NODE_TOTAL) % 2 == 0, 1.0, -1.0),
    "one_hot": np.eye(NODE_TOTAL)[3],
}


@settings(derandomize=True, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=200),
    name=st.sampled_from(sorted(CONSTRAINTS)),
    normalize=st.booleans(),
)
def test_the_projection_leaves_every_column_orthogonal_to_any_non_zero_constraint(
    seed, name, normalize
):
    weights = torch.tensor(np.random.default_rng(seed).standard_normal((NODE_TOTAL, 4)))
    constraint = torch.tensor(CONSTRAINTS[name])

    projected = project_orthogonal(weights, constraint, normalize)

    norms = torch.linalg.vector_norm(projected, dim=0)
    cosine = torch.abs(constraint @ projected) / (torch.linalg.vector_norm(constraint) * norms)
    assert float(cosine.max()) < 1e-9
    if normalize:
        assert torch.allclose(norms, torch.ones(4, dtype=torch.float64), rtol=1e-9, atol=0.0)


class SqrtDegreeProblem(TwoHotSpanProblem):
    """The BL-41 seam: the ncut constraint vector, sqrt(d), with nothing else overridden."""

    @property
    def constraint_vector(self) -> np.ndarray:
        return np.sqrt(np.diag(self.laplacian))


def test_a_problem_with_the_sqrt_degree_constraint_runs_its_budget_without_editing_the_optimizer():
    """The constraint check is cT v_j, so a subclass that changes c never trips it (BL-41)."""
    problem = SqrtDegreeProblem(roach_graph(5), 2)

    result = TwoHotSpanOptimizer(
        problem,
        training_cost(problem),
        [CollisionPenalty(weight=1.0)],
        settings=TwoHotSpanSettings(step_count=5, seed=0),
    ).run()

    assert result.stop_reason is StopReason.STEP_BUDGET
    assert result.steps_taken == 5
    assert result.max_zero_sum_violation <= 1e-12
    residual = problem.constraint_vector @ result.spanning_set
    assert np.abs(residual).max() <= 1e-12
