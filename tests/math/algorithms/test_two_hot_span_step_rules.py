"""Adam as an injected step rule (BL-48 slice 3, D-35 (1), (3)).

`AdamStepRule` wraps the same `torch.optim.Adam` and `LambdaLR` the optimizer used to build inside
its loop, so the first claim is bit identity: a leaf moved by the rule is, after every step,
`torch.equal` to the same leaf moved by a hand-built Adam under the same schedule. The rest is the
contract of a stateful injected object: it binds once and refuses twice, it reports the learning
rate the next step will use, it compares on its knobs and not on its moments, and a copy or a
pickle arrives unbound. The two transitional adapters that still build the rule from the config
are pinned so slice 4 can remove them knowingly.

Deterministic: one seeded leaf, one fixed quadratic loss, no clock.
"""

from __future__ import annotations

import copy
import pickle

import pytest

pytest.importorskip("torch")

import torch

from mllib.describe import describe
from mllib.math.algorithms.two_hot_span.step_rules import AdamStepRule
from mllib.math.learning_rate_schedule import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    WarmupCosineSchedule,
)
from mllib.math.step_rule import AbstractStepRule

STEP_COUNT = 20
LEARNING_RATE = 0.05
SHAPE = (6, 4)


def seeded_leaf(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*SHAPE, dtype=torch.float64, generator=generator).requires_grad_(True)


def quadratic(leaf: torch.Tensor) -> torch.Tensor:
    """A fixed bowl with a gradient at every point the runs visit."""
    return torch.sum((leaf - 0.5) ** 2 * torch.arange(1, SHAPE[1] + 1, dtype=torch.float64))


SCHEDULES = [
    pytest.param(ConstantSchedule(), id="constant"),
    pytest.param(LinearSchedule(final_fraction=0.1), id="linear"),
    pytest.param(CosineSchedule(final_fraction=0.1), id="cosine"),
    # Warm-up: the learning rate is exactly 0.0 at step 0, so step 0 is a pure moment update.
    pytest.param(WarmupCosineSchedule(warmup_steps=3, final_fraction=0.1), id="warmup_cosine"),
]


# ------------------------------------------------------------------------------------------------
# Bit identity with the hand-built pair.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_rule_moves_the_leaf_exactly_as_a_hand_built_adam_and_lambda_lr_do(schedule):
    ruled = seeded_leaf()
    by_hand = seeded_leaf()
    assert torch.equal(ruled, by_hand)

    rule = AdamStepRule(LEARNING_RATE, schedule)
    rule.bind(ruled, STEP_COUNT)
    optimizer = torch.optim.Adam([by_hand], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule.multiplier(step, STEP_COUNT)
    )

    for _ in range(STEP_COUNT):
        rule.zero_gradient()
        quadratic(ruled).backward()
        rule.step()

        optimizer.zero_grad(set_to_none=True)
        quadratic(by_hand).backward()
        optimizer.step()
        scheduler.step()

        assert torch.equal(ruled, by_hand)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_learning_rate_in_force_is_the_scheduled_rate_at_every_step(schedule):
    leaf = seeded_leaf()
    rule = AdamStepRule(LEARNING_RATE, schedule)
    rule.bind(leaf, STEP_COUNT)
    by_hand = seeded_leaf()
    optimizer = torch.optim.Adam([by_hand], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule.multiplier(step, STEP_COUNT)
    )

    for step in range(STEP_COUNT):
        in_force = rule.learning_rate_in_force()
        # `LambdaLR` sets `base_lr * multiplier`, the same product, so this is bit identity.
        assert in_force == LEARNING_RATE * schedule.multiplier(step, STEP_COUNT)
        assert in_force == float(optimizer.param_groups[0]["lr"])

        rule.zero_gradient()
        quadratic(leaf).backward()
        rule.step()
        optimizer.zero_grad(set_to_none=True)
        quadratic(by_hand).backward()
        optimizer.step()
        scheduler.step()


# ------------------------------------------------------------------------------------------------
# The contract of a stateful injected object.
# ------------------------------------------------------------------------------------------------


def test_zero_gradient_clears_the_leafs_gradient_to_none():
    leaf = seeded_leaf()
    rule = AdamStepRule(LEARNING_RATE)
    rule.bind(leaf, STEP_COUNT)
    quadratic(leaf).backward()
    assert leaf.grad is not None

    rule.zero_gradient()

    assert leaf.grad is None


def test_binding_a_tensor_that_does_not_require_grad_is_refused():
    rule = AdamStepRule(LEARNING_RATE)
    frozen_leaf = seeded_leaf().detach()

    with pytest.raises(ValueError, match="leaf tensor that requires grad"):
        rule.bind(frozen_leaf, STEP_COUNT)


def test_binding_a_non_leaf_is_refused():
    rule = AdamStepRule(LEARNING_RATE)
    non_leaf = seeded_leaf() * 2.0
    assert non_leaf.requires_grad and not non_leaf.is_leaf

    with pytest.raises(ValueError, match="leaf tensor that requires grad"):
        rule.bind(non_leaf, STEP_COUNT)


def test_the_learning_rate_in_force_after_the_last_step_is_the_multiplier_past_the_horizon():
    """The read no step uses: `LambdaLR` has advanced to ``step_count``, outside the schedule."""
    leaf = seeded_leaf()
    rule = AdamStepRule(LEARNING_RATE, LinearSchedule(final_fraction=0.0))
    rule.bind(leaf, 5)
    for _ in range(5):
        rule.zero_gradient()
        quadratic(leaf).backward()
        rule.step()

    assert rule.learning_rate_in_force() == LEARNING_RATE * -0.25


def test_a_rule_serves_one_run_and_refuses_a_second_bind():
    rule = AdamStepRule(LEARNING_RATE)
    rule.bind(seeded_leaf(), STEP_COUNT)

    with pytest.raises(RuntimeError, match="one run"):
        rule.bind(seeded_leaf(1), STEP_COUNT)


@pytest.mark.parametrize(
    "call",
    [
        lambda rule: rule.step(),
        lambda rule: rule.learning_rate_in_force(),
        lambda rule: rule.zero_gradient(),
    ],
    ids=["step", "learning_rate_in_force", "zero_gradient"],
)
def test_using_a_rule_before_binding_it_is_refused(call):
    rule = AdamStepRule(LEARNING_RATE)
    assert not rule.is_bound

    with pytest.raises(RuntimeError, match="not bound"):
        call(rule)


def test_describe_reads_the_learning_rate_and_the_schedule_as_the_knobs():
    descriptor = describe(AdamStepRule)

    assert descriptor["params"] == ["learning_rate", "schedule"]
    assert descriptor["kind"] == "math"


def test_the_defaults_are_the_two_hot_learning_rate_under_the_constant_schedule():
    rule = AdamStepRule()

    assert isinstance(rule, AbstractStepRule)
    # 0.05 is the learning rate every two-hot run has used since slice 2.2.
    assert rule.learning_rate == 0.05
    assert rule.schedule == ConstantSchedule()


def test_rules_compare_on_their_knobs_whether_or_not_they_have_run():
    bound = AdamStepRule(LEARNING_RATE)
    bound.bind(seeded_leaf(), STEP_COUNT)
    quadratic(bound._bound()[0].param_groups[0]["params"][0]).backward()
    bound.step()
    fresh = AdamStepRule(LEARNING_RATE)

    assert bound == fresh
    assert hash(bound) == hash(fresh)
    assert AdamStepRule(LEARNING_RATE) != AdamStepRule(2 * LEARNING_RATE)
    assert AdamStepRule(LEARNING_RATE) != AdamStepRule(LEARNING_RATE, CosineSchedule(0.1))


def test_a_negative_learning_rate_is_refused_at_construction():
    with pytest.raises(ValueError, match="learning_rate"):
        AdamStepRule(-0.01)


@pytest.mark.parametrize("learning_rate", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_non_finite_learning_rate_is_refused_at_construction(learning_rate):
    with pytest.raises(ValueError, match="finite"):
        AdamStepRule(learning_rate)


@pytest.mark.parametrize(
    "duplicate",
    [copy.deepcopy, lambda rule: pickle.loads(pickle.dumps(rule))],
    ids=["copy", "pickle"],
)
def test_a_copied_or_pickled_rule_keeps_its_knobs_and_arrives_unbound(duplicate):
    rule = AdamStepRule(LEARNING_RATE, CosineSchedule(0.1))
    rule.bind(seeded_leaf(), STEP_COUNT)

    twin = duplicate(rule)

    assert twin == rule
    assert not twin.is_bound
    twin.bind(seeded_leaf(), STEP_COUNT)  # a fresh binding is allowed on the twin
    assert twin.is_bound


# ------------------------------------------------------------------------------------------------
# The transitional adapters, pinned so slice 4 removes them knowingly.
# ------------------------------------------------------------------------------------------------
