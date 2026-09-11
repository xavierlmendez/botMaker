"""The four learning-rate schedules against the closure they replace (BL-48 slice 3, D-35 (1)).

The optimizer used to carry one `learning_rate_lambda(config)` closure that branched on a name.
That closure is pasted here verbatim as the oracle, its config reads turned into parameters, and
each schedule object's multiplier sequence is asserted equal to it to the last bit over a thousand
steps -- the equality is against the old code, not the new. The rest is the contract: knobs read by
`describe()`, refusal at construction, a one-step run that still divides, and the shape claims
(where each schedule starts, where it ends, that it never rises) as hypothesis properties over the
knob domain.

Deterministic: pure Python floats, no arrays, no randomness beyond ``derandomize=True``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mllib.describe import describe
from mllib.math.learning_rate_schedule import (
    AbstractLearningRateSchedule,
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    WarmupCosineSchedule,
)

STEP_COUNT = 1_000
FRACTION = 0.1
WARMUP = 50
# 1 and 2 are where `max(step_count - 1, 1)` clamps; 1000 is a run long enough to read a shape off.
STEP_COUNTS = [1, 2, STEP_COUNT]


def _old_learning_rate_lambda(
    name: str, final_fraction: float, step_count: int, warmup_steps: int
) -> Callable[[int], float]:
    """`learning_rate_lambda` as the optimizer had it before the slice, verbatim.

    Only the reads off ``config`` became the four parameters; every expression is the original.
    """
    fraction = float(final_fraction)
    last_step = max(step_count - 1, 1)
    warmup = int(warmup_steps)

    def factor(step: int) -> float:
        if name == "constant":
            return 1.0
        if name == "linear":
            return 1.0 + (fraction - 1.0) * (step / last_step)
        if name == "cosine":
            return fraction + (1.0 - fraction) * 0.5 * (1.0 + math.cos(math.pi * step / last_step))
        # warmup_cosine
        if warmup > 0 and step < warmup:
            return step / warmup
        remaining = step_count - 1 - warmup
        if remaining <= 0:
            return 1.0
        progress = (step - warmup) / remaining
        return fraction + (1.0 - fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


# (name the old closure branched on, the schedule at the default knobs, the same at one non-default
# setting, the old closure's knobs for that non-default setting)
SCHEDULES = [
    ("constant", ConstantSchedule(), ConstantSchedule(), (0.0, 0)),
    ("linear", LinearSchedule(), LinearSchedule(final_fraction=FRACTION), (FRACTION, 0)),
    ("cosine", CosineSchedule(), CosineSchedule(final_fraction=FRACTION), (FRACTION, 0)),
    (
        "warmup_cosine",
        WarmupCosineSchedule(),
        WarmupCosineSchedule(warmup_steps=WARMUP, final_fraction=FRACTION),
        (FRACTION, WARMUP),
    ),
]
IDS = [entry[0] for entry in SCHEDULES]


def _sequence(schedule: AbstractLearningRateSchedule, step_count: int) -> list[float]:
    return [schedule.multiplier(step, step_count) for step in range(step_count)]


# ------------------------------------------------------------------------------------------------
# The old closure, to the last bit.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("step_count", STEP_COUNTS)
@pytest.mark.parametrize(("name", "default", "_tuned", "_knobs"), SCHEDULES, ids=IDS)
def test_the_default_schedule_is_the_old_closure_at_its_defaults_to_the_last_bit(
    name, default, _tuned, _knobs, step_count
):
    old = _old_learning_rate_lambda(name, 0.0, step_count, 0)

    assert _sequence(default, step_count) == [old(step) for step in range(step_count)]


@pytest.mark.parametrize("step_count", STEP_COUNTS)
@pytest.mark.parametrize(("name", "_default", "tuned", "knobs"), SCHEDULES, ids=IDS)
def test_a_tuned_schedule_is_the_old_closure_at_the_same_knobs_to_the_last_bit(
    name, _default, tuned, knobs, step_count
):
    fraction, warmup = knobs
    old = _old_learning_rate_lambda(name, fraction, step_count, warmup)

    assert _sequence(tuned, step_count) == [old(step) for step in range(step_count)]


@pytest.mark.parametrize("step_count", [5, 10])
def test_a_warm_up_at_or_past_the_horizon_is_the_old_closure_to_the_last_bit(step_count):
    """`remaining <= 0`: the branch the closure took when the warm-up outlasted the run."""
    warmup = 10
    old = _old_learning_rate_lambda("warmup_cosine", FRACTION, step_count, warmup)
    schedule = WarmupCosineSchedule(warmup_steps=warmup, final_fraction=FRACTION)

    assert _sequence(schedule, step_count) == [old(step) for step in range(step_count)]


# ------------------------------------------------------------------------------------------------
# The contract.
# ------------------------------------------------------------------------------------------------


def test_the_abstract_schedule_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractLearningRateSchedule()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("schedule_class", "knobs"),
    [
        (ConstantSchedule, []),
        (LinearSchedule, ["final_fraction"]),
        (CosineSchedule, ["final_fraction"]),
        (WarmupCosineSchedule, ["warmup_steps", "final_fraction"]),
    ],
    ids=IDS,
)
def test_describe_reads_exactly_the_schedules_knobs(schedule_class, knobs):
    descriptor = describe(schedule_class)

    assert descriptor["params"] == knobs
    assert descriptor["kind"] == "math"


@pytest.mark.parametrize(("_name", "_default", "tuned", "_knobs"), SCHEDULES, ids=IDS)
def test_every_schedule_is_a_frozen_schedule(_name, _default, tuned, _knobs):
    assert isinstance(tuned, AbstractLearningRateSchedule)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        # A schedule with a knob raises FrozenInstanceError; the knob-free one has no field to
        # freeze and its ``__slots__`` refuse the attribute with TypeError instead.
        tuned.final_fraction = 0.5  # type: ignore[misc]


def test_schedules_compare_on_their_knobs():
    assert CosineSchedule(final_fraction=0.1) == CosineSchedule(final_fraction=0.1)
    assert CosineSchedule(final_fraction=0.1) != CosineSchedule(final_fraction=0.2)
    assert CosineSchedule(final_fraction=0.1) != LinearSchedule(final_fraction=0.1)
    assert ConstantSchedule() == ConstantSchedule()
    assert hash(WarmupCosineSchedule(5, 0.1)) == hash(WarmupCosineSchedule(5, 0.1))


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
@pytest.mark.parametrize(
    "build",
    [
        lambda fraction: LinearSchedule(final_fraction=fraction),
        lambda fraction: CosineSchedule(final_fraction=fraction),
        lambda fraction: WarmupCosineSchedule(final_fraction=fraction),
    ],
    ids=["linear", "cosine", "warmup_cosine"],
)
def test_a_final_fraction_outside_the_unit_interval_is_refused_at_construction(build, fraction):
    with pytest.raises(ValueError, match="final_fraction"):
        build(fraction)


def test_a_negative_warmup_is_refused_at_construction():
    with pytest.raises(ValueError, match="warmup_steps"):
        WarmupCosineSchedule(warmup_steps=-1)


@pytest.mark.parametrize(("_name", "_default", "tuned", "_knobs"), SCHEDULES, ids=IDS)
def test_a_one_step_run_still_divides(_name, _default, tuned, _knobs):
    """`max(step_count - 1, 1)` is what keeps step 0 of a 1-step run off a division by zero."""
    assert math.isfinite(tuned.multiplier(0, 1))


@pytest.mark.parametrize(("warmup", "step_count"), [(10, 5), (10, 10), (3, 3)])
def test_a_warm_up_past_the_horizon_holds_the_multiplier_at_one(warmup, step_count):
    """The decay never runs when the warm-up outlasts the run: 1.0 from the warm-up on."""
    schedule = WarmupCosineSchedule(warmup_steps=warmup, final_fraction=FRACTION)

    for step in range(warmup, warmup + 5):
        assert schedule.multiplier(step, step_count) == 1.0


def test_a_decaying_schedule_read_past_the_horizon_is_outside_its_domain():
    """The one read a step rule makes at ``step_count`` after its last step, pinned as it is.

    `LambdaLR` advances once per step, so the rule evaluates the schedule at the horizon once, and
    no step uses the value. A linear ramp to 0 lands at -0.25 there. The number is inherited from
    the closure the schedules replaced and the refactor snapshot forbids clamping it now; this
    test says so, so a later slice changes it knowingly.
    """
    assert LinearSchedule(final_fraction=0.0).multiplier(5, 5) == -0.25


# ------------------------------------------------------------------------------------------------
# The shapes, over the knob domain.
# ------------------------------------------------------------------------------------------------

fractions = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
step_counts = st.integers(min_value=2, max_value=500)


@settings(derandomize=True, deadline=None)
@given(fraction=fractions, step_count=step_counts)
def test_the_cosine_schedule_starts_at_one_ends_at_the_fraction_and_never_rises(
    fraction, step_count
):
    sequence = _sequence(CosineSchedule(final_fraction=fraction), step_count)

    assert sequence[0] == pytest.approx(1.0, rel=1e-12)
    assert sequence[-1] == pytest.approx(fraction, rel=1e-12, abs=1e-15)
    assert all(later <= earlier for earlier, later in pairwise(sequence))


@settings(derandomize=True, deadline=None)
@given(fraction=fractions, step_count=step_counts)
def test_the_linear_schedule_starts_at_one_ends_at_the_fraction_and_is_affine_in_the_step(
    fraction, step_count
):
    sequence = _sequence(LinearSchedule(final_fraction=fraction), step_count)
    slope = (fraction - 1.0) / (step_count - 1)

    assert sequence[0] == pytest.approx(1.0, rel=1e-12)
    assert sequence[-1] == pytest.approx(fraction, rel=1e-12, abs=1e-15)
    for step, value in enumerate(sequence):
        assert value == pytest.approx(1.0 + slope * step, rel=1e-12, abs=1e-15)


@settings(derandomize=True, deadline=None)
@given(fraction=fractions, step_count=st.integers(min_value=3, max_value=500), data=st.data())
def test_the_warmup_cosine_schedule_is_zero_then_one_at_the_warmup_then_never_rises(
    fraction, step_count, data
):
    warmup = data.draw(st.integers(min_value=1, max_value=step_count - 2), label="warmup")
    sequence = _sequence(WarmupCosineSchedule(warmup, fraction), step_count)

    assert sequence[0] == 0.0
    assert sequence[warmup] == 1.0
    tail = sequence[warmup:]
    assert all(later <= earlier for earlier, later in pairwise(tail))
    assert sequence[-1] == pytest.approx(fraction, rel=1e-12, abs=1e-15)


def test_the_constant_schedule_is_one_everywhere():
    assert _sequence(ConstantSchedule(), 37) == [1.0] * 37
