"""The loop `AbstractOptimizer` owns, proved on a one-dimensional quadratic (BL-48 slice 4).

A subclass small enough to read fills the three seams — `_begin`, `_step`, `_assemble` — with a
hand-written descent on f(x) = (x - 3)², and the tests ask only about what the base promises: the
budget, the stop as a value, the recorder guard, the result. No torch: the contract is
array-agnostic (D-35 (6)).
"""

from __future__ import annotations

import pytest

from mllib.math.algorithms.abstract_optimizer import (
    AbstractOptimizer,
    OptimizerResult,
    Stop,
    StopReason,
)
from mllib.math.recorder import AbstractStepRecorder, NullRecorder

TARGET = 3.0
LEARNING_RATE = 0.25


class QuadraticDescent(AbstractOptimizer):
    """x <- x - lr * 2 (x - 3), stopping on request at a named step or before any step."""

    def __init__(self, *, step_count, start=0.0, stop_at=None, stop_before_start=False, **kwargs):
        super().__init__(step_count=step_count, **kwargs)
        self.start = float(start)
        self.stop_at = stop_at
        self.stop_before_start = stop_before_start
        self.x = None
        self.begun = False

    @property
    def configuration(self):
        return {"name": type(self).__name__, "step_count": self.step_count}

    def _begin(self):
        self.begun = True
        self.x = self.start
        if self.stop_before_start:
            return Stop(StopReason.NON_FINITE, "asked to stop before the first step")
        return None

    def _step(self, step):
        if step == self.stop_at:
            return Stop(StopReason.CONSTRAINT_VIOLATION, f"asked to stop at step {step}")
        loss = (self.x - TARGET) ** 2
        self.x = self.x - LEARNING_RATE * 2.0 * (self.x - TARGET)
        return loss

    def _iterate(self):
        return self.x

    def _recorder_extras(self):
        return {"x": self.x}

    def _assemble(self, steps_taken, final_training_loss, stop):
        return OptimizerResult(
            parameters=self.x,
            final_training_loss=final_training_loss,
            steps_taken=steps_taken,
            stop_reason=stop.reason,
            stop_detail=stop.detail,
            configuration=self.configuration,
        )


class SpyRecorder(AbstractStepRecorder):
    """Every call the loop makes, in order, with what it was handed."""

    def __init__(self):
        super().__init__()
        self.steps = []
        self.ends = []

    def record_step(self, step, training_loss, spanning_set, **extras):
        self.steps.append((step, training_loss, spanning_set, dict(extras)))

    def record_end(self, run, **extras):
        self.ends.append(run)

    def describe_result(self, result):
        return {}


def test_a_run_takes_its_whole_budget_and_says_so():
    result = QuadraticDescent(step_count=8).run()

    assert result.stop_reason is StopReason.STEP_BUDGET
    assert result.steps_taken == 8
    assert "8" in result.stop_detail
    assert result.final_training_loss == pytest.approx((TARGET * 0.5**7) ** 2)
    # x_8 - 3 = -3 / 2^8, the geometric tail of eight halvings.
    assert result.parameters == pytest.approx(TARGET - TARGET * 0.5**8)


def test_a_stop_from_a_step_ends_the_run_with_the_completed_steps_and_that_reason():
    result = QuadraticDescent(step_count=8, stop_at=3).run()

    assert result.stop_reason is StopReason.CONSTRAINT_VIOLATION
    assert result.stop_detail == "asked to stop at step 3"
    assert result.steps_taken == 3
    assert result.final_training_loss == pytest.approx((TARGET * 0.5**2) ** 2)


def test_a_stop_from_begin_gives_no_steps_and_no_loss():
    result = QuadraticDescent(step_count=8, stop_before_start=True).run()

    assert result.stop_reason is StopReason.NON_FINITE
    assert result.steps_taken == 0
    assert result.final_training_loss is None
    assert result.parameters == 0.0


def test_the_recorder_sees_one_step_per_completed_step_with_the_extras_and_one_end():
    spy = SpyRecorder()
    result = QuadraticDescent(step_count=4, stop_at=3, recorder=spy).run()

    assert [step for step, *_ in spy.steps] == [0, 1, 2]
    assert [loss for _, loss, *_ in spy.steps] == pytest.approx(
        [TARGET**2, (TARGET * 0.5) ** 2, (TARGET * 0.25) ** 2]
    )
    assert all(extras == {"x": iterate} for _, _, iterate, extras in spy.steps)
    assert spy.ends == [result]


def test_the_default_recorder_is_the_null_one_and_no_call_reaches_it():
    """`NullRecorder` raises on any call, so a run that finishes proves the guard held."""
    optimizer = QuadraticDescent(step_count=5)

    assert isinstance(optimizer.recorder, NullRecorder)
    assert optimizer.run().steps_taken == 5


def test_a_zero_budget_runs_begin_only():
    optimizer = QuadraticDescent(step_count=0, start=1.5)
    result = optimizer.run()

    assert optimizer.begun
    assert result.steps_taken == 0
    assert result.final_training_loss is None
    assert result.parameters == 1.5
    assert result.stop_reason is StopReason.STEP_BUDGET


def test_a_negative_budget_is_refused():
    with pytest.raises(ValueError, match="step_count must not be negative"):
        QuadraticDescent(step_count=-1)


def test_an_optimizer_serves_one_run():
    optimizer = QuadraticDescent(step_count=2)
    optimizer.run()
    with pytest.raises(RuntimeError, match="one run"):
        optimizer.run()


def test_the_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractOptimizer(step_count=1)  # type: ignore[abstract]


def test_the_result_names_the_configuration_it_ran_under():
    result = QuadraticDescent(step_count=3).run()
    assert result.configuration == {"name": "QuadraticDescent", "step_count": 3}


class RaisingRecorder(AbstractStepRecorder):
    """A recorder that fails on its first step, as an injected collaborator might."""

    def record_step(self, step, training_loss, spanning_set, **extras):
        raise RuntimeError("the watcher fell over")

    def record_end(self, run, **extras):
        raise AssertionError("never reached")

    def describe_result(self, result):
        return {}


def test_a_recorder_that_raises_mid_run_leaves_the_optimizer_spent():
    """The run is over either way: the state belongs to it, so a second run is still refused."""
    optimizer = QuadraticDescent(step_count=3, recorder=RaisingRecorder())

    with pytest.raises(RuntimeError, match="fell over"):
        optimizer.run()

    assert optimizer.begun
    assert optimizer._has_run is True
    with pytest.raises(RuntimeError, match="one run"):
        optimizer.run()


class BareDescent(QuadraticDescent):
    """The quadratic without its own extras: the base's default is an empty mapping."""

    def _recorder_extras(self):
        return super(QuadraticDescent, self)._recorder_extras()


def test_the_recorder_extras_default_to_an_empty_mapping():
    spy = SpyRecorder()
    BareDescent(step_count=2, recorder=spy).run()

    assert [extras for _, _, _, extras in spy.steps] == [{}, {}]
