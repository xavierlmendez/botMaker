"""Learning-rate schedules: the multiplier a step rule applies to its learning rate at each step.

One concept, four implementations (D-35 (1)), each a frozen dataclass whose knobs are its shape
(D-35 (3)). The run's length is not a knob of the schedule: the optimizer passes its own step
budget at each call, so the horizon has one owner. Pure Python floats, no array library.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class AbstractLearningRateSchedule(ABC):
    """The multiplier on a step rule's learning rate at step ``step`` of a ``step_count``-step run.

    ``multiplier(step, step_count)`` is dimensionless: 1.0 means the learning rate as set. A
    schedule shapes the trajectory and never enters a reported number (D-31).
    """

    __slots__ = ()

    @abstractmethod
    def multiplier(self, step: int, step_count: int) -> float:
        """The factor in force for ``step`` in ``0 .. step_count - 1``.

        A step rule may evaluate it once past the horizon, at ``step_count``, after its last step;
        what a decaying schedule returns there is not a learning rate any step uses.
        """
        raise NotImplementedError


def _last_step(step_count: int) -> int:
    """The index the decaying schedules interpolate to, never below 1 so a 1-step run divides."""
    return max(step_count - 1, 1)


def _check_fraction(final_fraction: float) -> None:
    if not 0.0 <= final_fraction <= 1.0:
        raise ValueError(f"final_fraction must lie in [0, 1]; got {final_fraction!r}")


@dataclass(frozen=True, slots=True)
class ConstantSchedule(AbstractLearningRateSchedule):
    """1.0 at every step: the learning rate as set, the default."""

    def multiplier(self, step: int, step_count: int) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class LinearSchedule(AbstractLearningRateSchedule):
    """A straight ramp from 1.0 at step 0 to ``final_fraction`` at the last step."""

    final_fraction: float = 0.0

    def __post_init__(self) -> None:
        _check_fraction(self.final_fraction)

    def multiplier(self, step: int, step_count: int) -> float:
        fraction = float(self.final_fraction)
        return 1.0 + (fraction - 1.0) * (step / _last_step(step_count))


@dataclass(frozen=True, slots=True)
class CosineSchedule(AbstractLearningRateSchedule):
    """A half cosine from 1.0 at step 0 down to ``final_fraction`` at the last step, never rising."""

    final_fraction: float = 0.0

    def __post_init__(self) -> None:
        _check_fraction(self.final_fraction)

    def multiplier(self, step: int, step_count: int) -> float:
        fraction = float(self.final_fraction)
        # `math.pi * step / last_step`, left to right: the rounding the refactor snapshot pins.
        last_step = _last_step(step_count)
        return fraction + (1.0 - fraction) * 0.5 * (1.0 + math.cos(math.pi * step / last_step))


@dataclass(frozen=True, slots=True)
class WarmupCosineSchedule(AbstractLearningRateSchedule):
    """0 at step 0, a linear rise to 1.0 at step ``warmup_steps``, then the cosine down.

    With ``warmup_steps`` past the last step the decay never runs and the multiplier holds at 1.0
    after the warm-up; the optimizer refuses ``warmup_steps >= step_count`` up front.
    """

    warmup_steps: int = 0
    final_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must not be negative; got {self.warmup_steps!r}")
        _check_fraction(self.final_fraction)

    def multiplier(self, step: int, step_count: int) -> float:
        fraction = float(self.final_fraction)
        warmup = int(self.warmup_steps)
        if warmup > 0 and step < warmup:
            return step / warmup
        remaining = step_count - 1 - warmup
        if remaining <= 0:
            return 1.0
        progress = (step - warmup) / remaining
        return fraction + (1.0 - fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
