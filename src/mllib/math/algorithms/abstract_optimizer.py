"""The optimizer contract: a descent loop over injected math objects, the analogue of a search.

The base owns what every optimizer shares — the step budget, the recorder guard (D-32), the stop
check and the assembly of the result — and leaves to the subclass what its problem decides: how
the parameters start, what one step does, and which numbers the run delivers. It mirrors
`AbstractGraphAlgorithm`: `run()` is owned here and never overridden; `_step` is the seam a
variant fills. A run that stops early is still a run and returns a result with its reason
(D-35 (9)); nothing here raises for a numeric stop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mllib.math.recorder import AbstractStepRecorder, NullRecorder


class StopReason(Enum):
    """Why a run ended: its budget, or a stop condition a step tripped."""

    STEP_BUDGET = "step_budget"
    NON_FINITE = "non_finite"
    CONSTRAINT_VIOLATION = "constraint_violation"


@dataclass(frozen=True, slots=True)
class Stop:
    """A stop condition as a value a step returns, with the sentence that says what tripped it."""

    reason: StopReason
    detail: str


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """What a run paid, how it was set up, and what it delivered (D-28 applied to descent, D-35 (9)).

    ``parameters`` are the moved parameters as delivered, in the reporting arithmetic;
    ``final_training_loss`` is the loss of the last completed step, ``None`` when no step completed;
    ``steps_taken`` counts the steps that completed and passed their checks; ``stop_reason`` and
    ``stop_detail`` say why the run ended; ``configuration`` is the assembled record of every knob
    that produced it (D-35 (4)). Per-step observation is the recorder's, never a field here.
    """

    parameters: Any
    final_training_loss: float | None
    steps_taken: int
    stop_reason: StopReason
    stop_detail: str
    configuration: dict[str, Any]


class AbstractOptimizer(ABC):
    """A descent loop: ``run()`` is owned here, a subclass fills ``_begin``, ``_step`` and ``_assemble``.

    ``_begin`` prepares the parameters and may stop the run before any step; ``_step(step)`` takes
    one step and returns the training loss it descended, or a ``Stop``; ``_iterate()`` is the
    current parameters in the reporting arithmetic, handed raw to the recorder; ``_assemble`` turns
    the run into the subclass's result. ``recorder`` defaults to a fresh ``NullRecorder`` and every
    call site is guarded, so an unwatched run pays one attribute read per step (D-32).
    """

    def __init__(self, *, step_count: int, recorder: AbstractStepRecorder | None = None):
        if step_count < 0:
            raise ValueError(f"step_count must not be negative; got {step_count!r}")
        self.step_count = int(step_count)
        self.recorder: AbstractStepRecorder = recorder if recorder is not None else NullRecorder()
        self._has_run = False

    @property
    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        """Every knob that produced the run, the optimizer's own and its injected objects'."""
        raise NotImplementedError

    def run(self) -> OptimizerResult:
        """Begin, step until the budget or a stop, assemble; record each step and the end.

        One optimizer serves one run: its step rule's state and its recorder's frames belong to
        that run, so a second call is refused and a grid builds fresh objects (D-35 (5)).
        """
        if self._has_run:
            raise RuntimeError("an optimizer serves one run; build a fresh one for the next")
        self._has_run = True
        stop = self._begin()
        steps_taken = 0
        final_training_loss: float | None = None
        if stop is None:
            for step in range(self.step_count):
                outcome = self._step(step)
                if isinstance(outcome, Stop):
                    stop = outcome
                    break
                steps_taken = step + 1
                final_training_loss = float(outcome)
                if self.recorder.enabled:
                    self.recorder.record_step(
                        step, final_training_loss, self._iterate(), **self._recorder_extras()
                    )
        if stop is None:
            stop = Stop(StopReason.STEP_BUDGET, f"the step budget of {self.step_count} ran out")
        result = self._assemble(steps_taken, final_training_loss, stop)
        if self.recorder.enabled:
            self.recorder.record_end(result)
        return result

    @abstractmethod
    def _begin(self) -> Stop | None:
        """Prepare the parameters and the step rule; a ``Stop`` here ends the run before any step."""
        raise NotImplementedError

    @abstractmethod
    def _step(self, step: int) -> float | Stop:
        """Take step ``step``; return the training loss it descended, or a ``Stop``."""
        raise NotImplementedError

    @abstractmethod
    def _iterate(self) -> Any:
        """The current parameters in the reporting arithmetic, for the recorder."""
        raise NotImplementedError

    def _recorder_extras(self) -> dict[str, Any]:
        """What this optimizer alone tracks per step, handed to the recorder as keywords."""
        return {}

    @abstractmethod
    def _assemble(
        self, steps_taken: int, final_training_loss: float | None, stop: Stop
    ) -> OptimizerResult:
        """The run's result: parameters, the delivered numbers, the stop and the configuration."""
        raise NotImplementedError
