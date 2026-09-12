"""Adam as a step rule: torch's Adam with a schedule, bound once per run.

The torch member of the step-rule concept (`mllib.math.step_rule`, D-35 (6)). The learning rate
and the schedule are knobs on the rule (D-35 (3)); the moment estimates are the state a run
accumulates and are never part of a configuration.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from mllib.math.learning_rate_schedule import AbstractLearningRateSchedule, ConstantSchedule
from mllib.math.step_rule import AbstractStepRule


class AdamStepRule(AbstractStepRule):
    """Adam at ``learning_rate``, scaled each step by ``schedule``; one instance for the whole run.

    One `torch.optim.Adam` under one `LambdaLR` for the whole run, so the moments survive; the
    bind-once contract is `AbstractStepRule`'s. Equality is on the knobs, not on the state, so two
    rules set up alike compare equal before and after they have run.
    """

    __slots__ = ("_optimizer", "_scheduler", "learning_rate", "schedule")

    def __init__(
        self,
        learning_rate: float = 0.05,
        schedule: AbstractLearningRateSchedule | None = None,
    ):
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError(
                f"learning_rate must be finite and not negative; got {learning_rate!r}"
            )
        self.learning_rate = float(learning_rate)
        self.schedule = ConstantSchedule() if schedule is None else schedule
        self._optimizer: torch.optim.Adam | None = None
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    @property
    def is_bound(self) -> bool:
        return self._optimizer is not None

    def bind(self, parameters: torch.Tensor, step_count: int) -> None:
        if self.is_bound:
            raise RuntimeError("an AdamStepRule serves one run; build a fresh rule for the next")
        if not parameters.requires_grad or not parameters.is_leaf:
            raise ValueError(
                "the parameters must be a leaf tensor that requires grad, or a step moves nothing"
            )
        schedule = self.schedule
        self._optimizer = torch.optim.Adam([parameters], lr=self.learning_rate)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda step: schedule.multiplier(step, step_count)
        )

    def _bound(self) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.LambdaLR]:
        if self._optimizer is None or self._scheduler is None:
            raise RuntimeError("the step rule is not bound to a run; call bind() first")
        return self._optimizer, self._scheduler

    def learning_rate_in_force(self) -> float:
        """The scheduled rate for the next step; after the last step, the multiplier past the horizon.

        `LambdaLR` advances once per `step()`, so a read after step ``step_count - 1`` evaluates the
        schedule at ``step_count``, outside its domain; the loop never uses that value.
        """
        optimizer, _ = self._bound()
        return float(optimizer.param_groups[0]["lr"])

    def zero_gradient(self) -> None:
        optimizer, _ = self._bound()
        optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        optimizer, scheduler = self._bound()
        optimizer.step()
        scheduler.step()

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.learning_rate == other.learning_rate and self.schedule == other.schedule

    def __hash__(self) -> int:
        return hash((type(self), self.learning_rate, self.schedule))

    def __repr__(self) -> str:
        return f"AdamStepRule(learning_rate={self.learning_rate!r}, schedule={self.schedule!r})"

    def __getstate__(self) -> dict[str, Any]:
        """Only the knobs travel: a copied or pickled rule arrives unbound."""
        return {"learning_rate": self.learning_rate, "schedule": self.schedule}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.learning_rate = state["learning_rate"]
        self.schedule = state["schedule"]
        self._optimizer = None
        self._scheduler = None
