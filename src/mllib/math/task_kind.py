"""The kind of learning task a math component is meant for (BL-10)."""

from __future__ import annotations

from enum import Enum


class TaskKind(Enum):
    """Regression predicts a continuous value; classification predicts a label."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
