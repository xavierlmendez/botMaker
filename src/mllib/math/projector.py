"""The projector contract: the projection onto a spanning set's span, seen through its residual.

One concept with two arithmetics (D-35 (6)): the exact pseudo-inverse projector here, for every
reported number, and the ridge projector the training loss descends
(`math/algorithms/two_hot_span/projectors.py`). D-31 fixes which is which: a reported number never
comes through the ridge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class AbstractProjector(ABC):
    """P_V, the projection onto the column span of a spanning set V, as the residual it leaves.

    ``residual(X, spanning_set)`` is ‖X - P_V X‖²_F: how much of X the span of V fails to explain.
    The arithmetic of P_V is the implementation's; the base names no array library.
    """

    __slots__ = ()

    @abstractmethod
    def residual(self, X: Any, spanning_set: Any) -> Any:
        """‖X - P_V X‖²_F at ``spanning_set``, in the implementation's arithmetic."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ExactProjector(AbstractProjector):
    """P_V = V V⁺ through the numpy pseudo-inverse: the reporting arithmetic, no knob (D-31).

    The projector onto the span whatever the rank of V, which the ridge arithmetic approximates.
    """

    def residual(self, X: np.ndarray, spanning_set: np.ndarray) -> float:
        """E(V) = ‖X - V V⁺ X‖²_F, float64. There is no epsilon here."""
        X = np.asarray(X, dtype=float)
        spanning_set = np.asarray(spanning_set, dtype=float)
        residual = X - spanning_set @ (np.linalg.pinv(spanning_set) @ X)
        return float(np.sum(residual * residual))
