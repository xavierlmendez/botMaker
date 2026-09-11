"""The ridge projector: the training arithmetic of the projector, with its epsilon as a knob.

The torch member of the projector concept (`mllib.math.projector`, D-35 (6)). ``epsilon`` is a
training knob and never reaches a reported number (D-31); the derivation is the learning-log entry
of 2026-09-11.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mllib.math.projector import AbstractProjector


@dataclass(frozen=True, slots=True)
class RidgeProjector(AbstractProjector):
    """P_ε = V (VᵀV + εI)⁻¹ Vᵀ, float64 torch: the projector with filter factors s²/(s²+ε).

    Smooth in V where the exact projector is not, so the gradient stays defined as columns go
    dependent. The r x r system is solved directly; nothing is orthogonalised.
    """

    epsilon: float = 1e-6

    def residual(self, X: torch.Tensor, spanning_set: torch.Tensor) -> torch.Tensor:
        """‖X - P_ε X‖²_F at ``spanning_set``, as a tensor on the gradient path."""
        spanning_count = spanning_set.shape[1]
        gram = spanning_set.T @ spanning_set
        ridge = gram + self.epsilon * torch.eye(
            spanning_count, dtype=spanning_set.dtype, device=spanning_set.device
        )
        coefficients = torch.linalg.solve(ridge, spanning_set.T @ X)
        residual = X - spanning_set @ coefficients
        return torch.sum(residual * residual)
