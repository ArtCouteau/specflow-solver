from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Spectral, Vector
from specflow.core.physics.quantity.sparse import SparseField
from specflow.simulation.terms.term import Term, TermRegistry


@TermRegistry.register("Forcing/OrnsteinUhlenbeck")
@dataclass(frozen=True)
class OrnsteinUhlenbeckForcingTerm(Term[Vector]):
    sigma: float
    tau: float
    previousTime: float
    coefficients: SparseField[Vector]

    def _update_coefficients(self, dt: float) -> None:
        values = self.coefficients.data.values()
        values.add_(-dt / self.tau * values + self.sigma * math.sqrt(2.0 * dt / self.tau) * torch.randn_like(values))
        self.coefficients.apply_Hermitian_symmetry_()

    def compute(self, rhs: Field[Vector, Spectral], time: float) -> None:
        if time > self.previousTime:
            self._update_coefficients(time - self.previousTime)
            object.__setattr__(self, "previousTime", time)

        rhs.data.add_(self.coefficients.data.to_dense())

    def diagnostic_fields(self, time: float) -> dict[str, Field[Vector, Spectral]]:
        return {"coefficients": self.coefficients.to_dense()}
