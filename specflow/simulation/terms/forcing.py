from __future__ import annotations

import math
import torch

from . import term
from ...core.numerics import context, sparse
from ...core.physics import variable


@term.TermRegistry.register('forcing', 'OU')
class OrnsteinUhlenbeckForcingTerm(term.Term):
    sigma: float
    tau: float
    previousTime: float
    mapping: torch.Tensor
    coefficients: torch.Tensor

    def __init__(self, equation: str, tag: str, rank: variable.Rank):
        super().__init__(equation, tag, rank)
        self.sigma = context.get().config.get(f'equations/{equation}/terms/{tag}/sigma', float)
        self.tau = context.get().config.get(f'equations/{equation}/terms/{tag}/tau', float)

        _range = tuple(context.get().config.get(f'equations/{equation}/terms/{tag}/range', list))
        self.indices = sparse.indices_from_range(rank, _range[0], _range[1]).to(
            device=context.get().config.get('numerics/device', str)
        )
        self.mapping = sparse.compute_Hermitian_symmetric_mapping(self.indices)
        self.coefficients = self._init_coefficients().to(device=context.get().config.get('numerics/device', str))

        self.previousTime = context.get().sim.control.time.current

    def _init_coefficients(self) -> torch.Tensor:
        if context.get().config.get('time/restart', bool):
            return context.get().io.h5.read_tensor(
                f'output/{context.get().sim.control.time.timestamp()}/simulation',
                f'{self.tag}/coefficients',
                device=context.get().config.get('numerics/device', str)
            )
        else:
            return (
                torch.zeros(self.indices.size(1)).to(
                    dtype=context.get().config.get('numerics/precision/complex', torch.dtype))
            )

    def _update_coefficients(self, dt: float):
        self.coefficients.add_(
            - dt/self.tau * self.coefficients
            + self.sigma * math.sqrt(2.0*dt/self.tau) * torch.randn_like(self.coefficients)
        )

    def compute(self, variable: torch.Tensor, time: float) -> torch.Tensor:
        if time > self.previousTime:
            self._update_coefficients(time - self.previousTime)
            self.previousTime = time

        return torch.sparse_coo_tensor(  # type: ignore
            self.indices,
            sparse.apply_Hermitian_symmetric_mapping(self.coefficients, self.mapping),
            size=variable.size(),
        ).to_dense()
