from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal

import torch

from specflow.core.numerics.backend.backend import NumericsBackend
from specflow.core.numerics.mesh import Mesh
from specflow.core.physics.quantity.field import Field
from specflow.core.physics.quantity.quantity import Quantity, Scalar, Spectral, TRank


@dataclass(frozen=True, eq=False)
class SparseField(Quantity, Generic[TRank]):
    rank: type[TRank]
    component: Literal[1, 2, 3]
    dimension: Literal[1, 2, 3]
    backend: NumericsBackend = field(repr=False)
    mesh: Mesh = field(repr=False)
    HermitianMapping: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.data.layout != torch.sparse_coo or not self.data.is_coalesced():
            raise ValueError("SparseField expects a coalesced sparse COO tensor")
        object.__setattr__(self, "HermitianMapping", self._compute_Hermitian_symmetric_mapping())

    @classmethod
    def zeros(cls, support: Field[Scalar, Spectral], rank: type[TRank]) -> SparseField[TRank]:
        is_scalar = rank is Scalar
        component = 1 if is_scalar else support.dimension
        mask = support.data.bool()
        if not is_scalar:
            mask = mask.expand(component, *mask.shape)
        indices = mask.nonzero().T
        values = torch.zeros(
            indices.size(1),
            dtype=support.backend.parameters.precision.COMPLEX,
            device=indices.device,
        )
        return SparseField(
            torch.sparse_coo_tensor(indices, values, mask.shape).coalesce(),
            rank,
            component,
            support.dimension,
            support.backend,
            support.mesh,
        )

    def _compute_Hermitian_symmetric_mapping(self) -> torch.Tensor:
        N, D = self.mesh.layout.N, self.mesh.layout.DIMENSION
        nnz = self.data.indices().size(1)
        positions = torch.arange(nnz, device=self.data.indices().device)

        # Reflection k -> -k restricted to the stored half-spectrum: only acts on the
        # rfft planes k_last = 0 (and k_last = N/2 for even N), negating the full axes.
        last = self.data.indices()[-1]
        onPlane = (last == 0) | ((N % 2 == 0) & (last == N // 2))
        full = self.data.indices()[-D:-1]
        mirroredFull = (N - full) % N
        mirrored = self.data.indices().clone()
        mirrored[-D:-1] = mirroredFull

        # Locate each mirror in the support
        inverse: torch.Tensor
        _, inverse = torch.unique(torch.cat([self.data.indices(), mirrored], dim=1), dim=1, return_inverse=True)  # type: ignore
        lookup = torch.full((2 * nnz,), -1, dtype=torch.int64, device=self.data.indices().device)
        lookup[inverse[:nnz]] = positions
        partner = lookup[inverse[nnz:]]

        # Of each (k, -k) pair, the mode whose last differing full axis is in (0, N/2) is dependent
        weights = N ** torch.arange(D - 1, device=self.data.indices().device)
        key = (full * weights[:, None]).sum(dim=0)
        mirroredKey = (mirroredFull * weights[:, None]).sum(dim=0)
        dependent = onPlane & (key < mirroredKey)

        if (onPlane & (partner < 0)).any():
            raise ValueError("Support is not symmetric under k -> -k")
        if (onPlane & (key == mirroredKey)).any():
            raise ValueError("Support contains self-conjugate modes (k = 0 or Nyquist)")

        # Positions of the dependent modes (row 0) and of the free modes they mirror (row 1)
        return torch.stack([positions[dependent], partner[dependent]])

    def apply_Hermitian_symmetry_(self) -> SparseField[TRank]:
        dependent, partner = self.HermitianMapping
        values = self.data.values()
        values[dependent] = values[partner].conj()
        return self

    def to_dense(self) -> Field[TRank, Spectral]:
        return Field(
            self.data.to_dense(),
            self.rank,
            Spectral,
            self.component,
            self.dimension,
            self.backend,
            self.mesh,
        )
