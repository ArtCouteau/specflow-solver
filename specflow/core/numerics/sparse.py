import torch

from . import dims
from .operations import mask
from .. import context
from ..physics import variable


def indices_from_range(rank: variable.Rank, lower: float, upper: float) -> torch.Tensor:
    tmp = mask.shell(lower, upper)
    if rank.name == 'SCALAR':
        return tmp.to_sparse().indices()
    elif rank.name == 'VECTOR':
        return torch.stack(
            context.get().config.get('grid/dimension', int)*[tmp],
            dim=dims.VECTOR
        ).to_sparse().indices()
    else:
        raise NotImplementedError()


def _compute_Hermitian_symmetric_mapping_1D(indices: torch.Tensor) -> torch.Tensor:
    return torch.zeros(indices.size(1), dtype=torch.int64, device=indices.device)


def _compute_Hermitian_symmetric_mapping_2D(indices: torch.Tensor) -> torch.Tensor:
    mapping = torch.zeros(indices.size(1), dtype=torch.int64, device=indices.device)

    for mode in range(indices.size(1)):
        if indices[-1, mode] == 0:
            if indices[-2, mode] < context.get().config.get('grid/N', int)//2:
                symmetry = indices[:, mode].clone()
                symmetry[-2] = context.get().config.get('grid/N', int) - indices[-2, mode]

                mapping[mode] = (indices.T == symmetry).all(dim=1).nonzero()

    return mapping


def _compute_Hermitian_symmetric_mapping_3D(indices: torch.Tensor) -> torch.Tensor:
    mapping = torch.zeros(indices.size(1), dtype=torch.int64, device=indices.device)
    for mode in range(indices.size(1)):
        if indices[-1, mode] == 0:
            if (indices[-2, mode] > 0) and (indices[-2, mode] < context.get().config.get('grid/N', int)//2):
                symmetry = indices[:, mode].clone()
                symmetry[-2] = context.get().config.get('grid/N', int) - indices[-2, mode]
                if indices[-3, mode] > 0:
                    symmetry[-3] = context.get().config.get('grid/N', int) - indices[-3, mode]

                mapping[mode] = (indices.T == symmetry).all(dim=1).nonzero()

            elif indices[-2, mode] == 0:
                if (indices[-3, mode] < 0) and (indices[-3, mode] < context.get().config.get('grid/N', int)//2):
                    symmetry = indices[:, mode].clone()
                    symmetry[-3] = context.get().config.get('grid/N', int) - indices[-3, mode]

                    mapping[mode] = (indices.T == symmetry).all(dim=1).nonzero()

    return mapping


def compute_Hermitian_symmetric_mapping(indices: torch.Tensor) -> torch.Tensor:
    if context.get().config.get('grid/dimension', int) == 1:
        return _compute_Hermitian_symmetric_mapping_1D(indices)
    elif context.get().config.get('grid/dimension', int) == 2:
        return _compute_Hermitian_symmetric_mapping_2D(indices)
    elif context.get().config.get('grid/dimension', int) == 3:
        return _compute_Hermitian_symmetric_mapping_3D(indices)
    else:
        raise ValueError('Invalid grid dimension')


def apply_Hermitian_symmetric_mapping(values: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    return torch.where(
        mapping == 0,
        values,
        values[mapping].conj()
    )
