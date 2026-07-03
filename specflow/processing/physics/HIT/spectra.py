import torch

from ...numerics import shell
from ....core.numerics import dims
from ....core.numerics.operations import volume


def spectrum(field: torch.Tensor, shells: shell.SpectralShells) -> torch.Tensor:
    tmp = torch.zeros((dims.rank_shape(field) + (shells.nShells,)), dtype=field.dtype, device=field.device)

    for index in range(shells.nShells):
        tmp[..., index] = volume.sum_over_spectral_domain(field*shells.shell(index))

    return tmp
