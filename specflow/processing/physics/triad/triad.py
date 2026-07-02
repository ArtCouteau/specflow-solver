import torch

from ...numerics import shell
from ....core import context
from ....core.numerics import dims
from ....core.numerics.operations import algebra, volume


def flip_and_shift(data: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    if context.get().config.get('grid/N', int) % 2 == 0:
        _data = data.roll(
            shifts=(-1,)*context.get().config.get('grid/dimension', int),
            dims=dims.domain()
        )
    else:
        _data = data

    return _data.flip(
        dims.domain()
    ).roll(
        shifts=kappa.tolist(),  # type: ignore
        dims=dims.domain()
    )


def mode_to_mode_energy_transfer(velocityKappa: torch.Tensor, modeToModeContribution: torch.Tensor) -> torch.Tensor:
    return (
        algebra.inner_vector_vector(
            dims.view_in_domain(velocityKappa.conj()),
            modeToModeContribution
        )
    ).real


def mode_to_shell_energy_transfer(modeToModeEnergyTransfer: torch.Tensor, shells: shell.SpectralShells) -> torch.Tensor:
    tmp = torch.zeros(
        size=(shells.nShells,),
        dtype=modeToModeEnergyTransfer.dtype,
        device=modeToModeEnergyTransfer.device
    )

    for index in range(shells.nShells):
        tmp[index] = volume.sum(modeToModeEnergyTransfer*shells.full_shell(index))

    return tmp
