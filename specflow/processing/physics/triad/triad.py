import torch

from ...numerics import shell
from ....core.numerics import context, dims
from ....core.numerics.decomposition import communication
from ....core.numerics.operations import algebra, volume


def _reverse(data: torch.Tensor, axes: tuple[int, ...], shifts: list[int] | int) -> torch.Tensor:
    if context.get().config.get('grid/N', int) % 2 == 0:
        data = data.roll(shifts=(-1,)*len(axes), dims=axes)

    return data.flip(axes).roll(shifts=shifts, dims=axes)


def flip_and_shift(data: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    if not context.get().distribution.enabled:
        return _reverse(data, dims.domain(), kappa.tolist())  # type: ignore

    # v1 scope: only kappa vectors with a zero component on the split axis are
    # supported; the split-axis exchange below is a pure mirror with no shift.
    assert int(kappa[0].item()) == 0, (
        'Distributed flip_and_shift only supports kappa with a zero component on the split axis'
    )

    splitAxis, *localAxes = dims.domain()
    exchanged = _reverse(communication.mirror_exchange(data), (splitAxis,), 0)
    return _reverse(exchanged, tuple(localAxes), kappa[1:].tolist())  # type: ignore


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
