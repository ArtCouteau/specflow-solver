import torch

from .. import context, dims
from ..decomposition import communication


def to_physical(data: torch.Tensor) -> torch.Tensor:
    if not context.get().distribution.enabled:
        return torch.fft.irfftn(data, dim=dims.domain(), norm='forward')  # type: ignore

    N: int = context.get().config.get('grid/N', int)
    splitAxis, middleAxis, lastAxis = dims.domain()

    stage = torch.fft.ifft(data, dim=middleAxis, norm='forward')
    stage = communication.transpose(
        stage, splitAxis=middleAxis, splitAxisGlobalSize=N, gatherAxis=splitAxis, gatherAxisGlobalSize=N
    )
    return torch.fft.irfftn(stage, dim=(splitAxis, lastAxis), norm='forward')  # type: ignore


def to_spectral(data: torch.Tensor) -> torch.Tensor:
    if not context.get().distribution.enabled:
        return torch.fft.rfftn(data, dim=dims.domain(), norm='forward')  # type: ignore

    N: int = context.get().config.get('grid/N', int)
    splitAxis, middleAxis, lastAxis = dims.domain()

    stage = torch.fft.rfftn(data, dim=(splitAxis, lastAxis), norm='forward')  # type: ignore
    stage = communication.transpose(
        stage, splitAxis=splitAxis, splitAxisGlobalSize=N, gatherAxis=middleAxis, gatherAxisGlobalSize=N
    )
    return torch.fft.fft(stage, dim=middleAxis, norm='forward')


def to_full_spectral(data: torch.Tensor) -> torch.Tensor:
    # Assumes that the data is half spectral
    if not context.get().distribution.enabled:
        return torch.fft.fftn(to_physical(data), dim=dims.domain(), norm='forward')  # type: ignore

    N: int = context.get().config.get('grid/N', int)
    splitAxis, middleAxis, lastAxis = dims.domain()

    physical = to_physical(data)

    stage = torch.fft.fftn(physical, dim=(splitAxis, lastAxis), norm='forward')  # type: ignore
    stage = communication.transpose(
        stage, splitAxis=splitAxis, splitAxisGlobalSize=N, gatherAxis=middleAxis, gatherAxisGlobalSize=N
    )
    return torch.fft.fft(stage, dim=middleAxis, norm='forward')
