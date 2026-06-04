import torch

from ...numerics import shell
from ....core.numerics import context
from ....core.numerics.operations import algebra, norm, volume


def _compute(field: torch.Tensor, term: torch.Tensor, shells: shell.SpectralShells) -> torch.Tensor:
    tmp = torch.zeros(
        (shells.nShells,),
        dtype=context.get().config.get('numerics/precision/float', torch.dtype),
        device=field.device
    )

    for index in range(shells.nShells):
        tmp[index] = volume.sum_over_spectral_domain(
            shells.spherical(index)
            * algebra.inner_vector_vector(field.conj(), term).real
        )

    return tmp


def kinetic_energy(velocity: torch.Tensor, term: torch.Tensor, shells: shell.SpectralShells) -> torch.Tensor:
    return _compute(velocity, term, shells)


def enstrophy(velocity: torch.Tensor, term: torch.Tensor, shells: shell.SpectralShells) -> torch.Tensor:
    return _compute(
        algebra.mul_scalar_vector(
            norm.square_l2_norm(context.get().mesh.grid),
            velocity
        ),
        term,
        shells
    )
