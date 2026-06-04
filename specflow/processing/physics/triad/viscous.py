
import torch

from . import triad
from ....core.numerics import dims, mesh
from ....core.numerics.operations import algebra, fourier, norm
from ....core.physics import fields


def mode_to_mode_contribution(velocity: torch.Tensor, viscosity: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    _fullGrid = mesh.construct_full_spectral_grid()

    return (
        - algebra.mul_scalar_vector(
            fourier.to_full_spectral(viscosity),
            (
                algebra.mul_scalar_vector(
                    algebra.inner_vector_vector(
                        dims.view_in_domain(kappa),
                        triad.flip_and_shift(fourier.to_full_spectral(velocity), kappa)
                    ),
                    triad.flip_and_shift(_fullGrid, kappa)
                )
                + algebra.mul_scalar_vector(
                    algebra.inner_vector_vector(
                        dims.view_in_domain(kappa),
                        triad.flip_and_shift(_fullGrid, kappa)
                    ),
                    triad.flip_and_shift(fourier.to_full_spectral(velocity), kappa)
                )
            )
        )
    )


def mode_to_mode_energy_transfer_maximum_estimate(
    velocity: torch.Tensor,
    viscosity: torch.Tensor,
    kappa: torch.Tensor
) -> torch.Tensor:
    _fullGrid = mesh.construct_full_spectral_grid()

    return (
        norm.square_l2_norm(kappa).sqrt()
        * norm.square_l2_norm(triad.flip_and_shift(_fullGrid, kappa)).sqrt()
        * fourier.to_full_spectral(viscosity).abs()
        * triad.flip_and_shift(
            fields.spectral.velocity.modal_kinetic_energy(
                fourier.to_full_spectral(velocity)
            ).sqrt(),
            kappa
        )
    )


def mode_to_mode_energy_transfer_symmetry(
    velocity: torch.Tensor,
    viscosity: torch.Tensor,
    kappa: torch.Tensor
) -> torch.Tensor:
    # Here we return energy transfers directly, not contributions!
    _fullGrid = mesh.construct_full_spectral_grid()

    return (
        dims.view_in_domain(viscosity[..., *tuple(kappa)])
        * (
            (
                algebra.inner_vector_vector(
                    _fullGrid,
                    triad.flip_and_shift(fourier.to_full_spectral(velocity).conj(), kappa)
                )
                * algebra.inner_vector_vector(
                    triad.flip_and_shift(_fullGrid, kappa),
                    fourier.to_full_spectral(velocity).conj()
                )
                + algebra.inner_vector_vector(
                    _fullGrid,
                    triad.flip_and_shift(_fullGrid, kappa)
                )
                * algebra.inner_vector_vector(
                    triad.flip_and_shift(fourier.to_full_spectral(velocity).conj(), kappa),
                    fourier.to_full_spectral(velocity).conj()
                )
            )
        )
    ).real
