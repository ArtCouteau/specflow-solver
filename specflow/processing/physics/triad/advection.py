import torch

from . import triad
from ....core import defs
from ....core.numerics import context, dims
from ....core.numerics.operations import algebra, fourier, norm
from ....core.physics import fields


def mode_to_mode_contribution(velocity: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    return (
        1.0j
        * fourier.to_full_spectral(velocity)
        * algebra.inner_vector_vector(
            dims.view_in_domain(kappa),
            triad.flip_and_shift(fourier.to_full_spectral(velocity), kappa)
        )
    )


def mode_to_mode_energy_transfer_maximum_estimate(
    velocity: torch.Tensor,
    kappa: torch.Tensor
) -> torch.Tensor:
    return (
        norm.square_l2_norm(kappa).sqrt()
        * (
            fields.spectral.velocity.modal_kinetic_energy(
                velocity[:, *tuple(kappa)]
            ).sqrt()
        )
        * (
            fields.spectral.velocity.modal_kinetic_energy(
                fourier.to_full_spectral(velocity)
            ).sqrt()
        )
        * triad.flip_and_shift(
            fields.spectral.velocity.modal_kinetic_energy(
                fourier.to_full_spectral(velocity)
            ).sqrt(),
            kappa
        )
    )


def EDQNM_mode_to_mode_energy_transfer_estimate(
    velocity: torch.Tensor,
    kappa: torch.Tensor,
    viscosity: float
) -> torch.Tensor:
    # Taken from Domaradzki & Rogallo (1990), eq (21)
    return (
        EDQNM_theta(kappa, velocity, viscosity)
        * (
            dims.view_in_domain(norm.l2_norm(kappa))
            / (
                norm.l2_norm(context.get().mesh.grid)
                * triad.flip_and_shift(norm.l2_norm(context.get().mesh.grid), kappa)
                + defs.SMALL
            )
        )
        * EDQNM_b(
            dims.view_in_domain(kappa),
            context.get().mesh.grid,
            triad.flip_and_shift(context.get().mesh.grid, kappa)
        )
        * EDQNM_f(kappa, velocity)

    )


def EDQNM_theta(kappa: torch.Tensor, velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    # Taken from Domaradzki & Rogallo (1990), eq (23)
    return 1.0/(
        dims.view_in_domain(EDQNM_eta(kappa, velocity[:, *tuple(kappa)], viscosity))
        + EDQNM_eta(context.get().mesh.grid, velocity, viscosity)
        + EDQNM_eta(
            triad.flip_and_shift(context.get().mesh.grid, kappa),
            triad.flip_and_shift(velocity, kappa),
            viscosity
        )
    )


def EDQNM_eta(kappa: torch.Tensor, velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    # Taken from Domaradzki & Rogallo (1990), eq (24)
    _lambda = 0.31
    return (
        viscosity * norm.square_l2_norm(kappa)
        + _lambda * (
            norm.square_l2_norm(kappa)**1.5
            * fields.spectral.velocity.modal_kinetic_energy(velocity)
        ).sqrt()
    )


def EDQNM_b(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Taken from Domaradzki & Rogallo (1990), eq (22)
    def _cos_opposite_to_first(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (
            (norm.square_l2_norm(b) + norm.square_l2_norm(c) - norm.square_l2_norm(a))
            / (2.0 * norm.l2_norm(b) * norm.l2_norm(c) + defs.SMALL)
        )

    return (
        (norm.l2_norm(b)/(norm.l2_norm(a) + defs.SMALL))
        * (
            _cos_opposite_to_first(a, b, c)*_cos_opposite_to_first(b, a, c)
            + _cos_opposite_to_first(c, a, b)**3
        )
    )


def EDQNM_f(kappa: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    # Taken from Domaradzki & Rogallo (1990), new factor from (21)
    # f(k,kP) = E[k-kP] * ( |k|^2*E[kP] - |kP|^2*E[k] )
    return (
        triad.flip_and_shift(fields.spectral.velocity.modal_kinetic_energy(velocity), kappa)
        * (
            (
                norm.square_l2_norm(dims.view_in_domain(kappa))
                * fields.spectral.velocity.modal_kinetic_energy(velocity)
            )
            - (
                norm.square_l2_norm(context.get().mesh.grid)
                * fields.spectral.velocity.modal_kinetic_energy(velocity[:, *tuple(kappa)])
            )
        )
    )
