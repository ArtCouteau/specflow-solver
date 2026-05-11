import torch

from ....numerics.operations import algebra, derivative, fourier, norm, rank
from ... import fields


def physical_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return fourier.to_physical(velocity)


def vorticity(velocity: torch.Tensor) -> torch.Tensor:
    return fourier.to_physical(fields.spectral.velocity.vorticity(velocity))


def strain_rate_tensor(velocity: torch.Tensor) -> torch.Tensor:
    velocity_gradient = fourier.to_physical(derivative.gradient_vector(velocity))
    return 0.5 * (velocity_gradient + rank.transpose(velocity_gradient))


def kinetic_energy(velocity: torch.Tensor) -> torch.Tensor:
    return 0.5*norm.square_l2_norm(physical_velocity(velocity))


def Reynolds_stress_tensor(velocity: torch.Tensor) -> torch.Tensor:
    return algebra.outer_vector_vector(
        physical_velocity(velocity),
        physical_velocity(velocity)
    )


def enstrophy(velocity: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * norm.square_l2_norm(vorticity(velocity))
    )


def dissipation(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    return (
        2.0
        * viscosity
        * norm.square_frobenius_norm(strain_rate_tensor(velocity))
    )


def dissipation_from_velocity_gradient(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    return (
        viscosity
        * norm.square_frobenius_norm(fourier.to_physical(derivative.gradient_vector(velocity)))
    )


def dissipation_from_vorticity(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    return (
        viscosity
        * norm.square_l2_norm(vorticity(velocity))
    )


def rate_of_strain(physicalStrainRateTensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(2.0 * norm.square_frobenius_norm(physicalStrainRateTensor))
