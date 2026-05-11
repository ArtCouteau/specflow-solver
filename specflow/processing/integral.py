import torch

from ..core.numerics.operations import volume
from ..core.physics import fields


def _rms_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return total_kinetic_energy(velocity).sqrt()


def total_kinetic_energy(velocity: torch.Tensor) -> torch.Tensor:
    return volume.mean(
        fields.physical.velocity.kinetic_energy(velocity)
    )


def Reynolds_stress_tensor(velocity: torch.Tensor) -> torch.Tensor:
    return volume.mean(
        fields.physical.velocity.Reynolds_stress_tensor(velocity)
    )


def dissipation_rate(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    return volume.mean(
        fields.physical.velocity.dissipation(velocity, viscosity)
    )


def residual_dissipation_rate(velocity: torch.Tensor, eddyViscosity: torch.Tensor) -> torch.Tensor:
    return volume.mean(
        fields.physical.les.residual_dissipation(velocity, eddyViscosity)
    )


def total_enstrophy(velocity: torch.Tensor) -> torch.Tensor:
    return volume.mean(
        fields.physical.velocity.enstrophy(velocity)
    )


def Taylor_microscale(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    epsilon = dissipation_rate(velocity, viscosity)

    if epsilon.item() > 0.0:
        return _rms_velocity(velocity) * (15.0 * viscosity / epsilon).sqrt()
    else:
        return torch.tensor(float('inf'), device=velocity.device)


def Reynolds_lambda(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    return _rms_velocity(velocity) * Taylor_microscale(velocity, viscosity) / viscosity


def Kolmogorov_length_scale(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    epsilon = dissipation_rate(velocity, viscosity)

    if epsilon.item() > 0.0:
        return (viscosity**3 / epsilon).pow(0.25)
    else:
        return torch.tensor(float('inf'), device=velocity.device)


def Kolmogorov_time_scale(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    epsilon = dissipation_rate(velocity, viscosity)

    if epsilon.item() > 0.0:
        return (viscosity / epsilon).pow(0.5)
    else:
        return torch.tensor(float('inf'), device=velocity.device)


def largest_eddy_turnover_time(velocity: torch.Tensor, viscosity: float) -> torch.Tensor:
    epsilon = dissipation_rate(velocity, viscosity)

    if epsilon.item() > 0.0:
        return total_kinetic_energy(velocity) / epsilon
    else:
        return torch.tensor(float('inf'), device=velocity.device)
