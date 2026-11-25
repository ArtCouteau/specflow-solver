import math

import pytest
import torch

from specflow import diagnostics
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh

POINTS = 8
VISCOSITY = 1 / 1600


def make_mesh(dimension: int) -> Mesh:
    return Mesh(POINTS, 2 * math.pi, dimension, SerialDecomposition(POINTS, dimension))


def taylor_green_spectral(mesh: Mesh) -> torch.Tensor:
    coordinates = mesh.coordinates
    x, y = coordinates[0], coordinates[1]
    shape = torch.ones_like(x)
    if mesh.dimension == 3:
        shape = torch.cos(coordinates[2])
    components = [
        torch.sin(x) * torch.cos(y) * shape,
        -torch.cos(x) * torch.sin(y) * shape,
    ]
    if mesh.dimension == 3:
        components.append(torch.zeros_like(x))
    return mesh.decomposition.to_spectral(torch.stack(components))


def test_the_3d_taylor_green_vortex_matches_the_reference_first_row() -> None:
    mesh = make_mesh(3)
    velocity_spectral = taylor_green_spectral(mesh)

    energy = diagnostics.kinetic_energy(mesh, velocity_spectral)
    dissipation = diagnostics.dissipation(mesh, velocity_spectral, VISCOSITY)
    enstrophy = diagnostics.enstrophy(mesh, velocity_spectral)

    assert energy.item() == pytest.approx(0.125, rel=1e-14)
    assert dissipation.item() == pytest.approx(0.75 * VISCOSITY, rel=1e-14)
    assert enstrophy.item() == pytest.approx(0.375, rel=1e-14)


def test_the_2d_taylor_green_vortex_has_known_energy_and_enstrophy() -> None:
    mesh = make_mesh(2)
    velocity_spectral = taylor_green_spectral(mesh)

    energy = diagnostics.kinetic_energy(mesh, velocity_spectral)
    dissipation = diagnostics.dissipation(mesh, velocity_spectral, VISCOSITY)
    enstrophy = diagnostics.enstrophy(mesh, velocity_spectral)

    assert energy.item() == pytest.approx(0.25, rel=1e-14)
    assert dissipation.item() == pytest.approx(VISCOSITY, rel=1e-14)
    assert enstrophy.item() == pytest.approx(0.5, rel=1e-14)


def test_the_enstrophy_of_a_1d_mesh_raises() -> None:
    mesh = make_mesh(1)

    with pytest.raises(ValueError, match="1D"):
        diagnostics.enstrophy(mesh, torch.zeros(1, 5, dtype=mesh.complex_dtype))


def test_a_divergence_free_field_has_zero_rms_divergence() -> None:
    mesh = make_mesh(3)

    rms = diagnostics.divergence_rms(mesh, taylor_green_spectral(mesh))

    assert rms.item() <= 1e-15


def test_the_rms_divergence_of_a_gradient_is_known() -> None:
    mesh = make_mesh(2)
    x = mesh.coordinates[0]
    velocity_physical = torch.stack([-torch.sin(x), torch.zeros_like(x)])

    rms = diagnostics.divergence_rms(
        mesh, mesh.decomposition.to_spectral(velocity_physical)
    )

    assert rms.item() == pytest.approx(1 / math.sqrt(2), rel=1e-14)


@pytest.mark.parametrize("component_axes", [0, 1])
def test_the_relative_error_of_a_scaled_field_is_the_scale_minus_one(
    component_axes: int,
) -> None:
    mesh = make_mesh(2)
    reference_spectral = taylor_green_spectral(mesh)
    if component_axes == 0:
        reference_spectral = reference_spectral[0]

    error = diagnostics.relative_error(
        mesh, 1.1 * reference_spectral, reference_spectral, component_axes
    )

    assert error.shape == ()
    assert error.item() == pytest.approx(0.1, rel=1e-12)


def test_diagnostics_keep_the_leading_axes_on_the_device() -> None:
    mesh = make_mesh(2)
    velocity_spectral = taylor_green_spectral(mesh)
    ensemble_spectral = torch.stack([velocity_spectral, 2 * velocity_spectral])

    energy = diagnostics.kinetic_energy(mesh, ensemble_spectral)

    assert energy.shape == (2,)
    assert energy.dtype == mesh.dtype
    assert energy.device == mesh.wavenumbers.device
    assert energy.tolist() == pytest.approx([0.25, 1.0], rel=1e-14)
