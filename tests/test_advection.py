import math

import pytest
import torch

from specflow.dealiasing import TwoThirdsDealiasing
from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.mesh import Mesh
from specflow.terms import Advection, Diffusion

POINTS = 12
TOLERANCE = 1e-12
DIMENSIONS = [2, 3]


def make_mesh(dimension: int) -> Mesh:
    decomposition = SerialDecomposition(POINTS, dimension)
    return Mesh(POINTS, 2 * math.pi, dimension, decomposition)


def band_limited_spectral(mesh: Mesh, *components: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (*components, *mesh.physical_shape)
    field_physical = torch.randn(shape, generator=generator, dtype=mesh.dtype)
    field_spectral = mesh.decomposition.to_spectral(field_physical)
    band = mesh.wavenumbers_squared < (POINTS / 3) ** 2
    field_spectral.mul_(band)
    return field_spectral.div_(norm(mesh, field_spectral))


def velocity_spectral(mesh: Mesh, *leading: int, seed: int = 0) -> torch.Tensor:
    field_spectral = band_limited_spectral(mesh, *leading, mesh.dimension, seed=seed)
    projected_spectral = mesh.project(field_spectral)
    return projected_spectral.div_(norm(mesh, projected_spectral))


def inner_product(
    mesh: Mesh, a_spectral: torch.Tensor, b_spectral: torch.Tensor
) -> float:
    products = (a_spectral.conj() * b_spectral).real * mesh.parseval_weights
    return mesh.decomposition.sum(products).sum().item()


def norm(mesh: Mesh, field_spectral: torch.Tensor) -> float:
    return math.sqrt(inner_product(mesh, field_spectral, field_spectral))


def advection(mesh: Mesh, variable: str = "velocity") -> Advection:
    return Advection(mesh, TwoThirdsDealiasing(mesh), variable)


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    scale = max(expected.abs().max().item(), 1.0)
    assert (actual - expected).abs().max().item() <= TOLERANCE * scale


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_advection_conserves_energy(dimension: int) -> None:
    mesh = make_mesh(dimension)
    equation = Equation("velocity", [advection(mesh)], projection=mesh.project)
    u_spectral = velocity_spectral(mesh)

    a_spectral = equation.contributions({"velocity": u_spectral}, 0.0)["advection"]

    rate = inner_product(mesh, u_spectral, a_spectral)
    assert norm(mesh, a_spectral) > 0.1
    assert abs(rate) <= TOLERANCE * norm(mesh, u_spectral) * norm(mesh, a_spectral)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_projected_advection_is_divergence_free(dimension: int) -> None:
    mesh = make_mesh(dimension)
    equation = Equation("velocity", [advection(mesh)], projection=mesh.project)

    contributions = equation.contributions({"velocity": velocity_spectral(mesh)}, 0.0)

    divergence_spectral = mesh.divergence(contributions["advection"])
    assert_close(divergence_spectral, torch.zeros(()))


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_advection_of_a_shear_by_a_uniform_flow(dimension: int) -> None:
    mesh = make_mesh(dimension)
    x = mesh.coordinates[0]
    u_physical = torch.zeros(dimension, *mesh.physical_shape, dtype=mesh.dtype)
    u_physical[0] = 0.7
    u_physical[-1] = 1.3 * torch.sin(x)
    state = {"velocity": mesh.decomposition.to_spectral(u_physical)}

    contributions = Equation("velocity", [advection(mesh)]).contributions(state, 0.0)

    expected_physical = torch.zeros_like(u_physical)
    expected_physical[-1] = -0.7 * 1.3 * torch.cos(x)
    expected_spectral = mesh.decomposition.to_spectral(expected_physical)
    assert_close(contributions["advection"], expected_spectral)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_advection_of_a_scalar_by_a_uniform_flow(dimension: int) -> None:
    mesh = make_mesh(dimension)
    y = mesh.coordinates[1]
    u_physical = torch.zeros(dimension, *mesh.physical_shape, dtype=mesh.dtype)
    u_physical[1] = 0.6
    state = {
        "velocity": mesh.decomposition.to_spectral(u_physical),
        "scalar": mesh.decomposition.to_spectral(torch.sin(2 * y)),
    }
    equation = Equation("scalar", [advection(mesh, "scalar")])

    contributions = equation.contributions(state, 0.0)

    expected_spectral = mesh.decomposition.to_spectral(-0.6 * 2 * torch.cos(2 * y))
    assert_close(contributions["advection"], expected_spectral)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_advection_of_a_scalar_conserves_its_variance(dimension: int) -> None:
    mesh = make_mesh(dimension)
    scalar_spectral = band_limited_spectral(mesh, seed=1)
    state = {"velocity": velocity_spectral(mesh), "scalar": scalar_spectral}
    equation = Equation("scalar", [advection(mesh, "scalar")])

    a_spectral = equation.contributions(state, 0.0)["advection"]

    rate = inner_product(mesh, scalar_spectral, a_spectral)
    assert norm(mesh, a_spectral) > 0.1
    assert abs(rate) <= TOLERANCE * norm(mesh, scalar_spectral) * norm(mesh, a_spectral)


@pytest.mark.parametrize("variable", ["velocity", "scalar"])
def test_advection_acts_on_each_realization_of_the_leading_axes(variable: str) -> None:
    mesh = make_mesh(2)
    state = {
        "velocity": velocity_spectral(mesh, 2),
        "scalar": band_limited_spectral(mesh, 2, seed=1),
    }
    equation = Equation(variable, [advection(mesh, variable)])

    batch_spectral = equation.contributions(state, 0.0)["advection"]

    for index in range(2):
        single = {key: tensor[index] for key, tensor in state.items()}
        single_spectral = equation.contributions(single, 0.0)["advection"]
        assert_close(batch_spectral[index], single_spectral)


def test_projected_contributions_sum_to_the_tendency() -> None:
    mesh = make_mesh(3)
    u_spectral = velocity_spectral(mesh)
    diffusion = Diffusion(mesh, "velocity", 0.1, components=(3,))
    terms = [diffusion, advection(mesh)]
    equation = Equation("velocity", terms, projection=mesh.project)
    state = {"velocity": u_spectral}
    nonlinear_spectral = torch.full_like(u_spectral, math.nan)

    equation.nonlinear_tendency(state, 0.0, nonlinear_spectral)

    tendency_spectral = equation.linear_operator * u_spectral + nonlinear_spectral
    contributions = equation.contributions(state, 0.0)
    assert list(contributions) == ["diffusion", "advection"]
    assert_close(sum(contributions.values(), torch.zeros(())), tendency_spectral)
    expected_diffusion = -0.1 * mesh.wavenumbers_squared * u_spectral
    assert_close(contributions["diffusion"], expected_diffusion)
