import math
from collections.abc import Mapping

import pytest
import torch

from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.mesh import Mesh
from specflow.terms import Diffusion, State

TOLERANCE = 1e-12


def make_mesh(points: int, dimension: int, length: float = 2 * math.pi) -> Mesh:
    return Mesh(points, length, dimension, SerialDecomposition(points, dimension))


def random_spectral(mesh: Mesh, *components: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (*components, *mesh.physical_shape)
    field_physical = torch.randn(shape, generator=generator, dtype=mesh.dtype)
    return mesh.decomposition.to_spectral(field_physical)


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    scale = max(expected.abs().max().item(), 1.0)
    assert (actual - expected).abs().max().item() <= TOLERANCE * scale


class Source:
    def __init__(
        self,
        variable: str,
        source_spectral: torch.Tensor,
        components: tuple[int, ...] = (),
    ) -> None:
        self.variable = variable
        self.source_spectral = source_spectral
        self.components = components

    @property
    def name(self) -> str:
        return "source"

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        return {self.variable: self.components}

    def contribution(self, state: State, time: float) -> torch.Tensor:
        return self.source_spectral * (1.0 + time) + state[self.variable] ** 2


def test_diffusion_contribution_is_minus_diffusivity_times_k_squared() -> None:
    mesh = make_mesh(16, 1, length=0.7)
    scalar_spectral = random_spectral(mesh)
    equation = Equation("scalar", [Diffusion(mesh, "scalar", 0.3)])

    contributions = equation.contributions({"scalar": scalar_spectral}, 0.0)

    expected = -0.3 * mesh.wavenumbers_squared * scalar_spectral
    assert list(contributions) == ["diffusion"]
    assert_close(contributions["diffusion"], expected)


def test_contributions_sum_to_the_tendency() -> None:
    mesh = make_mesh(16, 1)
    scalar_spectral = random_spectral(mesh)
    source = Source("scalar", random_spectral(mesh, seed=1))
    equation = Equation("scalar", [Diffusion(mesh, "scalar", 0.3), source])
    state = {"scalar": scalar_spectral}
    nonlinear_spectral = torch.full_like(scalar_spectral, math.nan)

    equation.nonlinear_tendency(state, 0.5, nonlinear_spectral)

    tendency_spectral = equation.linear_operator * scalar_spectral + nonlinear_spectral
    contributions = equation.contributions(state, 0.5)
    assert_close(sum(contributions.values(), torch.zeros(())), tendency_spectral)
    assert_close(nonlinear_spectral, source.contribution(state, 0.5))


def test_projection_applies_to_each_contribution_and_the_nonlinear_tendency() -> None:
    mesh = make_mesh(8, 2)
    velocity_spectral = random_spectral(mesh, 2)
    source = Source("velocity", random_spectral(mesh, 2, seed=1), (2,))
    diffusion = Diffusion(mesh, "velocity", 0.3, components=(2,))
    equation = Equation("velocity", [diffusion, source], projection=mesh.project)
    state = {"velocity": velocity_spectral}
    nonlinear_spectral = torch.empty_like(velocity_spectral)

    equation.nonlinear_tendency(state, 0.0, nonlinear_spectral)
    contributions = equation.contributions(state, 0.0)

    expected_source = mesh.project(source.contribution(state, 0.0))
    expected_diffusion = mesh.project(diffusion.multiplier * velocity_spectral)
    assert_close(nonlinear_spectral, expected_source)
    assert_close(contributions["source"], expected_source)
    assert_close(contributions["diffusion"], expected_diffusion)
    assert_close(mesh.divergence(contributions["diffusion"]), torch.zeros(()))


def test_check_projected_raises_for_a_variable_the_projection_changes() -> None:
    mesh = make_mesh(8, 2)
    velocity_spectral = random_spectral(mesh, 3, 2)
    state = {"velocity": mesh.project(velocity_spectral) + 1e-6 * velocity_spectral}
    equation = Equation("velocity", [], projection=mesh.project)

    with pytest.raises(ValueError, match="initial 'velocity' is not divergence-free"):
        equation.check_projected(mesh, state)


@pytest.mark.parametrize("scale", [1.0, 0.0])
def test_check_projected_passes_a_solenoidal_or_a_zero_variable(scale: float) -> None:
    mesh = make_mesh(8, 2)
    velocity_spectral = scale * mesh.project(random_spectral(mesh, 3, 2))
    equation = Equation("velocity", [], projection=mesh.project)

    equation.check_projected(mesh, {"velocity": velocity_spectral})
