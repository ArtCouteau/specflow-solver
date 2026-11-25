import math
from collections.abc import Mapping, Sequence

import pytest
import torch

from specflow.decomposition import SerialDecomposition
from specflow.equation import Equation
from specflow.integrators import IFRK4
from specflow.mesh import Mesh
from specflow.simulation import Simulation
from specflow.terms import Diffusion, State
from specflow.time_step import FixedTimeStep


class Reader:
    def __init__(self, key: str, components: tuple[int, ...]) -> None:
        self.key = key
        self.components = components

    @property
    def name(self) -> str:
        return f"reader of {self.key}"

    @property
    def reads(self) -> Mapping[str, tuple[int, ...]]:
        return {self.key: self.components}

    def contribution(self, state: State, time: float) -> torch.Tensor:
        return torch.zeros_like(state[self.key])


def make_mesh(dimension: int = 1) -> Mesh:
    return Mesh(8, 2 * math.pi, dimension, SerialDecomposition(8, dimension))


def make_simulation(mesh: Mesh, equations: Sequence[Equation]) -> Simulation:
    return Simulation(mesh, equations, IFRK4(), FixedTimeStep(0.1))


def scalar_equation(mesh: Mesh) -> Equation:
    return Equation("scalar", [Diffusion(mesh, "scalar", 0.1)])


def zeros_spectral(mesh: Mesh, *components: int) -> torch.Tensor:
    shape = (*components, *mesh.spectral_shape)
    return torch.zeros(shape, dtype=mesh.complex_dtype)


def test_a_run_leaves_the_initial_state_untouched() -> None:
    mesh = make_mesh()
    scalar_physical = torch.cos(mesh.coordinates[0])
    initial = {"scalar": mesh.decomposition.to_spectral(scalar_physical)}
    copy = initial["scalar"].clone()

    final = make_simulation(mesh, [scalar_equation(mesh)]).run(initial, 0.0, 0.3)

    assert torch.equal(initial["scalar"], copy)
    assert not torch.equal(final["scalar"], copy)
    assert final["scalar"].is_inference()


def test_leading_axes_are_free() -> None:
    mesh = make_mesh()
    initial = {"scalar": zeros_spectral(mesh, 3)}

    final = make_simulation(mesh, [scalar_equation(mesh)]).run(initial, 0.0, 0.1)

    assert final["scalar"].shape == initial["scalar"].shape


def test_a_missing_key_raises() -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, [scalar_equation(mesh)])

    with pytest.raises(ValueError, match="missing.*scalar"):
        simulation.run({}, 0.0, 1.0)


def test_a_key_evolved_by_no_equation_raises() -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, [scalar_equation(mesh)])
    state = {"scalar": zeros_spectral(mesh), "other": zeros_spectral(mesh)}

    with pytest.raises(ValueError, match="other.*no equation"):
        simulation.run(state, 0.0, 1.0)


def test_a_key_evolved_by_two_equations_raises() -> None:
    mesh = make_mesh()

    with pytest.raises(ValueError, match="scalar.*more than one equation"):
        make_simulation(mesh, [scalar_equation(mesh), scalar_equation(mesh)])


def test_a_read_key_that_no_equation_evolves_raises() -> None:
    mesh = make_mesh()
    equation = Equation("scalar", [Reader("velocity", (1,))])

    with pytest.raises(ValueError, match="velocity.*no equation"):
        make_simulation(mesh, [equation])


def test_terms_reading_a_key_with_different_component_shapes_raise() -> None:
    mesh = make_mesh()
    terms = [Diffusion(mesh, "scalar", 0.1), Reader("scalar", (1,))]

    with pytest.raises(ValueError, match="scalar.*component shapes"):
        make_simulation(mesh, [Equation("scalar", terms)])


def test_two_terms_with_one_name_raise() -> None:
    mesh = make_mesh()
    terms = [Diffusion(mesh, "scalar", 0.1), Diffusion(mesh, "scalar", 0.2)]

    with pytest.raises(ValueError, match="diffusion.*twice"):
        make_simulation(mesh, [Equation("scalar", terms)])


def test_a_linear_term_of_another_variable_raises() -> None:
    mesh = make_mesh()
    terms = [Diffusion(mesh, "other", 0.1)]

    with pytest.raises(ValueError, match="linear term.*other"):
        make_simulation(mesh, [Equation("scalar", terms)])


@pytest.mark.parametrize(
    "shape",
    [(2, 8), (3, 5), (5,)],
    ids=["physical shape", "wrong component", "missing component axis"],
)
def test_a_wrong_shape_raises(shape: tuple[int, ...]) -> None:
    mesh = make_mesh()
    equation = Equation("scalar", [Diffusion(mesh, "scalar", 0.1, components=(2,))])
    simulation = make_simulation(mesh, [equation])
    state = {"scalar": torch.zeros(shape, dtype=mesh.complex_dtype)}

    with pytest.raises(ValueError, match="scalar.*shape"):
        simulation.run(state, 0.0, 1.0)


def test_a_wrong_shape_of_a_key_no_term_reads_raises() -> None:
    mesh = make_mesh(2)
    simulation = make_simulation(mesh, [Equation("scalar", [])])
    state = {"scalar": torch.zeros(8, 8, dtype=mesh.complex_dtype)}

    with pytest.raises(ValueError, match="scalar.*shape"):
        simulation.run(state, 0.0, 1.0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.float64])
def test_a_wrong_dtype_raises(dtype: torch.dtype) -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, [scalar_equation(mesh)])
    state = {"scalar": torch.zeros(mesh.spectral_shape, dtype=dtype)}

    with pytest.raises(ValueError, match="scalar.*dtype"):
        simulation.run(state, 0.0, 1.0)


def test_a_wrong_device_raises() -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, [scalar_equation(mesh)])
    state = {"scalar": zeros_spectral(mesh).to("meta")}

    with pytest.raises(ValueError, match="scalar.*device"):
        simulation.run(state, 0.0, 1.0)


def test_an_end_before_the_start_raises() -> None:
    mesh = make_mesh()
    simulation = make_simulation(mesh, [scalar_equation(mesh)])

    with pytest.raises(ValueError, match="end"):
        simulation.run({"scalar": zeros_spectral(mesh)}, 1.0, 0.5)


@pytest.mark.parametrize("value", [0.0, -0.1, math.inf, math.nan])
def test_a_fixed_time_step_that_is_not_positive_and_finite_raises(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="time step"):
        FixedTimeStep(value)
