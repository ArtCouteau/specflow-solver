import math

import pytest
import torch

from specflow import diagnostics, distributions
from specflow.decomposition import SerialDecomposition
from specflow.mesh import Mesh

POINTS = 12
VISCOSITY = 0.01
TOLERANCE = 1e-12
DIMENSIONS = [2, 3]


def make_mesh(dimension: int, points: int = POINTS) -> Mesh:
    decomposition = SerialDecomposition(points, dimension)
    return Mesh(points, 2 * math.pi, dimension, decomposition)


def random_velocity_spectral(mesh: Mesh, *leading: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (*leading, mesh.dimension, *mesh.physical_shape)
    field_physical = torch.randn(shape, generator=generator, dtype=mesh.dtype)
    field_spectral = mesh.decomposition.to_spectral(field_physical)
    field_spectral.mul_(mesh.wavenumbers_squared < (mesh.points / 3) ** 2)
    return mesh.project(field_spectral)


def domain_mean(mesh: Mesh, field_physical: torch.Tensor) -> torch.Tensor:
    return distributions.raw_moments(mesh, field_physical, 1)[0]


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    scale = max(expected.abs().max().item(), 1.0)
    assert (actual - expected).abs().max().item() <= TOLERANCE * scale


def test_raw_moments_lead_with_the_order_axis_and_keep_the_others() -> None:
    mesh = make_mesh(2)
    field_physical = torch.randn(2, 3, *mesh.physical_shape, dtype=mesh.dtype)

    moments = distributions.raw_moments(mesh, field_physical)

    assert moments.shape == (4, 2, 3)
    assert moments.dtype == mesh.dtype
    for order in range(1, 5):
        expected = field_physical.pow(order).mean(dim=(-2, -1))
        assert_close(moments[order - 1], expected)


def test_raw_moments_take_the_highest_order() -> None:
    mesh = make_mesh(1)
    field_physical = torch.randn(*mesh.physical_shape, dtype=mesh.dtype)

    moments = distributions.raw_moments(mesh, field_physical, 6)

    assert moments.shape == (6,)
    assert_close(moments[5], field_physical.pow(6).mean())


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_second_moment_of_each_longitudinal_derivative_matches_the_spectrum(
    dimension: int,
) -> None:
    mesh = make_mesh(dimension)
    velocity_spectral = random_velocity_spectral(mesh)

    derivatives_physical = diagnostics.longitudinal_derivatives(mesh, velocity_spectral)
    moments = distributions.raw_moments(mesh, derivatives_physical)

    density_spectral = (
        velocity_spectral * mesh.derivative_wavenumbers
    ).abs().square() * mesh.parseval_weights
    assert moments.shape == (4, dimension)
    assert_close(moments[1], mesh.decomposition.sum(density_spectral))


def test_the_longitudinal_derivatives_of_a_shear_vanish_and_of_a_strain_do_not() -> (
    None
):
    mesh = make_mesh(2)
    x, y = mesh.coordinates[0], mesh.coordinates[1]
    velocity_physical = torch.stack([torch.sin(x) + torch.sin(y), torch.cos(x)])

    derivatives_physical = diagnostics.longitudinal_derivatives(
        mesh, mesh.decomposition.to_spectral(velocity_physical)
    )

    assert_close(derivatives_physical[0], torch.cos(x))
    assert_close(derivatives_physical[1], torch.zeros_like(x))


def test_transverse_derivatives_are_row_major_off_the_diagonal() -> None:
    mesh = make_mesh(3)
    x, y, z = mesh.coordinates[0], mesh.coordinates[1], mesh.coordinates[2]
    velocity_physical = torch.stack(
        [torch.sin(y) + 2 * torch.sin(z), 3 * torch.sin(x), 4 * torch.sin(y)]
    )

    derivatives_physical = diagnostics.transverse_derivatives(
        mesh, mesh.decomposition.to_spectral(velocity_physical)
    )

    expected = torch.stack(
        [
            torch.cos(y),
            2 * torch.cos(z),
            3 * torch.cos(x),
            torch.zeros_like(x),
            torch.zeros_like(x),
            4 * torch.cos(y),
        ]
    )
    assert derivatives_physical.shape == (6, *mesh.physical_shape)
    assert_close(derivatives_physical, expected)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_mean_local_dissipation_is_the_dissipation(dimension: int) -> None:
    mesh = make_mesh(dimension)
    velocity_spectral = random_velocity_spectral(mesh, 2)

    local_physical = diagnostics.local_dissipation(mesh, velocity_spectral, VISCOSITY)

    assert local_physical.shape == (2, *mesh.physical_shape)
    assert local_physical.min().item() >= 0
    expected = diagnostics.dissipation(mesh, velocity_spectral, VISCOSITY)
    assert_close(domain_mean(mesh, local_physical), expected)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_the_mean_local_enstrophy_is_the_enstrophy(dimension: int) -> None:
    mesh = make_mesh(dimension)
    velocity_spectral = random_velocity_spectral(mesh, 2)

    local_physical = diagnostics.local_enstrophy(mesh, velocity_spectral)

    assert local_physical.shape == (2, *mesh.physical_shape)
    expected = diagnostics.enstrophy(mesh, velocity_spectral)
    assert_close(domain_mean(mesh, local_physical), expected)


def test_the_local_enstrophy_of_a_1d_mesh_raises() -> None:
    mesh = make_mesh(1)

    with pytest.raises(ValueError, match="1D"):
        diagnostics.local_enstrophy(mesh, torch.zeros(1, 7, dtype=mesh.complex_dtype))


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_histogram_counts_total_the_points_per_component(dimension: int) -> None:
    mesh = make_mesh(dimension)
    velocity_spectral = random_velocity_spectral(mesh)
    derivatives_physical = diagnostics.longitudinal_derivatives(mesh, velocity_spectral)
    edges = torch.linspace(-0.5, 0.5, 11, dtype=mesh.dtype)

    counts = distributions.histogram(mesh, derivatives_physical, edges)

    assert counts.shape == (dimension, 12)
    assert counts.dtype == torch.int64
    assert counts[:, 0].sum().item() > 0
    assert counts[:, -1].sum().item() > 0
    assert counts.sum(-1).tolist() == dimension * [POINTS**dimension]


def test_histogram_bins_are_half_open() -> None:
    mesh = make_mesh(1, 8)
    field_physical = torch.arange(8, dtype=mesh.dtype)
    edges = torch.tensor([1.0, 3.0, 4.0, 6.0], dtype=mesh.dtype)

    counts = distributions.histogram(mesh, field_physical, edges)

    assert counts.tolist() == [1, 2, 1, 2, 2]


def test_histogram_takes_log_spaced_edges() -> None:
    mesh = make_mesh(1, 8)
    field_physical = torch.tensor(
        [0.5, 1.0, 5.0, 10.0, 50.0, 99.0, 100.0, 1000.0], dtype=mesh.dtype
    )
    edges = torch.logspace(0, 2, 3, dtype=mesh.dtype)

    counts = distributions.histogram(mesh, field_physical, edges)

    assert counts.tolist() == [1, 2, 3, 2]


def test_standardized_moments_of_a_sine_match_their_closed_forms() -> None:
    mesh = make_mesh(1, 16)
    field_physical = 1 + 2 * torch.sin(mesh.coordinates[0])

    mean, variance, skewness, flatness = distributions.standardized_moments(
        distributions.raw_moments(mesh, field_physical)
    )

    assert mean.item() == pytest.approx(1, rel=1e-14)
    assert variance.item() == pytest.approx(2, rel=1e-14)
    assert skewness.item() == pytest.approx(0, abs=1e-14)
    assert flatness.item() == pytest.approx(1.5, rel=1e-14)


def test_standardized_moments_keep_the_axes_after_the_order_axis() -> None:
    mesh = make_mesh(1, 16)
    x = mesh.coordinates[0]
    field_physical = torch.stack([torch.sin(x), 3 * torch.cos(x) - 1])

    moments = distributions.standardized_moments(
        distributions.raw_moments(mesh, field_physical)
    )

    expected = [[0, -1], [0.5, 4.5], [0, 0], [1.5, 1.5]]
    for actual, values in zip(moments, expected, strict=True):
        assert actual.shape == (2,)
        assert actual.tolist() == pytest.approx(values, abs=1e-14)


def test_standardized_moments_of_a_constant_component_are_undefined() -> None:
    mesh = make_mesh(1, 16)
    x = mesh.coordinates[0]
    field_physical = torch.stack([torch.sin(x), torch.zeros_like(x), 3 + 0 * x])

    _, variance, skewness, flatness = distributions.standardized_moments(
        distributions.raw_moments(mesh, field_physical)
    )

    assert variance.tolist() == pytest.approx([0.5, 0, 0], abs=1e-14)
    assert skewness[0].item() == pytest.approx(0, abs=1e-14)
    assert flatness[0].item() == pytest.approx(1.5, rel=1e-14)
    assert skewness[1:].isnan().all()
    assert flatness[1:].isnan().all()


def test_a_round_off_negative_variance_is_zero() -> None:
    moments = torch.tensor([1.0, 1.0 - 2**-52, 1.0, 1.0], dtype=torch.float64)

    _, variance, skewness, flatness = distributions.standardized_moments(moments)

    assert variance.item() == 0
    assert skewness.isnan().item()
    assert flatness.isnan().item()


def test_standardized_moments_need_four_orders() -> None:
    with pytest.raises(ValueError, match="4"):
        distributions.standardized_moments(torch.ones(3, 2))


def test_the_density_of_uniform_counts_integrates_to_the_in_range_fraction() -> None:
    edges = torch.logspace(-2, 1, 7, dtype=torch.float64)
    counts = torch.ones(2, 8, dtype=torch.float64)

    density = distributions.probability_density(counts, edges)

    assert density.shape == (2, 6)
    integral = (density * edges.diff()).sum(-1)
    assert integral.tolist() == pytest.approx([6 / 8, 6 / 8], rel=1e-14)


def test_the_density_of_counts_divides_each_bin_by_its_width() -> None:
    edges = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    counts = torch.tensor([0, 2, 2, 0])

    density = distributions.probability_density(counts, edges)

    assert density.dtype == torch.float64
    assert density.tolist() == pytest.approx([0.5, 0.25], rel=1e-14)
