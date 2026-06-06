import torch

from ...numerics import shell
from ....core.numerics import context
from ....core.numerics.operations import algebra, volume
from ....simulation import equation, solution


def kinetic_energy(
        equation: equation.Equation,
        solution: solution.SolutionVariable,
        time: float,
        shells: shell.SpectralShells
) -> dict[str, torch.Tensor]:
    tmp: dict[str, torch.Tensor] = {}

    for term in equation.terms.values():
        tmp[term.tag] = torch.zeros(
            (shells.nShells,),
            dtype=context.get().config.get('numerics/precision/float', torch.dtype),
            device=solution.data[term.variable].device
        )

        for index in range(shells.nShells):
            tmp[term.tag][index] = volume.sum_over_spectral_domain(
                shells.shell(index)
                * algebra.inner_vector_vector(solution.get('velocity').conj(), term.reconstruct(solution, time)).real
            )

    return tmp
