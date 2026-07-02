import torch

from ...numerics import shell
from ....core import context
from ....core.numerics.operations import algebra, volume
from ....simulation import equation, solution


def kinetic_energy(
        equation: equation.NavierStokesEquation,
        velocity: torch.Tensor,
        timestamp: str,
        shells: shell.SpectralShells
) -> dict[str, torch.Tensor]:
    tmp: dict[str, torch.Tensor] = {}

    for term in equation.terms.values():
        tmp[term.tag] = torch.zeros(
            (shells.nShells,),
            dtype=context.get().config.get('numerics/precision/float', torch.dtype),
            device=velocity.device
        )

        for index in range(shells.nShells):
            tmp[term.tag][index] = volume.sum_over_spectral_domain(
                shells.shell(index)
                * algebra.inner_vector_vector(
                    velocity.conj(),
                    term.reconstruct(solution.SolutionVariable({'velocity': velocity}), timestamp)
                ).real
            )

    return tmp
