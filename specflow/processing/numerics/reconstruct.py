
from ...core import context
from ...simulation import equation


def build_equation(tag: str) -> equation.Equation:
    return equation.EquationRegistry.get(
        context.get().config.get(f'equations/{tag}/prebuild', str, 'custom')
    )(tag)
