from __future__ import annotations

from .terms import term
from ..core.physics import field


class Equation:
    terms: dict[str, term.Term]
    variable: field.Field

    def __init__(self):
        pass
