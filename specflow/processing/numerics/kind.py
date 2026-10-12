from __future__ import annotations

from typing import Any, TypeVar, get_args, get_origin

# A kind is the generic alias of a wrapper, e.g. Field[Vector, Spectral]. In a pattern, a TypeVar matches any type.


def unify(pattern: Any, kind: Any, bindings: dict[TypeVar, Any] | None = None) -> dict[TypeVar, Any] | None:
    bindings = {} if bindings is None else dict(bindings)

    if isinstance(pattern, TypeVar):
        if pattern in bindings:
            return bindings if bindings[pattern] == kind else None
        if pattern.__bound__ is not None and not (isinstance(kind, type) and issubclass(kind, pattern.__bound__)):
            return None
        bindings[pattern] = kind
        return bindings

    if get_origin(pattern) is None:
        return bindings if pattern == kind else None

    if get_origin(pattern) is not get_origin(kind) or len(get_args(pattern)) != len(get_args(kind)):
        return None

    for p, k in zip(get_args(pattern), get_args(kind)):
        bindings = unify(p, k, bindings)
        if bindings is None:
            return None

    return bindings


def substitute(pattern: Any, bindings: dict[TypeVar, Any]) -> Any:
    if isinstance(pattern, TypeVar):
        if pattern not in bindings:
            raise TypeError(f"Unbound type variable {pattern}")
        return bindings[pattern]

    if get_origin(pattern) is None:
        return pattern

    return get_origin(pattern)[tuple(substitute(arg, bindings) for arg in get_args(pattern))]


def is_concrete(kind: Any) -> bool:
    if isinstance(kind, TypeVar):
        return False
    return all(is_concrete(arg) for arg in get_args(kind))


def kind_tag(kind: Any) -> str:
    if isinstance(kind, TypeVar):
        return kind.__name__
    if get_origin(kind) is None:
        return getattr(kind, "__name__", repr(kind))
    return f"{get_origin(kind).__name__}[{', '.join(kind_tag(arg) for arg in get_args(kind))}]"
