from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    _registry: dict[str, type[T]]

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: str):
        def decorator(klass: type[T]) -> type[T]:
            cls._registry[key] = klass
            return klass

        return decorator

    @classmethod
    def get(cls, key: str) -> type[T]:
        if key not in cls._registry:
            raise KeyError(
                f"'{key}' not found in {cls.__name__}, available:\n" + "\n".join(f"  - {k}" for k in cls._registry)
            )
        return cls._registry[key]
