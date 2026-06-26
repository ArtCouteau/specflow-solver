from __future__ import annotations

from typing import Generic, Type, TypeVar

T = TypeVar('T')


class Registry(Generic[T]):
    _registry: dict[str, Type[T]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: str):
        def decorator(klass: Type[T]) -> Type[T]:
            cls._registry[key] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, key: str) -> Type[T]:
        if key not in cls._registry:
            raise KeyError(
                f"'{key}' not found in {cls.__name__}, available:\n"
                + '\n'.join(f"  - {k}" for k in cls._registry)
            )
        return cls._registry[key]
