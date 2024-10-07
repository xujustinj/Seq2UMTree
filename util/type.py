from typing import Any, Type, TypeVar


T = TypeVar("T")

def assert_type(x: Any, t: Type[T]) -> T:
    assert isinstance(x, t)
    return x
