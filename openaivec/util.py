from typing import List, TypeVar, Callable

T: TypeVar = TypeVar("T")
U: TypeVar = TypeVar("U")


def split_to_minibatch(b: List[T], batch_size: int) -> List[List[T]]:
    return [b[i:i + batch_size] for i in range(0, len(b), batch_size)]


def map_with_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    return [item for batch in split_to_minibatch(b, batch_size) for item in f(batch)]
