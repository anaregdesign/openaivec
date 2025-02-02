from typing import List, TypeVar, Callable


T: TypeVar = TypeVar("T")
U: TypeVar = TypeVar("U")


def split_to_minibatch(b: List[T], batch_size: int) -> List[List[T]]:
    return [b[i:i + batch_size] for i in range(0, len(b), batch_size)]


def map_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    return [item for batch in split_to_minibatch(b, batch_size) for item in f(batch)]


def map_unique(b: List[T], f: Callable[[List[T]], List[U]]) -> List[U]:
    unique_b: List[T] = list(dict.fromkeys(b))
    results: List[U] = f(unique_b)
    return [results[unique_b.index(item)] for item in b]
