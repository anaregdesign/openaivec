import functools
import re
import time
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Optional, TypeVar

import numpy as np
import tiktoken


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def split_to_minibatch(b: List[T], batch_size: int) -> List[List[T]]:
    """Splits the list into sublists of size `batch_size`."""
    return [b[i : i + batch_size] for i in range(0, len(b), batch_size)]


def map_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Splits the list `b` into batches of size `batch_size` and applies the function `f` to each batch.
    The results (each a list) are then flattened into a single list.
    """
    batches = split_to_minibatch(b, batch_size)
    return list(chain.from_iterable(f(batch) for batch in batches))


def map_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Splits the list `b` into batches of size `batch_size` and applies the function `f` to each batch.
    The results (each a list) are then flattened into a single list.
    This version uses parallel processing to apply the function to each batch.
    """
    batches = split_to_minibatch(b, batch_size)
    with ThreadPoolExecutor() as executor:
        results = executor.map(f, batches)
    return list(chain.from_iterable(results))


def map_unique(b: List[T], f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Applies the function `f` only once to the unique values in the list `b` (preserving their order),
    and then maps the resulting values back to match the original list.
    This avoids repeated execution of `f` for duplicate values.
    """
    # Use dict.fromkeys to remove duplicates while preserving the order
    unique_values = list(dict.fromkeys(b))
    value_to_index = {v: i for i, v in enumerate(unique_values)}
    results = f(unique_values)
    return [results[value_to_index[value]] for value in b]


def map_unique_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Uses minibatch processing on the unique values of the list `b`.
    The function `f` is applied to these unique values in batches,
    and the results are mapped back to match the order of the original list.
    """
    return map_unique(b, lambda x: map_minibatch(x, batch_size, f))


def map_unique_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Uses minibatch processing on the unique values of the list `b`.
    The function `f` is applied to these unique values in batches using parallel processing,
    and the results are mapped back to match the order of the original list.
    """
    return map_unique(b, lambda x: map_minibatch_parallel(x, batch_size, f))


def get_exponential_with_cutoff(scale: float) -> float:
    gen = np.random.default_rng()

    while True:
        v = gen.exponential(scale)
        if v < scale * 3:
            return v


def backoff(exception: Exception, scale: int = None, max_retries: Optional[int] = None) -> Callable[..., V]:
    def decorator(func: Callable[..., V]) -> Callable[..., V]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> V:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exception:
                    attempt += 1
                    if max_retries is not None and attempt >= max_retries:
                        raise

                    interval = get_exponential_with_cutoff(scale)
                    time.sleep(interval)

        return wrapper

    return decorator


@dataclass(frozen=True)
class TextChunker:
    enc: tiktoken.Encoding

    def split(self, original: str, max_tokens: int, sep: List[str]) -> List[str]:
        sentences = re.split(f"({'|'.join(sep)})", original)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [(s, len(self.enc.encode(s))) for s in sentences]

        chunks = []
        sentence = ""
        token_count = 0
        for s, n in sentences:
            if token_count + n > max_tokens:
                if sentence:
                    chunks.append(sentence)
                sentence = ""
                token_count = 0

            sentence += s
            token_count += n

        if sentence:
            chunks.append(sentence)

        return chunks
