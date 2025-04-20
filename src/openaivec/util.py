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
    """Split a list into mini‑batches.

    Args:
        b (List[T]): Input list to split.
        batch_size (int): Maximum number of elements per batch.

    Returns:
        List[List[T]]: The input list divided into sub‑lists of length
        `batch_size` (the final batch may be smaller).
    """
    return [b[i : i + batch_size] for i in range(0, len(b), batch_size)]


def map_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Apply a function to each mini‑batch sequentially and flatten the result.

    Args:
        b (List[T]): Input list.
        batch_size (int): Batch size passed to :func:`split_to_minibatch`.
        f (Callable[[List[T]], List[U]]): Function executed on every batch.

    Returns:
        List[U]: Flattened list obtained by concatenating the results returned
        by ``f`` for each batch.
    """
    batches = split_to_minibatch(b, batch_size)
    return list(chain.from_iterable(f(batch) for batch in batches))


def map_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Parallel variant of :func:`map_minibatch`.

    Args:
        b (List[T]): Input list.
        batch_size (int): Batch size passed to :func:`split_to_minibatch`.
        f (Callable[[List[T]], List[U]]): Function executed on every batch.

    Returns:
        List[U]: Flattened list of results produced by ``f`` for every batch,
        evaluated in parallel threads.
    """
    batches = split_to_minibatch(b, batch_size)
    with ThreadPoolExecutor() as executor:
        results = executor.map(f, batches)
    return list(chain.from_iterable(results))


def map_unique(b: List[T], f: Callable[[List[T]], List[U]]) -> List[U]:
    """Call a function once per unique value and broadcast the result.

    Args:
        b (List[T]): Input list that may contain duplicates.
        f (Callable[[List[T]], List[U]]): Function applied to the unique values
            extracted from ``b``.

    Returns:
        List[U]: Result list aligned with the original order of ``b``.
    """
    unique_values = list(dict.fromkeys(b))
    value_to_index = {v: i for i, v in enumerate(unique_values)}
    results = f(unique_values)
    return [results[value_to_index[value]] for value in b]


def map_unique_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Combine :func:`map_unique` and :func:`map_minibatch`.

    The function ``f`` is executed on unique values only, processed in
    mini‑batches.

    Args:
        b (List[T]): Input list that may contain duplicates.
        batch_size (int): Batch size for mini‑batch processing.
        f (Callable[[List[T]], List[U]]): Function to apply.

    Returns:
        List[U]: Result list aligned with the original order of ``b``.
    """
    return map_unique(b, lambda x: map_minibatch(x, batch_size, f))


def map_unique_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Parallel version of :func:`map_unique_minibatch`.

    Args:
        b (List[T]): Input list that may contain duplicates.
        batch_size (int): Batch size for mini‑batch processing.
        f (Callable[[List[T]], List[U]]): Function to apply.

    Returns:
        List[U]: Result list aligned with the original order of ``b``.
    """
    return map_unique(b, lambda x: map_minibatch_parallel(x, batch_size, f))


def get_exponential_with_cutoff(scale: float) -> float:
    """Sample an exponential random variable with an upper cutoff.

    A value is repeatedly drawn from an exponential distribution with rate
    ``1/scale`` until it is smaller than ``3 * scale``.

    Args:
        scale (float): Scale parameter of the exponential distribution.

    Returns:
        float: Sampled value bounded by ``3 * scale``.
    """
    gen = np.random.default_rng()

    while True:
        v = gen.exponential(scale)
        if v < scale * 3:
            return v


def backoff(exception: Exception, scale: int | None = None, max_retries: Optional[int] = None) -> Callable[..., V]:
    """Decorator implementing exponential back‑off retry logic.

    Args:
        exception (Exception): Exception type that triggers a retry.
        scale (int | None): Scale parameter forwarded to
            :func:`get_exponential_with_cutoff`. If ``None``, the default scale
            of the RNG is used.
        max_retries (Optional[int]): Maximum number of retries. ``None`` means
            retry indefinitely.

    Returns:
        Callable[..., V]: A decorated function that retries on the specified
        exception with exponential back‑off.

    Raises:
        exception: Re‑raised when the maximum number of retries is exceeded.
    """

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
    """Utility for splitting text into token‑bounded chunks."""

    enc: tiktoken.Encoding

    def split(self, original: str, max_tokens: int, sep: List[str]) -> List[str]:
        """Token‑aware sentence segmentation.

        The text is first split by the given separators, then greedily packed
        into chunks whose token counts do not exceed ``max_tokens``.

        Args:
            original (str): Original text to split.
            max_tokens (int): Maximum number of tokens allowed per chunk.
            sep (List[str]): List of separator patterns used by
                :pyfunc:`re.split`.

        Returns:
            List[str]: List of text chunks respecting the ``max_tokens`` limit.
        """
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
