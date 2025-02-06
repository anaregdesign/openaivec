"""Vectorize OpenAI Utilities.

This module provides utility functions for executing common operations:
    - Splitting lists into minibatches.
    - Applying functions to minibatches sequentially or in parallel.
    - Mapping functions only to unique elements in a list.

Example:
    >>> from openaivec.util import split_to_minibatch
    >>> batches = split_to_minibatch([1, 2, 3, 4], 2)
    >>> print(batches)
    [[1, 2], [3, 4]]
"""

from concurrent.futures.thread import ThreadPoolExecutor
from itertools import chain
from typing import List, TypeVar, Callable

T = TypeVar("T")
U = TypeVar("U")


def split_to_minibatch(b: List[T], batch_size: int) -> List[List[T]]:
    """Splits the list into sublists of a given batch size.

    Args:
        b (List[T]): The input list.
        batch_size (int): The desired size of each sublist.

    Returns:
        List[List[T]]: A list of sublists.
    """
    return [b[i : i + batch_size] for i in range(0, len(b), batch_size)]


def map_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Applies a function to each minibatch of the list and flattens the results.

    Splits the list `b` into batches of the given size and applies function `f` to each batch.

    Args:
        b (List[T]): The input list.
        batch_size (int): The size of each batch.
        f (Callable[[List[T]], List[U]]): A function that processes a list of T and returns a list of U.

    Returns:
        List[U]: A flattened list after processing each batch.
    """
    batches = split_to_minibatch(b, batch_size)
    return list(chain.from_iterable(f(batch) for batch in batches))


def map_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Applies a function to each minibatch of the list in parallel and flattens the results.

    Splits the list `b` into batches of the given size, processes each batch in parallel using function `f`,
    and then flattens the resulting lists into a single list.

    Args:
        b (List[T]): The input list.
        batch_size (int): The size of each batch.
        f (Callable[[List[T]], List[U]]): A function that processes a list of T and returns a list of U.

    Returns:
        List[U]: A flattened list of results from parallel execution.
    """
    batches = split_to_minibatch(b, batch_size)
    with ThreadPoolExecutor() as executor:
        results = executor.map(f, batches)
    return list(chain.from_iterable(results))


def map_unique(b: List[T], f: Callable[[List[T]], List[U]]) -> List[U]:
    """Applies a function once to the unique elements of a list and remaps the results.

    Removes duplicate values (preserving order), processes the unique elements with function `f`,
    and then maps the resulting values back to correspond to the original list.

    Args:
        b (List[T]): The input list.
        f (Callable[[List[T]], List[U]]): A function that processes a list of T and returns a list of U.

    Returns:
        List[U]: The list with results mapped back to the original values.
    """
    # Use dict.fromkeys to remove duplicates while preserving the order
    unique_values = list(dict.fromkeys(b))
    value_to_index = {v: i for i, v in enumerate(unique_values)}
    results = f(unique_values)
    return [results[value_to_index[value]] for value in b]


def map_unique_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Processes unique elements of a list in minibatches using a function and remaps the results.

    Splits the unique values into minibatches, applies function `f` to each,
    and reassembles the results to match the order of the original list.

    Args:
        b (List[T]): The input list.
        batch_size (int): The size of each minibatch.
        f (Callable[[List[T]], List[U]]): A function that processes a list of T and returns a list of U.

    Returns:
        List[U]: The processed list with results corresponding to the original list order.
    """
    return map_unique(b, lambda x: map_minibatch(x, batch_size, f))


def map_unique_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """Processes unique elements of a list in minibatches in parallel using a function and remaps the results.

    Splits the unique values into minibatches, processes each batch in parallel with function `f`,
    and reassembles the results to maintain the original order.

    Args:
        b (List[T]): The input list.
        batch_size (int): The size of each minibatch.
        f (Callable[[List[T]], List[U]]): A function that processes a list of T and returns a list of U.

    Returns:
        List[U]: The processed list with results corresponding to the original list order.
    """
    return map_unique(b, lambda x: map_minibatch_parallel(x, batch_size, f))
