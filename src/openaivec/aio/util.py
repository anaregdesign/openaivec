import asyncio
from functools import wraps
from typing import Awaitable, Callable, Dict, List, TypeVar, Coroutine, Any
import time  # Added for potential sleep fallback

__all__ = ["map", "as_sync"]  # Corrected __all__

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")


async def map(inputs: List[T], f: Callable[[List[T]], Awaitable[List[U]]], batch_size: int = 128) -> List[U]:
    """Asynchronously map a function `f` over a list of inputs in batches.

    This function divides the input list into smaller batches and applies the
    asynchronous function `f` to each batch concurrently. It gathers the results
    and returns them in the same order as the original inputs.

    Args:
        inputs (List[T]): List of inputs to be processed.
        f (Callable[[List[T]], Awaitable[List[U]]]): Asynchronous function to apply.
            It takes a batch of inputs (List[T]) and must return a list of
            corresponding outputs (List[U]) of the same size.
        batch_size (int): Size of each batch for processing.

    Returns:
        List[U]: List of outputs corresponding to the original inputs, in order.
    """
    original_hashes: List[int] = [hash(str(v)) for v in inputs]  # Use str(v) for hash if T is not hashable
    hash_inputs: Dict[int, T] = {k: v for k, v in zip(original_hashes, inputs)}
    unique_hashes: List[int] = list(hash_inputs.keys())
    unique_inputs: List[T] = list(hash_inputs.values())
    input_batches: List[List[T]] = [unique_inputs[i : i + batch_size] for i in range(0, len(unique_inputs), batch_size)]
    # Ensure f is awaited correctly within gather
    tasks = [f(batch) for batch in input_batches]
    output_batches: List[List[U]] = await asyncio.gather(*tasks)
    unique_outputs: List[U] = [u for batch in output_batches for u in batch]
    if len(unique_hashes) != len(unique_outputs):
        raise ValueError(
            f"Number of unique inputs ({len(unique_hashes)}) does not match number of unique outputs ({len(unique_outputs)}). Check the function f."
        )
    hash_outputs: Dict[int, U] = {k: v for k, v in zip(unique_hashes, unique_outputs)}
    outputs: List[U] = [hash_outputs[k] for k in original_hashes]
    return outputs


def as_sync(func: Callable[..., Coroutine[Any, Any, S]]) -> Callable[..., S]:
    """Decorator to run an async function synchronously.

    This decorator wraps an asynchronous function, allowing it to be called
    from synchronous code. It attempts to use the existing running event loop
    if one exists, otherwise it creates a new event loop using `asyncio.run`.

    Args:
        func: The asynchronous function to wrap.

    Returns:
        A synchronous wrapper function that executes the original async function
        and returns its result.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> S:
        try:
            loop = asyncio.get_running_loop()

        except RuntimeError:
            return asyncio.run(func(*args, **kwargs))

        else:
            return loop.run_until_complete(func(*args, **kwargs))

    return wrapper
