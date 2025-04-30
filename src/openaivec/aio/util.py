import asyncio
from typing import Awaitable, Callable, Dict, List, TypeVar

__all__ = ["map"]

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
    original_hashes: List[int] = [v.__hash__() for v in inputs]
    hash_inputs: Dict[int, T] = {k: v for k, v in zip(original_hashes, inputs)}
    unique_hashes: List[int] = list(hash_inputs.keys())
    unique_inputs: List[T] = list(hash_inputs.values())
    input_batches: List[List[T]] = [unique_inputs[i : i + batch_size] for i in range(0, len(unique_inputs), batch_size)]
    output_batches: List[List[U]] = await asyncio.gather(*[f(batch) for batch in input_batches])
    unique_outputs: List[U] = [u for batch in output_batches for u in batch]
    if len(unique_hashes) != len(unique_outputs):
        raise ValueError(
            f"Number of unique inputs ({len(unique_hashes)}) does not match number of unique outputs ({len(unique_outputs)}). Check the function f."
        )
    hash_outputs: Dict[int, U] = {k: v for k, v in zip(unique_hashes, unique_outputs)}
    outputs: List[U] = [hash_outputs[k] for k in original_hashes]
    return outputs
