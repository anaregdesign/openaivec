import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class MinibatchUtil(Generic[T], Generic[U]):
    """Utility class for applying an asynchronous function to a list of inputs in minibatches.

    This class helps manage batching and asynchronous execution of a function `f`
    over a potentially large list of inputs. It internally uses hashes to handle
    potential duplicates (last occurrence wins in case of hash collision) before processing.
    The results are stored maintaining the original input order.

    Attributes:
        inputs (List[T]): The original list of inputs provided.
        hashes (List[int]): List of hash values corresponding to the original inputs.
        hash_inputs (Dict[int, T]): Dictionary mapping unique hash values to input objects.
        hash_outputs (Dict[int, U]): Dictionary mapping unique hash values to output objects.
        outputs (List[U]): List of outputs corresponding to the original inputs, in order.
        batch_size (int): The size of each minibatch for processing.
    """

    inputs: List[T] = field(default_factory=list)
    hashes: List[int] = field(default_factory=list)
    hash_inputs: Dict[int, T] = field(default_factory=dict)
    hash_outputs: Dict[int, U] = field(default_factory=dict)
    outputs: List[U] = field(default_factory=list)
    batch_size: int = 128

    def set_inputs(self, inputs: List[T]) -> None:
        """Set the inputs for the minibatch utility and precompute hashes.

        This method stores the original inputs and computes their hashes.
        It also creates an internal mapping (`hash_inputs`) from hash values to inputs,
        effectively deduplicating inputs based on their hash (last occurrence wins).

        Args:
            inputs (List[T]): List of inputs to be processed.
        """
        self.inputs = inputs
        self.hashes = [v.__hash__() for v in self.inputs]
        self.hash_inputs = {k: v for k, v in zip(self.hashes, self.inputs)}

    async def map(self, f: Callable[[List[T]], Awaitable[List[U]]]) -> None:
        """Asynchronously map a function `f` over the unique inputs in batches.

        This method applies the asynchronous function `f` to the unique inputs
        (derived from `hash_inputs`) divided into batches of `batch_size`.
        It gathers the results concurrently and stores them in the `outputs`
        attribute, ensuring the order corresponds to the original `inputs` list.

        Args:
            f (Callable[[List[T]], Awaitable[List[U]]]): Asynchronous function to apply.
                It takes a batch of inputs (List[T]) and must return a list of
                corresponding outputs (List[U]) of the same size.

        Raises:
            ValueError: If the function `f` does not return a list of outputs
                        with the same number of elements as the input batch.
        """
        keys: List[int] = list(self.hash_inputs.keys())
        unique_inputs: List[T] = list(self.hash_inputs.values())

        if not unique_inputs:
            self.outputs = []
            return

        input_batches: List[List[T]] = [
            unique_inputs[i : i + self.batch_size] for i in range(0, len(unique_inputs), self.batch_size)
        ]
        output_batches: List[List[U]] = await asyncio.gather(*[f(batch) for batch in input_batches])
        unique_outputs: List[U] = [u for batch in output_batches for u in batch]

        if len(keys) != len(unique_outputs):
            raise ValueError(
                f"Number of unique inputs ({len(keys)}) does not match number of unique outputs ({len(unique_outputs)}). Check the function f."
            )

        self.hash_outputs = {k: v for k, v in zip(keys, unique_outputs)}
        self.outputs = [self.hash_outputs[k] for k in self.hashes]
