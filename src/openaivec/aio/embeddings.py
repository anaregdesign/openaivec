"""Embedding utilities built on top of OpenAI’s embedding endpoint.

This module provides ``VectorizedEmbeddingsOpenAI``, a concrete implementation
of the ``VectorizedEmbeddings`` abstract base class. It leverages the OpenAI
SDK for embedding generation, supporting both sequential and asynchronous
execution with concurrency control and exponential back-off for rate limits.
"""

import asyncio
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, RateLimitError

from openaivec.embeddings import VectorizedEmbeddings
from openaivec.log import observe
from openaivec.util import backoff, map_unique_minibatch_async

__all__ = ["VectorizedEmbeddingsOpenAI"]

_LOGGER: Logger = getLogger(__name__)


@dataclass(frozen=True)
class VectorizedEmbeddingsOpenAI(VectorizedEmbeddings):
    """Thin wrapper around the OpenAI /embeddings endpoint using async operations.

    This class provides an asynchronous interface for generating embeddings using
    OpenAI models. It manages concurrency and handles rate limits automatically.

    Attributes:
        client: An already‑configured ``openai.AsyncOpenAI`` client.
        model_name: The model identifier, e.g. ``"text-embedding-3-small"``.
        max_concurrency: Maximum number of concurrent requests to the OpenAI API.
    """

    client: AsyncOpenAI
    model_name: str
    max_concurrency: int = 8  # Default concurrency limit
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self):
        # Initialize the semaphore after the object is created
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.max_concurrency))

    @observe(_LOGGER)
    @backoff(exception=RateLimitError, scale=60, max_retries=16)
    async def _embed_chunk(self, inputs: List[str]) -> List[NDArray[np.float32]]:
        """Embed one minibatch of sentences asynchronously, respecting concurrency limits.

        This private helper handles the actual API call for a batch of inputs.
        Exponential back-off is applied automatically when ``openai.RateLimitError``
        is raised.

        Args:
            inputs: Input strings to be embedded. Duplicates are allowed.

        Returns:
            List of embedding vectors (``np.ndarray`` with dtype ``float32``)
            in the same order as *inputs*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted.
        """
        # Acquire semaphore before making the API call
        async with self._semaphore:
            responses = await self.client.embeddings.create(input=inputs, model=self.model_name)
            return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_LOGGER)
    async def create_async(self, inputs: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        """Asynchronous public API: generate embeddings for a list of inputs.

        Uses ``map_unique_minibatch_async`` to efficiently handle batching and
        de-duplication.

        Args:
            inputs: A list of input strings. Duplicates are handled efficiently.
            batch_size: Maximum number of unique inputs per API call.

        Returns:
            A list of ``np.ndarray`` objects (dtype ``float32``) where each entry
            is the embedding of the corresponding string in *inputs*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted during API calls.
        """
        return await map_unique_minibatch_async(inputs, batch_size, self._embed_chunk)

    @observe(_LOGGER)
    def create(self, inputs: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        """Synchronous public API: generate embeddings for a list of inputs.

        This method wraps the asynchronous ``create_async`` method using ``asyncio.run``.

        Args:
            inputs: A list of input strings. Duplicates are handled efficiently.
            batch_size: Maximum number of unique inputs per API call.

        Returns:
            A list of ``np.ndarray`` objects (dtype ``float32``) where each entry
            is the embedding of the corresponding string in *inputs*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted during API calls.
        """
        return asyncio.run(self.create_async(inputs, batch_size))
