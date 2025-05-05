"""Embedding utilities built on top of OpenAI’s embedding endpoint.

This module defines an abstract base class ``VectorizedEmbeddings`` and a
concrete implementation ``VectorizedEmbeddingsOpenAI`` that delegates the
actual embedding work to the OpenAI SDK.  The implementation supports
sequential as well as multiprocess execution (via
``map_unique_minibatch_parallel``) and applies a generic
exponential‑back‑off policy when OpenAI’s rate limits are hit.
"""

from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI, RateLimitError

from openaivec.log import observe
from openaivec.util import backoff, map_unique_minibatch

__all__ = ["BatchEmbeddings"]

_logger: Logger = getLogger(__name__)


@dataclass(frozen=True)
class BatchEmbeddings:
    """Thin wrapper around the OpenAI /embeddings endpoint.

    Attributes:
        client: An already‑configured ``openai.OpenAI`` client.
        model_name: The model identifier, e.g. ``"text-embedding-3-small"``.
    """

    client: OpenAI
    model_name: str

    @observe(_logger)
    @backoff(exception=RateLimitError, scale=60, max_retries=16)
    def _embed_chunk(self, inputs: List[str]) -> List[NDArray[np.float32]]:
        """Embed one minibatch of sentences.

        This private helper is the unit of work used by the map/parallel
        utilities.  Exponential back‑off is applied automatically when
        ``openai.RateLimitError`` is raised.

        Args:
            inputs: Input strings to be embedded.  Duplicates are allowed; the
                implementation may decide to de‑duplicate internally.

        Returns:
            List of embedding vectors with the same ordering as *sentences*.
        """
        responses = self.client.embeddings.create(input=inputs, model=self.model_name)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_logger)
    def create(self, inputs: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        """See ``VectorizedEmbeddings.create`` for contract details.

        The call is internally delegated to either ``map_unique_minibatch`` or
        its parallel counterpart depending on *is_parallel*.

        Args:
            inputs: A list of input strings. Duplicates are allowed; the
                implementation may decide to de‑duplicate internally.
            batch_size: Maximum number of sentences to be sent to the underlying
                model in one request.

        Returns:
            A list of ``np.ndarray`` objects (dtype ``float32``) where each entry
                is the embedding of the corresponding sentence in *sentences*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted.
        """
        return map_unique_minibatch(inputs, batch_size, self._embed_chunk)
