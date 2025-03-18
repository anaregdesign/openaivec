from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI, RateLimitError

from openaivec.log import observe
from openaivec.util import backoff, map_unique_minibatch_parallel

__all__ = ["EmbeddingOpenAI"]

_logger: Logger = getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingOpenAI:
    client: OpenAI
    model_name: str

    @observe(_logger)
    @backoff(exception=RateLimitError, interval=60, max_retries=32)
    def embed(self, sentences: List[str]) -> List[NDArray[np.float32]]:
        responses = self.client.embeddings.create(input=sentences, model=self.model_name)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_logger)
    def embed_minibatch(self, sentences: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        return map_unique_minibatch_parallel(sentences, batch_size, self.embed)
