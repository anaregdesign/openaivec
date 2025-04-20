from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI, RateLimitError

from openaivec.log import observe
from openaivec.util import backoff, map_unique_minibatch, map_unique_minibatch_parallel

__all__ = ["EmbeddingOpenAI"]

_logger: Logger = getLogger(__name__)


class EmbeddingLLM(metaclass=ABCMeta):
    @abstractmethod
    def embed(self, sentences: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        pass


@dataclass(frozen=True)
class EmbeddingOpenAI(EmbeddingLLM):
    client: OpenAI
    model_name: str
    is_parallel: bool = False

    @observe(_logger)
    @backoff(exception=RateLimitError, scale=60, max_retries=16)
    def _embed_chunk(self, sentences: List[str]) -> List[NDArray[np.float32]]:
        responses = self.client.embeddings.create(input=sentences, model=self.model_name)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_logger)
    def embed(self, sentences: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        if self.is_parallel:
            return map_unique_minibatch_parallel(sentences, batch_size, self._embed_chunk)
        else:
            return map_unique_minibatch(sentences, batch_size, self._embed_chunk)
