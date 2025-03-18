import time
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI, RateLimitError

from openaivec.log import observe
from openaivec.util import map_unique_minibatch_parallel

__all__ = ["EmbeddingOpenAI"]

_logger: Logger = getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingOpenAI:
    client: OpenAI
    model_name: str

    @observe(_logger)
    def embed(self, sentences: List[str]) -> List[NDArray[np.float32]]:
        while True:
            try:
                responses = self.client.embeddings.create(input=sentences, model=self.model_name)
                break
            except RateLimitError:
                _logger.info("429 RateLimit encountered; retrying in 60 seconds")
                time.sleep(60)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_logger)
    def embed_minibatch(self, sentences: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        return map_unique_minibatch_parallel(sentences, batch_size, self.embed)
