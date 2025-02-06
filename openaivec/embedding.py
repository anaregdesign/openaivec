from dataclasses import dataclass
from logging import getLogger, Logger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from openaivec.log import observe
from openaivec.util import map_unique_minibatch_parallel

__ALL__ = ["EmbeddingOpenAI"]

_logger: Logger = getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingOpenAI:
    """Class for embedding sentences using the OpenAI API.

    Attributes:
        client (OpenAI): The OpenAI API client.
        model_name (str): The name of the model to use for embeddings.
    """

    client: OpenAI
    model_name: str

    @observe(_logger)
    def embed(self, sentences: List[str]) -> List[NDArray[np.float32]]:
        """Embeds sentences using the OpenAI API.

        Args:
            sentences (List[str]): A list of sentences to embed.

        Returns:
            List[NDArray[np.float32]]: A list of embeddings as numpy arrays.
        """
        responses = self.client.embeddings.create(input=sentences, model=self.model_name)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_logger)
    def embed_minibatch(self, sentences: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        """Embeds sentences in minibatches in parallel.

        Args:
            sentences (List[str]): A list of sentences to embed.
            batch_size (int): The number of sentences per minibatch.

        Returns:
            List[NDArray[np.float32]]: A list of embeddings as numpy arrays.
        """
        return map_unique_minibatch_parallel(sentences, batch_size, self.embed)
