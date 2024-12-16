from dataclasses import dataclass
from typing import List

from openai import OpenAI

from openaivec.util import map_with_minibatch


@dataclass(frozen=True)
class EmbeddingOpenAI:
    client: OpenAI
    model_name: str

    def embed(self, sentences: List[str]) -> List[List[float]]:
        responses = self.client.embeddings.create(input=sentences, model=self.model_name)
        return [d["embedding"] for d in responses.data]

    def embed_minibatch(self, sentences: List[str], batch_size: int) -> List[List[float]]:
        return map_with_minibatch(sentences, batch_size, self.embed)
