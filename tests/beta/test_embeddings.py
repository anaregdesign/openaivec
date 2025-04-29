from unittest import TestCase
from openai import AsyncOpenAI

from openaivec.beta.embeddings import VectorizedEmbeddingsOpenAI


class TestEmbeddingsOpenAI(TestCase):
    def setUp(self):
        self.openai_client = AsyncOpenAI()
        self.model_name = "text-embedding-3-small"

    def test_create(self):
        client = VectorizedEmbeddingsOpenAI(
            client=self.openai_client,
            model_name=self.model_name,
            is_parallel=False,
        )
        response = client.create(["apple", "banana", "orange", "pineapple"], batch_size=1)

        self.assertEqual(len(response), 4)
        for embedding in response:
            self.assertEqual(embedding.shape, (1536,))  # Assuming the embedding size is 1536
