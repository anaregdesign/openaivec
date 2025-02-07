import unittest
from unittest.mock import MagicMock
import numpy as np
from openai import OpenAI
from openaivec.embedding import EmbeddingOpenAI

class TestEmbeddingOpenAI(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=OpenAI)
        self.model_name = "test-model"
        self.embedding = EmbeddingOpenAI(client=self.mock_client, model_name=self.model_name)

    def test_embed(self):
        sentences = ["Hello world", "Test sentence"]
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6])
        ]
        self.mock_client.embeddings.create.return_value = mock_response

        result = self.embedding.embed(sentences)
        expected = [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32)]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r, e)

    def test_embed_minibatch(self):
        sentences = ["Hello world", "Test sentence", "Another sentence"]
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
            MagicMock(embedding=[0.7, 0.8, 0.9])
        ]
        self.mock_client.embeddings.create.return_value = mock_response

        result = self.embedding.embed_minibatch(sentences, batch_size=2)
        expected = [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32), np.array([0.7, 0.8, 0.9], dtype=np.float32)]

        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            np.testing.assert_array_equal(r, e)

if __name__ == "__main__":
    unittest.main()
