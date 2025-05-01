import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, call

import numpy as np
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding

from openaivec.aio.embeddings import AsyncBatchEmbeddings


# Helper function to create mock embedding responses
def _create_mock_embedding_response(input_texts: list[str]) -> CreateEmbeddingResponse:
    embeddings = [
        Embedding(
            embedding=[0.1] * 1536,  # Dummy embedding vector
            index=i,
            object="embedding",
        )
        for i, _ in enumerate(input_texts)
    ]
    return CreateEmbeddingResponse(
        data=embeddings,
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=len(input_texts) * 5, total_tokens=len(input_texts) * 5),  # Dummy usage
    )


class TestAsyncBatchEmbeddings(IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock the AsyncOpenAI client and its embeddings.create method
        self.mock_openai_client = AsyncMock()
        self.mock_openai_client.embeddings.create = AsyncMock()
        self.model_name = "text-embedding-3-small"
        self.embedding_dim = 1536  # Define expected dimension

    async def test_create_basic(self):
        """Test basic embedding creation with a small batch size."""
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "orange", "pineapple"]
        batch_size = 2

        # Configure the mock response
        self.mock_openai_client.embeddings.create.side_effect = [
            _create_mock_embedding_response(inputs[0:2]),
            _create_mock_embedding_response(inputs[2:4]),
        ]

        response = await client.create(inputs, batch_size=batch_size)

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))
            self.assertEqual(embedding.dtype, np.float32)

        # Check if the mock was called correctly
        expected_calls = [
            call(input=inputs[0:2], model=self.model_name),
            call(input=inputs[2:4], model=self.model_name),
        ]
        self.mock_openai_client.embeddings.create.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(self.mock_openai_client.embeddings.create.call_count, 2)

    async def test_create_empty_input(self):
        """Test embedding creation with an empty input list."""
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client,
            model_name=self.model_name,
        )
        inputs = []
        response = await client.create(inputs, batch_size=1)

        self.assertEqual(len(response), 0)
        self.mock_openai_client.embeddings.create.assert_not_called()

    async def test_create_with_duplicates(self):
        """Test embedding creation with duplicate inputs."""
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "apple", "orange", "banana"]
        unique_inputs = ["apple", "banana", "orange"]
        batch_size = 2

        # Mock response for unique inputs
        self.mock_openai_client.embeddings.create.side_effect = [
            _create_mock_embedding_response(unique_inputs[0:2]),  # apple, banana
            _create_mock_embedding_response(unique_inputs[2:3]),  # orange
        ]

        response = await client.create(inputs, batch_size=batch_size)

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))

        # Check embeddings for duplicates are the same object (due to map's caching)
        self.assertIs(response[0], response[2])  # 'apple'
        self.assertIs(response[1], response[4])  # 'banana'
        self.assertIsNot(response[0], response[1])  # 'apple' vs 'banana'
        self.assertIsNot(response[1], response[3])  # 'banana' vs 'orange'

        # Check API calls (only unique inputs should be sent)
        expected_calls = [
            call(input=unique_inputs[0:2], model=self.model_name),
            call(input=unique_inputs[2:3], model=self.model_name),
        ]
        self.mock_openai_client.embeddings.create.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(self.mock_openai_client.embeddings.create.call_count, 2)

    async def test_create_batch_size_larger_than_unique(self):
        """Test when batch_size is larger than the number of unique inputs."""
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "orange", "apple"]
        unique_inputs = ["apple", "banana", "orange"]
        batch_size = 5  # Larger than unique inputs (3)

        # Mock response for unique inputs in one batch
        self.mock_openai_client.embeddings.create.side_effect = [
            _create_mock_embedding_response(unique_inputs),
        ]

        response = await client.create(inputs, batch_size=batch_size)

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))

        # Check API calls (only one call with unique inputs)
        expected_calls = [
            call(input=unique_inputs, model=self.model_name),
        ]
        self.mock_openai_client.embeddings.create.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(self.mock_openai_client.embeddings.create.call_count, 1)

    async def test_create_batch_size_one(self):
        """Test embedding creation with batch_size = 1."""
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "orange"]
        batch_size = 1

        # Configure the mock response for individual calls
        self.mock_openai_client.embeddings.create.side_effect = [
            _create_mock_embedding_response([inputs[0]]),
            _create_mock_embedding_response([inputs[1]]),
            _create_mock_embedding_response([inputs[2]]),
        ]

        response = await client.create(inputs, batch_size=batch_size)

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))

        # Check if the mock was called correctly for each input
        expected_calls = [
            call(input=[inputs[0]], model=self.model_name),
            call(input=[inputs[1]], model=self.model_name),
            call(input=[inputs[2]], model=self.model_name),
        ]
        self.mock_openai_client.embeddings.create.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(self.mock_openai_client.embeddings.create.call_count, 3)

    def test_initialization_default_concurrency(self):
        """Test initialization uses default max_concurrency."""
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client,
            model_name=self.model_name,
        )
        self.assertEqual(client.max_concurrency, 8)  # Default value
        self.assertIsInstance(client._semaphore, asyncio.Semaphore)
        self.assertEqual(client._semaphore._value, 8)

    def test_initialization_custom_concurrency(self):
        """Test initialization with custom max_concurrency."""
        custom_concurrency = 4
        client = AsyncBatchEmbeddings(
            client=self.mock_openai_client, model_name=self.model_name, max_concurrency=custom_concurrency
        )
        self.assertEqual(client.max_concurrency, custom_concurrency)
        self.assertIsInstance(client._semaphore, asyncio.Semaphore)
        self.assertEqual(client._semaphore._value, custom_concurrency)
