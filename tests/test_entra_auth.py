import unittest
from unittest.mock import MagicMock, patch

from openaivec import pandas_ext
from openaivec.spark import ResponsesUDFBuilder, EmbeddingsUDFBuilder


class TestEntraAuthentication(unittest.TestCase):
    """Test Entra ID authentication functionality."""

    def setUp(self):
        # Reset global clients before each test
        pandas_ext._CLIENT = None
        pandas_ext._ASYNC_CLIENT = None

    def test_use_azure_openai_entra_import_error(self):
        """Test that ImportError is raised when azure-identity is not available."""
        with patch('openaivec.pandas_ext._AZURE_IDENTITY_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                pandas_ext.use_azure_openai_entra(
                    endpoint="https://test.openai.azure.com",
                    api_version="2024-02-01"
                )
            self.assertIn("azure-identity package is required", str(context.exception))

    def test_use_azure_openai_service_principal_import_error(self):
        """Test that ImportError is raised when azure-identity is not available for Service Principal auth."""
        with patch('openaivec.pandas_ext._AZURE_IDENTITY_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                pandas_ext.use_azure_openai_service_principal(
                    endpoint="https://test.openai.azure.com",
                    api_version="2024-02-01",
                    client_id="test-client-id",
                    client_secret="test-client-secret",
                    tenant_id="test-tenant-id"
                )
            self.assertIn("azure-identity package is required", str(context.exception))

    @patch('openaivec.pandas_ext._AZURE_IDENTITY_AVAILABLE', True)
    @patch('openaivec.pandas_ext.DefaultAzureCredential')
    @patch('openaivec.pandas_ext.AzureOpenAI')
    @patch('openaivec.pandas_ext.AsyncAzureOpenAI')
    def test_use_azure_openai_entra_success(self, mock_async_azure, mock_azure, mock_credential_class):
        """Test successful Entra ID authentication setup."""
        # Mock the credential
        mock_credential = MagicMock()
        mock_token = MagicMock()
        mock_token.token = "test_token"
        mock_credential.get_token.return_value = mock_token
        mock_credential_class.return_value = mock_credential

        # Mock the clients
        mock_client = MagicMock()
        mock_async_client = MagicMock()
        mock_azure.return_value = mock_client
        mock_async_azure.return_value = mock_async_client

        # Call the function
        pandas_ext.use_azure_openai_entra(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01"
        )

        # Verify credential was created
        mock_credential_class.assert_called_once()

        # Verify clients were created with correct parameters
        self.assertTrue(mock_azure.called)
        self.assertTrue(mock_async_azure.called)

        # Check that the clients are set globally
        self.assertEqual(pandas_ext._CLIENT, mock_client)
        self.assertEqual(pandas_ext._ASYNC_CLIENT, mock_async_client)

    @patch('openaivec.pandas_ext._AZURE_IDENTITY_AVAILABLE', True)
    @patch('openaivec.pandas_ext.ClientSecretCredential')
    @patch('openaivec.pandas_ext.AzureOpenAI')
    @patch('openaivec.pandas_ext.AsyncAzureOpenAI')
    def test_use_azure_openai_service_principal_success(self, mock_async_azure, mock_azure, mock_credential_class):
        """Test successful Service Principal authentication setup."""
        # Mock the credential
        mock_credential = MagicMock()
        mock_token = MagicMock()
        mock_token.token = "test_token"
        mock_credential.get_token.return_value = mock_token
        mock_credential_class.return_value = mock_credential

        # Mock the clients
        mock_client = MagicMock()
        mock_async_client = MagicMock()
        mock_azure.return_value = mock_client
        mock_async_azure.return_value = mock_async_client

        # Call the function
        pandas_ext.use_azure_openai_service_principal(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id"
        )

        # Verify credential was created with correct parameters
        mock_credential_class.assert_called_once_with(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret"
        )

        # Verify clients were created with correct parameters
        self.assertTrue(mock_azure.called)
        self.assertTrue(mock_async_azure.called)

        # Check that the clients are set globally
        self.assertEqual(pandas_ext._CLIENT, mock_client)
        self.assertEqual(pandas_ext._ASYNC_CLIENT, mock_async_client)

    def test_responses_udf_builder_of_azure_openai_entra(self):
        """Test ResponsesUDFBuilder.of_azure_openai_entra creates correct instance."""
        builder = ResponsesUDFBuilder.of_azure_openai_entra(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="gpt-4"
        )

        self.assertIsNone(builder.api_key)
        self.assertEqual(builder.endpoint, "https://test.openai.azure.com")
        self.assertEqual(builder.api_version, "2024-02-01")
        self.assertEqual(builder.model_name, "gpt-4")
        self.assertTrue(builder.use_entra_id)
        # Service Principal fields should be None for Entra ID auth
        self.assertIsNone(builder.client_id)
        self.assertIsNone(builder.client_secret)
        self.assertIsNone(builder.tenant_id)

    def test_responses_udf_builder_of_azure_openai_service_principal(self):
        """Test ResponsesUDFBuilder.of_azure_openai_service_principal creates correct instance."""
        builder = ResponsesUDFBuilder.of_azure_openai_service_principal(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="gpt-4",
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id"
        )

        self.assertIsNone(builder.api_key)
        self.assertEqual(builder.endpoint, "https://test.openai.azure.com")
        self.assertEqual(builder.api_version, "2024-02-01")
        self.assertEqual(builder.model_name, "gpt-4")
        self.assertTrue(builder.use_entra_id)
        # Service Principal fields should be set
        self.assertEqual(builder.client_id, "test-client-id")
        self.assertEqual(builder.client_secret, "test-client-secret")
        self.assertEqual(builder.tenant_id, "test-tenant-id")

    def test_embeddings_udf_builder_of_azure_openai_entra(self):
        """Test EmbeddingsUDFBuilder.of_azure_openai_entra creates correct instance."""
        builder = EmbeddingsUDFBuilder.of_azure_openai_entra(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="text-embedding-ada-002"
        )

        self.assertIsNone(builder.api_key)
        self.assertEqual(builder.endpoint, "https://test.openai.azure.com")
        self.assertEqual(builder.api_version, "2024-02-01")
        self.assertEqual(builder.model_name, "text-embedding-ada-002")
        self.assertTrue(builder.use_entra_id)
        # Service Principal fields should be None for Entra ID auth
        self.assertIsNone(builder.client_id)
        self.assertIsNone(builder.client_secret)
        self.assertIsNone(builder.tenant_id)

    def test_embeddings_udf_builder_of_azure_openai_service_principal(self):
        """Test EmbeddingsUDFBuilder.of_azure_openai_service_principal creates correct instance."""
        builder = EmbeddingsUDFBuilder.of_azure_openai_service_principal(
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="text-embedding-ada-002",
            client_id="test-client-id",
            client_secret="test-client-secret",
            tenant_id="test-tenant-id"
        )

        self.assertIsNone(builder.api_key)
        self.assertEqual(builder.endpoint, "https://test.openai.azure.com")
        self.assertEqual(builder.api_version, "2024-02-01")
        self.assertEqual(builder.model_name, "text-embedding-ada-002")
        self.assertTrue(builder.use_entra_id)
        # Service Principal fields should be set
        self.assertEqual(builder.client_id, "test-client-id")
        self.assertEqual(builder.client_secret, "test-client-secret")
        self.assertEqual(builder.tenant_id, "test-tenant-id")

    def test_existing_builders_still_work(self):
        """Test that existing builder methods still work with API keys."""
        resp_builder = ResponsesUDFBuilder.of_azure_openai(
            api_key="test_key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="gpt-4"
        )

        self.assertEqual(resp_builder.api_key, "test_key")
        self.assertFalse(resp_builder.use_entra_id)

        emb_builder = EmbeddingsUDFBuilder.of_azure_openai(
            api_key="test_key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="text-embedding-ada-002"
        )

        self.assertEqual(emb_builder.api_key, "test_key")
        self.assertFalse(emb_builder.use_entra_id)

    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
        'AZURE_OPENAI_API_VERSION': '2024-02-01',
        'AZURE_OPENAI_USE_ENTRA_ID': 'true'
    })
    @patch('openaivec.pandas_ext.use_azure_openai_entra')
    def test_environment_variable_entra_id_detection(self, mock_use_entra):
        """Test that environment variables trigger Entra ID authentication."""
        # This should trigger the Entra ID path
        try:
            pandas_ext._get_openai_client()
        except Exception:
            # Expected to fail due to mocking, but should call use_azure_openai_entra
            pass

        mock_use_entra.assert_called_once_with(
            endpoint='https://test.openai.azure.com',
            api_version='2024-02-01'
        )

    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
        'AZURE_OPENAI_API_VERSION': '2024-02-01',
        'AZURE_CLIENT_ID': 'test-client-id',
        'AZURE_CLIENT_SECRET': 'test-client-secret',
        'AZURE_TENANT_ID': 'test-tenant-id'
    })
    @patch('openaivec.pandas_ext.use_azure_openai_service_principal')
    def test_environment_variable_service_principal_detection(self, mock_use_sp):
        """Test that Service Principal environment variables trigger Service Principal authentication."""
        # This should trigger the Service Principal path
        try:
            pandas_ext._get_openai_client()
        except Exception:
            # Expected to fail due to mocking, but should call use_azure_openai_service_principal
            pass

        mock_use_sp.assert_called_once_with(
            endpoint='https://test.openai.azure.com',
            api_version='2024-02-01',
            client_id='test-client-id',
            client_secret='test-client-secret',
            tenant_id='test-tenant-id'
        )


if __name__ == '__main__':
    unittest.main()