"""Asynchronous Spark UDFs for the OpenAI and Azure OpenAI APIs.

This module provides builder classes for creating asynchronous Spark UDFs
that communicate with either the public OpenAI API or Azure OpenAI using
the `aio` subpackage. It supports UDFs for generating responses and
creating embeddings asynchronously. The UDFs operate on Spark DataFrames
and leverage asyncio for potentially improved performance in I/O-bound
operations.

Note: This module relies on the `openaivec.aio.pandas_ext` extension.
"""

import asyncio
from dataclasses import dataclass
from typing import Iterator, Optional, Type, TypeVar
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.udf import UserDefinedFunction
from pyspark.sql.types import StringType, ArrayType, FloatType
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openaivec.aio import pandas_ext
import pandas as pd
from pydantic import BaseModel

from openaivec.serialize import deserialize_base_model, serialize_base_model
from openaivec.spark import _pydantic_to_spark_schema, _safe_cast_str, _safe_dump

ResponseFormat = BaseModel | Type[str]
T = TypeVar("T", bound=BaseModel)

_INITIALIZED = False


def _initialize(api_key: str, endpoint: Optional[str], api_version: Optional[str]) -> None:
    """Initializes the OpenAI client for asynchronous operations.

    This function sets up the global asynchronous OpenAI client instance
    (either `AsyncOpenAI` or `AsyncAzureOpenAI`) used by the UDFs in this
    module. It ensures the client is initialized only once.

    Args:
        api_key: The OpenAI or Azure OpenAI API key.
        endpoint: The Azure OpenAI endpoint URL. Required for Azure.
        api_version: The Azure OpenAI API version. Required for Azure.
    """
    global _INITIALIZED
    if not _INITIALIZED:
        if endpoint and api_version:
            pandas_ext.use(AsyncAzureOpenAI(api_key=api_key, endpoint=endpoint, api_version=api_version))
        else:
            pandas_ext.use(AsyncOpenAI(api_key=api_key))
        _INITIALIZED = True


@dataclass(frozen=True)
class ResponsesUDFBuilder:
    """Builder for asynchronous Spark pandas UDFs for generating responses.

    Configures and builds UDFs that leverage `openaivec.aio.pandas_ext.responses`
    to generate text or structured responses from OpenAI models asynchronously.

    Attributes:
        api_key: OpenAI or Azure API key.
        endpoint: Azure endpoint base URL or None for public OpenAI.
        api_version: Azure API version, ignored for public OpenAI.
        model_name: Deployment (Azure) or model (OpenAI) name for responses.
    """

    # Params for OpenAI SDK
    api_key: str
    endpoint: Optional[str]
    api_version: Optional[str]

    # Params for Responses API
    model_name: str

    @classmethod
    def of_openai(cls, api_key: str, model_name: str) -> "ResponsesUDFBuilder":
        """Creates a builder configured for the public OpenAI API.

        Args:
            api_key: The OpenAI API key.
            model_name: The OpenAI model name (e.g., "gpt-4o").

        Returns:
            A ResponsesUDFBuilder instance configured for OpenAI.
        """
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "ResponsesUDFBuilder":
        """Creates a builder configured for Azure OpenAI.

        Args:
            api_key: The Azure OpenAI API key.
            endpoint: The Azure OpenAI endpoint URL.
            api_version: The Azure OpenAI API version.
            model_name: The Azure OpenAI deployment name.

        Returns:
            A ResponsesUDFBuilder instance configured for Azure OpenAI.
        """
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(self, instructions: str, response_format: Type[T] = str, batch_size: int = 256) -> UserDefinedFunction:
        """Builds the asynchronous pandas UDF for generating responses.

        Args:
            instructions: The system prompt or instructions for the model.
            response_format: The desired output format. Either `str` for plain text
                or a Pydantic `BaseModel` for structured JSON output.
            batch_size: The number of rows to process in each asynchronous batch
                passed to the underlying pandas extension.

        Returns:
            A Spark pandas UDF configured to generate responses asynchronously.

        Raises:
            ValueError: If `response_format` is not `str` or a Pydantic `BaseModel`.
        """
        if issubclass(response_format, BaseModel):
            spark_schema = _pydantic_to_spark_schema(response_format)
            json_schema_string = serialize_base_model(response_format)

            @pandas_udf(returnType=spark_schema)
            def structure_udf(col: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
                _initialize(self.api_key, self.endpoint, self.api_version)
                pandas_ext.responses_model(self.model_name)

                for part in col:
                    predictions: pd.Series = asyncio.run(
                        part.aio.responses(
                            instructions=instructions,
                            response_format=deserialize_base_model(json_schema_string),
                            batch_size=batch_size,
                        )
                    )
                    yield pd.DataFrame(predictions.map(_safe_dump).tolist())

            return structure_udf

        elif issubclass(response_format, str):

            @pandas_udf(returnType=StringType())
            def string_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
                _initialize(self.api_key, self.endpoint, self.api_version)
                pandas_ext.responses_model(self.model_name)

                for part in col:
                    predictions: pd.Series = asyncio.run(
                        part.aio.responses(
                            instructions=instructions,
                            response_format=str,
                            batch_size=batch_size,
                        )
                    )
                    yield predictions.map(_safe_cast_str)

            return string_udf

        else:
            raise ValueError(f"Unsupported response_format: {response_format}")


@dataclass(frozen=True)
class EmbeddingsUDFBuilder:
    """Builder for asynchronous Spark pandas UDFs for creating embeddings.

    Configures and builds UDFs that leverage `openaivec.aio.pandas_ext.embeddings`
    to generate vector embeddings from OpenAI models asynchronously.

    Attributes:
        api_key: OpenAI or Azure API key.
        endpoint: Azure endpoint base URL or None for public OpenAI.
        api_version: Azure API version, ignored for public OpenAI.
        model_name: Deployment (Azure) or model (OpenAI) name for embeddings.
    """

    # Params for OpenAI SDK
    api_key: str
    endpoint: Optional[str]
    api_version: Optional[str]

    # Params for Embeddings API
    model_name: str

    @classmethod
    def of_openai(cls, api_key: str, model_name: str) -> "EmbeddingsUDFBuilder":
        """Creates a builder configured for the public OpenAI API.

        Args:
            api_key: The OpenAI API key.
            model_name: The OpenAI model name (e.g., "text-embedding-3-small").

        Returns:
            An EmbeddingsUDFBuilder instance configured for OpenAI.
        """
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "EmbeddingsUDFBuilder":
        """Creates a builder configured for Azure OpenAI.

        Args:
            api_key: The Azure OpenAI API key.
            endpoint: The Azure OpenAI endpoint URL.
            api_version: The Azure OpenAI API version.
            model_name: The Azure OpenAI deployment name for embeddings.

        Returns:
            An EmbeddingsUDFBuilder instance configured for Azure OpenAI.
        """
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(self, batch_size: int = 256) -> UserDefinedFunction:
        """Builds the asynchronous pandas UDF for generating embeddings.

        Args:
            batch_size: The number of rows to process in each asynchronous batch
                passed to the underlying pandas extension. Defaults to 256.

        Returns:
            A Spark pandas UDF configured to generate embeddings asynchronously,
            returning an ArrayType(FloatType()).
        """

        @pandas_udf(returnType=ArrayType(FloatType()))
        def embeddings_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            _initialize(self.api_key, self.endpoint, self.api_version)
            pandas_ext.embeddings_model(self.model_name)

            for part in col:
                embeddings: pd.Series = asyncio.run(part.aio.embeddings(batch_size=batch_size))
                yield embeddings.map(lambda x: x.tolist())

        return embeddings_udf
