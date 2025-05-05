"""Asynchronous Spark UDFs for the OpenAI and Azure OpenAI APIs.

This module provides builder classes (`ResponsesUDFBuilder`, `EmbeddingsUDFBuilder`)
for creating asynchronous Spark UDFs that communicate with either the public
OpenAI API or Azure OpenAI using the `openaivec.aio` subpackage.
It supports UDFs for generating responses and creating embeddings asynchronously.
The UDFs operate on Spark DataFrames and leverage asyncio for potentially
improved performance in I/O-bound operations.

## Setup

First, obtain a Spark session:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

Next, instantiate UDF builders with your OpenAI API key (or Azure credentials)
and model/deployment names, then register the desired UDFs:

```python
import os
from openaivec.aio.spark import ResponsesUDFBuilder, EmbeddingsUDFBuilder
from pydantic import BaseModel

# Option 1: Using OpenAI
resp_builder = ResponsesUDFBuilder.of_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini", # Model for responses
)
emb_builder = EmbeddingsUDFBuilder.of_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small", # Model for embeddings
)

# Option 2: Using Azure OpenAI
# resp_builder = ResponsesUDFBuilder.of_azure_openai(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     model_name="your-resp-deployment-name", # Deployment for responses
# )
# emb_builder = EmbeddingsUDFBuilder.of_azure_openai(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     model_name="your-emb-deployment-name", # Deployment for embeddings
# )

# Define a Pydantic model for structured responses (optional)
class Translation(BaseModel):
    en: str
    fr: str
    # ... other languages

# Register the asynchronous responses UDF
spark.udf.register(
    "translate_async",
    resp_builder.build(
        instructions="Translate the text to multiple languages.",
        response_format=Translation,
    ),
)

# Register the asynchronous embeddings UDF
spark.udf.register(
    "embed_async",
    emb_builder.build(),
)
```

You can now invoke the UDFs from Spark SQL:

```sql
SELECT
    text,
    translate_async(text) AS translation,
    embed_async(text) AS embedding
FROM your_table;
```

Note: This module relies on the `openaivec.aio.pandas_ext` extension for its core asynchronous logic.
"""

import asyncio
from dataclasses import dataclass
from typing import Iterator, Optional, Type, TypeVar
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.udf import UserDefinedFunction
from pyspark.sql.types import StringType, ArrayType, FloatType
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openaivec import pandas_ext
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
            pandas_ext.use(AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version))
        else:
            pandas_ext.use(AsyncOpenAI(api_key=api_key))
        _INITIALIZED = True


@dataclass(frozen=True)
class ResponsesUDFBuilder:
    """Builder for asynchronous Spark pandas UDFs for generating responses.

    Configures and builds UDFs that leverage `openaivec.aio.pandas_ext.responses`
    to generate text or structured responses from OpenAI models asynchronously.
    An instance stores authentication parameters and the model name.

    Attributes:
        api_key (str): OpenAI or Azure API key.
        endpoint (Optional[str]): Azure endpoint base URL. None for public OpenAI.
        api_version (Optional[str]): Azure API version. Ignored for public OpenAI.
        model_name (str): Deployment name (Azure) or model name (OpenAI) for responses.
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
            api_key (str): The OpenAI API key.
            model_name (str): The OpenAI model name for responses (e.g., "gpt-4o-mini").

        Returns:
            ResponsesUDFBuilder: A builder instance configured for OpenAI responses.
        """
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "ResponsesUDFBuilder":
        """Creates a builder configured for Azure OpenAI.

        Args:
            api_key (str): The Azure OpenAI API key.
            endpoint (str): The Azure OpenAI endpoint URL.
            api_version (str): The Azure OpenAI API version (e.g., "2024-02-01").
            model_name (str): The Azure OpenAI deployment name for responses.

        Returns:
            ResponsesUDFBuilder: A builder instance configured for Azure OpenAI responses.
        """
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,  # Default batch size for async might differ
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> UserDefinedFunction:
        """Builds the asynchronous pandas UDF for generating responses.

        Args:
            instructions (str): The system prompt or instructions for the model.
            response_format (Type[T]): The desired output format. Either `str` for plain text
                or a Pydantic `BaseModel` for structured JSON output. Defaults to `str`.
            batch_size (int): Number of rows per async batch request passed to the underlying
                `pandas_ext` function. Defaults to 128.
            temperature (float): Sampling temperature (0.0 to 2.0). Defaults to 0.0.
            top_p (float): Nucleus sampling parameter. Defaults to 1.0.

        Returns:
            UserDefinedFunction: A Spark pandas UDF configured to generate responses asynchronously.
                Output schema is `StringType` or a struct derived from `response_format`.

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
                            temperature=temperature,
                            top_p=top_p,
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
                            temperature=temperature,
                            top_p=top_p,
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
    An instance stores authentication parameters and the model name.

    Attributes:
        api_key (str): OpenAI or Azure API key.
        endpoint (Optional[str]): Azure endpoint base URL. None for public OpenAI.
        api_version (Optional[str]): Azure API version. Ignored for public OpenAI.
        model_name (str): Deployment name (Azure) or model name (OpenAI) for embeddings.
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
            api_key (str): The OpenAI API key.
            model_name (str): The OpenAI model name for embeddings (e.g., "text-embedding-3-small").

        Returns:
            EmbeddingsUDFBuilder: A builder instance configured for OpenAI embeddings.
        """
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "EmbeddingsUDFBuilder":
        """Creates a builder configured for Azure OpenAI.

        Args:
            api_key (str): The Azure OpenAI API key.
            endpoint (str): The Azure OpenAI endpoint URL.
            api_version (str): The Azure OpenAI API version (e.g., "2024-02-01").
            model_name (str): The Azure OpenAI deployment name for embeddings.

        Returns:
            EmbeddingsUDFBuilder: A builder instance configured for Azure OpenAI embeddings.
        """
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(self, batch_size: int = 128) -> UserDefinedFunction:  # Default batch size for async might differ
        """Builds the asynchronous pandas UDF for generating embeddings.

        Args:
            batch_size (int): Number of rows per async batch request passed to the underlying
                `pandas_ext` function. Defaults to 128.

        Returns:
            UserDefinedFunction: A Spark pandas UDF configured to generate embeddings asynchronously,
                returning an `ArrayType(FloatType())` column.
        """

        @pandas_udf(returnType=ArrayType(FloatType()))
        def embeddings_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            _initialize(self.api_key, self.endpoint, self.api_version)
            pandas_ext.embeddings_model(self.model_name)

            for part in col:
                embeddings: pd.Series = asyncio.run(part.aio.embeddings(batch_size=batch_size))
                yield embeddings.map(lambda x: x.tolist())

        return embeddings_udf
