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
    global _INITIALIZED
    if not _INITIALIZED:
        if endpoint and api_version:
            pandas_ext.use(AsyncAzureOpenAI(api_key=api_key, endpoint=endpoint, api_version=api_version))
        else:
            pandas_ext.use(AsyncOpenAI(api_key=api_key))
        _INITIALIZED = True


@dataclass(frozen=True)
class ResponsesUDFBuilder:
    # Params for OpenAI SDK
    api_key: str
    endpoint: Optional[str]
    api_version: Optional[str]

    # Params for Responses API
    model_name: str

    @classmethod
    def of_openai(cls, api_key: str, model_name: str) -> "ResponsesUDFBuilder":
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "ResponsesUDFBuilder":
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(self, instructions: str, response_format: Type[T] = str, batch_size: int = 256) -> UserDefinedFunction:
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
    # Params for OpenAI SDK
    api_key: str
    endpoint: Optional[str]
    api_version: Optional[str]

    # Params for Embeddings API
    model_name: str

    @classmethod
    def of_openai(cls, api_key: str, model_name: str) -> "EmbeddingsUDFBuilder":
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "EmbeddingsUDFBuilder":
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(self, batch_size: int) -> UserDefinedFunction:
        @pandas_udf(returnType=ArrayType(FloatType()))
        def embeddings_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            _initialize(self.api_key, self.endpoint, self.api_version)
            pandas_ext.embeddings_model(self.model_name)

            for part in col:
                embeddings: pd.Series = asyncio.run(part.aio.embeddings(batch_size=batch_size))
                yield embeddings.map(lambda x: x.tolist())

        return embeddings_udf
