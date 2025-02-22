import os
from dataclasses import dataclass
from logging import getLogger, Logger
from typing import Iterator, Optional, TypeVar, Type, Any

import httpx
import pandas as pd
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType, ArrayType, FloatType

from openaivec import VectorizedOpenAI, EmbeddingOpenAI
from openaivec.log import observe
from openaivec.util import serialize_base_model, deserialize_base_model
from openaivec.vectorize import VectorizedLLM

__ALL__ = ["UDFBuilder"]

_logger: Logger = getLogger(__name__)

# Global Singletons
_openai_client: Optional[OpenAI] = None
_vectorized_client: Optional[VectorizedLLM] = None
_embedding_client: Optional[EmbeddingOpenAI] = None


T = TypeVar("T")


def get_openai_client(conf: "UDFBuilder", http_client: httpx.Client) -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if conf.endpoint is None:
            _openai_client = OpenAI(
                api_key=conf.api_key,
                http_client=http_client,
            )
        else:
            _openai_client = AzureOpenAI(
                api_key=conf.api_key,
                api_version=conf.api_version,
                azure_endpoint=conf.endpoint,
                http_client=http_client,
            )
    return _openai_client


def get_vectorized_openai_client(
    conf: "UDFBuilder", system_message: str, response_format: Type[T], http_client: httpx.Client
) -> VectorizedLLM:
    global _vectorized_client
    if _vectorized_client is None:
        _vectorized_client = VectorizedOpenAI(
            client=get_openai_client(conf, http_client),
            model_name=conf.model_name,
            system_message=system_message,
            temperature=conf.temperature,
            top_p=conf.top_p,
            response_format=response_format,
        )
    return _vectorized_client


def get_vectorized_embedding_client(conf: "UDFBuilder", http_client: httpx.Client) -> EmbeddingOpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingOpenAI(
            client=get_openai_client(conf, http_client),
            model_name=conf.model_name,
        )
    return _embedding_client


@dataclass(frozen=True)
class UDFBuilder:
    # Params for Constructor
    api_key: str
    endpoint: str
    api_version: str

    # Params for chat_completion
    model_name: str  # it would be the name of deployment for Azure
    temperature: float = 0.0
    top_p: float = 1.0

    # Params for minibatch
    batch_size: int = 256

    # Params for httpx.Client
    http2: bool = True
    ssl_verify: bool = False

    @classmethod
    def of_environment(cls, batch_size: int = 256) -> "UDFBuilder":
        return cls(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            model_name=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
            batch_size=batch_size,
        )

    def __post_init__(self):
        assert self.api_key, "api_key must be set"
        assert self.api_version, "api_version must be set"
        assert self.endpoint, "endpoint must be set"
        assert self.model_name, "model_name must be set"

    @observe(_logger)
    def completion(self, system_message: str, response_format: Type[T] = str):
        format_source = None
        format_class_name = None

        if issubclass(response_format, BaseModel):
            format_source = serialize_base_model(response_format)
            format_class_name = response_format.__name__

        @pandas_udf(StringType())
        def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            cls = str
            if format_source is not None:
                cls = deserialize_base_model(format_source, format_class_name)

            http_client = httpx.Client(http2=self.http2, verify=self.ssl_verify)
            client_vec = get_vectorized_openai_client(
                conf=self,
                system_message=system_message,
                response_format=cls,
                http_client=http_client,
            )

            def _cast_to_string(x: Any) -> Optional[str]:
                match x:
                    case None:
                        return None

                    case str():
                        return x

                    case BaseModel():
                        return x.model_dump_json()

                    case _:
                        try:
                            return str(x)

                        except Exception as e:
                            _logger.warning(f"Failed to cast {x} to string: {e}")
                            return None

            for part in col:
                yield (pd.Series(client_vec.predict_minibatch(part.tolist(), self.batch_size)).map(_cast_to_string))

        return fn

    @observe(_logger)
    def embedding(self):
        @pandas_udf(ArrayType(FloatType()))
        def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            http_client = httpx.Client(http2=self.http2, verify=self.ssl_verify)
            client_emb = get_vectorized_embedding_client(self, http_client)

            for part in col:
                yield pd.Series(client_emb.embed_minibatch(part.tolist(), self.batch_size))

        return fn
