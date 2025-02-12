import os
from dataclasses import dataclass
from logging import getLogger, Logger
from typing import Iterator, Optional

import httpx
import pandas as pd
from openai import OpenAI, AzureOpenAI
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType, ArrayType, FloatType

from openaivec import VectorizedOpenAI, EmbeddingOpenAI
from openaivec.log import observe
from openaivec.vectorize import VectorizedLLM

__ALL__ = ["UDFBuilder"]

_logger: Logger = getLogger(__name__)

# Global Singletons
_http_client: Optional[httpx.Client] = None
_openai_client: Optional[OpenAI] = None
_vectorized_client: Optional[VectorizedLLM] = None
_embedding_client: Optional[EmbeddingOpenAI] = None


def get_http_client(http2: bool, verify: bool) -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(http2=http2, verify=verify)
    return _http_client


def get_azureopenai_client(api_version: str, azure_endpoint: str, api_key: str) -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            http_client=get_http_client(http2=True, verify=False),
            api_key=api_key,
        )
    return _openai_client


def get_openai_client(api_key: str) -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=api_key,
            http_client=get_http_client(http2=True, verify=False),
        )
    return _openai_client


def get_vectorized_azureopenai_client(api_version: str, azure_endpoint: str, apy_key: str, model_name: str,
                          system_message: str) -> VectorizedLLM:
    global _vectorized_client
    if _vectorized_client is None:
        _vectorized_client = VectorizedOpenAI(
            client=get_azureopenai_client(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=apy_key,
            ),
            model_name=model_name,
            system_message=system_message,
            top_p=1.0,
            temperature=0.0,
        )
    return _vectorized_client


def get_vectorized_embedding_client(api_version: str, azure_endpoint: str, api_key: str,
                                    model_name: str) -> VectorizedLLM:
    global _vectorized_client
    if _vectorized_client is None:
        _vectorized_client = EmbeddingOpenAI(
            client=get_azureopenai_client(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            ),
            model_name=model_name,
        )
    return _vectorized_client


@dataclass(frozen=True)
class UDFBuilder:
    api_key: str
    api_version: str
    endpoint: str
    model_name: str
    batch_size: int = 256

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
    def completion(self, system_message: str):
        @pandas_udf(StringType())
        def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            import pandas as pd

            client_vec = get_vectorized_azureopenai_client(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                apy_key=self.api_key,
                model_name=self.model_name,
                system_message=system_message,
            )

            for part in col:
                yield pd.Series(client_vec.predict_minibatch(part.tolist(), self.batch_size))

        return fn

    @observe(_logger)
    def embedding(self):
        @pandas_udf(ArrayType(FloatType()))
        def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            import httpx
            from openai import AzureOpenAI

            from openaivec.embedding import EmbeddingOpenAI

            client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                http_client=httpx.Client(http2=True, verify=False),
                api_key=self.api_key,
            )

            client_emb = EmbeddingOpenAI(
                client=client,
                model_name=self.model_name,
            )

            for part in col:
                yield pd.Series(client_emb.embed_minibatch(part.tolist(), self.batch_size))

        return fn
