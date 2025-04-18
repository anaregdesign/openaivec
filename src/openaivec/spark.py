from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Iterator, List, Optional, Type, TypeVar, Union, get_args, get_origin

import httpx
import pandas as pd
import tiktoken
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import ArrayType, BooleanType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec import EmbeddingOpenAI, VectorizedOpenAI
from openaivec.log import observe
from openaivec.serialize import deserialize_base_model, serialize_base_model
from openaivec.util import TextChunker
from openaivec.vectorize import VectorizedLLM

__all__ = [
    "UDFBuilder",
    "count_tokens_udf",
]

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
    conf: "UDFBuilder",
    system_message: str,
    response_format: Type[T],
    temperature: float,
    top_p: float,
    http_client: httpx.Client,
) -> VectorizedLLM:
    global _vectorized_client
    if _vectorized_client is None:
        _vectorized_client = VectorizedOpenAI(
            client=get_openai_client(conf, http_client),
            model_name=conf.model_name,
            system_message=system_message,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            is_parallel=conf.is_parallel,
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


def _safe_dump(x: BaseModel) -> Optional[dict]:
    try:
        return x.model_dump()
    except Exception as e:
        _logger.error(f"Error during model_dump: {e}")
        return None


def _safe_cast_str(x: str) -> Optional[str]:
    try:
        return str(x)
    except Exception as e:
        _logger.error(f"Error during casting to str: {e}")
        return None


def python_type_to_spark(python_type):
    origin = get_origin(python_type)

    # For list types (e.g., List[int])
    if origin is list or origin is List:
        # Retrieve the inner type and recursively convert it
        inner_type = get_args(python_type)[0]
        return ArrayType(python_type_to_spark(inner_type))

    # For Optional types (Union[..., None])
    elif origin is Union:
        non_none_args = [arg for arg in get_args(python_type) if arg is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_spark(non_none_args[0])
        else:
            raise ValueError(f"Unsupported Union type with multiple non-None types: {python_type}")

    # For nested Pydantic models (to be treated as Structs)
    elif isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return pydantic_to_spark_schema(python_type)

    # Basic type mapping
    elif python_type is int:
        return IntegerType()
    elif python_type is float:
        return FloatType()
    elif python_type is str:
        return StringType()
    elif python_type is bool:
        return BooleanType()
    else:
        raise ValueError(f"Unsupported type: {python_type}")


def pydantic_to_spark_schema(model: Type[BaseModel]) -> StructType:
    fields = []
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        # Use outer_type_ to correctly handle types like Optional
        spark_type = python_type_to_spark(field_type)
        # Set nullable to True (adjust logic as needed)
        fields.append(StructField(field_name, spark_type, nullable=True))
    return StructType(fields)


@dataclass(frozen=True)
class UDFBuilder:
    # Params for Constructor
    api_key: str
    endpoint: Optional[str]
    api_version: Optional[str]

    # Params for chat_completion
    model_name: str  # it would be the name of deployment for Azure

    # Params for minibatch
    batch_size: int = 256
    is_parallel: bool = False

    # Params for httpx.Client
    http2: bool = True
    ssl_verify: bool = False

    @classmethod
    def of_azureopenai(
        cls,
        api_key: str,
        api_version: str,
        endpoint: str,
        model_name: str,
        batch_size: int = 256,
        http2: bool = True,
        ssl_verify: bool = False,
        is_parallel: bool = False,
    ) -> "UDFBuilder":
        return cls(
            api_key=api_key,
            api_version=api_version,
            endpoint=endpoint,
            model_name=model_name,
            batch_size=batch_size,
            http2=http2,
            ssl_verify=ssl_verify,
            is_parallel=is_parallel,
        )

    @classmethod
    def of_openai(
        cls,
        api_key: str,
        model_name: str,
        batch_size: int = 256,
        http2: bool = True,
        ssl_verify: bool = False,
        is_parallel: bool = False,
    ) -> "UDFBuilder":
        return cls(
            api_key=api_key,
            api_version=None,
            endpoint=None,
            model_name=model_name,
            batch_size=batch_size,
            http2=http2,
            ssl_verify=ssl_verify,
            is_parallel=is_parallel,
        )

    def __post_init__(self):
        assert self.api_key, "api_key must be set"
        assert self.model_name, "model_name must be set"

    @observe(_logger)
    def completion(
        self, system_message: str, response_format: Type[T] = str, temperature: float = 0.0, top_p: float = 1.0
    ):
        if issubclass(response_format, BaseModel):
            spark_schema = pydantic_to_spark_schema(response_format)
            json_schema_string = serialize_base_model(response_format)

        elif issubclass(response_format, str):
            spark_schema = StringType()
            json_schema_string = None

        else:
            raise ValueError(f"Unsupported response_format: {response_format}")

        @pandas_udf(spark_schema)
        def fn_struct(col: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            cls = str
            if json_schema_string:
                cls = deserialize_base_model(json_schema_string)

            http_client = httpx.Client(http2=self.http2, verify=self.ssl_verify)
            client_vec = get_vectorized_openai_client(
                conf=self,
                system_message=system_message,
                response_format=cls,
                temperature=temperature,
                top_p=top_p,
                http_client=http_client,
            )

            for part in col:
                predictions = client_vec.predict_minibatch(part.tolist(), self.batch_size)
                result = pd.Series(predictions)
                yield pd.DataFrame(result.map(_safe_dump).tolist())

        @pandas_udf(spark_schema)
        def fn_str(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            http_client = httpx.Client(http2=self.http2, verify=self.ssl_verify)
            client_vec = get_vectorized_openai_client(
                conf=self,
                system_message=system_message,
                response_format=str,
                temperature=temperature,
                top_p=top_p,
                http_client=http_client,
            )

            for part in col:
                predictions = client_vec.predict_minibatch(part.tolist(), self.batch_size)
                result = pd.Series(predictions)
                yield result.map(_safe_cast_str)

        if issubclass(response_format, str):
            return fn_str

        else:
            return fn_struct

    @observe(_logger)
    def embedding(self):
        @pandas_udf(ArrayType(FloatType()))
        def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            http_client = httpx.Client(http2=self.http2, verify=self.ssl_verify)
            client_emb = get_vectorized_embedding_client(self, http_client)

            for part in col:
                yield pd.Series(client_emb.embed_minibatch(part.tolist(), self.batch_size))

        return fn


# singleton for tiktoken
_tiktoken_enc: Optional[tiktoken.Encoding] = None


def count_tokens_udf(model_name: str = "gpt-4o"):
    @pandas_udf(IntegerType())
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        global _tiktoken_enc
        if _tiktoken_enc is None:
            _tiktoken_enc = tiktoken.encoding_for_model(model_name)

        for part in col:
            yield part.map(lambda x: len(_tiktoken_enc.encode(x)) if isinstance(x, str) else 0)

    return fn


def split_to_chunks_udf(model_name: str, max_tokens: int, sep: List[str]):
    @pandas_udf(ArrayType(StringType()))
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        global _tiktoken_enc
        if _tiktoken_enc is None:
            _tiktoken_enc = tiktoken.encoding_for_model(model_name)

        chunker = TextChunker(_tiktoken_enc)

        for part in col:
            yield part.map(lambda x: chunker.split(x, max_tokens=max_tokens, sep=sep) if isinstance(x, str) else [])

    return fn
