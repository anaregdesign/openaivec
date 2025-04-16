import json
import os
from typing import Type, TypeVar

import pandas as pd
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
import tiktoken

from openaivec.embedding import EmbeddingLLM, EmbeddingOpenAI
from openaivec.vectorize import VectorizedLLM, VectorizedOpenAI

__all__ = [
    "use",
    "responses_model",
    "embedding_model",
    "use_openai",
    "use_azure_openai",
]


T = TypeVar("T")

_CLIENT: OpenAI | None = None
_RESPONSES_MODEL_NAME = "gpt-4o-mini"
_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

_TIKTOKEN_ENCODING = tiktoken.encoding_for_model(_RESPONSES_MODEL_NAME)


def use(client: OpenAI) -> None:
    """
    Set the OpenAI client to use for OpenAI.
    """
    global _CLIENT
    _CLIENT = client


def use_openai(api_key: str) -> None:
    """
    Set the OpenAI API key to use for OpenAI.
    """
    global _CLIENT
    _CLIENT = OpenAI(api_key=api_key)


def use_azure_openai(api_key: str, endpoint: str, api_version: str) -> None:
    """
    Set the Azure OpenAI API key to use for Azure OpenAI.
    """
    global _CLIENT
    _CLIENT = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def responses_model(name: str) -> None:
    """
    Set the OpenAI responses model name to use for OpenAI.
    """
    global _RESPONSES_MODEL_NAME, _TIKTOKEN_ENCODING
    _RESPONSES_MODEL_NAME = name
    _TIKTOKEN_ENCODING = tiktoken.encoding_for_model(name)


def embedding_model(name: str) -> None:
    """
    Set the OpenAI embedding model name to use for OpenAI.
    """
    global _EMBEDDING_MODEL_NAME
    _EMBEDDING_MODEL_NAME = name


def get_openai_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    if "OPENAI_API_KEY" in os.environ:
        _CLIENT = OpenAI()
        return _CLIENT

    aoai_param_names = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ]

    if all(param in os.environ for param in aoai_param_names):
        _CLIENT = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        return _CLIENT

    raise ValueError(
        "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable or provide Azure OpenAI parameters."
        "If using Azure OpenAI, ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION are set."
        "If using OpenAI, ensure OPENAI_API_KEY is set."
    )


@pd.api.extensions.register_series_accessor("ai")
class OpenAIVecSeriesAccessor:
    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def response(
        self, instructions: str, response_format: Type[T] = str, model_name=_RESPONSES_MODEL_NAME, batch_size: int = 128
    ) -> pd.Series:
        client: VectorizedLLM = VectorizedOpenAI(
            client=get_openai_client(),
            model_name=model_name,
            system_message=instructions,
            is_parallel=True,
            response_format=response_format,
            temperature=0,
            top_p=1,
        )

        return pd.Series(
            client.predict_minibatch(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def embed(self, model_name: str = _EMBEDDING_MODEL_NAME, batch_size: int = 128) -> pd.Series:
        client: EmbeddingLLM = EmbeddingOpenAI(
            client=get_openai_client(),
            model_name=model_name,
        )

        return pd.Series(
            client.embed_minibatch(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def count_tokens(self) -> pd.Series:
        return self._obj.map(_TIKTOKEN_ENCODING.encode).map(len).rename("num_tokens")

    def extract(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._obj.map(lambda x: x.model_dump() if isinstance(x, BaseModel) else {self._obj.name: x}).tolist(),
            index=self._obj.index,
        )


@pd.api.extensions.register_dataframe_accessor("ai")
class OpenAIVecDataFrameAccessor:
    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    def extract(self, column: str) -> pd.DataFrame:
        if column not in self._obj.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        return (
            self._obj.pipe(lambda df: df.reset_index(drop=True))
            .pipe(lambda df: df.join(df[column].ai.extract()))
            .pipe(lambda df: df.set_index(self._obj.index))
            .pipe(lambda df: df.drop(columns=[column], axis=1))
        )

    def response(
        self,
        instructions: str,
        response_format: Type[T] = str,
        model_name: str = _RESPONSES_MODEL_NAME,
        batch_size: int = 128,
    ) -> pd.Series:
        return self._obj.pipe(
            lambda df: (
                df.pipe(lambda df: pd.Series(df.to_dict(orient="records"), index=df.index))
                .map(lambda x: json.dumps(x, ensure_ascii=False))
                .ai.predict(
                    instructions=instructions,
                    response_format=response_format,
                    model_name=model_name,
                    batch_size=batch_size,
                )
            )
        )
