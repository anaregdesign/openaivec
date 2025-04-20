import json
import os
import logging
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

_LOGGER = logging.getLogger(__name__)


T = TypeVar("T")

_CLIENT: OpenAI | None = None
_RESPONSES_MODEL_NAME = "gpt-4o-mini"
_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

_TIKTOKEN_ENCODING = tiktoken.encoding_for_model(_RESPONSES_MODEL_NAME)


def use(client: OpenAI) -> None:
    """Register a custom ``openai.OpenAI``‑compatible client.

    Parameters
    ----------
    client : OpenAI
        An already configured OpenAI/AzureOpenAI client instance that will be
        reused by every helper in this module.
    """
    global _CLIENT
    _CLIENT = client


def use_openai(api_key: str) -> None:
    """Create and register a default ``openai.OpenAI`` client.

    Parameters
    ----------
    api_key : str
        Value passed to the ``api_key`` argument of ``openai.OpenAI``.
    """
    global _CLIENT
    _CLIENT = OpenAI(api_key=api_key)


def use_azure_openai(api_key: str, endpoint: str, api_version: str) -> None:
    """Create and register an ``openai.AzureOpenAI`` client.

    Parameters
    ----------
    api_key : str
        Azure OpenAI subscription key.
    endpoint : str
        Resource endpoint (e.g. ``https://<resource>.openai.azure.com``).
    api_version : str
        REST API version such as ``2024‑02‑15-preview``.
    """
    global _CLIENT
    _CLIENT = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def responses_model(name: str) -> None:
    """Override the model used for text responses.

    Parameters
    ----------
    name : str
        Model name listed in the OpenAI API (e.g. ``gpt-4o-mini``).
    """
    global _RESPONSES_MODEL_NAME, _TIKTOKEN_ENCODING
    _RESPONSES_MODEL_NAME = name

    try:
        _TIKTOKEN_ENCODING = tiktoken.encoding_for_model(name)

    except KeyError:
        _LOGGER.warning(
            "The model name '%s' is not supported by tiktoken. Instead, using the 'o200k_base' encoding.",
            name,
        )
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


def embedding_model(name: str) -> None:
    """Override the model used for text embeddings.

    Parameters
    ----------
    name : str
        Embedding model name such as ``text-embedding-3-small``.
    """
    global _EMBEDDING_MODEL_NAME
    _EMBEDDING_MODEL_NAME = name


def get_openai_client() -> OpenAI:
    """Return a configured OpenAI client.

    The priority is:

    1. A client registered via :func:`use`, :func:`use_openai`,
       or :func:`use_azure_openai`.
    2. Environment variable ``OPENAI_API_KEY`` (plain OpenAI).
    3. Environment variables ``AZURE_OPENAI_*`` (Azure OpenAI).

    Raises
    ------
    ValueError
        If no credentials are found.
    """
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


def _extract_value(x, series_name):
    """Convert heterogeneous objects in a Series to a homogeneous ``dict``.

    Parameters
    ----------
    x :
        Individual element of the Series.
    series_name : str
        Name of the Series – used only for logging.

    Returns
    -------
    dict
        Dictionary representation or an empty dict when the value cannot be
        coerced.
    """
    if x is None:
        return {}
    elif isinstance(x, BaseModel):
        return x.model_dump()
    elif isinstance(x, dict):
        return x

    _LOGGER.warning(f"The value '{x}' in the series is not a dict or BaseModel. Returning an empty dict.")
    return {}


@pd.api.extensions.register_series_accessor("ai")
class OpenAIVecSeriesAccessor:
    """pandas ``Series`` accessor that adds OpenAI helpers (``.ai``)."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def response(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
    ) -> pd.Series:
        """Call an LLM for every element of the Series.

        Parameters
        ----------
        instructions : str
            System prompt placed before each user message.
        response_format : Type[T], default ``str``
            Pydantic model or builtin type the assistant should return.
        batch_size : int, default ``128``
            Number of prompts submitted in a single request.

        Returns
        -------
        pandas.Series
            Series whose values are of type ``response_format``.
        """
        client: VectorizedLLM = VectorizedOpenAI(
            client=get_openai_client(),
            model_name=_RESPONSES_MODEL_NAME,
            system_message=instructions,
            is_parallel=True,
            response_format=response_format,
            temperature=0,
            top_p=1,
        )

        return pd.Series(
            client.predict(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def embed(self, batch_size: int = 128) -> pd.Series:
        """Compute OpenAI embeddings for every element.

        Parameters
        ----------
        batch_size : int, default ``128``
            Number of inputs sent in one embedding request.

        Returns
        -------
        pandas.Series
            Series of ``list[float]`` (one embedding vector per element).
        """
        client: EmbeddingLLM = EmbeddingOpenAI(
            client=get_openai_client(),
            model_name=_EMBEDDING_MODEL_NAME,
            is_parallel=True,
        )

        return pd.Series(
            client.embed(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def count_tokens(self) -> pd.Series:
        """Return the number of tokens in each row using *tiktoken*."""
        return self._obj.map(_TIKTOKEN_ENCODING.encode).map(len).rename("num_tokens")

    def extract(self) -> pd.DataFrame:
        """Expand a Series of Pydantic models/dicts into a DataFrame.

        When the Series has a name, extracted column names are prefixed with it.
        """
        extracted = pd.DataFrame(
            self._obj.map(lambda x: _extract_value(x, self._obj.name)).tolist(),
            index=self._obj.index,
        )

        if self._obj.name:
            # If the Series has a name and all elements are dict or BaseModel, use it as the prefix for the columns
            extracted.columns = [f"{self._obj.name}_{col}" for col in extracted.columns]
        return extracted


@pd.api.extensions.register_dataframe_accessor("ai")
class OpenAIVecDataFrameAccessor:
    """pandas ``DataFrame`` accessor that adds OpenAI helpers (``.ai``)."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    def extract(self, column: str) -> pd.DataFrame:
        """Flatten one column of Pydantic models/dicts into top‑level columns.

        Parameters
        ----------
        column : str
            Name of the column to expand.

        Returns
        -------
        pandas.DataFrame
            Original DataFrame plus the extracted columns (source column
            removed).
        """
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
        batch_size: int = 128,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON.

        Parameters
        ----------
        instructions : str
            System prompt for the assistant.
        response_format : Type[T], default ``str``
            Desired return type.
        batch_size : int, default ``128``
            Request batch size.

        Returns
        -------
        pandas.Series
            Series of responses aligned with the original index.
        """
        return self._obj.pipe(
            lambda df: (
                df.pipe(lambda df: pd.Series(df.to_dict(orient="records"), index=df.index))
                .map(lambda x: json.dumps(x, ensure_ascii=False))
                .ai.response(
                    instructions=instructions,
                    response_format=response_format,
                    batch_size=batch_size,
                )
            )
        )
