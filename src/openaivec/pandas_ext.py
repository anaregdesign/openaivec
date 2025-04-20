"""Pandas Series / DataFrame extension for OpenAI.

## Setup
```python
from openai import OpenAI
from openaivec import pandas_ext

# Set up the OpenAI client to use with pandas_ext
pandas_ext.use(OpenAI())

# Set up the model_name for responses and embeddings
pandas_ext.responses_model("gpt-4.1-nano")
pandas_ext.embeddings_model("text-embedding-3-small")
```

"""

import json
import os
import logging
from typing import Type, TypeVar

import pandas as pd
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
import tiktoken

from openaivec.embeddings import VectorizedEmbeddings, VectorizedEmbeddingsOpenAI
from openaivec.responses import VectorizedResponses, VectorizedResponsesOpenAI

__all__ = [
    "use",
    "responses_model",
    "embeddings_model",
    "use_openai",
    "use_azure_openai",
]

_LOGGER = logging.getLogger(__name__)


T = TypeVar("T")

_CLIENT: OpenAI | None = None
_RESPONSES_MODEL_NAME = "gpt-4o-mini"
_EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"

_TIKTOKEN_ENCODING = tiktoken.encoding_for_model(_RESPONSES_MODEL_NAME)


def use(client: OpenAI) -> None:
    """Register a custom OpenAI‑compatible client.

    Args:
        client (OpenAI): A pre‑configured `openai.OpenAI` or
            `openai.AzureOpenAI` instance.
            The same instance is reused by every helper in this module.
    """
    global _CLIENT
    _CLIENT = client


def use_openai(api_key: str) -> None:
    """Create and register a default `openai.OpenAI` client.

    Args:
        api_key (str): Value forwarded to the ``api_key`` parameter of
            `openai.OpenAI`.
    """
    global _CLIENT
    _CLIENT = OpenAI(api_key=api_key)


def use_azure_openai(api_key: str, endpoint: str, api_version: str) -> None:
    """Create and register an `openai.AzureOpenAI` client.

    Args:
        api_key (str): Azure OpenAI subscription key.
        endpoint (str): Resource endpoint, e.g.
            ``https://<resource>.openai.azure.com``.
        api_version (str): REST API version such as ``2024‑02‑15-preview``.
    """
    global _CLIENT
    _CLIENT = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def responses_model(name: str) -> None:
    """Override the model used for text responses.

    Args:
        name (str): Model name as listed in the OpenAI API
            (for example, ``gpt-4o-mini``).
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


def embeddings_model(name: str) -> None:
    """Override the model used for text embeddings.

    Args:
        name (str): Embedding model name, e.g. ``text-embedding-3-small``.
    """
    global _EMBEDDINGS_MODEL_NAME
    _EMBEDDINGS_MODEL_NAME = name


def _get_openai_client() -> OpenAI:
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
    """Return a homogeneous ``dict`` representation of any Series value.

    Args:
        x: Single element taken from the Series.
            series_name (str): Name of the Series (only used for logging).

    Returns:
        dict: A dictionary representation or an empty ``dict`` if ``x`` cannot
            be coerced.
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
    """pandas Series accessor (``.ai``) that adds OpenAI helpers."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def responses(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
    ) -> pd.Series:
        """Call an LLM once for every Series element.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.responses("translate to French")
            ```
            This method returns a Series of strings, each containing the
            assistant's response to the corresponding input.
            The model used is set by the `responses_model` function.
            The default model is `gpt-4o-mini`.

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (Type[T], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int, optional): Number of prompts grouped into a single
                request. Defaults to ``128``.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.
        """
        client: VectorizedResponses = VectorizedResponsesOpenAI(
            client=_get_openai_client(),
            model_name=_RESPONSES_MODEL_NAME,
            system_message=instructions,
            is_parallel=True,
            response_format=response_format,
            temperature=0,
            top_p=1,
        )

        return pd.Series(
            client.parse(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def embeddings(self, batch_size: int = 128) -> pd.Series:
        """Compute OpenAI embeddings for every Series element.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.embeddings()
            ```
            This method returns a Series of numpy arrays, each containing the
            embedding vector for the corresponding input.
            The embedding model is set by the `embeddings_model` function.
            The default embedding model is `text-embedding-3-small`.

        Args:
            batch_size (int, optional): Number of inputs grouped into a
                single request. Defaults to ``128``.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).
        """
        client: VectorizedEmbeddings = VectorizedEmbeddingsOpenAI(
            client=_get_openai_client(),
            model_name=_EMBEDDINGS_MODEL_NAME,
            is_parallel=True,
        )

        return pd.Series(
            client.create(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def count_tokens(self) -> pd.Series:
        """Count `tiktoken` tokens per row.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.count_tokens()
            ```
            This method uses the `tiktoken` library to count tokens based on the
            model name set by `responses_model`.

        Returns:
            pandas.Series: Token counts for each element.
        """
        return self._obj.map(_TIKTOKEN_ENCODING.encode).map(len).rename("num_tokens")

    def extract(self) -> pd.DataFrame:
        """Expand a Series of Pydantic models/dicts into columns.

        Example:
            ```python
            animals = pd.Series([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            animals.ai.extract()
            ```
            This method returns a DataFrame with the same index as the Series,
            where each column corresponds to a key in the dictionaries.
            If the Series has a name, extracted columns are prefixed with it.

        Returns:
            pandas.DataFrame: Expanded representation.
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
    """pandas DataFrame accessor (``.ai``) that adds OpenAI helpers."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    def extract(self, column: str) -> pd.DataFrame:
        """Flatten one column of Pydantic models/dicts into top‑level columns.

        Example:
            ```python
            df = pd.DataFrame([
                {"animal": {"name": "cat", "legs": 4}},
                {"animal": {"name": "dog", "legs": 4}},
                {"animal": {"name": "elephant", "legs": 4}},
            ])
            df.ai.extract("animal")
            ```
            This method returns a DataFrame with the same index as the original,
            where each column corresponds to a key in the dictionaries.
            The source column is dropped.

        Args:
            column (str): Column to expand.

        Returns:
            pandas.DataFrame: Original DataFrame with the extracted columns; the source column is dropped.
        """
        if column not in self._obj.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        return (
            self._obj.pipe(lambda df: df.reset_index(drop=True))
            .pipe(lambda df: df.join(df[column].ai.extract()))
            .pipe(lambda df: df.set_index(self._obj.index))
            .pipe(lambda df: df.drop(columns=[column], axis=1))
        )

    def responses(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON.

        Example:
            ```python
            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            df.ai.responses("what is the animal's name?")
            ```
            This method returns a Series of strings, each containing the
            assistant's response to the corresponding input.
            Each row is serialised to JSON before being sent to the assistant.
            The model used is set by the `responses_model` function.
            The default model is `gpt-4o-mini`.

        Args:
            instructions (str): System prompt for the assistant.
            response_format (Type[T], optional): Desired Python type of the
                responses. Defaults to ``str``.
            batch_size (int, optional): Number of requests sent in one batch.
                Defaults to ``128``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame’s original index.
        """
        return self._obj.pipe(
            lambda df: (
                df.pipe(lambda df: pd.Series(df.to_dict(orient="records"), index=df.index, name="record"))
                .map(lambda x: json.dumps(x, ensure_ascii=False))
                .ai.responses(
                    instructions=instructions,
                    response_format=response_format,
                    batch_size=batch_size,
                )
            )
        )
