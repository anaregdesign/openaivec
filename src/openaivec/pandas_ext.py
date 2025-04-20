"""Pandas Series / DataFrame extension for OpenAI.

## Setup
```python
from openai import OpenAI
from openaivec import pandas_ext

# Set up the OpenAI client to use with pandas_ext
pandas_ext.use(OpenAI())

# Set up the model_name for responses and embeddings
pandas_ext.responses_model("gpt-4.1-nano")
pandas_ext.embedding_model("text-embedding-3-small"
```

## Usage for Series

This is a simple dummy data with `pd.Series`.
```python
import pandas as pd
animals: pd.Series = pd.Series(["panda", "koala", "python", "dog", "cat"])
```

You can mutate the column with natural language instructions.

```python
# Translate animal names to Chinese
animals.ai.response(instructions="Translate the animal names to Chinese.")
```

and its results are `['熊猫', '考拉', '蟒蛇', '狗', '猫']` (Not sure that's right, I can't read Chinese).

Embedding is also available.

```python
animals.ai.embed()
# 0    [-0.008575918, -0.07940717, -0.011005879, 0.00...
# 1    [0.0008873118, -0.015903357, -0.021896126, -0....
# 2    [-0.010200691, -0.011314859, 0.009946684, -0.0...
# 3    [0.051125195, -0.018667098, -0.00435894, 0.072...
# 4    [0.025523458, -0.02345273, -0.016077219, 0.039...
# Name: animal, dtype: object
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


def embedding_model(name: str) -> None:
    """Override the model used for text embeddings.

    Args:
        name (str): Embedding model name, e.g. ``text-embedding-3-small``.
    """
    global _EMBEDDING_MODEL_NAME
    _EMBEDDING_MODEL_NAME = name


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

    def response(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
    ) -> pd.Series:
        """Call an LLM once for every Series element.

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (Type[T], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int, optional): Number of prompts grouped into a single
                request. Defaults to ``128``.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.
        """
        client: VectorizedLLM = VectorizedOpenAI(
            client=_get_openai_client(),
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
        """Compute OpenAI embeddings for every Series element.

        Args:
            batch_size (int, optional): Number of inputs sent per request.
                Defaults to ``128``.

        Returns:
            pandas.Series: Each value is a list of floats (the embedding vector).
        """
        client: EmbeddingLLM = EmbeddingOpenAI(
            client=_get_openai_client(),
            model_name=_EMBEDDING_MODEL_NAME,
            is_parallel=True,
        )

        return pd.Series(
            client.embed(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def count_tokens(self) -> pd.Series:
        """Count `tiktoken` tokens per row.

        Returns:
            pandas.Series: Token counts for each element.
        """
        return self._obj.map(_TIKTOKEN_ENCODING.encode).map(len).rename("num_tokens")

    def extract(self) -> pd.DataFrame:
        """Expand a Series of Pydantic models/dicts into columns.

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

    def response(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON.

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
                df.pipe(lambda df: pd.Series(df.to_dict(orient="records"), index=df.index))
                .map(lambda x: json.dumps(x, ensure_ascii=False))
                .ai.response(
                    instructions=instructions,
                    response_format=response_format,
                    batch_size=batch_size,
                )
            )
        )
