import os
from typing import Type, TypeVar

import pandas as pd
from openai import AzureOpenAI, OpenAI

from openaivec.embedding import EmbeddingOpenAI
from openaivec.vectorize import VectorizedLLM, VectorizedOpenAI

__all__ = []


T = TypeVar("T")

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    if "OPENAI_API_KEY" in os.environ:
        _client = OpenAI()
        return _client

    aoai_param_names = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ]

    if all(param in os.environ for param in aoai_param_names):
        _client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        return _client

    raise ValueError(
        "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable or provide Azure OpenAI parameters."
        "If using Azure OpenAI, ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION are set."
        "If using OpenAI, ensure OPENAI_API_KEY is set."
    )


@pd.api.extensions.register_series_accessor("ai")
class OpenAIVecSeriesAccessor:
    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def predict(self, model_name: str, prompt: str, respnse_format: Type[T] = str, batch_size: int = 128):
        client: VectorizedLLM = VectorizedOpenAI(
            client=get_openai_client(),
            model_name=model_name,
            system_message=prompt,
            is_parallel=True,
            response_format=respnse_format,
            temperature=0,
            top_p=1,
        )

        return pd.Series(
            client.predict_minibatch(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
        )

    def embed(self, model_name: str, batch_size: int = 128):
        client: VectorizedLLM = EmbeddingOpenAI(
            client=get_openai_client(),
            model_name=model_name,
        )

        return pd.Series(
            client.embed_minibatch(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
        )
