import os
from dataclasses import dataclass
from logging import getLogger, Logger
from typing import Iterator

import pandas as pd
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType, ArrayType, FloatType

from openaivec.log import observe

__ALL__ = ["UDFBuilder"]

_logger: Logger = getLogger(__name__)


@dataclass(frozen=True)
class UDFBuilder:
    """Helper class to build UDFs for OpenAI completion and embedding tasks using Spark.

    Attributes:
        api_key (str): Azure API key for OpenAI.
        api_version (str): API version.
        endpoint (str): The Azure endpoint.
        model_name (str): OpenAI model name.
        batch_size (int, optional): Batch size for UDF operations (default: 256).
    """

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
        """Creates a pandas UDF for generating completions.

        Args:
            system_message (str): The system prompt for generating completions.

        Returns:
            function: A pandas UDF that processes a pandas Series.
        """

        @pandas_udf(StringType())
        def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            import httpx
            import pandas as pd
            from openai import AzureOpenAI

            from openaivec import VectorizedOpenAI

            client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                http_client=httpx.Client(http2=True, verify=False),
                api_key=self.api_key,
            )

            client_vec = VectorizedOpenAI(
                client=client,
                model_name=self.model_name,
                system_message=system_message,
                top_p=1.0,
                temperature=0.0,
            )

            for part in col:
                yield pd.Series(client_vec.predict_minibatch(part.tolist(), self.batch_size))

        return fn

    @observe(_logger)
    def embedding(self):
        """Creates a pandas UDF for generating embeddings.

        Returns:
            function: A pandas UDF that processes a pandas Series.
        """

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
