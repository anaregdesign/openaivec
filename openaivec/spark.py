import os
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType

__ALL__ = ["UDFConfig", "openai_udf"]


@dataclass(frozen=True)
class UDFConfig:
    api_key: str
    api_version: str
    endpoint: str
    model_name: str

    @classmethod
    def of_environment(cls) -> "UDFConfig":
        return cls(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            model_name=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
        )

    def __post_init__(self):
        assert self.api_key, "api_key must be set"
        assert self.api_version, "api_version must be set"
        assert self.endpoint, "endpoint must be set"
        assert self.model_name, "model_name must be set"


def completion_udf(conf: UDFConfig, system_message: str, batch_size: int = 128):
    @pandas_udf(StringType())
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:

        import httpx
        import pandas as pd
        from openai import AzureOpenAI

        from openaivec import VectorizedOpenAI

        client = AzureOpenAI(
            api_version=conf.api_version,
            azure_endpoint=conf.endpoint,
            http_client=httpx.Client(http2=True, verify=False),
            api_key=conf.api_key,
        )

        client_vec = VectorizedOpenAI(
            client=client,
            model_name=conf.model_name,
            system_message=system_message,
            top_p=1.0,
            temperature=0.0,
        )

        for part in col:
            yield pd.Series(
                client_vec.predict_minibatch(part.tolist(), batch_size)
            )

    return fn
