import os
from typing import Iterator

import httpx
import pandas as pd
from openai import AzureOpenAI
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType

from openaivec import VectorizedOpenAI


def openai_udf(system_message: str, batch_size: int = 128):
    @pandas_udf(StringType())
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        client = AzureOpenAI(
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-10-21"),
            azure_endpoint=os.environ.get("OPENAI_AZURE_ENDPOINT"),
            http_client=httpx.Client(http2=True, verify=False),
        )

        client_vec = VectorizedOpenAI(
            client=client,
            model_name=os.environ.get("OPENAI_MODEL_NAME"),
            system_message=system_message,
            top_p=1.0,
            temperature=0.0,
        )

        for part in col:
            yield pd.Series(
                client_vec.predict_minibatch(part.tolist(), batch_size)
            )

    return fn
