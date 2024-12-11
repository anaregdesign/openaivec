from typing import Iterator

import pandas as pd
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType


def openai_udf(system_message: str, batch_size: int = 128):
    @pandas_udf(StringType())
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        import os

        import httpx
        import pandas as pd
        from openai import AzureOpenAI

        from openaivec import VectorizedOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")

        assert api_key, f"AZURE_OPENAI_API_KEY environment variable must be set"
        assert api_version, "AZURE_OPENAI_API_VERSION environment variable must be set"
        assert endpoint, "AZURE_OPENAI_ENDPOINT environment variable must be set"
        assert model_name, "AZURE_OPENAI_MODEL_NAME environment variable must be set"

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            http_client=httpx.Client(http2=True, verify=False),
        )

        client_vec = VectorizedOpenAI(
            client=client,
            model_name=model_name,
            system_message=system_message,
            top_p=1.0,
            temperature=0.0,
        )

        for part in col:
            yield pd.Series(
                client_vec.predict_minibatch(part.tolist(), batch_size)
            )

    return fn
