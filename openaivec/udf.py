from typing import Iterator

import pandas as pd
from openai import OpenAI
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType

from openaivec import VectorizedOpenAI


def openai_udf(client: OpenAI, model_name: str, system_message: str, top_p: float = 1.0, temperature: float = 0.0,
               batch_size: int = 128):
    @pandas_udf(StringType())
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        client_vec = VectorizedOpenAI(
            client=client,
            model_name=model_name,
            system_message=system_message,
            top_p=top_p,
            temperature=temperature,
        )

        for part in col:
            yield pd.Series(
                client_vec.predict_minibatch(part.tolist(), batch_size)
            )

    return fn
