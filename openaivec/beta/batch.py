import json
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType

__all__ = ["JsonlUDFBuilder"]


@dataclass(frozen=True)
class JsonlUDFBuilder:
    model_name: str

    def to_jsonl(self, system_message: str, temperature: float = 0.0, top_p: float = 1.0):
        @pandas_udf(StringType())
        def to_jsonl_udf(custom_id: Iterator[pd.Series], user_message: Iterator[pd.Series]) -> Iterator[pd.Series]:
            for i, m in zip(custom_id, user_message):
                df: pd.DataFrame = pd.DataFrame({"custom_id": i, "user_message": m})

                yield pd.Series(
                    df.apply(
                        lambda row: json.dumps(
                            {
                                "custom_id": row["custom_id"],
                                "method": "POST",
                                "url": "/chat/completions",
                                "body": {
                                    "model": self.model_name,
                                    "messages": [
                                        {"role": "system", "content": system_message},
                                        {"role": "user", "content": row["user_message"]},
                                    ],
                                    "temperature": temperature,
                                    "top_p": top_p,
                                },
                            }
                        ),
                        axis=1,
                    )
                )

        return to_jsonl_udf
