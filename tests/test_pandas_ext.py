import unittest

import numpy as np
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd

from openaivec import pandas_ext

pandas_ext.use(OpenAI())


class TestPandasExt(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["apple", "banana", "cherry"],
            }
        )

    def test_embed(self):
        embeddings: pd.Series = self.df["name"].ai.embed("text-embedding-3-large")

        # assert all values are elements of np.ndarray
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))

    def test_predict(self):
        names_fr: pd.Series = self.df["name"].ai.predict("gpt-4o-mini", "translate to French")

        # assert all values are elements of str
        self.assertTrue(all(isinstance(name_fr, str) for name_fr in names_fr))

    def test_extract(self):
        class Fruit(BaseModel):
            color: str
            flavor: str
            taste: str

        self.df.assign(
            fruit=lambda df: df.name.ai.predict(
                model_name="gpt-4o-mini", prompt="extract fruit information", response_format=Fruit
            )
        ).pipe(lambda df: df.ai.extract("fruit"))
