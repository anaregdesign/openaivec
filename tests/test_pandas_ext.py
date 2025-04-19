from typing import List
import unittest

import numpy as np
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd

from openaivec import pandas_ext

pandas_ext.use(OpenAI())
pandas_ext.responses_model("gpt-4o-mini")
pandas_ext.embedding_model("text-embedding-3-small")


class TestPandasExt(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["apple", "banana", "cherry"],
            }
        )

    def test_embed(self):
        embeddings: pd.Series = self.df["name"].ai.embed()

        # assert all values are elements of np.ndarray
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))

    def test_predict(self):
        names_fr: pd.Series = self.df["name"].ai.response("translate to French")

        # assert all values are elements of str
        self.assertTrue(all(isinstance(name_fr, str) for name_fr in names_fr))

    def test_extract(self):
        class Fruit(BaseModel):
            color: str
            flavor: str
            taste: str

        columns: List[str] = (
            self.df.assign(
                fruit=lambda df: df.name.ai.response(instructions="extract fruit information", response_format=Fruit)
            )
            .pipe(lambda df: df.ai.extract("fruit"))
            .columns
        )

        # assert columns are ['name', 'fruit_color', 'fruit_flavor', 'fruit_taste']
        # extracted_field names are f{original_field_name}_{extracted_field_name}
        for column in columns:
            self.assertIn(column, ["name", "fruit_color", "fruit_flavor", "fruit_taste"])

    def test_extract_dict(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": {"name": "banana", "color": "yellow", "flavor": "sweet", "taste": "soft"}},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")

        # assert columns are exactly ['fruit_name', 'fruit_color', 'fruit_flavor', 'fruit_taste']
        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_extract_dict_with_none(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": None},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")

        # assert columns are ['fruit_name', 'fruit_color', 'fruit_flavor', 'fruit_taste']
        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

        # assert the row with None is filled with NaN
        self.assertTrue(sample_df.iloc[1].isna().all())

    def test_count_tokens(self):
        num_tokens: pd.Series = self.df.name.ai.count_tokens()

        # assert all values are elements of int
        self.assertTrue(all(isinstance(num_token, int) for num_token in num_tokens))
