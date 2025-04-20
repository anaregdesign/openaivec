import unittest

import numpy as np
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd

from openaivec import pandas_ext

pandas_ext.use(OpenAI())
pandas_ext.responses_model("gpt-4o-mini")
pandas_ext.embeddings_model("text-embedding-3-small")


class Fruit(BaseModel):
    color: str
    flavor: str
    taste: str


class TestPandasExt(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["apple", "banana", "cherry"],
            }
        )

    def test_embeddings(self):
        embeddings: pd.Series = self.df["name"].ai.embeddings()

        # assert all values are elements of np.ndarray
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))

    def test_responses(self):
        names_fr: pd.Series = self.df["name"].ai.responses("translate to French")

        # assert all values are elements of str
        self.assertTrue(all(isinstance(x, str) for x in names_fr))

    def test_responses_dataframe(self):
        names_fr: pd.Series = self.df.ai.responses("translate to French")

        # assert all values are elements of str
        self.assertTrue(all(isinstance(x, str) for x in names_fr))

    def test_extract_series(self):
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                Fruit(color="yellow", flavor="sweet", taste="soft"),
                Fruit(color="red", flavor="sweet", taste="tart"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

    def test_extract_series_without_name(self):
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                Fruit(color="yellow", flavor="sweet", taste="soft"),
                Fruit(color="red", flavor="sweet", taste="tart"),
            ]
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["color", "flavor", "taste"]  # without prefix
        self.assertListEqual(list(extracted_df.columns), expected_columns)

    def test_extract_series_dict(self):
        sample_series = pd.Series(
            [
                {"color": "red", "flavor": "sweet", "taste": "crunchy"},
                {"color": "yellow", "flavor": "sweet", "taste": "soft"},
                {"color": "red", "flavor": "sweet", "taste": "tart"},
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

    def test_extract_series_with_none(self):
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                None,
                Fruit(color="yellow", flavor="sweet", taste="soft"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()

        # assert columns are ['fruit_color', 'fruit_flavor', 'fruit_taste']
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

        # assert the row with None is filled with NaN
        self.assertTrue(extracted_df.iloc[1].isna().all())

    def test_extract_series_with_invalid_row(self):
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                123,  # Invalid row
                Fruit(color="yellow", flavor="sweet", taste="soft"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()

        # assert columns are ['fruit_color', 'fruit_flavor', 'fruit_taste']
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

        # assert the invalid row is filled with NaN
        self.assertTrue(extracted_df.iloc[1].isna().all())

    def test_extract(self):
        sample_df = pd.DataFrame(
            [
                {"name": "apple", "fruit": Fruit(color="red", flavor="sweet", taste="crunchy")},
                {"name": "banana", "fruit": Fruit(color="yellow", flavor="sweet", taste="soft")},
                {"name": "cherry", "fruit": Fruit(color="red", flavor="sweet", taste="tart")},
            ]
        ).ai.extract("fruit")

        expected_columns = ["name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

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

    def test_extract_with_invalid_row(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": 123},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        )

        expected_columns = ["fruit"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_count_tokens(self):
        num_tokens: pd.Series = self.df.name.ai.count_tokens()

        # assert all values are elements of int
        self.assertTrue(all(isinstance(num_token, int) for num_token in num_tokens))
