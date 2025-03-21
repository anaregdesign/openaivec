import unittest

import numpy as np
import pandas as pd

from openaivec import pandas_ext


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
