import pytest
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel
import pandas as pd

# Import the module to test
from openaivec.aio import pandas_ext

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Setup the async client and models for testing
# Ensure you have OPENAI_API_KEY set in your environment or use pandas_ext.use()
try:
    pandas_ext.use(AsyncOpenAI())  # Attempts to use environment variables if no args
except ValueError:
    pytest.skip(
        "OpenAI client setup failed, skipping async pandas tests. Ensure API keys are set.", allow_module_level=True
    )

pandas_ext.responses_model("gpt-4o-mini")
pandas_ext.embeddings_model("text-embedding-3-small")


class Fruit(BaseModel):
    color: str
    flavor: str
    taste: str


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "name": ["apple", "banana", "cherry"],
        }
    )


@pytest.fixture
def sample_series_fruit_model():
    return pd.Series(
        [
            Fruit(color="red", flavor="sweet", taste="crunchy"),
            Fruit(color="yellow", flavor="sweet", taste="soft"),
            Fruit(color="red", flavor="sweet", taste="tart"),
        ],
        name="fruit",
    )


@pytest.fixture
def sample_series_fruit_model_with_none():
    return pd.Series(
        [
            Fruit(color="red", flavor="sweet", taste="crunchy"),
            None,
            Fruit(color="yellow", flavor="sweet", taste="soft"),
        ],
        name="fruit",
    )


@pytest.fixture
def sample_series_fruit_model_with_invalid():
    return pd.Series(
        [
            Fruit(color="red", flavor="sweet", taste="crunchy"),
            123,  # Invalid row
            Fruit(color="yellow", flavor="sweet", taste="soft"),
        ],
        name="fruit",
    )


@pytest.fixture
def sample_series_dict():
    return pd.Series(
        [
            {"color": "red", "flavor": "sweet", "taste": "crunchy"},
            {"color": "yellow", "flavor": "sweet", "taste": "soft"},
            {"color": "red", "flavor": "sweet", "taste": "tart"},
        ],
        name="fruit",
    )


@pytest.fixture
def sample_df_extract():
    return pd.DataFrame(
        [
            {"name": "apple", "fruit": Fruit(color="red", flavor="sweet", taste="crunchy")},
            {"name": "banana", "fruit": Fruit(color="yellow", flavor="sweet", taste="soft")},
            {"name": "cherry", "fruit": Fruit(color="red", flavor="sweet", taste="tart")},
        ]
    )


@pytest.fixture
def sample_df_extract_dict():
    return pd.DataFrame(
        [
            {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
            {"fruit": {"name": "banana", "color": "yellow", "flavor": "sweet", "taste": "soft"}},
            {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
        ]
    )


@pytest.fixture
def sample_df_extract_dict_with_none():
    return pd.DataFrame(
        [
            {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
            {"fruit": None},
            {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
        ]
    )


@pytest.fixture
def sample_df_extract_with_invalid():
    return pd.DataFrame(
        [
            {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
            {"fruit": 123},  # Invalid data
            {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
        ]
    )


async def test_embeddings(sample_df):
    embeddings: pd.Series = await sample_df["name"].ai.embeddings()
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert embeddings.shape == (3,)
    assert embeddings.index.equals(sample_df.index)


async def test_responses_series(sample_df):
    names_fr: pd.Series = await sample_df["name"].ai.responses("translate to French")
    assert all(isinstance(x, str) for x in names_fr)
    assert names_fr.shape == (3,)
    assert names_fr.index.equals(sample_df.index)


async def test_responses_dataframe(sample_df):
    # Test DataFrame.ai.responses
    names_fr: pd.Series = await sample_df.ai.responses("translate the 'name' field to French")
    assert all(isinstance(x, str) for x in names_fr)
    assert names_fr.shape == (3,)
    assert names_fr.index.equals(sample_df.index)


def test_extract_series_model(sample_series_fruit_model):
    extracted_df = sample_series_fruit_model.ai.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 3)
    assert extracted_df.index.equals(sample_series_fruit_model.index)


def test_extract_series_model_with_none(sample_series_fruit_model_with_none):
    extracted_df = sample_series_fruit_model_with_none.ai.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 3)


def test_extract_series_model_with_invalid(sample_series_fruit_model_with_invalid):
    extracted_df = sample_series_fruit_model_with_invalid.ai.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 3)


def test_extract_series_dict(sample_series_dict):
    extracted_df = sample_series_dict.ai.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 3)
    assert extracted_df.index.equals(sample_series_dict.index)


def test_extract_series_without_name(sample_series_fruit_model):
    # Test extraction when Series has no name
    series_no_name = sample_series_fruit_model.copy()
    series_no_name.name = None
    extracted_df = series_no_name.ai.extract()
    expected_columns = ["color", "flavor", "taste"]  # No prefix
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 3)


def test_extract_dataframe(sample_df_extract):
    extracted_df = sample_df_extract.ai.extract("fruit")
    expected_columns = ["name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 4)
    assert extracted_df.index.equals(sample_df_extract.index)


def test_extract_dataframe_dict(sample_df_extract_dict):
    extracted_df = sample_df_extract_dict.ai.extract("fruit")
    expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 4)
    assert extracted_df.index.equals(sample_df_extract_dict.index)


def test_extract_dataframe_dict_with_none(sample_df_extract_dict_with_none):
    extracted_df = sample_df_extract_dict_with_none.ai.extract("fruit")
    expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 4)


def test_extract_dataframe_with_invalid(sample_df_extract_with_invalid):
    # Test DataFrame.ai.extract with a column containing non-dict/BaseModel data
    # It should raise a warning but produce NaNs for the invalid row's extracted columns
    extracted_df = sample_df_extract_with_invalid.ai.extract("fruit")
    expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 4)


def test_count_tokens(sample_df):
    num_tokens: pd.Series = sample_df.name.ai.count_tokens()
    assert all(isinstance(num_token, int) for num_token in num_tokens)
    assert num_tokens.name == "num_tokens"
    assert num_tokens.shape == (3,)
    assert num_tokens.index.equals(sample_df.index)
