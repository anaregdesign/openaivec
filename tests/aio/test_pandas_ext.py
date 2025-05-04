import pytest
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel
import pandas as pd

# Import the module to test
from openaivec.aio import pandas_ext


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


@pytest.mark.asyncio
async def test_embeddings(sample_df):
    # Use .aio for the async embeddings method
    embeddings: pd.Series = await sample_df["name"].aio.embeddings()
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert embeddings.shape == (3,)
    assert embeddings.index.equals(sample_df.index)


@pytest.mark.asyncio
async def test_responses_series(sample_df):
    # Use .aio for the async responses method
    names_fr: pd.Series = await sample_df["name"].aio.responses("translate to French")
    assert all(isinstance(x, str) for x in names_fr)
    assert names_fr.shape == (3,)
    assert names_fr.index.equals(sample_df.index)


@pytest.mark.asyncio
async def test_responses_dataframe(sample_df):
    # Test DataFrame.aio.responses
    # Use .aio for the async responses method
    names_fr: pd.Series = await sample_df.aio.responses("translate the 'name' field to French")
    assert all(isinstance(x, str) for x in names_fr)
    assert names_fr.shape == (3,)
    assert names_fr.index.equals(sample_df.index)


def test_extract_series_model(sample_series_fruit_model):
    # Use .aio for extract method
    extracted_df = sample_series_fruit_model.aio.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 3)
    assert extracted_df.index.equals(sample_series_fruit_model.index)


def test_extract_series_model_with_none(sample_series_fruit_model_with_none):
    # Use .aio for extract method
    extracted_df = sample_series_fruit_model_with_none.aio.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 3)


def test_extract_series_model_with_invalid(sample_series_fruit_model_with_invalid):
    # Use .aio for extract method
    extracted_df = sample_series_fruit_model_with_invalid.aio.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 3)


def test_extract_series_dict(sample_series_dict):
    # Use .aio for extract method
    extracted_df = sample_series_dict.aio.extract()
    expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 3)
    assert extracted_df.index.equals(sample_series_dict.index)


def test_extract_series_without_name(sample_series_fruit_model):
    # Test extraction when Series has no name
    series_no_name = sample_series_fruit_model.copy()
    series_no_name.name = None
    # Use .aio for extract method
    extracted_df = series_no_name.aio.extract()
    expected_columns = ["color", "flavor", "taste"]  # No prefix
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 3)


def test_extract_dataframe(sample_df_extract):
    # Use .aio for extract method
    extracted_df = sample_df_extract.aio.extract("fruit")
    expected_columns = ["name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 4)
    assert extracted_df.index.equals(sample_df_extract.index)


def test_extract_dataframe_dict(sample_df_extract_dict):
    # Use .aio for extract method
    extracted_df = sample_df_extract_dict.aio.extract("fruit")
    expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.shape == (3, 4)
    assert extracted_df.index.equals(sample_df_extract_dict.index)


def test_extract_dataframe_dict_with_none(sample_df_extract_dict_with_none):
    # Use .aio for extract method
    extracted_df = sample_df_extract_dict_with_none.aio.extract("fruit")
    expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 4)


def test_extract_dataframe_with_invalid(sample_df_extract_with_invalid):
    # Test DataFrame.aio.extract with a column containing non-dict/BaseModel data
    # It should raise a warning but produce NaNs for the invalid row's extracted columns
    # Use .aio for extract method
    extracted_df = sample_df_extract_with_invalid.aio.extract("fruit")
    expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
    assert list(extracted_df.columns) == expected_columns
    assert extracted_df.iloc[1].isna().all()
    assert extracted_df.shape == (3, 4)


def test_count_tokens(sample_df):
    # Use .aio for count_tokens method
    num_tokens: pd.Series = sample_df.name.aio.count_tokens()
    assert all(isinstance(num_token, int) for num_token in num_tokens)
    assert num_tokens.name == "num_tokens"
    assert num_tokens.shape == (3,)
    assert num_tokens.index.equals(sample_df.index)


@pytest.mark.asyncio
async def test_async_pipe(sample_df):
    # Use .aio for async pipe method
    async def dummy_func(df: pd.DataFrame) -> pd.DataFrame:
        return df

    result = await sample_df.aio.pipe(dummy_func)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_df.shape
    assert result.index.equals(sample_df.index)


@pytest.mark.asyncio
async def test_async_pipe_with_sync(sample_df):
    # Use .aio for async pipe method with a sync function
    def dummy_func(df: pd.DataFrame) -> pd.DataFrame:
        return df

    result = await sample_df.aio.pipe(dummy_func)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_df.shape
    assert result.index.equals(sample_df.index)


@pytest.mark.asyncio
async def test_async_assign(sample_df):
    # Use .aio for async assign method
    async def dummy_func(df: pd.DataFrame) -> pd.Series:
        return df["name"].str.upper()

    async def static_value_func(df: pd.DataFrame) -> str:
        return "static_value"

    result = await sample_df.aio.assign(
        upper_name=dummy_func,
        static_value=static_value_func,
        first_letter=lambda df: df["name"].map(lambda x: x[0]),
    )
    assert isinstance(result, pd.DataFrame)
    assert "upper_name" in result.columns
    assert "first_letter" in result.columns
    assert "static_value" in result.columns
    assert result["upper_name"].equals(sample_df["name"].str.upper())
    assert result["first_letter"].equals(sample_df["name"].map(lambda x: x[0]))
    assert result["static_value"].equals(pd.Series(["static_value"] * len(sample_df)))
