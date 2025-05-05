import os
from typing import List
from unittest import TestCase

from pydantic import BaseModel
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec.spark import EmbeddingsUDFBuilder, ResponsesUDFBuilder, _pydantic_to_spark_schema, count_tokens_udf


class TestResponsesUDFBuilder(TestCase):
    def setUp(self):
        self.responses = ResponsesUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4.1-nano",
        )
        self.embeddings = EmbeddingsUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        self.spark: SparkSession = SparkSession.builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")

    def tearDown(self):
        if self.spark:
            self.spark.stop()

    def test_responses(self):
        self.spark.udf.register(
            "repeat",
            self.responses.build("Repeat twice input string."),
        )
        dummy_df = self.spark.range(31)
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            SELECT id, repeat(cast(id as STRING)) as v from dummy
            """
        )

        df_pandas = df.toPandas()
        assert df_pandas.shape == (31, 2)

    def test_responses_structured(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        self.spark.udf.register(
            "fruit",
            self.responses.build(
                instructions="return the color and taste of given fruit",
                response_format=Fruit,
            ),
        )

        fruit_data = [("apple",), ("banana",), ("cherry",)]
        dummy_df = self.spark.createDataFrame(fruit_data, ["name"])
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            with t as (SELECT fruit(name) as info from dummy)
            select info.name, info.color, info.taste from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 3)

    def test_embeddings(self):
        self.spark.udf.register(
            "embed",
            self.embeddings.build(batch_size=8),
        )
        dummy_df = self.spark.range(31)
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            SELECT id, embed(cast(id as STRING)) as v from dummy
            """
        )

        df_pandas = df.toPandas()
        assert df_pandas.shape == (31, 2)


class TestMappingFunctions(TestCase):
    def test_pydantic_to_spark_schema(self):
        class InnerModel(BaseModel):
            inner_id: int
            description: str

        class OuterModel(BaseModel):
            id: int
            name: str
            values: List[float]
            inner: InnerModel

        schema = _pydantic_to_spark_schema(OuterModel)

        expected = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), True),
                StructField("values", ArrayType(FloatType(), True), True),
                StructField(
                    "inner",
                    StructType(
                        [StructField("inner_id", IntegerType(), True), StructField("description", StringType(), True)]
                    ),
                    True,
                ),
            ]
        )

        self.assertEqual(schema, expected)


class TestCountTokensUDF(TestCase):
    def setUp(self):
        self.spark: SparkSession = SparkSession.builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")
        self.spark.udf.register(
            "count_tokens",
            count_tokens_udf("gpt-4o"),
        )

    def test_count_token(self):
        self.spark.udf.register(
            "count_tokens",
            count_tokens_udf("gpt-4o"),
        )
        sentences = [
            ("How many tokens in this sentence?",),
            ("Understanding token counts helps optimize language model inputs",),
            ("Tokenization is a crucial step in natural language processing tasks",),
        ]
        dummy_df = self.spark.createDataFrame(sentences, ["sentence"])
        dummy_df.createOrReplaceTempView("sentences")

        self.spark.sql(
            """
            SELECT sentence, count_tokens(sentence) as token_count from sentences
            """
        ).show(truncate=False)
