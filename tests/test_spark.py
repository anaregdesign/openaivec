import os
from typing import List
from unittest import TestCase

from openai import BaseModel
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec.spark import UDFBuilder, _pydantic_to_spark_schema, count_tokens_udf


class TestUDFBuilder(TestCase):
    def setUp(self):
        self.udf = UDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            batch_size=8,
        )
        self.spark: SparkSession = SparkSession.builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")

    def tearDown(self):
        if self.spark:
            self.spark.stop()

    def test_responses(self):
        self.spark.udf.register(
            "repeat",
            self.udf.responses(
                """
                Repeat twice input string.
                """,
            ),
        )
        dummy_df = self.spark.range(31)
        dummy_df.createOrReplaceTempView("dummy")

        self.spark.sql(
            """
            SELECT id, repeat(cast(id as STRING)) as v from dummy
            """
        ).show()

    def test_responses_structured(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        self.spark.udf.register(
            "fruit",
            self.udf.responses(
                instructions="return the color and taste of given fruit",
                response_format=Fruit,
            ),
        )

        fruit_data = [("apple",), ("banana",), ("cherry",)]
        dummy_df = self.spark.createDataFrame(fruit_data, ["name"])
        dummy_df.createOrReplaceTempView("dummy")

        self.spark.sql(
            """
            with t as (SELECT name, fruit(name) as info from dummy)
            select name, info.name, info.color, info.taste from t
            """
        ).show(truncate=False)

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
