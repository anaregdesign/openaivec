import os
from unittest import TestCase

from openai import BaseModel
from pyspark.sql.session import SparkSession

from openaivec.spark import UDFBuilder, count_tokens_udf


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

    def test_completion(self):
        self.spark.udf.register(
            "repeat",
            self.udf.completion(
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

    def test_completion_structured(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        self.spark.udf.register(
            "fruit",
            self.udf.completion(
                system_message="return the color and taste of given fruit",
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
