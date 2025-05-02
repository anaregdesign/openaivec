import os
from unittest import TestCase

from openai import BaseModel
from pyspark.sql.session import SparkSession

from openaivec.aio.spark import EmbeddingsUDFBuilder, ResponsesUDFBuilder


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

        df.show()

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
            select name, info.name, info.color, info.taste from t
            """
        )

        df.show()

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

        df.show()
