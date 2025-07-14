import os
from typing import List
from unittest import TestCase

from pydantic import BaseModel
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec.spark import (
    EmbeddingsUDFBuilder,
    ResponsesUDFBuilder,
    TaskUDFBuilder,
    _pydantic_to_spark_schema,
    count_tokens_udf,
    similarity_udf,
)
from openaivec.task import nlp


class TestUDFBuilder(TestCase):
    def setUp(self):
        self.responses = ResponsesUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4.1-nano",
        )
        self.embeddings = EmbeddingsUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        self.task_builder = TaskUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4.1-nano",
        )
        self.spark: SparkSession = SparkSession.builder \
            .appName("TestTaskUDF") \
            .master("local[*]") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.adaptive.enabled", "false") \
            .getOrCreate()
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

    def test_task_sentiment_analysis(self):
        self.spark.udf.register(
            "analyze_sentiment",
            self.task_builder.build(task=nlp.SENTIMENT_ANALYSIS),
        )
        
        text_data = [
            ("I love this product!",),
            ("This is terrible and disappointing.",),
            ("It's okay, nothing special.",),
        ]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("reviews")

        df = self.spark.sql(
            """
            with t as (SELECT analyze_sentiment(text) as sentiment from reviews)
            select sentiment.sentiment, sentiment.confidence, sentiment.polarity from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 3)


class TestTaskUDFBuilderUnit(TestCase):
    """Unit tests for TaskUDFBuilder that don't require Spark execution."""
    
    def test_task_udf_builder_creation(self):
        """Test TaskUDFBuilder can be created with various configurations."""
        # Test OpenAI builder
        openai_builder = TaskUDFBuilder.of_openai(
            api_key="test-key",
            model_name="gpt-4o-mini",
        )
        self.assertEqual(openai_builder.api_key, "test-key")
        self.assertEqual(openai_builder.model_name, "gpt-4o-mini")
        self.assertIsNone(openai_builder.endpoint)
        self.assertIsNone(openai_builder.api_version)
        
        # Test Azure builder
        azure_builder = TaskUDFBuilder.of_azure_openai(
            api_key="azure-key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            model_name="gpt-4o-deployment",
        )
        self.assertEqual(azure_builder.api_key, "azure-key")
        self.assertEqual(azure_builder.endpoint, "https://test.openai.azure.com")
        self.assertEqual(azure_builder.api_version, "2024-02-01")
        self.assertEqual(azure_builder.model_name, "gpt-4o-deployment")

    def test_task_serialization_in_build(self):
        """Test that PreparedTask is properly serialized within build method."""
        from openaivec.task import nlp
        from openaivec.serialize import serialize_base_model, deserialize_base_model
        
        # Test that we can serialize/deserialize the task's response format
        task = nlp.SENTIMENT_ANALYSIS
        serialized = serialize_base_model(task.response_format)
        deserialized = deserialize_base_model(serialized)
        
        # Check that sentiment analysis fields are preserved
        schema = deserialized.model_json_schema()
        self.assertIn('sentiment', schema['properties'])
        self.assertIn('confidence', schema['properties'])
        self.assertIn('emotions', schema['properties'])
        
        # Check descriptions are preserved
        self.assertIsNotNone(schema['properties']['sentiment'].get('description'))
        self.assertIsNotNone(schema['properties']['confidence'].get('description'))


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


class TestSimilarityUDF(TestCase):
    def setUp(self):
        self.spark: SparkSession = SparkSession.builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")
        self.spark.udf.register("similarity", similarity_udf())

    def test_similarity(self):
        df = self.spark.createDataFrame(
            [
                (1, [0.1, 0.2, 0.3]),
                (2, [0.4, 0.5, 0.6]),
                (3, [0.7, 0.8, 0.9]),
            ],
            ["id", "vector"],
        )
        df.createOrReplaceTempView("vectors")
        result_df = self.spark.sql(
            """
            SELECT id, similarity(vector, vector) as similarity_score
            FROM vectors
            """
        )
        result_df.show(truncate=False)
        df_pandas = result_df.toPandas()
        assert df_pandas.shape == (3, 2)
