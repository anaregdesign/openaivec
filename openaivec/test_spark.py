from pathlib import Path
from unittest import TestCase

from pyspark.sql.session import SparkSession

from openaivec.spark import UDFBuilder


class TestUDFBuilder(TestCase):
    def setUp(self):
        project_root = Path(__file__).parent.parent
        policy_path = project_root / "spark.policy"
        self.udf = UDFBuilder.of_environment(batch_size=8)
        self.spark: SparkSession = (
            SparkSession.builder.appName("test")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config(
                "spark.driver.extraJavaOptions",
                "-Djava.security.manager "
                + f"-Djava.security.policy=file://{policy_path} "
                + "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED "
                + "--add-opens=java.base/java.nio=ALL-UNNAMED "
                + "-Darrow.enable_unsafe=true",
            )
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("INFO")

    def tearDown(self):
        if self.spark:
            self.spark.stop()

    def test_completion(self):
        self.spark.udf.register(
            "repeat",
            self.udf.completion(
                """
            Just repeat input string.
        """
            ),
        )
        dummy_df = self.spark.range(31)
        dummy_df.createOrReplaceTempView("dummy")

        self.spark.sql(
            """
            SELECT repeat(cast(id as STRING)) as v from dummy
            """
        ).show()
