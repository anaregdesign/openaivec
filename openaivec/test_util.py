from typing import List
from unittest import TestCase

from openai import BaseModel
from pyspark.sql.types import StructField, IntegerType, StringType, ArrayType, FloatType, StructType

from openaivec.util import (
    split_to_minibatch,
    map_minibatch,
    map_unique,
    map_unique_minibatch,
    map_unique_minibatch_parallel,
    map_minibatch_parallel,
    pydantic_to_spark_schema,
)


class TestMappingFunctions(TestCase):

    def test_split_to_minibatch_normal(self):
        b = [1, 2, 3, 4, 5]
        batch_size = 2
        expected = [[1, 2], [3, 4], [5]]
        self.assertEqual(split_to_minibatch(b, batch_size), expected)

    def test_split_to_minibatch_empty(self):
        b: List[int] = []
        batch_size = 3
        expected: List[List[int]] = []
        self.assertEqual(split_to_minibatch(b, batch_size), expected)

    def test_map_minibatch(self):
        # Function that doubles each element in the batch.
        def double_list(lst: List[int]) -> List[int]:
            return [x * 2 for x in lst]

        b = [1, 2, 3, 4, 5]
        batch_size = 2
        # Batches: [1,2] -> [2,4], [3,4] -> [6,8], [5] -> [10]
        expected = [2, 4, 6, 8, 10]
        self.assertEqual(map_minibatch(b, batch_size, double_list), expected)

    def test_map_minibatch_parallel(self):
        # Function that squares each element in the batch.
        def square_list(lst: List[int]) -> List[int]:
            return [x * x for x in lst]

        b = [1, 2, 3, 4, 5]
        batch_size = 2
        # Batches: [1,2] -> [1,4], [3,4] -> [9,16], [5] -> [25]
        expected = [1, 4, 9, 16, 25]
        self.assertEqual(map_minibatch_parallel(b, batch_size, square_list), expected)

    def test_map_minibatch_batch_size_one(self):
        # Identity function: returns the list as is.
        def identity(lst: List[int]) -> List[int]:
            return lst

        b = [1, 2, 3, 4]
        batch_size = 1
        expected = [1, 2, 3, 4]
        self.assertEqual(map_minibatch(b, batch_size, identity), expected)

    def test_map_minibatch_batch_size_greater_than_list(self):
        def identity(lst: List[int]) -> List[int]:
            return lst

        b = [1, 2, 3]
        batch_size = 5
        expected = [1, 2, 3]
        self.assertEqual(map_minibatch(b, batch_size, identity), expected)

    def test_map_unique(self):
        # Function that squares each element.
        def square_list(lst: List[int]) -> List[int]:
            return [x * x for x in lst]

        b = [3, 2, 3, 1]
        # Unique order preserved using dict.fromkeys: [3, 2, 1]
        # After applying f: [9, 4, 1]
        # Mapping back for original list: [9, 4, 9, 1]
        expected = [9, 4, 9, 1]
        self.assertEqual(map_unique(b, square_list), expected)

    def test_map_unique_minibatch(self):
        # Function that doubles each element.
        def double_list(lst: List[int]) -> List[int]:
            return [x * 2 for x in lst]

        b = [1, 2, 1, 3]
        batch_size = 2
        # Unique order: [1, 2, 3]
        # Using map_minibatch on unique values:
        #  Split [1,2,3] with batch_size=2 -> [[1,2], [3]]
        #  Apply function: [[2,4], [6]] -> flattened to [2,4,6]
        # Mapping back for original list: [2, 4, 2, 6]
        expected = [2, 4, 2, 6]
        self.assertEqual(map_unique_minibatch(b, batch_size, double_list), expected)

    def test_map_unique_minibatch_parallel(self):
        # Function that squares each element.
        def square_list(lst: List[int]) -> List[int]:
            return [x * x for x in lst]

        b = [3, 2, 3, 1]
        batch_size = 2
        # Unique order preserved using dict.fromkeys: [3, 2, 1]
        # After applying f: [9, 4, 1]
        # Mapping back for original list: [9, 4, 9, 1]
        expected = [9, 4, 9, 1]
        self.assertEqual(map_unique_minibatch_parallel(b, batch_size, square_list), expected)

    def test_pydantic_to_spark_schema(self):
        class InnerModel(BaseModel):
            inner_id: int
            description: str

        class OuterModel(BaseModel):
            id: int
            name: str
            values: List[float]
            inner: InnerModel

        schema = pydantic_to_spark_schema(OuterModel)

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
