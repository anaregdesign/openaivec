import asyncio
from typing import Any, List
from unittest import TestCase

import tiktoken
from openai import BaseModel
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec.spark import _pydantic_to_spark_schema
from openaivec.util import (
    TextChunker,
    map_async,
    map_minibatch,
    map_minibatch_parallel,
    map_unique,
    map_unique_minibatch,
    map_unique_minibatch_parallel,
    split_to_minibatch,
)


# Helper async function for testing map
async def double_items(items: List[int]) -> List[int]:
    await asyncio.sleep(0.01)  # Simulate async work
    return [item * 2 for item in items]


async def double_items_str(items: List[str]) -> List[str]:
    await asyncio.sleep(0.01)
    return [item * 2 for item in items]


async def raise_exception(items: List[Any]) -> List[Any]:
    await asyncio.sleep(0.01)
    raise ValueError("Test exception")


async def return_wrong_count(items: List[Any]) -> List[Any]:
    await asyncio.sleep(0.01)
    return items[:-1]  # Return one less item


class TestAioMap(TestCase):
    def test_empty_list(self):
        inputs = []
        outputs = asyncio.run(map_async(inputs, double_items))
        self.assertEqual(outputs, [])

    def test_smaller_than_batch_size(self):
        inputs = [1, 2, 3]
        outputs = asyncio.run(map_async(inputs, double_items, batch_size=5))
        self.assertEqual(outputs, [2, 4, 6])

    def test_multiple_batches(self):
        inputs = [1, 2, 3, 4, 5, 6]
        outputs = asyncio.run(map_async(inputs, double_items, batch_size=2))
        self.assertEqual(outputs, [2, 4, 6, 8, 10, 12])

    def test_with_duplicates(self):
        inputs = [1, 2, 1, 3, 2, 3]
        outputs = asyncio.run(map_async(inputs, double_items, batch_size=2))
        self.assertEqual(outputs, [2, 4, 2, 6, 4, 6])

    def test_with_custom_objects(self):
        class MyObject:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

            def __eq__(self, other):
                return isinstance(other, MyObject) and self.value == other.value

        async def process_objects(items: List[MyObject]) -> List[str]:
            await asyncio.sleep(0.01)
            return [f"Processed: {item.value}" for item in items]

        inputs = [MyObject("a"), MyObject("b"), MyObject("a")]
        outputs = asyncio.run(map_async(inputs, process_objects, batch_size=2))
        self.assertEqual(outputs, ["Processed: a", "Processed: b", "Processed: a"])

    def test_batch_size_one(self):
        inputs = [1, 2, 3]
        outputs = asyncio.run(map_async(inputs, double_items, batch_size=1))
        self.assertEqual(outputs, [2, 4, 6])

    def test_function_raises_exception(self):
        inputs = [1, 2, 3]
        with self.assertRaises(ValueError) as cm:
            asyncio.run(map_async(inputs, raise_exception, batch_size=2))
        self.assertEqual(str(cm.exception), "Test exception")

    def test_function_returns_wrong_count(self):
        inputs = [1, 2, 3, 4]
        with self.assertRaises(ValueError) as cm:
            asyncio.run(map_async(inputs, return_wrong_count, batch_size=2))
        self.assertTrue("does not match number of unique outputs" in str(cm.exception))

    def test_string_inputs(self):
        inputs = ["a", "b", "c", "a"]
        outputs = asyncio.run(map_async(inputs, double_items_str, batch_size=2))
        self.assertEqual(outputs, ["aa", "bb", "cc", "aa"])

    def test_large_input_list(self):
        inputs = list(range(1000))
        start_time = time.time()
        outputs = asyncio.run(map_async(inputs, double_items, batch_size=50))
        end_time = time.time()
        self.assertEqual(outputs, [i * 2 for i in range(1000)])
        logging.info(f"Large list test took {end_time - start_time:.2f} seconds")
        self.assertLess(end_time - start_time, 10)


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


class TestTextChunker(TestCase):
    def setUp(self):
        self.sep = TextChunker(
            enc=tiktoken.encoding_for_model("text-embedding-3-large"),
        )

    def test_split(self):
        text = """
Kubernetes was announced by Google on June 6, 2014.[10] The project was conceived and created by Google employees Joe Beda, Brendan Burns, and Craig McLuckie. Others at Google soon joined to help build the project including Ville Aikas, Dawn Chen, Brian Grant, Tim Hockin, and Daniel Smith.[11][12] Other companies such as Red Hat and CoreOS joined the effort soon after, with notable contributors such as Clayton Coleman and Kelsey Hightower.[10]

The design and development of Kubernetes was inspired by Google's Borg cluster manager and based on Promise Theory.[13][14] Many of its top contributors had previously worked on Borg;[15][16] they codenamed Kubernetes "Project 7" after the Star Trek ex-Borg character Seven of Nine[17] and gave its logo a seven-spoked ship's wheel (designed by Tim Hockin). Unlike Borg, which was written in C++,[15] Kubernetes is written in the Go language.

Kubernetes was announced in June, 2014 and version 1.0 was released on July 21, 2015.[18] Google worked with the Linux Foundation to form the Cloud Native Computing Foundation (CNCF)[19] and offered Kubernetes as the seed technology.

Google was already offering a managed Kubernetes service, GKE, and Red Hat was supporting Kubernetes as part of OpenShift since the inception of the Kubernetes project in 2014.[20] In 2017, the principal competitors rallied around Kubernetes and announced adding native support for it:

VMware (proponent of Pivotal Cloud Foundry)[21] in August,
Mesosphere, Inc. (proponent of Marathon and Mesos)[22] in September,
Docker, Inc. (proponent of Docker)[23] in October,
Microsoft Azure[24] also in October,
AWS announced support for Kubernetes via the Elastic Kubernetes Service (EKS)[25] in November.
Cisco Elastic Kubernetes Service (EKS)[26] in November.
On March 6, 2018, Kubernetes Project reached ninth place in the list of GitHub projects by the number of commits, and second place in authors and issues, after the Linux kernel.[27]

Until version 1.18, Kubernetes followed an N-2 support policy, meaning that the three most recent minor versions receive security updates and bug fixes.[28] Starting with version 1.19, Kubernetes follows an N-3 support policy.[29]
"""

        chunks = self.sep.split(text, max_tokens=256, sep=[".", "\n\n"])

        # Assert that the number of chunks is as expected
        enc = tiktoken.encoding_for_model("text-embedding-3-large")

        for chunk in chunks:
            self.assertLessEqual(len(enc.encode(chunk)), 256)
