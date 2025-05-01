import logging
import unittest
from typing import List, Any
import time
import asyncio

from openaivec.aio.util import map


# Helper async function for testing
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


class TestAioMap(unittest.TestCase):
    def test_empty_list(self):
        inputs = []
        outputs = asyncio.run(map(inputs, double_items))
        self.assertEqual(outputs, [])

    def test_smaller_than_batch_size(self):
        inputs = [1, 2, 3]
        outputs = asyncio.run(map(inputs, double_items, batch_size=5))
        self.assertEqual(outputs, [2, 4, 6])

    def test_multiple_batches(self):
        inputs = [1, 2, 3, 4, 5, 6]
        outputs = asyncio.run(map(inputs, double_items, batch_size=2))
        self.assertEqual(outputs, [2, 4, 6, 8, 10, 12])

    def test_with_duplicates(self):
        inputs = [1, 2, 1, 3, 2, 3]
        outputs = asyncio.run(map(inputs, double_items, batch_size=2))
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
        outputs = asyncio.run(map(inputs, process_objects, batch_size=2))
        self.assertEqual(outputs, ["Processed: a", "Processed: b", "Processed: a"])

    def test_batch_size_one(self):
        inputs = [1, 2, 3]
        outputs = asyncio.run(map(inputs, double_items, batch_size=1))
        self.assertEqual(outputs, [2, 4, 6])

    def test_function_raises_exception(self):
        inputs = [1, 2, 3]
        with self.assertRaises(ValueError) as cm:
            asyncio.run(map(inputs, raise_exception, batch_size=2))
        self.assertEqual(str(cm.exception), "Test exception")

    def test_function_returns_wrong_count(self):
        inputs = [1, 2, 3, 4]
        with self.assertRaises(ValueError) as cm:
            asyncio.run(map(inputs, return_wrong_count, batch_size=2))
        self.assertTrue("does not match number of unique outputs" in str(cm.exception))

    def test_string_inputs(self):
        inputs = ["a", "b", "c", "a"]
        outputs = asyncio.run(map(inputs, double_items_str, batch_size=2))
        self.assertEqual(outputs, ["aa", "bb", "cc", "aa"])

    def test_large_input_list(self):
        inputs = list(range(1000))
        start_time = time.time()
        outputs = asyncio.run(map(inputs, double_items, batch_size=50))
        end_time = time.time()
        self.assertEqual(outputs, [i * 2 for i in range(1000)])
        logging.info(f"Large list test took {end_time - start_time:.2f} seconds")
        self.assertLess(end_time - start_time, 5)
