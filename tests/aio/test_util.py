import asyncio
from unittest import IsolatedAsyncioTestCase
from typing import List, Any
import time

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


class TestAioMap(IsolatedAsyncioTestCase):
    async def test_empty_list(self):
        inputs = []
        outputs = await map(inputs, double_items)
        self.assertEqual(outputs, [])

    async def test_smaller_than_batch_size(self):
        inputs = [1, 2, 3]
        outputs = await map(inputs, double_items, batch_size=5)
        self.assertEqual(outputs, [2, 4, 6])

    async def test_multiple_batches(self):
        inputs = [1, 2, 3, 4, 5, 6]
        outputs = await map(inputs, double_items, batch_size=2)
        self.assertEqual(outputs, [2, 4, 6, 8, 10, 12])

    async def test_with_duplicates(self):
        inputs = [1, 2, 1, 3, 2, 3]
        outputs = await map(inputs, double_items, batch_size=2)
        # Should return results in the original order, respecting duplicates
        self.assertEqual(outputs, [2, 4, 2, 6, 4, 6])

    async def test_with_custom_objects(self):
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
        outputs = await map(inputs, process_objects, batch_size=2)
        self.assertEqual(outputs, ["Processed: a", "Processed: b", "Processed: a"])

    async def test_batch_size_one(self):
        inputs = [1, 2, 3]
        outputs = await map(inputs, double_items, batch_size=1)
        self.assertEqual(outputs, [2, 4, 6])

    async def test_function_raises_exception(self):
        inputs = [1, 2, 3]
        with self.assertRaises(ValueError) as cm:
            await map(inputs, raise_exception, batch_size=2)
        self.assertEqual(str(cm.exception), "Test exception")

    async def test_function_returns_wrong_count(self):
        inputs = [1, 2, 3, 4]
        with self.assertRaises(ValueError) as cm:
            await map(inputs, return_wrong_count, batch_size=2)
        # The exact error message might depend on which batch fails first
        self.assertTrue("does not match number of unique outputs" in str(cm.exception))

    async def test_string_inputs(self):
        inputs = ["a", "b", "c", "a"]
        outputs = await map(inputs, double_items_str, batch_size=2)
        self.assertEqual(outputs, ["aa", "bb", "cc", "aa"])

    async def test_large_input_list(self):
        inputs = list(range(1000))
        start_time = time.time()
        outputs = await map(inputs, double_items, batch_size=50)
        end_time = time.time()
        self.assertEqual(outputs, [i * 2 for i in range(1000)])
        # Check if it runs reasonably fast (e.g., less than a few seconds)
        # This depends on the machine, but ensures concurrency is working
        print(f"Large list test took {end_time - start_time:.2f} seconds")
        self.assertLess(end_time - start_time, 5)  # Adjust time limit if needed
