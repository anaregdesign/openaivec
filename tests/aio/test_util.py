import asyncio
from unittest import TestCase
from typing import List

from openaivec.aio.util import MinibatchUtil


# Async function for testing
async def async_double(inputs: List[int]) -> List[int]:
    await asyncio.sleep(0.01)  # Simulate asynchronous processing
    return [i * 2 for i in inputs]


# Async function for testing (returns incorrect number of outputs)
async def async_invalid_output(inputs: List[int]) -> List[int]:
    await asyncio.sleep(0.01)
    return [i * 2 for i in inputs[:-1]]  # Intentionally remove the last element


class TestMinibatchUtil(TestCase):
    def setUp(self):
        """Setup executed before each test method"""
        self.minibatch_util = MinibatchUtil[int, int](batch_size=2)

    def test_set_inputs(self):
        """Test if set_inputs works correctly"""
        inputs = [1, 2, 3, 2, 4]
        self.minibatch_util.set_inputs(inputs)

        self.assertEqual(self.minibatch_util.inputs, inputs)
        self.assertEqual(len(self.minibatch_util.hashes), len(inputs))
        # Deduplication (last occurrence takes precedence)
        expected_hash_inputs = {
            hash(1): 1,
            hash(2): 2,  # The last 2 is used
            hash(3): 3,
            hash(4): 4,
        }
        self.assertDictEqual(self.minibatch_util.hash_inputs, expected_hash_inputs)
        self.assertEqual(len(self.minibatch_util.hash_inputs), 4)  # Number of unique elements

    def test_map_empty_input(self):
        """Test calling map with an empty input list"""
        self.minibatch_util.set_inputs([])
        asyncio.run(self.minibatch_util.map(async_double))
        self.assertListEqual(self.minibatch_util.outputs, [])

    def test_map_normal(self):
        """Test calling map with a normal input list"""
        inputs = [1, 2, 3, 4, 5]
        self.minibatch_util.set_inputs(inputs)
        asyncio.run(self.minibatch_util.map(async_double))

        self.assertEqual(len(self.minibatch_util.outputs), len(inputs))
        self.assertListEqual(self.minibatch_util.outputs, [2, 4, 6, 8, 10])
        # Check internal state as well (optional)
        self.assertEqual(len(self.minibatch_util.hash_outputs), len(set(inputs)))
        self.assertEqual(self.minibatch_util.hash_outputs[hash(1)], 2)
        self.assertEqual(self.minibatch_util.hash_outputs[hash(5)], 10)

    def test_map_with_duplicates(self):
        """Test calling map with an input list containing duplicates"""
        inputs = [1, 2, 1, 3, 2]  # Contains duplicates
        self.minibatch_util.set_inputs(inputs)
        asyncio.run(self.minibatch_util.map(async_double))

        self.assertEqual(len(self.minibatch_util.outputs), len(inputs))
        # Ensure output corresponds to the original input order
        self.assertListEqual(self.minibatch_util.outputs, [2, 4, 2, 6, 4])
        # Check internal state as well (optional)
        self.assertEqual(len(self.minibatch_util.hash_outputs), 3)  # Number of unique elements (1, 2, 3)
        self.assertEqual(self.minibatch_util.hash_outputs[hash(1)], 2)
        self.assertEqual(self.minibatch_util.hash_outputs[hash(2)], 4)
        self.assertEqual(self.minibatch_util.hash_outputs[hash(3)], 6)
        self.assertListEqual(
            self.minibatch_util.outputs,
            [2, 4, 2, 6, 4],
        )

    def test_map_value_error_on_invalid_output_count(self):
        """Test that ValueError is raised when calling map with a function that returns an incorrect number of outputs"""
        inputs = [1, 2, 3, 4]
        self.minibatch_util.set_inputs(inputs)

        # Use assertRaises within asyncio.run
        async def run_map_with_invalid_output():
            await self.minibatch_util.map(async_invalid_output)

        with self.assertRaises(ValueError) as cm:
            asyncio.run(run_map_with_invalid_output())

        self.assertIn("Number of unique inputs", str(cm.exception))
        self.assertIn("does not match number of unique outputs", str(cm.exception))
