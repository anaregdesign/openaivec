from typing import List
from unittest import TestCase

from openaivec.util import (
    split_to_minibatch,
    map_minibatch,
    map_unique,
    map_unique_minibatch,
    map_unique_minibatch_parallel,
    map_minibatch_parallel,
)


class TestMappingFunctions(TestCase):

    def test_split_to_minibatch_normal(self):
        """Test splitting a list into minibatches in a normal scenario.

        Ensures that a list whose length is not evenly divisible by the batch size is split correctly.
        """
        b = [1, 2, 3, 4, 5]
        batch_size = 2
        expected = [[1, 2], [3, 4], [5]]
        self.assertEqual(split_to_minibatch(b, batch_size), expected)

    def test_split_to_minibatch_empty(self):
        """Test splitting an empty list.

        Verifies that an empty list returns an empty set of minibatches.
        """
        b: List[int] = []
        batch_size = 3
        expected: List[List[int]] = []
        self.assertEqual(split_to_minibatch(b, batch_size), expected)

    def test_map_minibatch(self):
        """Test mapping a function over minibatches.

        Checks that elements are correctly processed in each minibatch when doubled.
        """

        def double_list(lst: List[int]) -> List[int]:
            return [x * 2 for x in lst]

        b = [1, 2, 3, 4, 5]
        batch_size = 2
        expected = [2, 4, 6, 8, 10]
        self.assertEqual(map_minibatch(b, batch_size, double_list), expected)

    def test_map_minibatch_parallel(self):
        """Test parallel mapping of a function over minibatches.

        Ensures that squaring each element in parallel produces the correct results.
        """

        def square_list(lst: List[int]) -> List[int]:
            return [x * x for x in lst]

        b = [1, 2, 3, 4, 5]
        batch_size = 2
        expected = [1, 4, 9, 16, 25]
        self.assertEqual(map_minibatch_parallel(b, batch_size, square_list), expected)

    def test_map_minibatch_batch_size_one(self):
        """Test mapping with a batch size of one.

        Confirms that the identity function returns the original list without changes.
        """

        def identity(lst: List[int]) -> List[int]:
            return lst

        b = [1, 2, 3, 4]
        batch_size = 1
        expected = [1, 2, 3, 4]
        self.assertEqual(map_minibatch(b, batch_size, identity), expected)

    def test_map_minibatch_batch_size_greater_than_list(self):
        """Test mapping when the batch size exceeds the list length.

        Checks that the entire list is returned unchanged when the batch size is larger than the list.
        """

        def identity(lst: List[int]) -> List[int]:
            return lst

        b = [1, 2, 3]
        batch_size = 5
        expected = [1, 2, 3]
        self.assertEqual(map_minibatch(b, batch_size, identity), expected)

    def test_map_unique(self):
        """Test mapping with unique value preservation.

        Validates that duplicate elements are processed correctly while maintaining their original positions.
        """

        def square_list(lst: List[int]) -> List[int]:
            return [x * x for x in lst]

        b = [3, 2, 3, 1]
        expected = [9, 4, 9, 1]
        self.assertEqual(map_unique(b, square_list), expected)

    def test_map_unique_minibatch(self):
        """Test mapping unique elements via minibatches.

        Ensures that unique elements are processed in minibatches and mapped back to the original order.
        """

        def double_list(lst: List[int]) -> List[int]:
            return [x * 2 for x in lst]

        b = [1, 2, 1, 3]
        batch_size = 2
        expected = [2, 4, 2, 6]
        self.assertEqual(map_unique_minibatch(b, batch_size, double_list), expected)

    def test_map_unique_minibatch_parallel(self):
        """Test parallel mapping of unique elements via minibatches.

        Confirms that parallel processing of unique elements produces correct results.
        """

        def square_list(lst: List[int]) -> List[int]:
            return [x * x for x in lst]

        b = [3, 2, 3, 1]
        batch_size = 2
        expected = [9, 4, 9, 1]
        self.assertEqual(map_unique_minibatch_parallel(b, batch_size, square_list), expected)
