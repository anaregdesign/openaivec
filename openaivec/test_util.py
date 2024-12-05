from unittest import TestCase

from openaivec.util import split_to_minibatch, map_with_minibatch


class Test(TestCase):
    def test_split_to_minibatch(self):
        all = [str(i) for i in range(10)]

        result = split_to_minibatch(all, 3)
        self.assertEqual(result, [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8'], ['9']])

    def test_map_with_minibatch(self):
        all = [str(i) for i in range(10)]
        result = map_with_minibatch(all, 3, lambda x: [int(i) for i in x])
        self.assertEqual(result, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
