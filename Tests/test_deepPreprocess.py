from unittest import TestCase
import numpy as np
from math import sqrt

from DeepPreprocess import DeepPreprocess


class TestDeepPreprocess(TestCase):
    def test_normalize(self):
        train_data = [1, 0, 5]
        test_data = 1
        train_norm, test_norm = DeepPreprocess.normalize(train_data, test_data)
        # test the train data
        self.assertAlmostEqual(train_norm.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(train_norm.std(), 1.0, delta=0.01)

        # test the test data

        mean = sum(train_data) / len(train_data)
        test_data = test_data - mean
        s = sum((x - mean) ** 2 for x in train_data)
        s = s / len(train_data)
        std = sqrt(s)
        test_data = test_data / std
        self.assertAlmostEqual(test_norm, test_data, delta=0.05)

    def test_test_train_val_split(self):
        train, test, val = DeepPreprocess.test_train_val_split(
            [1, 2, 3, 4, 5, 6], 0.2, 0.2)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 1)
        self.assertEqual(len(val), 1)

    def test_get_one_hot(self):
        labels = [1, 2, 3]
        expected = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        p = DeepPreprocess(0, labels)
        result = p.get_one_hot()
        np.testing.assert_array_equal(expected, result)
