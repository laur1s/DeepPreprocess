from unittest import TestCase
import numpy as np

from DeepPreprocess import DeepPreprocess


class TestDeepPreprocess(TestCase):
    def test_normalize(self):
        p = DeepPreprocess(1, 2)
        train_norm, test_norm = p.normalize([1, 0, 2], [1])
        # test the train data
        self.assertAlmostEqual(train_norm.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(train_norm.std(), 1.0, delta=0.01)
        # test the test data
        self.assertAlmostEqual(test_norm.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(test_norm.std(), 1.0, delta=0.01)

    def test_test_train_val_split(self):
        p = DeepPreprocess(1,2)
        train,test,val = p.test_train_val_split([1,2,3,4,5,6], 0.2, 0.2)
        self.assertEqual(train, 4)
        self.assertEqual(test, 1)
        self.assertEqual(val, 1)

    def test_get_one_hot(self):
        labels = [1, 2, 3]
        expected = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        p = DeepPreprocess(0, labels)
        result = p.get_one_hot()
        np.testing.assert_array_equal(expected, result)
