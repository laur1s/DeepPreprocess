from unittest import TestCase

from DeepPreprocess import DeepPreprocess


class TestDeepPreprocess(TestCase):
    def test_normalize(self):
        p = DeepPreprocess(1, 2)
        train_norm, test_norm = p.normalize([1, 0, 2], [1])
        print (train_norm, test_norm)

    def test_test_train_val_split(self):
        p = DeepPreprocess(1,2)
        train,test,val = p.test_train_val_split([1,2,3,4,5,6], 0.2, 0.2)
        self.assertEqual(train, 4)
        self.assertEqual(test, 1)
        self.assertEqual(val, 1)


