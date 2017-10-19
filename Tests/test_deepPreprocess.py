from unittest import TestCase
from DeepPreprocess import DeepPreprocess


class TestDeepPreprocess(TestCase):
    def test_normalize(self):
        self.fail()

    def test_test_train_val_split(self):
        p = DeepPreprocess(1,2)
        a,b,c = p.test_train_val_split([1,2,3,4,5,6], 0.1, 0.1)
        print (a,b,c)

