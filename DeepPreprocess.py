import numpy as np


class DeepPreprocess:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def normalize (self, train_data, test_data):
        """Normalizes training and test data

        """

        mean = train_data.mean(axis=0)
        train_data = train_data - mean
        std = train_data.std(axis=0)
        train_data = train_data / std

        test_data = test_data - mean
        test_data = test_data / std
        
        return train_data, test_data

