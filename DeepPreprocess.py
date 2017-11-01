import numpy as np

#test travis
# n_values = np.max(values) + 1
class DeepPreprocess:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.n_class = np.max(labels) + 1

    def normalize(self, train_data, test_data):
        """
        Normalizes training and test data
        """
        train_data = np.asarray(train_data)  # convert python lists to np arrays
        test_data = np.asarray(test_data)

        mean = train_data.mean(axis=0)
        train_data = train_data - mean
        std = train_data.std(axis=0)
        train_data = train_data / std

        test_data = test_data - mean
        test_data = test_data / std

        return train_data, test_data

    def test_train_val_split(self, data, test_perc, val_perc):
        """
        :return:
        """

        # shuffle the data
        np.random.shuffle(data)
        len_test_data = int(len(data) * test_perc)
        len_val_data = int(len(data) * val_perc)
        len_train_data = int(len(data) - len_test_data - len_val_data)
        return len_train_data, len_test_data, len_val_data

    def get_one_hot(self):
        """Get one hot encoding

            if labels = np.array([1,2,3])

            it will return 

            [[ 0.  1.  0.  0.]
            [ 0.  0.  1.  0.]
            [ 0.  0.  0.  1.]]

        """
        return np.eye(self.n_class)[self.labels]
