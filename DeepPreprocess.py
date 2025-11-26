import numpy as np

class DeepPreprocess:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.n_class = np.max(labels) + 1

    @staticmethod
    def normalize(train_data, test_data):
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

    @staticmethod
    def test_train_val_split(data, test_perc, val_perc):
        """
        Splits the data into training, testing, and validation sets.
        :param data: The data to be split.
        :param test_perc: The percentage of data to be used for testing.
        :param val_perc: The percentage of data to be used for validation.
        :return: A tuple containing the training, testing, and validation data.
        """

        # shuffle the data
        np.random.shuffle(data)
        len_test_data = int(len(data) * test_perc)
        len_val_data = int(len(data) * val_perc)

        test_data = data[:len_test_data]
        val_data = data[len_test_data:len_test_data + len_val_data]
        train_data = data[len_test_data + len_val_data:]

        return train_data, test_data, val_data

    def get_one_hot(self):
        """Get one hot encoding

            if labels = np.array([1,2,3])

            it will return 

            [[ 0.  1.  0.  0.]
            [ 0.  0.  1.  0.]
            [ 0.  0.  0.  1.]]

        """
        return np.eye(self.n_class)[self.labels]
