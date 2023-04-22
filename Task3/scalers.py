import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mins = data.min(axis=0)
        self.div = data.max(axis=0) - self.mins

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mins) / self.div


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mean) / self.std
