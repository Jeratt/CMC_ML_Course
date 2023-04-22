import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.val_count = X.nunique().to_numpy()
        self.unique_elem = np.array([])
        for col in X.items():
            self.unique_elem = np.concatenate((self.unique_elem, np.sort(col[1].unique())))

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        new_X = np.repeat(X.to_numpy(), self.val_count, axis=1)
        for i in range(X.shape[0]):
            new_X[i] = (new_X[i] == self.unique_elem).astype(int)
        return new_X

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.params = []

        for i in range(X.shape[1]):
            self.params.append({})
            unique_val = X.iloc[:, i].unique()
            for j in range(len(unique_val)):
                self.params[i][unique_val[j]] = {}
                self.params[i][unique_val[j]]['inds'] = (X.iloc[:, i] == unique_val[j]).sum()
                self.params[i][unique_val[j]]['sum_y'] = (Y[X.iloc[:, i] == unique_val[j]].sum())
                self.params[i][unique_val[j]]['suc'] = self.params[i][unique_val[j]]['sum_y'] / \
                    self.params[i][unique_val[j]]['inds']
                self.params[i][unique_val[j]]['counters'] = self.params[i][unique_val[j]]['inds'] / X.shape[0]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        new_X = np.zeros((X.shape[0], X.shape[1] * 3))
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                new_X[j][i * 3] = self.params[i][X.iloc[j, i]]['suc']
                new_X[j][i * 3 + 1] = self.params[i][X.iloc[j, i]]['counters']
                new_X[j][i * 3 + 2] = (new_X[j][i * 3] + a) / (new_X[j][i * 3 + 1] + b)
        return new_X

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.folds = [(u, v) for u, v in group_k_fold(X.shape[0], n_splits=self.n_folds, seed=seed)]
        self.transformers = [SimpleCounterEncoder(dtype=self.dtype) for i in range(self.n_folds)]
        for i in range(self.n_folds):
            self.transformers[i].fit(X.iloc[self.folds[i][1], :], Y.iloc[self.folds[i][1]])

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        new_X = np.zeros((X.shape[0], X.shape[1] * 3))
        for i in range(self.n_folds):
            new_X[self.folds[i][0], :] = self.transformers[i].transform(X.iloc[self.folds[i][0], :], a, b)
        return new_X

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    unique_values = np.unique(x)
    out = np.zeros_like(unique_values, dtype='f')
    for i in range(len(unique_values)):
        out[i] = (y[x == unique_values[i]].sum() / (x == unique_values[i]).sum())
    return out
