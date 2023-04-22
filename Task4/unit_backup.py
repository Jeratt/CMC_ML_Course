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
        return new_X;


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
        self.inds = X.apply(lambda col: col.map(col.value_counts()))
        self.suc = X.copy()
        self.suc.insert(X.shape[1], X.shape[1], Y)
        for i in range(self.suc.shape[1] - 1):
            self.suc.iloc[:, i] = X.iloc[:, i].map(self.suc.groupby(by=self.suc.columns[-1])[X.shape[1]].sum())
        self.suc.drop(self.suc.columns[-1], inplace=True, axis=1)



    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        print("X.shape: ", X.shape)
        successes = self.suc.to_numpy() / self.inds.to_numpy()
        counters = self.inds / X.shape[0]
        relation = (successes + a)/(counters + b)
        new_X = np.zeros((X.shape[0], X.shape[1]*3))
        print("New_X.shape:", new_X.shape)
        for i in range(new_X.shape[1] // 3):
            new_X[:, i*3] = successes[:, i]
            new_X[:, i*3 + 1] = counters[:, i]
            new_X[:, i*3+2] = relation[:, i]
        return new_X;

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
        # your code here

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        # your code here

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    # your code here
