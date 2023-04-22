import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    indices, fold_len = np.array([x for x in range(num_objects)]), num_objects // num_folds
    out = []
    for i in range(num_folds):
        if i + 1 != num_folds:
            out.append(tuple([np.delete(indices, indices[i*fold_len:(i+1)*fold_len].tolist()),
                              indices[i*fold_len:(i+1)*fold_len]]))
        else:
            out.append(tuple([np.delete(indices, indices[i*fold_len:].tolist()),
                              indices[i*fold_len:]]))
    return out


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    out = {}
    fold_out = np.zeros(len(folds))
    for n_neighbors in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weights in parameters['weights']:
                for normalizer in parameters['normalizers']:
                    for i in range(len(folds)):
                        key = (normalizer[1], n_neighbors, metric, weights)
                        knn_classifier = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weights)
                        if normalizer[0] is not None:
                            normalizer[0].fit(X[folds[i][0]])
                            knn_classifier.fit(normalizer[0].transform(X[folds[i][0]]), y[folds[i][0]])
                            fold_out[i] = score_function(y[folds[i][1]],
                                                         knn_classifier.predict(normalizer[0].transform(X[folds[i][1]])))
                        else:
                            knn_classifier.fit(X[folds[i][0]], y[folds[i][0]])
                            fold_out[i] = score_function(y[folds[i][1]], knn_classifier.predict(X[folds[i][1]]))
                    out[key] = fold_out.mean()
    return out
