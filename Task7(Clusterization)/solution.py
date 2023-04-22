import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    distances = sklearn.metrics.pairwise_distances(x)
    labels_stack = np.tile(labels, [x.shape[0], 1])
    pair_labels = (labels_stack == labels_stack.T)
    s = np.sum(distances * pair_labels, axis=1, keepdims=True)
    clst_sizes = np.count_nonzero(pair_labels, axis=1, keepdims=True)
    clst_sizes_reduced = np.where(clst_sizes > 1, clst_sizes - 1, clst_sizes)
    s /= clst_sizes_reduced
    s *= (clst_sizes != 1)
    num_clst = np.unique(labels).size
    clsts = np.unique(labels)
    d = np.zeros((x.shape[0], num_clst))
    for i, c in enumerate(clsts):
        d_i = np.sum(distances * (labels_stack == c), axis=1, keepdims=True)
        c_size = np.count_nonzero(labels_stack == c, axis=1, keepdims=True)
        d_i /= c_size
        d_i[labels.T == c] = 1e20
        d[:, i] = d_i.T
    d = np.min(d, axis=1, keepdims=True)
    max_d_s = np.maximum(d, s)
    max_d_s[max_d_s == 0] = 1
    sil_score = (d - s)/max_d_s
    sil_score[d == 1e20] = 0
    sil_score[s == 0] = 0
    return np.mean(sil_score)


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    pred_labels = np.expand_dims(predicted_labels, axis=0)
    pred_same_clst = np.tile(predicted_labels, [predicted_labels.shape[0], 1])
    pred_same_clst = pred_same_clst == pred_labels.T

    true_labels_exp = np.expand_dims(true_labels, axis=0)
    true_same_clst = np.tile(true_labels, [true_labels.shape[0], 1])
    true_same_clst = true_same_clst == true_labels_exp.T

    correct = pred_same_clst * true_same_clst

    precision = np.mean(correct, where=pred_same_clst, axis=1)
    precision = np.mean(precision)

    recall = np.mean(correct, where=true_same_clst, axis=1)
    recall = np.mean(recall)

    if precision + recall == 0:
        score = 0
    else:
        score = 2 * precision * recall / (precision + recall)
    return score
