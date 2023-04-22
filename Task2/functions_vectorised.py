import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    out = X.diagonal()
    if(np.sum(out>=0)):
      return out[out >= 0].sum()
    else:
      return -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    pair_arr = np.roll(x, -1) * x
    pair_arr[-1] = -1
    if np.sum(pair_arr % 3 == 0):
        return pair_arr[pair_arr % 3 == 0].max()
    else:
        return -1



def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    im_h, im_w, im_c = image.shape
    y = np.reshape(weights, (im_c, 1, 1))
    np.transpose(image.T * y, (0, 2, 1)).sum(axis=0)
    return np.transpose(image.T * y, (0, 2, 1)).sum(axis=0)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_unpacked = np.repeat(x.T[0], x.T[1])
    y_unpacked = np.repeat(y.T[0], y.T[1])
    if x_unpacked.shape != y_unpacked.shape:
        return -1
    else:
        return x_unpacked @ y_unpacked



def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    X_norm = np.expand_dims(np.linalg.norm(X, axis=1), 0)
    Y_norm = np.expand_dims(np.linalg.norm(Y, axis=1), 0)
    scal = X @ Y.T
    norms = X_norm.T @ Y_norm
    norms_without_zeros = norms.copy()
    norms_without_zeros[norms == 0] = 1
    out = scal / norms_without_zeros
    out[norms == 0] = 1
    return out
