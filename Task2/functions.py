from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    key = False
    out = 0
    diag = min(len(X),len(X[0]))
    for i in range(diag):
        if X[i][i] >= 0:
            out += X[i][i]
            key = True
    if not key:
        return -1
    else:
        return out


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    superset_x = {i: x.count(i) for i in set(x)}
    superset_y = {i: y.count(i) for i in set(y)}
    for x_val, y_val in zip(superset_x.items(), superset_y.items()):
        if x_val != y_val:
            return False
    return True


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    pair_lst = [x[i] * x[i+1] for i in range(len(x)-1)]
    max_prod = -1
    for p in pair_lst:
        if not p % 3 and p > max_prod:
            max_prod = p
    return max_prod

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    out = [[0]*len(image[0]) for i in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                out[i][j] += image[i][j][k] * weights[k]
    return out



def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_unpacked = [pair[0] for pair in x for i in range(pair[1])]
    y_unpacked = [pair[0] for pair in y for i in range(pair[1])]
    if(len(x_unpacked) != len(y_unpacked)):
        return -1
    else:
        out = [x_unpacked[i]*y_unpacked[i] for i in range(len(x_unpacked))]
    return sum(out)


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    out = [[1]*len(Y) for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            abs_x = sum([x ** 2 for x in X[i]]) ** 0.5
            abs_y = sum([y ** 2 for y in Y[j]]) ** 0.5
            scal = sum([X[i][k]*Y[j][k] for k in range(len(X[0]))])
            if abs_x * abs_y:
                out[i][j] = scal / (abs_x * abs_y)
    return out