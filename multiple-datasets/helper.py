import numpy as np
from itertools import combinations
import random
import math


def ismember(A, B):
    """
    returns array with elements equal to logical 1 where array elements of A is found in B. Elsewhere it is logical 0
    :param A:
    :type A:
    :param B:
    :type B:
    :return:
    :rtype:
    """
    ret_list = []
    for rows in A:
        ret_list.append([np.sum(a in B) for a in rows])
    return np.array(ret_list)


def comb(n, r):
    """

    :param n:
    :type n:
    :param r:
    :type r:
    :return:
    :rtype:
    """
    f = math.factorial
    return int(f(n) / (f(r) * f(n - r)))


def arr_sort(x,order='ascending'):
    """

    Args:
        order (str): ascending or descending
        x (np.array): array  to be sorted

    Returns: sorted array and sort index

    """
    if order == 'ascending':
        sign = 1
    elif order == 'descending':
        sign = -1
    else:
        raise Exception("unknown order specified")

    idx = np.argsort(np.array(sign*x))
    sorted_arr = x[idx]
    return sorted_arr, idx
