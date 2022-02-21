#!/usr/bin/env python3
import timeit
from typing import List
from functools import reduce


def make_equal_length(x: List[int], y: List[int]) -> List[List[int]]:
    if len(x) > len(y):
        y = [0] * (len(x) - len(y)) + y
    elif len(x) < len(y):
        x = [0] * (len(y) - len(x)) + x
    return [x, y]

def sum_vectors(x: List[int], y: List[int]) -> List[int]:
    x, y = make_equal_length(x, y)
    result: List[int] = []
    shift = 0
    for idx in range(len(x)-1, -1, -1):
        _x = x[idx]
        _y = y[idx]
        _result = (_x + _y) + shift
        if _result > 9:
            shift = 1
            result.insert(0, _result % 10)
        else:
            shift = 0
            result.insert(0, _result)
    if shift > 0:
        result.insert(0, shift)
    return result

def diff_vectors(x: List[int], y: List[int]) -> List[int]:
    x, y = make_equal_length(x, y)
    result: List[int] = []
    debt = 0
    for idx in range(len(x)-1, -1, -1):
        _x = x[idx]
        _y = y[idx]
        _result = (_x - _y) - debt
        if _result < 0:
            debt = 1
            result.insert(0, _result + debt * 10)
        else:
            debt = 0
            result.insert(0, _result)
    return result

def vectorize(x: int) -> List[int]:
    return [int(_) for _ in str(x)]

def devectorize(x: List[int]) -> int:
    return int(''.join(str(_) for _ in x))

def classic(x: int, y: int) -> int:
    temp_results: List[List[int]] = [[] for _ in range(len(str(y)))]
    _x = vectorize(x)
    _y = vectorize(y)
    for idx, i in enumerate(_y[::-1]):
        shift = 0
        temp_results[idx] += [0] * idx
        for j in _x[::-1]:
            temp = (i * j) + shift
            shift = temp // 10
            temp_results[idx].insert(0, temp % 10)
    return devectorize(reduce(sum_vectors, temp_results))
big_x = int(
    "31415926535897932384626433832795"
    "02884197169399375105820974944592"
)
big_y = int(
    "27182818284590452353602874713526"
    "62497757247093699959574966967627"
)
classic(1234, 5678) == 7006652
print(timeit.timeit(
    "classic(1234, 5678) == 7006652",
    globals=globals()
))
# print(timeit.timeit(
#     f"classic({big_x}, {big_y})",
#     globals=globals()
# ))
