#!/usr/bin/env python3
import math
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

def divider(num: List[int]) -> List[List[int]]:
    return [num[:len(num) // 2], num[len(num) // 2:]]

def karatsuba(x: int, y: int) -> int:
    def _karatsuba(_x: List[int], _y: List[int]) -> List[int]:
        x, y = make_equal_length(_x, _y)
        if len(x) == 1:
            return vectorize(x[0] * y[0])
        a, b = divider(x)
        c, d = divider(y)
        step1 = _karatsuba(a, c)
        step2 = _karatsuba(b, d)
        step3 = _karatsuba(sum_vectors(a, b), sum_vectors(c, d))
        step4 = diff_vectors(diff_vectors(step3, step2), step1)
        line1 = step1 + [0] * len(x)
        line2 = step2
        line3 = step4 + [0] * math.ceil(len(x) / 2)
        result = sum_vectors(sum_vectors(line1, line2), line3)
        return result
    return devectorize(_karatsuba(vectorize(x), vectorize(y)))

print(karatsuba(
    1234,
    5678
))

assert karatsuba(1234, 5678) == 7006652
print(timeit.timeit(
    "karatsuba(1234, 5678) == 7006652",
    globals=globals()
))
# print(timeit.timeit(
#     f"karatsuba({big_x}, {big_y})",
#     globals=globals()
# ))
