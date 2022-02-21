#!/usr/bin/env python3
import sys
import time
import numpy as np

n = int(sys.argv[1])
start = time.time()
s = 1.0 / np.math.sqrt(n)
_a = (
    np.arange(n, dtype=float) * 2
    * np.math.pi / n
)
a = s * (np.sin(_a) + np.cos(_a))
b = s * np.sin(2.0 * _a)
c = a + b
norm2 = np.sum(np.power(c, 2))
finish = time.time()
time_used = finish - start
print(f"Time used for norm computation = {time_used:.5f}")
print(f"Norm-2 = {norm2:.5f}")
