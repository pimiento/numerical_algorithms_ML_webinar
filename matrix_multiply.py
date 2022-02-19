#!/usr/bin/env python3
import numpy as np
def mxm(A, X):
  n = len(A)    # A: n×m
  m = len(A[0])
  p = len(X[0]) # X: m×p
  B = [[0] * p for _ in range(n)]
  for i in range(n):
    for j in range(p):
      for k in range(m):
        B[i][j] += A[i][k]*X[k][j]
  return B

A = [
  [1, 2, 3, 4],
  [1, 2, 3, 4],
  [1, 2, 3, 4],
  [1, 2, 3, 4]
]

B = [
    [5, 5, 5, 5],
    [6, 6, 6, 6],
    [7, 7, 7, 7],
    [8, 8, 8, 8]
]

result = mxm(A, B)
print(result)

np_result = np.dot(np.array(A), np.array(B))
assert (np_result == np.array(result)).all(), [np_result, np.array(result)]
