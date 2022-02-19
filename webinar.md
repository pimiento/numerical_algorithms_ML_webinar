- [Умножение двух чисел](#org49f8ad5)
- [Умножение двоичных чисел](#org3ed2bf3)
- [Алгоритм Каратцубы](#org23fbe11)
- [Алгоритм Каратцубы](#org2d4a198)
- [Алгоритм Каратцубы](#org4ac9dd2)
- [Классическое умножение](#org85b21b9)
- [Классическое умножение](#org3471451)
- [Алгоритм Каратцубы](#org6c4212f)
- [Умножение матриц](#org52298cc)
- [Умножение матриц](#orgf32a40b)
- [Умножение матриц](#orgdea43d4)
- [Умножение матриц](#org3fb355b)
- [Алгоритм Штрассена](#orgc0d5f06)
- [Алгоритм Штрассена](#orgf742a41)
- [Алгоритм Штрассена](#org0e9ea0f)
- [векторизация](#orgd0a7402)
- [векторизация](#orga850663)
- [векторизация](#org8a88bb8)
- [векторизация](#orgcfaf45d)
- [NumPy](#orgdfcedbd)
- [Логистическая регрессия](#orgbb2f1f5)
- [Обучение](#org3523cd3)
- [Котики!](#org6b84ff0)
- [Tensorflow](#org041adbd)



<a id="org49f8ad5"></a>

# Умножение двух чисел

\begin{equation}
\opmul{5678}{1234}\qquad
\end{equation}

*Можно ли лучше?*


<a id="org3ed2bf3"></a>

# Умножение двоичных чисел

![img](binary_multiplication.png)


<a id="org23fbe11"></a>

# Алгоритм Каратцубы

```python
x = 5678
y = 1234

a = 56; b = 78
c = 12; d = 34
```


<a id="org2d4a198"></a>

# Алгоритм Каратцубы

```python

# step1
step1 = a * c
# step2
step2 = b * d
# step3
a_b = a + b
c_d = c + d
step3 = a_b * c_d
# step4:
# step3 - step2 - step1
step4 = step3 - step2 - step1
```


<a id="org4ac9dd2"></a>

# Алгоритм Каратцубы

```python

line1 = step1 * 10**4
line2 = step2
line3 = step4 * 10**2
result = (
    line1
    + line2
    + line3
)
print(result)
```

    7006652


<a id="org85b21b9"></a>

# Классическое умножение

\begin{equation}
\opmul{5678}{1234}\qquad
\end{equation}


<a id="org3471451"></a>

# Классическое умножение

```python

print(classic(1234, 5678))
```

    7006652

```python


print(timeit.timeit(
    "classic(1234, 5678) == 7006652",
    globals=globals()
))
# print(timeit.timeit(
#     f"classic({big_x}, {big_y})",
#     globals=globals()
# ))
```


<a id="org6c4212f"></a>

# Алгоритм Каратцубы

```python




print(timeit.timeit(
    "karatsuba(1234, 5678) == 7006652",
    globals=globals()
))
# print(timeit.timeit(
#     f"karatsuba({big_x}, {big_y})",
#     globals=globals()
# ))
```


<a id="org52298cc"></a>

# Умножение матриц

\begin{equation}
\left[ \begin{array}{ccc} A & B \\ C & D \\ \end{array} \right]
\times
\left[ \begin{array}{ccc} E & F \\ G & H \\ \end{array} \right]
= \left[ \begin{array}{ccc} AE + BG & AF + BH \\ CE + DG & CF + DH \\ \end{array} \right]
\end{equation}


<a id="orgf32a40b"></a>

# Умножение матриц

```python
def mxm(A, X):
  n = len(A)    # A: n×m
  m = len(A[0])
  p = len(X[0]) # X: m×p
  B = [[0] * p] * n
  for i in range(n):
    for j in range(p):
      for k in range(m):
        B[i][j] += A[i][k]*X[k][j]
  return B
```

**Где ошибка в этом коде?**


<a id="orgdea43d4"></a>

# Умножение матриц

```python
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
```


<a id="org3fb355b"></a>

# Умножение матриц

<div class="org-center">
\(O(n^{3})\)
*Можно ли лучше?*
</div>


<a id="orgc0d5f06"></a>

# Алгоритм Штрассена

\begin{normalsize}
\left[ \begin{array}{cccc} 11 & 12 & 13 & 14 \\ 21 & 22 & 23 & 24 \\ 31 & 32 & 33 & 34 \\ 41 & 42 & 43 & 44 \\ \end{array} \right] =
\left[ \begin{array}{cc} A & B \\ C & D \\ \end{array} \right]
\end{normalsize}

\begin{normalsize}
\left[ \begin{array}{cccc} 11 & 21 & 31 & 41 \\ 12 & 22 & 32 & 42 \\ 13 & 23 & 33 & 43 \\ 14 & 24 & 34 & 44 \\ \end{array} \right] =
\left[ \begin{array}{cc} E & F \\ G & H \\ \end{array} \right]
\end{normalsize}


<a id="orgf742a41"></a>

# Алгоритм Штрассена

\begin{array}{l}
P_{1} = A(F - H), \\
P_{2} = (A + B)H, \\
P_{3} = (C + D)E, \\
P_{4} = D(G - E), \\
P_{5} = (A + D)(E + H), \\
P_{6} = (B - D)(G + H), \\
P_{7} = (A - C)(E + F) \\
\end{array}


<a id="org0e9ea0f"></a>

# Алгоритм Штрассена

\begin{footnotesize}
\left[ \begin{array}{cc} AE+BG & AF+BH \\ CE+DG & CF+DH \end{array} \right] =
\left[ \begin{array}{ll} P_{5} + P_{4} - P_{2} + P_{6} & P_{1} + P_{2} \\ P_{3} + P_{4} & P_{1} + P_{5} - P_{3} + P_{7} \end{array} \right]
\end{footnotesize}


<a id="orgd0a7402"></a>

# векторизация

-   Большинство операций процессора это SISD: Single Instruction Single Data
-   Процессор может поддерживать специальные регистры для <span class="underline"><span class="underline">[SIMD](https://ru.wikipedia.org/wiki/SIMD)</span></span>: Single Instruction Multiple Data


<a id="orga850663"></a>

# векторизация

| 0     | 1        | 2        | 3        |
| a[0]= | not used | not used | not used |
| b[0]+ | not used | not used | not used |
| c[0]  | not used | not used | not used |


<a id="org8a88bb8"></a>

# векторизация

| 0     | 1     | 2     | 3     |
| a[0]= | a[1]= | a[2]= | a[3]= |
| b[0]+ | b[1]+ | b[2]+ | b[3]+ |
| c[0]  | c[1]  | c[2]  | c[3]  |


<a id="orgcfaf45d"></a>

# векторизация

-   Без векторизованных операций

```shell
g++ -o novec vecexample.cpp
echo "Без векторизации"
./novec 1000
```

    Без векторизации
    Time used for norm computation=8.6000E-05
      Norm-2  = 1.5000

```shell
g++ -O3 -mavx2 -o vec vecexample.cpp
echo "Используя векторизацию"
./vec 1000
```

    Используя векторизацию
    Time used for norm computation=7.2000E-05
      Norm-2  = 1.5000


<a id="orgdfcedbd"></a>

# NumPy

```python

_a = (
    np.arange(n, dtype=float) * 2
    * np.math.pi / n
)
a = s * (np.sin(_a) + np.cos(_a))
b = s * np.sin(2.0 * _a)
c = a + b
norm2 = np.sum(np.power(c, 2))

```

```shell
./numpy_vec.py 1000
```

    Time used for norm computation = 0.00019
    Norm-2 = 1.50000


<a id="orgbb2f1f5"></a>

# Логистическая регрессия

\(z = w_{0}x + w_{1}x + \dots + w_{n}x + b\)
\(a = \frac{1}{1+e^{-z}}\)

![img](sigmoid_fun.png "sigmoid")


<a id="org3523cd3"></a>

# Обучение

Чтобы минимизировать ошибку в ответах будем искать минимум функции, вычисляя градиент (производную) для каждой переменной.

\begin{normalsize}
w = w - \dfrac{\partial w}{\partial x_{n}}
\end{normalsize}


<a id="org6b84ff0"></a>

# Котики!

<span class="underline"><span class="underline">[GitHub](https://github.com/pimiento/numerical_algorithms_ML_webinar/blob/master/nn_model.py)</span></span>


<a id="org041adbd"></a>

# Tensorflow
