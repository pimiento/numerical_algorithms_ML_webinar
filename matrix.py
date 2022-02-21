class Matrix:
    def __init__(self, l):
        assert isinstance(l[0], list)
        self.n = len(l)
        self.m = len(l[0])
        self.container = l

    @classmethod
    def empty(cls, n, m, initial=None):
        container = [[initial] * m for _ in range(n)]
        return cls(container)

    def mmultiply(self, m):
        assert self.n == m.n
        assert self.m == m.m
        result = Matrix.empty(self.n, self.m)
        for idx_i, i in enumerate(self.container):
            for idx_j, j in enumerate(i):
                result.container[idx_i][idx_j] = self.container[idx_i][idx_j] * m[idx_i][idx_j]
        return result

    def multiply(self, m):
        assert self.m == m.n
        result = Matrix.empty(self.n, m.m, initial=0)
        for i in range(self.n):
            for k in range(m.m):
                for j in range(self.m):
                    result.container[i][j] += (
                        self.container[i][j] * m.container[j][k]
                    )
