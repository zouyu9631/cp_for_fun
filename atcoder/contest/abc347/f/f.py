standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 1  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

from typing import List, Callable


class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        self.dp = [[0] * n for _ in range(m)]
        self.dp[0][0] = matrix[0][0]
        for i in range(1, m):
            self.dp[i][0] = self.dp[i - 1][0] + matrix[i][0]
        for j in range(1, n):
            self.dp[0][j] = self.dp[0][j - 1] + matrix[0][j]
        for i in range(1, m):
            for j in range(1, n):
                self.dp[i][j] = (
                    self.dp[i - 1][j]
                    + self.dp[i][j - 1]
                    - self.dp[i - 1][j - 1]
                    + matrix[i][j]
                )

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        a = self.dp[row2][col2]
        b = self.dp[row1 - 1][col2] if row1 > 0 else 0
        c = self.dp[row2][col1 - 1] if col1 > 0 else 0
        d = self.dp[row1 - 1][col1 - 1] if row1 > 0 and col1 > 0 else 0
        return a - b - c + d


def display(mat):
    for row in mat:
        print(row)


def choose(ra, rb):
    if ra is None:
        return rb
    if rb is None:
        return ra

    sa = sum(ra)
    sb = sum(rb)
    return ra if sa > sb else rb


def assemble(a, b, c):
    return (a, b, c)


def solve():
    N, M = MII()
    G = [LII() for _ in range(N)]
    P = NumMatrix(G)
    grid = [[0] * (N - M + 1) for _ in range(N - M + 1)]
    for i in range(M - 1, N):
        for j in range(M - 1, N):
            grid[i - M + 1][j - M + 1] = P.sumRegion(i - M + 1, j - M + 1, i, j)
    rects = SplitThreeRectsInOne(grid, M, max, choose, assemble)

    res = rects.calc()
    return sum(res)


class SplitThreeRectsInOne:
    def __init__(
        self,
        grid: List[List[int]],
        d: int,
        merge: Callable,
        choose: Callable,
        assemble: Callable,
    ):
        self.d = d
        self.m, self.n = len(grid), len(grid[0])
        self.grid, self.merge = grid, merge
        self.choose, self.assemble = choose, assemble
        self.upper_left = self._init_dp_mat(1, 1)
        self.upper_right = self._init_dp_mat(1, -1)
        self.lower_left = self._init_dp_mat(-1, 1)
        self.lower_right = self._init_dp_mat(-1, -1)
        self.row_accum, self.col_accum = self._init_dp_lines()

    def calc(self):
        res = self._calc_vertical()
        self._rotate90()
        res2 = self._calc_vertical()
        return self.choose(res, res2)

    def _calc_vertical(self):
        res = None
        # 上中下
        res = self._calc_upper_middle_lower()

        # 上左右
        res = self.choose(res, self._calc_upper_left_right())

        # 上下颠倒后的上左右
        self._upside_down()
        res = self.choose(res, self._calc_upper_left_right())

        return res

    def _calc_upper_middle_lower(self):
        res = None

        for i1 in range(1, self.m - self.d * 2 + 1):
            upper_value = self.upper_left[i1 - 1][-1]

            middle_value = self.row_accum[i1 + self.d - 1]

            for i2 in range(i1 + self.d, self.m - self.d + 1):
                lower_value = self.lower_left[i2 + self.d - 1][-1]
                tr = self.assemble(upper_value, middle_value, lower_value)
                res = self.choose(res, tr)
                middle_value = self.merge(middle_value, self.row_accum[i2])

        return res

    def _calc_upper_left_right(self):
        res = None

        for i in range(1, self.m - self.d + 1):
            upper_value = self.upper_left[i - 1][-1]
            for j in range(1, self.n - self.d + 1):
                left_value = self.lower_left[i + self.d - 1][j - 1]
                right_value = self.lower_right[i + self.d - 1][j + self.d - 1]
                tr = self.assemble(upper_value, left_value, right_value)
                res = self.choose(res, tr)

        return res

    def _init_dp_mat(self, dx: int, dy: int) -> List[List[int]]:
        dp = [row[:] for row in self.grid]
        for i in range(self.m)[::dx]:
            for j in range(self.n)[::dy]:
                if 0 <= i - dx < self.m:
                    dp[i][j] = self.merge(dp[i][j], dp[i - dx][j])
                if 0 <= j - dy < self.n:
                    dp[i][j] = self.merge(dp[i][j], dp[i][j - dy])
        return dp

    def _init_dp_lines(self):
        return [reduce(self.merge, row) for row in self.grid], [
            reduce(self.merge, col) for col in zip(*self.grid)
        ]

    def _rotate(self, grid: List[List[int]]):
        for i in range(self.m):
            for j in range(i + 1, self.m):
                grid[i][j], grid[j][i] = grid[j][i], grid[i][j]

        for i in range(self.m):
            grid[i].reverse()

    def _rotate90(self):
        self.m, self.n = self.n, self.m

        self._rotate(self.grid)
        self._rotate(self.upper_left)
        self._rotate(self.upper_right)
        self._rotate(self.lower_right)
        self._rotate(self.lower_left)

        self.upper_left, self.upper_right, self.lower_right, self.lower_left = (
            self.lower_left,
            self.upper_left,
            self.upper_right,
            self.lower_right,
        )
        self.row_accum, self.col_accum = self.col_accum, self.row_accum

    def _upside_down(self):
        self.grid.reverse()
        self.upper_left.reverse()
        self.upper_right.reverse()
        self.lower_right.reverse()
        self.lower_left.reverse()

        self.upper_left, self.lower_left = self.lower_left, self.upper_left
        self.upper_right, self.lower_right = self.lower_right, self.upper_right
        self.row_accum.reverse()


def main():
    # region local test
    if read_from_file and "AW" in os.environ.get("COMPUTERNAME", ""):
        test_no = read_from_file
        f = open(os.path.dirname(__file__) + f"\\in{test_no}.txt", "r")
        global input
        input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = II() if multi_test else 1
    for t in range(T):
        if output_mode == 0:
            solve()
        elif output_mode == 1:
            print(solve())
        elif output_mode == 2:
            print("YES" if solve() else "NO")
        elif output_mode == 3:
            print("Yes" if solve() else "No")


# region
if standard_input:
    import os, sys, math

    input = lambda: sys.stdin.readline().strip()

    inf = math.inf

    def I():
        return input()

    def II():
        return int(input())

    def MII():
        return map(int, input().split())

    def LI():
        return list(input().split())

    def LII():
        return list(map(int, input().split()))

    def LFI():
        return list(map(float, input().split()))

    def GMI():
        return map(lambda x: int(x) - 1, input().split())

    def LGMI():
        return list(map(lambda x: int(x) - 1, input().split()))

    def GRAPH(n: int, m=-1):
        if m == -1:
            m = n - 1
        g = [[] for _ in range(n)]
        for _ in range(m):
            u, v = GMI()
            g[u].append(v)
            g[v].append(u)
        return g


if packages:
    from io import BytesIO, IOBase
    import math
    import random
    import bisect
    import typing
    from collections import Counter, defaultdict, deque
    from copy import deepcopy
    from functools import cmp_to_key, lru_cache, reduce
    from heapq import *
    from itertools import accumulate, combinations, permutations, count, product
    from operator import add, iand, ior, itemgetter, mul, xor
    from string import ascii_lowercase, ascii_uppercase, ascii_letters
    from typing import *

    BUFSIZE = 4096

if output_together:

    class FastIO(IOBase):
        newlines = 0

        def __init__(self, file):
            self._fd = file.fileno()
            self.buffer = BytesIO()
            self.writable = "x" in file.mode or "r" not in file.mode
            self.write = self.buffer.write if self.writable else None

        def read(self):
            while True:
                b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                if not b:
                    break
                ptr = self.buffer.tell()
                self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
            self.newlines = 0
            return self.buffer.read()

        def readline(self):
            while self.newlines == 0:
                b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                self.newlines = b.count(b"\n") + (not b)
                ptr = self.buffer.tell()
                self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
            self.newlines -= 1
            return self.buffer.readline()

        def flush(self):
            if self.writable:
                os.write(self._fd, self.buffer.getvalue())
                self.buffer.truncate(0), self.buffer.seek(0)

    class IOWrapper(IOBase):
        def __init__(self, file):
            self.buffer = FastIO(file)
            self.flush = self.buffer.flush
            self.writable = self.buffer.writable
            self.write = lambda s: self.buffer.write(s.encode("ascii"))
            self.read = lambda: self.buffer.read().decode("ascii")
            self.readline = lambda: self.buffer.readline().decode("ascii")

    sys.stdout = IOWrapper(sys.stdout)

if dfs_tag:
    from types import GeneratorType

    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if type(to) is GeneratorType:
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to

        return wrappedfunc


if int_hashing:
    RANDOM = random.getrandbits(20)

    class Wrapper(int):
        def __init__(self, x):
            int.__init__(x)

        def __hash__(self):
            return super(Wrapper, self).__hash__() ^ RANDOM


if True:

    def debug(*args, **kwargs):
        print("\033[92m", end="")
        print(*args, **kwargs)
        print("\033[0m", end="")


# endregion

if __name__ == "__main__":
    main()
