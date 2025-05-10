import sys
import io
import os
from string import *
from re import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from random import *
from itertools import *
from functools import *
from typing import *

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60


class UnionSet:
    def __init__(self, n: int):
        self.parent = [*range(n)]
        self.rank = [1] * n
        self.n = n

    def __len__(self):
        return self.n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def check(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def union(self, x, y):  # rank by deep
        x_0 = self.find(x)
        y_0 = self.find(y)
        if x_0 != y_0:
            if self.rank[x_0] < self.rank[y_0]:
                self.parent[x_0] = y_0
            elif self.rank[x_0] > self.rank[y_0]:
                self.parent[y_0] = x_0
            else:
                self.rank[x_0] += 1
                self.parent[y_0] = x_0
            self.n -= 1


def solve():
    H, W = read_int_tuple()
    grid = [read_encode_str(0) for _ in range(H)]  # '#' -> 35, '.' -> 46

    uf = UnionSet(H * W)
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    # 合并绿色单元格
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 35:  # 绿色单元格
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == 35:
                        uf.union(i * W + j, ni * W + nj)
    total_components = len(uf)

    # 计算红色单元格对连通区域的影响
    red_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == 46]
    total_red = len(red_cells)
    total_components -= total_red

    expected_value = 0
    for i, j in red_cells:
        unique_components = set()
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == 35:
                unique_components.add(uf.find(ni * W + nj))
        expected_value += total_components - len(unique_components) + 1
    #     print(total_components - len(unique_components) + 1)
    # print(expected_value, total_red)
    # 输出期望值
    print((expected_value * pow(total_red, MOD - 2, MOD)) % MOD)


def main():
    # region local test
    # if "AW" in os.environ.get("COMPUTERNAME", ""):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f"\\in{test_no}.txt", "r")

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1
    # T = read_int()
    for t in range(T):
        solve()
        # print('YES' if solve() else 'NO')
        # print('Yes' if solve() else 'No')


# region IO
BUFSIZE = 8192


class FastIO(io.IOBase):
    newlines = 0

    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.buffer = io.BytesIO()
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


class IOWrapper(io.IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


def print(*args, **kwargs):
    """Prints the values to a stream, or to sys.stdout by default."""
    sep, file = kwargs.pop("sep", " "), kwargs.pop("file", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop("end", "\n"))
    if kwargs.pop("flush", False):
        file.flush()


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip("\r\n")


def read_int_list():
    return list(map(int, input().split()))


def read_ints_minus_one():
    return [int(x) - 1 for x in input().split()]


def read_int_tuple():
    return map(int, input().split())


def read_encode_str(d=97):  # 'a': 97; 'A': 65
    return [ord(x) - d for x in input()]


def read_graph(n: int, m: int, d=1):
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        g[u - d].append(v - d)
        g[v - d].append(u - d)
    return g


def read_grid(m: int):
    return [input() for _ in range(m)]


def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()
