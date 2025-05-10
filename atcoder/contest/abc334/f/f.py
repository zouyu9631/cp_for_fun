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
from operator import *
from typing import *

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

from math import dist


def solve():
    N, K = read_int_tuple()
    SAINT = read_int_list()
    point = [read_int_list() for _ in range(N)]
    D = [dist(SAINT, p) for p in point]

    # 预处理距离的前缀和
    P = [0.0] + list(accumulate(dist(a, b) for a, b in pairwise(point)))

    # 初始化DP数组和单调队列
    dp = [float("inf")] * (N + 1)
    dp[0] = 0
    queue = deque([0])

    point.append(SAINT)
    P.append(0.0)

    def calc(i: int):
        return dp[i] + D[i] - P[i]

    # 动态规划计算
    for i in range(1, N + 1):
        # 更新队列，保持单调性和长度限制
        while queue and queue[0] < i - K:
            queue.popleft()

        j = queue[0]
        dp[i] = dp[j] + D[j] + (P[i - 1] - P[j]) + D[i - 1]

        # 维护单调队列
        if i == N:
            break

        while queue and calc(i) <= calc(queue[-1]):
            queue.pop()
        queue.append(i)

    # 输出最终结果
    # print(dp)
    print(dp[N])


def main():
    # region local test
    # if "AW" in os.environ.get("COMPUTERNAME", ""):
    #     test_no = 3
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
