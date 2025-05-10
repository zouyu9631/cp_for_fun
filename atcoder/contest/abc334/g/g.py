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

"""
无向图的强连通分量，去掉一个点后，剩下的连通分量的数目

三种情况 
1 孤立的一个点，去掉后剩下0个
2 非割点，去掉后还是1个
3 割点，去掉后是
"""

def lowlink(links):
    n = len(links)
    order = [-1] * n
    low = [n] * n
    parent = [-1] * n
    child = [[] for _ in range(n)]
    roots = set()
    x = 0
    for root in range(n):
        if order[root] != -1:
            continue
        roots.add(root)
        stack = [root]
        parent[root] = -2
        while stack:
            i = stack.pop()
            if i >= 0:
                if order[i] != -1:
                    continue
                order[i] = x
                low[i] = x
                x += 1
                if i != root:
                    child[parent[i]].append(i)
                stack.append(~i)
                check_p = 0
                for j in links[i]:
                    if j == parent[i] and check_p == 0:
                        check_p += 1
                        continue
                    elif order[j] != -1:
                        low[i] = min(low[i], order[j])
                    else:
                        parent[j] = i
                        stack.append(j)
            else:
                i = ~i
                if i == root:
                    continue
                p = parent[i]
                low[p] = min(low[p], low[i])

    return order, low, roots, child


def solve():
    H, W = read_int_tuple()
    grid = [input() for _ in range(H)]

    graph, book = [], dict()

    def get_idx(i, j):
        u = i * W + j
        if u not in book:
            book[u] = len(graph)
            graph.append([])
        return book[u]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == "#":
                u = get_idx(i, j)
                for d in directions:
                    ni, nj = i + d[0], j + d[1]
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == "#":
                        v = get_idx(ni, nj)
                        graph[u].append(v)

    # print(graph)
    # print(book)

    order, low, roots, child = lowlink(graph)
    # print(order, low, roots, child)
    n = len(graph)
    part_count = [0] * n
    for i in range(n):
        part_count[i] = sum(order[i] <= low[j] for j in child[i])
        if i not in roots:
            part_count[i] += 1

    # print(articulation)

    tot = len(roots) * n
    for x in part_count:
        tot += x - 1

    print(tot * pow(n, MOD - 2, MOD) % MOD)


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
