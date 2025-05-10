from functools import lru_cache
from heapq import merge
from itertools import accumulate
import sys
import io
import os
from typing import List

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        self.pre = [[0] + [*accumulate(r)] for r in matrix]
        self.pre = [*zip(*([0] + [*accumulate(c)] for c in zip(*self.pre)))]

    def sumRegion(self, i1: int, j1: int, i2: int, j2: int) -> int:
        return self.pre[i2][j2] - self.pre[i1][j2] - self.pre[i2][j1] + self.pre[i1][j1]

"""
二维前缀和优化
暴力枚举，每次枚举一个切割点，然后递归求解两个子问题
"""

def solve():
    H, W, T = read_int_tuple()
    G = [read_int_list() for _ in range(H)]
    PS = NumMatrix(G)
    
    @lru_cache(None)
    def calc(lx, ly, rx, ry, t):
        if (rx - lx) * (ry - ly) < t:
            return []
        if t == 1:
            return [PS.sumRegion(lx, ly, rx, ry)]

        res = [0, inf]
        for x in range(lx + 1, rx):
            for nt in range(1, t):
                a = calc(lx, ly, x, ry, nt)
                if nt != len(a): continue
                b = calc(x, ly, rx, ry, t - nt)
                if t - nt != len(b): continue
                
                if res[-1] - res[0] > max(a[-1], b[-1]) - min(a[0], b[0]):
                    res = list(merge(a, b))
        
        for y in range(ly + 1, ry):
            for nt in range(1, t):
                a = calc(lx, ly, rx, y, nt)
                if nt != len(a): continue
                b = calc(lx, y, rx, ry, t - nt)
                if t - nt != len(b): continue
                
                if res[-1] - res[0] > max(a[-1], b[-1]) - min(a[0], b[0]):
                    res = list(merge(a, b))
        
        return res

    res = calc(0, 0, H, W, T + 1)
    # print(res)
    print(res[-1] - res[0])

def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1
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

input = lambda: sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_ints_minus_one():
    return [int(x) - 1 for x in input().split()]


def read_int_tuple():
    return map(int, input().split())


def read_graph(n: int, m: int, d=1):
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        g[u - d].append(v - d)
        g[v - d].append(u - d)
    return g


def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()