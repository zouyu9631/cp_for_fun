import sys
import io
import os
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from itertools import *
from functools import *
from operator import *
from typing import *

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

# A = [1]
# N = 100
# for _ in range(N):
#     A.append(A[-1] * 10 + 1)

# R = []
# for i in range(N):
#     for j in range(i, N):
#         for k in range(j, N):
#             R.append(A[i] + A[j] + A[k])
# R.sort()
# print(R[:400])

R = [3, 13, 23, 33, 113, 123, 133, 223, 233, 333, 1113, 1123, 1133, 1223, 1233, 1333, 2223, 2233, 2333, 3333, 11113, 11123, 11133, 11223, 11233, 11333, 12223, 12233, 12333, 13333, 22223, 22233, 22333, 23333, 33333, 111113, 111123, 111133, 111223, 111233, 111333, 112223, 112233, 112333, 113333, 122223, 122233, 122333, 123333, 133333, 222223, 222233, 222333, 223333, 233333, 333333, 1111113, 1111123, 1111133, 1111223, 1111233, 1111333, 1112223, 1112233, 1112333, 1113333, 1122223, 1122233, 1122333, 1123333, 1133333, 1222223, 1222233, 1222333, 1223333, 1233333, 1333333, 2222223, 2222233, 2222333, 2223333, 2233333, 2333333, 3333333, 11111113, 11111123, 11111133, 11111223, 11111233, 11111333, 11112223, 11112233, 11112333, 11113333, 11122223, 11122233, 11122333, 11123333, 11133333, 11222223, 11222233, 11222333, 11223333, 11233333, 11333333, 12222223, 12222233, 12222333, 12223333, 12233333, 12333333, 13333333, 22222223, 22222233, 22222333, 22223333, 22233333, 22333333, 23333333, 33333333, 111111113, 111111123, 111111133, 111111223, 111111233, 111111333, 111112223, 111112233, 111112333, 111113333, 111122223, 111122233, 111122333, 111123333, 111133333, 111222223, 111222233, 111222333, 111223333, 111233333, 111333333, 112222223, 112222233, 112222333, 112223333, 112233333, 112333333, 113333333, 122222223, 122222233, 122222333, 122223333, 122233333, 122333333, 123333333, 133333333, 222222223, 222222233, 222222333, 222223333, 222233333, 222333333, 223333333, 233333333, 333333333, 1111111113, 1111111123, 1111111133, 1111111223, 1111111233, 1111111333, 1111112223, 1111112233, 1111112333, 1111113333, 1111122223, 1111122233, 1111122333, 1111123333, 1111133333, 1111222223, 1111222233, 1111222333, 1111223333, 1111233333, 1111333333, 1112222223, 1112222233, 1112222333, 1112223333, 1112233333, 1112333333, 1113333333, 1122222223, 1122222233, 1122222333, 1122223333, 1122233333, 1122333333, 1123333333, 1133333333, 1222222223, 1222222233, 1222222333, 1222223333, 1222233333, 1222333333, 1223333333, 1233333333, 1333333333, 2222222223, 2222222233, 2222222333, 2222223333, 2222233333, 2222333333, 2223333333, 2233333333, 2333333333, 3333333333, 11111111113, 11111111123, 11111111133, 11111111223, 11111111233, 11111111333, 11111112223, 11111112233, 11111112333, 11111113333, 11111122223, 11111122233, 11111122333, 11111123333, 11111133333, 11111222223, 11111222233, 11111222333, 11111223333, 11111233333, 11111333333, 11112222223, 11112222233, 11112222333, 11112223333, 11112233333, 11112333333, 11113333333, 11122222223, 11122222233, 11122222333, 11122223333, 11122233333, 11122333333, 11123333333, 11133333333, 11222222223, 11222222233, 11222222333, 11222223333, 11222233333, 11222333333, 11223333333, 11233333333, 11333333333, 12222222223, 12222222233, 12222222333, 12222223333, 12222233333, 12222333333, 12223333333, 12233333333, 12333333333, 13333333333, 22222222223, 22222222233, 22222222333, 22222223333, 22222233333, 22222333333, 22223333333, 22233333333, 22333333333, 23333333333, 33333333333, 111111111113, 111111111123, 111111111133, 111111111223, 111111111233, 111111111333, 111111112223, 111111112233, 111111112333, 111111113333, 111111122223, 111111122233, 111111122333, 111111123333, 111111133333, 111111222223, 111111222233, 111111222333, 111111223333, 111111233333, 111111333333, 111112222223, 111112222233, 111112222333, 111112223333, 111112233333, 111112333333, 111113333333, 111122222223, 111122222233, 111122222333, 111122223333, 111122233333, 111122333333, 111123333333, 111133333333, 111222222223, 111222222233, 111222222333, 111222223333, 111222233333, 111222333333, 111223333333, 111233333333, 111333333333, 112222222223, 112222222233, 112222222333, 112222223333, 112222233333, 112222333333, 112223333333, 112233333333, 112333333333, 113333333333, 122222222223, 122222222233, 122222222333, 122222223333, 122222233333, 122222333333, 122223333333, 122233333333, 122333333333, 123333333333, 133333333333, 222222222223, 222222222233, 222222222333, 222222223333, 222222233333, 222222333333, 222223333333, 222233333333, 222333333333, 223333333333, 233333333333, 333333333333, 1111111111113, 1111111111123, 1111111111133, 1111111111223, 1111111111233, 1111111111333, 1111111112223, 1111111112233, 1111111112333, 1111111113333, 1111111122223, 1111111122233, 1111111122333, 1111111123333, 1111111133333, 1111111222223, 1111111222233, 1111111222333, 1111111223333, 1111111233333, 1111111333333, 1111112222223, 1111112222233, 1111112222333, 1111112223333, 1111112233333, 1111112333333, 1111113333333, 1111122222223, 1111122222233, 1111122222333, 1111122223333, 1111122233333, 1111122333333, 1111123333333, 1111133333333]

def solve():
    n = read_int()
    print(R[n - 1])


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
