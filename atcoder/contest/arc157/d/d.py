from itertools import accumulate
import sys
import io
import os
from typing import List


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


def read_int_tuple():
    return map(int, input().split())


def read_int():
    return int(input())

read_str = input


# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        self.pre = [[0] + list(accumulate(r)) for r in matrix]
        self.pre = [*zip(*([0] + list(accumulate(c)) for c in zip(*self.pre)))]

    def sumRegion(self, i1: int, j1: int, i2: int, j2: int) -> int:
        return self.pre[i2][j2] - self.pre[i1][j2] - self.pre[i2][j1] + self.pre[i1][j1]

"""
根据二维前缀和以及数量约数 按照行和列寻找可能的构造 并且验证
"""

def solve():
    H, W = read_int_tuple()
    g = [[1 if ch == 'Y' else 0 for ch in input()] for _ in range(H)]
    tot = sum(map(sum, g))
    if tot & 1: return 0
    tot >>= 1

    pre2 = NumMatrix(g)
    res = 0

    for h in range(H):
        if tot % (h + 1): continue
        w = tot // (h + 1) - 1
        if not (0 <= w < W): continue
        left, tr, T = 0, 1, (w + 1) * 2
        
        rows = [0]
        
        for right in range(1, H + 1):
            block_sum = pre2.sumRegion(left, 0, right, W)
            if block_sum == T:
                rows.append(right)
                if left:
                    c = 1
                    for x in range(right - left):
                        if pre2.sumRegion(left + x, 0, left + x + 1, W) == 0:
                            c += 1
                        else:
                            break
                    tr = (tr * c) % MOD
                left = right
                    
            elif block_sum > T:
                tr = 0
                break
        
        if tr == 0: continue
        
        left, tc, T = 0, 1, (h + 1) * 2
        for right in range(1, W + 1):
            block_sum = pre2.sumRegion(0, left, H, right)
            if block_sum == T:
                if left:
                    if any(pre2.sumRegion(r0, left, r1, right) != 2 for r0, r1 in zip(rows, rows[1:])):
                        tc = 0
                        break
                        
                    c = 1
                    for x in range(right - left):
                        if pre2.sumRegion(0, left + x, H, left + x + 1) == 0:
                            c += 1
                        else:
                            break
                    tc = (tc * c) % MOD
                left = right
            elif block_sum > T:
                tc = 0
                break
        
        if tc:
            res = (res + tr * tc) % MOD
    
    return res

T = 1#read_int()
for t in range(T):
    print(solve())