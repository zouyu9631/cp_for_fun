from collections import defaultdict
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


# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
线性基
"""

class LinearBasis:
    __slots__ = ("Basis", "_rows", "_bit")
 
    @staticmethod
    def create(nums: List[int]) -> "LinearBasis":
        res = LinearBasis()
        for x in nums:
            res.add(x)
        res.build()
        return res
 
    def __init__(self, bit=62):
        self.Basis = []
        self._rows = defaultdict(int)
        self._bit = bit
 
    def add(self, x: int) -> None:
        x = self._normalize(x)
        if x == 0:
            return
        i = x.bit_length() - 1
        for j in range(self._bit):
            if (self._rows[j] >> i) & 1:
                self._rows[j] ^= x
        self._rows[i] = x
 
    def build(self) -> None:
        res = []
        for _, v in sorted(self._rows.items()):
            if v > 0:
                res.append(v)
        self.Basis = res
 
    def kth(self, k: int) -> int:
        """子序列第k小的异或 1<=k<=2**len(self._e)"""
        k -= 1
        res = 0
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                res ^= self.Basis[i]
        return res
 
    def _normalize(self, x: int) -> int:
        for i in range(x.bit_length() - 1, -1, -1):
            if (x >> i) & 1:
                x ^= self._rows[i]
        return x
 
    def __len__(self):
        return len(self.Basis)

def solve():
    N, L, R = read_int_tuple()
    lb = LinearBasis.create(read_int_list())
    print(*[lb.kth(i) for i in range(L, R + 1)])


T = 1#read_int()
for t in range(T):
    solve()