from itertools import accumulate
from random import randint
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


def input(): return sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 2
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007

n, m = read_int_tuple()

def C(D, R):
    if 0 in (D, R):
        return 0
    hd, hr = D // 2, R // 2
    J = hd * (1 + R) * R // 2 % MOD
    if D & 1:
        tr = (R + 1) // 2
        J += tr * tr % MOD
        J %= MOD
    
    I = hr * D * (D - 1) // 2 % MOD
    if R & 1:
        td = D - 1
        if td & 1: td -= 1
        I += td // 2 * (2 + td) // 2 % MOD
        I %= MOD
    I *= m
    I %= MOD
    # print(I, J)
    return I + J % MOD

# print(*maze,sep='\n')
# print(C(6, 5))
# print(nm.sumRegion(0, 0, 5, 4))

# up, left = randint(1, n), randint(1, n)
# down, right = randint(up, n), randint(left, n)
# print((down, right), (up - 1, left - 1), (up - 1, right), (down, left - 1))
# print(C(down, right), C(up - 1, left - 1), C(up - 1, right), C(down, left - 1))
# res = C(down, right) + C(up - 1, left - 1) - C(up - 1, right) - C(down, left - 1)
# print(res % MOD)
# print(nm.sumRegion(up-1,left-1,down-1,right-1))

for _ in range(read_int()):
    up, down, left, right = read_int_tuple()
    
    # print((down, right), (up - 1, left - 1), (up - 1, right), (down, left - 1))
    # print(C(down, right), C(up - 1, left - 1), C(up - 1, right), C(down, left - 1))
    res = C(down, right) + C(up - 1, left - 1) - C(up - 1, right) - C(down, left - 1)
    print(res % MOD)