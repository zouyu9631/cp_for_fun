from itertools import groupby
import sys
import io
import os
from typing import Counter


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
#     test_no = 2
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

def solve():
    n = read_int()
    A = read_int_list()
    B = read_int_list()
    if Counter(A) != Counter(B):
        return False
    
    if A == B:
        return True
    
    eA = [x for x in A if x & 1 == 0]
    eB = [x for x in B if x & 1 == 0]
    if not eA:
        return False
    if len(eA) == 2 and eA != eB:
        return False
    
    def check(idx: list):
        for a, b in zip(idx, idx[1:]):
            if a + 1 == b and (a > 0 or b < n - 1):
                return True
            if a + 2 == b:
                return True
        
        return False
    
    if check([i for i, x in enumerate(A) if x & 1]) and check([i for i, x in enumerate(B) if x & 1]):
        return True
    
    eA.clear(); eB.clear()
    A.append(1); B.append(1)
    
    for x, y in zip(A, B):
        if x & 1 != y & 1:
            return False
        
        if x & 1:
            if x != y or len(eA) == 2 and eA != eB:
                return False
            eA.clear(); eB.clear()
        else:
            eA.append(x)
            eB.append(y)
    
    return True

T = 1#read_int()
for t in range(T):
    print(['No', 'Yes'][solve()])