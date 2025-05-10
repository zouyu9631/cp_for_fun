from collections import defaultdict
from math import gcd
import sys
import io
import os


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

MOD = 1000000007
inf = 1 << 60

"""
计数
分组后，乘法原理，处理细节
"""

def solve():
    n = read_int()
    oc = 0
    cnt = defaultdict(int)
    for _ in range(n):
        a, b = read_int_tuple()
        if a == b == 0:
            oc += 1
            continue
        g = gcd(abs(a), abs(b))
        a //= g; b //= g
        if a < 0:
            a, b = -a, -b
        elif a == 0:
            b = abs(b)
        cnt[a, b] += 1
        if b < 0:
            tmp = cnt[-b, a]
    
    n -= oc
    
    p2 = [1] * (n + 1)
    for i in range(1, n + 1):
        p2[i] = (p2[i - 1] << 1) % MOD
    
    c1, c2 = cnt[0, 1], cnt[1, 0]
    cnt.pop((0, 1))
    cnt.pop((1, 0))
    
    res = (p2[c1] + p2[c2] - 1) % MOD
    
    for (a, b), c1 in cnt.items():
        if 0 in (a, b): continue
        if b < 0: continue
        c2 = cnt.get((b, -a), 0)
        res *= (p2[c1] + p2[c2] - 1) % MOD
        res %= MOD
    
    print((res - 1 + oc) % MOD)

T = 1#read_int()
for t in range(T):
    solve()