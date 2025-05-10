from itertools import product
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
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
暴力思路：
每个序列对应一个内向基环树。i的结果，就是计算从i开始，入环要几步。
然后oeis.org搜出结果
"""

def solve():
    N, M = read_int_tuple()
    res = 0
    pp = [1]
    fc = [N]
    for x in range(N - 1, 0, -1):
        pp.append(pp[-1] * N % M)
        fc.append(fc[-1] * x % M)
    fc.reverse()
    # print(pp)
    # print(fc)
    
    for q in range(N - 1):
        # print(q, pp[q] * fc[q] * (N - q) * (N - 1 - q))
        res += pp[q] * fc[q] * (N - q) * (N - 1 - q) // 2
        res %= M
        
    print(res)

#每个序列对应一个内向基环树。i的结果，就是计算从i开始，入环要几步。
def jihuan(n, t):
    indeg = [0] * n
    for x in t: indeg[x] += 1
    q = [x for x in range(n) if indeg[x] == 0]
    rg = [[] for _ in range(n)]
    while q:
        nq = []
        for u in q:
            v = t[u]
            rg[v].append(u)
            indeg[v] -= 1
            if indeg[v] == 0: nq.append(v)
        q = nq
    
    # 到环的距离
    dist = [0 if indeg[u] else -1 for u in range(n)]
    q = []
    for u in range(n):
        if indeg[u]:
            q.extend(rg[u])
    while q:
        nq = []
        for u in q:
            dist[u] = 1 + dist[t[u]]
            nq.extend(rg[u])
        q = nq
    return sum(dist)

def brute():
    N, M = read_int_tuple()
    res = 0
    for t in product(range(N), repeat=N):
        res += jihuan(N, t)    
    print(res)

T = 1#read_int()
for t in range(T):
    solve()
    # brute()