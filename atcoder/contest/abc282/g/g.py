from collections import defaultdict
from functools import lru_cache
from itertools import permutations
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
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
插入法DP 二维前缀和优化DP

转化成range(0, n)的排列
np[i][k][a][b]表示，排列到i，有k个相似对，A末尾是a，B末尾是b的数量，dp表示的是二维前缀和。
类似这题：https://atcoder.jp/contests/dp/tasks/dp_t
https://atcoder.jp/contests/abc282/editorial/5419
"""

def solve():
    N, K, P = read_int_tuple()
    dp = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(K + 1)]
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            dp[0][a][b] = 1
            
    np = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(K + 1)]
    
    for i in range(1, N):
        for k in range(K + 1):
            for a in range(i + 1):
                for b in range(i + 1):
                    np[k][a][b] = (dp[k][i][b] - dp[k][a][b]) + (dp[k][a][i] - dp[k][a][b])
                    if k: np[k][a][b] += (dp[k - 1][a][b]) + (dp[k - 1][i][i] - dp[k - 1][a][i] - dp[k - 1][i][b] + dp[k - 1][a][b])
                    np[k][a][b] %= P
 
        # 2-D presum
        for k in range(K + 1):
            for a in range(i + 1):
                for b in range(i + 1):
                    dp[k][a + 1][b + 1] = dp[k][a][b + 1] + dp[k][a + 1][b] - dp[k][a][b] + np[k][a][b]
                    dp[k][a + 1][b + 1] %= P
    
    print(dp[K][-1][-1])

T = 1#read_int()
for t in range(T):
    solve()