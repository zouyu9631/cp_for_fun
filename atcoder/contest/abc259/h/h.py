from collections import defaultdict
from itertools import combinations, product
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


def input(): return sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")

MOD = 998244353  # 1000000007

MOD = 998244353  # 1000000007

class combi:
    def __init__(self,max_n,mod):
        max_n+=1
        self.mod=mod
        self.fact=[0]*max_n
        self.rev=[0]*max_n
        self.fact_rev=[0]*max_n
        self.fact[0]=1
        self.rev[0]=1
        self.fact_rev[0]=1
        for i in range(max_n):
            if i<=1:
                self.fact[i]=1
                self.rev[i]=1
                self.fact_rev[i]=1
                continue
            self.fact[i]=(i*self.fact[i-1])%mod
            self.rev[i]=mod-((mod//i)*self.rev[mod%i])%mod
            self.fact_rev[i]=(self.fact_rev[i-1]*self.rev[i])%mod
    def cnk(self,n,k):
        if n<k:
            return 0
        ans=(self.fact_rev[n-k]*self.fact_rev[k])%self.mod
        return (ans*self.fact[n])%self.mod

for _ in range(1):
    n = read_int()
    grid = [read_int_list() for _ in range(n)]
    
    cnk = combi(n + n, MOD).cnk
    # C = [[0] * (n + n + 5) for _ in range((n + n + 5))]
    # for i in range(n + n + 2):
    #     C[i][0] = 1
    #     for j in range(1, i + 1):
    #         C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD
    
    d = defaultdict(list)
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            d[x].append((i, j))
    
    def dpfunc(points):
        dp = [[0] * n for _ in range(n)]
        for i, j in points:
            dp[i][j] = 1
        for i in range(n):
            for j in range(n):
                if i:
                    dp[i][j] += dp[i - 1][j]
                if j:
                    dp[i][j] += dp[i][j - 1]
                dp[i][j] %= MOD
        return sum(dp[i][j] for i, j in points) % MOD
    
    def enumfunc(points):
        res = len(points)
        for i in range(len(points)):
            a, b = points[i]
            for j in range(i + 1, len(points)):
                u, v = points[j]
                if a <= u and b <= v:
                    res += cnk(u - a + v - b, v - b)
                    # res += C[u - a + v - b][v - b]
                    res %= MOD
        return res
    
    res = 0
    for xlist in d.values():
        if len(xlist) > n:
            res += dpfunc(xlist)
        else:
            res += enumfunc(xlist)
        res %= MOD
    print(res)