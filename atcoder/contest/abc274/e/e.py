from math import sqrt, inf
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

def solve():
    n, m = read_int_tuple()
    nm = n + m
    N = 1 << nm
    
    bitcount = [0] * (1 << m)
    for mask in range(1, 1 << m):
        bitcount[mask] = 1 + bitcount[mask ^ (mask & -mask)]
    
    P = [read_int_tuple() for _ in range(nm)]
    dist = [[0] * nm for _ in range(nm)]
    dp = [inf] * (N * nm)
    for i in range(nm):
        dist[i][i] = dp[(1 << i) * nm + i] = sqrt(P[i][0] * P[i][0] + P[i][1] * P[i][1])
        for j in range(i + 1, nm):
            dist[j][i] = dist[i][j] = sqrt((P[i][0] - P[j][0]) * (P[i][0] - P[j][0]) + (P[i][1] - P[j][1]) * (P[i][1] - P[j][1]))

    for mask in range(1, N):
        v = 1 << bitcount[mask >> n]
        Z = mask * nm
        for j in range(nm):
            if mask & (1 << j): continue
            for i in range(nm):
                if mask & (1 << i) == 0: continue
                # nxt = mask | (1 << j)
                nxt = Z + (1 << j) * nm + j
                rt = dp[Z + i] + dist[i][j] / v
                if dp[nxt] > rt: dp[nxt] = rt
    
    # for mask in range(N):
    #     print(mask, dp[mask * nm: mask * nm + nm])
    
    R = (1 << n) - 1
    res = inf
    for mask in range(1 << m):
        v = 1 << bitcount[mask] # bin(mask).count('1')
        mask = mask << n | R
        Z = mask * nm
        tr = min(dp[Z + i] + dist[i][i] / v for i in range(nm))
        # tr = min(t + dist[i][i] / v for i, t in enumerate(dp[mask]))
        if res > tr: res = tr
    
    print(res)
   


T = 1#read_int()
for t in range(T):
    solve()