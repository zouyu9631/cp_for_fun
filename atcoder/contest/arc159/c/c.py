import random
import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

"""
题意： 给一个数组，每次整体加上一个排列，问最少多少次可以使得数组中的数全都一样
"""

def solve():
    n = read_int()
    A = read_int_list()
    res = []
    if sum(A) % n:
        if (sum(A) + n * (n - 1) // 2) % n:
            print('No')
            return
        P = [-1] * n
        for i, j in enumerate(sorted(range(n), key=A.__getitem__)):
            A[j] += i
            P[j] = i
        
        res.append(P)
        
        # P = list(range(n))
        # random.shuffle(P)
        
        # for i, p in enumerate(P):
        #     A[i] += p
        
        # res.append(P)

    print('Yes')
    while True:
        hi, hi_idx = -inf, -1
        lo, lo_idx = inf, -1
        for i, x in enumerate(A):
            if hi < x:
                hi = x
                hi_idx = i
            if lo > x:
                lo = x
                lo_idx = i
        if hi == lo:
            break
        
        t = min((hi - lo) // 2, n - 1)
        x_gen = iter(i for i in range(1, n) if i != t)
        
        P, Q = [-1] * n, [-1] * n
        
        for i in range(n):
            if i not in (hi_idx, lo_idx):
                x = next(x_gen)
                P[i] = n - 1 - x
                Q[i] = x
        P[lo_idx], P[hi_idx] = n - 1, n - 1 - t
        Q[lo_idx], Q[hi_idx] = t, 0
        for i in range(n):
            A[i] += P[i] + Q[i]
        res.append(P)
        res.append(Q)

    print(len(res))
    for R in res:
        print(*(x + 1 for x in R))


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 3
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1  #read_int()
    for t in range(T):
        solve()
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

input = lambda: sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_ints_minus_one():
    return [int(x) - 1 for x in input().split()]


def read_int_tuple():
    return map(int, input().split())


def read_graph(n: int, m: int, d=1):
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        g[u - d].append(v - d)
        g[v - d].append(u - d)
    return g


def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()