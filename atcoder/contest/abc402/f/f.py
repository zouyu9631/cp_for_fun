standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 1, 0, 0
multi_test = 0
output_mode = 0

MOD = 998244353

DIR4 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
DIR8 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]


from bisect import bisect_left


def solve():
    N, M = MII()
    A = [LII() for _ in range(N)]

    if N == 1:
        print(A[0][0] % M)
        return

    PT = [1] * N
    for i in range(1, N):
        PT[i] = (PT[i - 1] * 10) % M
    P = PT[-1]

    pref = [[[] for _ in range(N)] for _ in range(N)]
    suf = [[[] for _ in range(N)] for _ in range(N)]

    @bootstrap
    def dfs1(x, y, r):
        if x + y == N - 1:
            pref[x][y].append(r)
            yield None
        if x + 1 < N:
            yield dfs1(x + 1, y, (r * 10 + A[x + 1][y]) % M)
        if y + 1 < N:
            yield dfs1(x, y + 1, (r * 10 + A[x][y + 1]) % M)
        yield None

    dfs1(0, 0, A[0][0] % M)

    @bootstrap
    def dfs2(x, y, r, d):
        if x + y == N - 1:
            suf[x][y].append(r)
            yield None
        nr = (r + A[x][y] * PT[d]) % M
        if x - 1 >= 0:
            yield dfs2(x - 1, y, nr, d + 1)
        if y - 1 >= 0:
            yield dfs2(x, y - 1, nr, d + 1)
        yield None

    dfs2(N - 1, N - 1, 0, 0)

    res = 0

    for i in range(N):
        j = (N - 1) - i
        if 0 <= j < N and pref[i][j] and suf[i][j]:
            SF = sorted(suf[i][j])
            MX = SF[-1]
            for r1 in pref[i][j]:
                r1p = (r1 * P) % M
                t = (M - r1p) % M
                idx = bisect_left(SF, t)
                if idx > 0:
                    r2 = SF[idx - 1]
                    tmp = r1p + r2
                else:
                    r2 = MX
                    tmp = r1p + r2 - M
                res = max(res, tmp % M)
    print(res)


def main():

    if read_from_file and "AW" in os.environ.get("COMPUTERNAME", ""):
        test_no = read_from_file
        f = open(os.path.dirname(__file__) + f"\\in{test_no}.txt", "r")
        global input
        input = lambda: f.readline().rstrip("\r\n")

    T = II() if multi_test else 1
    for t in range(T):
        if output_mode == 0:
            solve()
        elif output_mode == 1:
            print(solve())
        elif output_mode == 2:
            print("YES" if solve() else "NO")
        elif output_mode == 3:
            print("Yes" if solve() else "No")


if standard_input:
    import os, sys, math

    input = lambda: sys.stdin.readline().strip()

    inf = math.inf

    def I():
        return input()

    def II():
        return int(input())

    def MII():
        return map(int, input().split())

    def LI():
        return list(input().split())

    def LII():
        return list(map(int, input().split()))

    def LFI():
        return list(map(float, input().split()))

    def GMI():
        return map(lambda x: int(x) - 1, input().split())

    def LGMI():
        return list(map(lambda x: int(x) - 1, input().split()))

    def GRAPH(n: int, m=-1):
        if m == -1:
            m = n - 1
        g = [[] for _ in range(n)]
        for _ in range(m):
            u, v = GMI()
            g[u].append(v)
            g[v].append(u)
        return g


if packages:
    from io import BytesIO, IOBase
    import math
    import random
    import bisect
    import typing
    from collections import Counter, defaultdict, deque
    from copy import deepcopy
    from functools import cmp_to_key, lru_cache, reduce
    from heapq import *
    from itertools import accumulate, combinations, permutations, count, product
    from operator import add, iand, ior, itemgetter, mul, xor
    from string import ascii_lowercase, ascii_uppercase, ascii_letters
    from typing import *

    BUFSIZE = 4096

if output_together:

    class FastIO(IOBase):
        newlines = 0

        def __init__(self, file):
            self._fd = file.fileno()
            self.buffer = BytesIO()
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

    class IOWrapper(IOBase):
        def __init__(self, file):
            self.buffer = FastIO(file)
            self.flush = self.buffer.flush
            self.writable = self.buffer.writable
            self.write = lambda s: self.buffer.write(s.encode("ascii"))
            self.read = lambda: self.buffer.read().decode("ascii")
            self.readline = lambda: self.buffer.readline().decode("ascii")

    sys.stdout = IOWrapper(sys.stdout)

if dfs_tag:
    from types import GeneratorType

    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if type(to) is GeneratorType:
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to

        return wrappedfunc


if int_hashing:
    RANDOM = random.getrandbits(20)

    class Wrapper(int):
        def __init__(self, x):
            int.__init__(x)

        def __hash__(self):
            return super(Wrapper, self).__hash__() ^ RANDOM


if True:

    def debug(*args, **kwargs):
        print("\033[92m", end="")
        print(*args, **kwargs)
        print("\033[0m", end="")


if __name__ == "__main__":
    main()
