standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

INF = 10**9
NEG = -INF


def solve():
    n, k = MII()
    s = I()

    dp0 = [0] * (n + 1)
    dp1 = [0] * (n + 1)
    mn0 = [0] * (n + 1)
    mn1 = [0] * (n + 1)
    dp0[n] = dp1[n] = 0
    mn0[n] = mn1[n] = 0
    for i in range(n - 1, -1, -1):
        c = s[i]
        if c == "o":
            dp0[i] = 1 + dp1[i + 1]
            mn0[i] = 1 + mn1[i + 1]
            dp1[i] = NEG
            mn1[i] = INF
        elif c == ".":
            dp0[i] = dp0[i + 1]
            mn0[i] = mn0[i + 1]
            dp1[i] = dp0[i + 1]
            mn1[i] = mn0[i + 1]
        else:
            dp0[i] = max(dp0[i + 1], 1 + dp1[i + 1])
            mn0[i] = min(mn0[i + 1], 1 + mn1[i + 1])
            dp1[i] = dp0[i + 1]
            mn1[i] = mn0[i + 1]

    def check(t):
        return t[0] <= t[1]

    def merge(t1, t2):
        if not check(t1):
            return t2
        if not check(t2):
            return t1
        return (min(t1[0], t2[0]), max(t1[1], t2[1]))

    F0, F1 = [(0, 0)], [(INF, -INF)]
    for i in range(n):
        c = s[i]
        p0, p1 = F0[-1], F1[-1]
        n0, n1 = (INF, -INF), (INF, -INF)
        if check(p0):
            if c == "o":
                n1 = (p0[0] + 1, p0[1] + 1)
            elif c == ".":
                n0 = p0
            else:
                n1 = (p0[0] + 1, p0[1] + 1)
                n0 = p0
        if check(p1) and c != "o":
            n0 = merge(n0, p1)
        F0.append(n0)
        F1.append(n1)

    res = []
    for i in range(n):
        if s[i] != "?":
            res.append(s[i])
        else:
            poss_o = poss_dot = False
            if check(F0[i]):
                lo = F0[i][0] + 1 + mn1[i + 1]
                hi = F0[i][1] + 1 + dp1[i + 1]
                if lo <= k <= hi:
                    poss_o = True
            pre = merge(F0[i], F1[i])
            if check(pre):
                lo = pre[0] + mn0[i + 1]
                hi = pre[1] + dp0[i + 1]
                if lo <= k <= hi:
                    poss_dot = True
            if poss_o and not poss_dot:
                res.append("o")
            elif poss_dot and not poss_o:
                res.append(".")
            else:
                res.append("?")
    print("".join(res))


def main():
    # region local test
    if read_from_file and "AW" in os.environ.get("COMPUTERNAME", ""):
        test_no = read_from_file
        f = open(os.path.dirname(__file__) + f"\\in{test_no}.txt", "r")
        global input
        input = lambda: f.readline().rstrip("\r\n")
    # endregion

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


# region
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


# endregion

if __name__ == "__main__":
    main()
