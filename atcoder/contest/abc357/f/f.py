standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 1
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

from atcoder.lazysegtree import LazySegTree

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)


class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, idx, value):
        while idx <= self.n:
            self.tree[idx] += value
            self.tree[idx] %= MOD
            idx += idx & -idx

    def _sum(self, idx):
        result = 0
        while idx > 0:
            result += self.tree[idx]
            result %= MOD
            idx -= idx & -idx
        return result


class RangeAddRangeSum:
    def __init__(self, nums):
        n = len(nums)
        self.B1 = FenwickTree(n)
        self.B2 = FenwickTree(n)
        for i, num in enumerate(nums, 1):
            self.add(i, i, num)

    def add(self, l, r, value):
        self.B1.add(l, value)
        self.B1.add(r + 1, -value)
        self.B2.add(l, value * (l - 1))
        self.B2.add(r + 1, -value * r)

    def prefix_sum(self, idx):
        return (self.B1._sum(idx) * idx - self.B2._sum(idx)) % MOD

    def range_sum(self, l, r):
        return (self.prefix_sum(r) - self.prefix_sum(l - 1)) % MOD

    def __repr__(self) -> str:
        return str([self.range_sum(i, i) for i in range(1, self.B1.n + 1)])


def solve_fenwick():
    N, Q = MII()
    A = LII()
    B = LII()
    C = [a * b % MOD for a, b in zip(A, B)]
    sa = RangeAddRangeSum(A)
    sb = RangeAddRangeSum(B)
    sc = RangeAddRangeSum(C)

    for _ in range(Q):
        ops = LII()
        if ops[0] == 3:
            _, l, r = ops
            res = sc.range_sum(l, r + 1)
            print(res)
        else:
            t, l, r, x = ops
            r += 1
            n = r - l
            if t == 1:
                # lst.apply(l, r, (x, 0))
                print(sa)
                sa.add(l, r, x)
                print(sa)
            else:  # t == 2
                # lst.apply(l, r, (0, x))
                print(sb)
                sb.add(l, r, x)
                print(sb)


def solve():
    N, Q = MII()
    A = LII()
    B = LII()

    MASK = (1 << 30) - 1

    def gen(a, b):
        return 1 << 90 | a << 60 | b << 30 | a * b % MOD

    e = 0

    def op(x, y):
        xl, xa, xb, xc = x >> 90, (x >> 60) & MASK, (x >> 30) & MASK, x & MASK
        yl, ya, yb, yc = y >> 90, (y >> 60) & MASK, (y >> 30) & MASK, y & MASK
        return (
            (xl + yl) << 90
            | ((xa + ya) % MOD) << 60
            | ((xb + yb) % MOD) << 30
            | (xc + yc) % MOD
        )

    id_ = 0

    def mapping(f, x):
        # n, sa, sb, sc = x
        # da, db = f
        # return (
        #     n,
        #     (sa + n * da) % MOD,
        #     (sb + n * db) % MOD,
        #     (sc + da * sb + db * sa + n * da * db) % MOD,
        # )

        xl, xa, xb, xc = x >> 90, (x >> 60) & MASK, (x >> 30) & MASK, x & MASK
        da, db = f >> 30, f & MASK
        return (
            xl << 90
            | (xa + xl * da) % MOD << 60
            | (xb + xl * db) % MOD << 30
            | (xc + da * xb + db * xa + xl * da * db) % MOD
        )

    def composition(f, g):
        fa, fb = f >> 30, f & MASK
        ga, gb = g >> 30, g & MASK
        return ((fa + ga) % MOD) << 30 | (fb + gb) % MOD

    lst = LazySegTree(
        op, e, mapping, composition, id_, [gen(a, b) for a, b in zip(A, B)]
    )

    for _ in range(Q):
        ops = LII()
        if ops[0] == 3:
            _, l, r = ops
            res = lst.prod(l - 1, r)
            print(res & MASK)
        else:
            t, l, r, x = ops
            l -= 1
            n = r - l
            if t == 1:
                lst.apply(l, r, x << 30)
            else:  # t == 2
                lst.apply(l, r, x)


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
