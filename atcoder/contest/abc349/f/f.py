from math import gcd
from random import randint


standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 1  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)


def solve():
    N, M = MII()
    A = LII()
    if M == 1:
        t = A.count(1)
        return pow(2, t, MOD) - 1

    pm = Factor.getPrimeFactors(M)
    pp = [pow(p, c) for p, c in pm.items()]
    ln = len(pp)
    cnt = [0] * (1 << ln)
    for x, c in Counter(A).items():
        if M % x:
            continue
        mask = sum(1 << i for i in range(ln) if x % pp[i] == 0)
        cnt[mask] += c
    # print(cnt)
    N = sum(cnt)
    pow2 = [1] * (N + 1)
    for i in range(1, N + 1):
        pow2[i] = pow2[i - 1] * 2 % MOD

    for b in range(ln):
        for mask in range(1 << ln):
            if mask & (1 << b):
                cnt[mask] += cnt[mask ^ (1 << b)]

    res = 0
    for mask in range(1 << ln):
        k = -1 if mask.bit_count() & 1 else 1
        res += k * (pow2[cnt[mask]] - 1)
        res %= MOD
    if ln & 1:
        res = (-res) % MOD
    return res


class Factor:
    @staticmethod
    def getPrimeFactors(n: int):
        """n 的质因数分解 基于PR算法 O(n^1/4*logn)"""
        res = defaultdict(int)
        while n > 1:
            p = Factor._PollardRho(n)
            while n % p == 0:
                res[p] += 1
                n //= p
        return res

    @staticmethod
    def _MillerRabin(n: int, k: int = 10) -> bool:
        """米勒-拉宾素性检验(MR)算法判断n是否是素数 O(k*logn*logn)"""
        if n == 2 or n == 3:
            return True
        if n < 2 or n % 2 == 0:
            return False
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(k):
            a = randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    @staticmethod
    def _PollardRho(n: int) -> int:
        """PR算法求n的一个因数 O(n^1/4)"""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if Factor._MillerRabin(n):
            return n

        x, c = randint(1, n - 1), randint(1, n - 1)
        y, res = x, 1
        while res == 1:
            x = (x * x % n + c) % n
            y = (y * y % n + c) % n
            y = (y * y % n + c) % n
            res = gcd(abs(x - y), n)

        return res if Factor._MillerRabin(res) else Factor._PollardRho(n)


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
