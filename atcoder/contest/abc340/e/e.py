standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

from atcoder import lazysegtree
from operator import add


def solve_lst():
    N, M = MII()
    A = LII()
    B = LII()

    lst = lazysegtree.LazySegTree(add, 0, add, add, 0, A)

    for i in B:
        c = lst.get(i)
        lst.set(i, 0)
        t, r = divmod(c, N)
        lst.apply(0, N, t)
        lst.apply(i + 1, min(N, i + r + 1), 1)
        lst.apply(0, max(0, i + r + 1 - N), 1)

    print(*[lst.get(i) for i in range(N)])


class BIT_RG:

    def __init__(self, nums):
        n = nums if type(nums) is int else len(nums)
        self.n = n
        self.sum = [0 for _ in range(n + 2)]
        self.ntimessum = [0 for _ in range(n + 2)]
        if type(nums) is not int:
            for i in range(1, self.n + 1):
                dif = nums[i - 1] - (nums[i - 2] if i > 1 else 0)
                self.sum[i] += dif
                self.ntimessum[i] += dif * (i - 1)
                j = i + (i & (-i))
                if j <= self.n:
                    self.sum[j] += self.sum[i]
                    self.ntimessum[j] += self.ntimessum[i]

    def _set(self, idx, x):
        i = idx - 1
        while idx <= self.n:
            self.sum[idx] += x
            self.ntimessum[idx] += x * i
            idx += idx & (-idx)

    def set(self, idx, x):
        self._set(idx, x)
        self._set(idx + 1, -x)

    def update_range(self, l, r, x):
        """[l, r)"""
        self._set(l, x)
        self._set(r, -x)

    def _get(self, idx):
        res = 0
        i = idx
        while idx:
            res += i * self.sum[idx] - self.ntimessum[idx]
            idx -= idx & (-idx)
        return res

    def get(self, idx):
        return self._get(idx) - self._get(idx - 1)


def solve():
    N, M = MII()
    A = LII()
    B = LII()

    bit = BIT_RG(N + 2)

    for i, x in enumerate(A, 1):
        bit.set(i, x)

    for i in B:
        i += 1
        c = bit.get(i)
        bit.set(i, -c)
        t, r = divmod(c, N)
        bit.update_range(1, N + 1, t)
        bit.update_range(i + 1, min(N + 1, i + r + 1), 1)
        bit.update_range(1, max(1, i + r + 1 - N), 1)

    print(*[bit.get(i) for i in range(1, N + 1)])



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
