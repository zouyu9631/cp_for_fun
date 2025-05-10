standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

"""
euler tour
欧拉序 树上差分模板
"""

def solve():
    n = II()

    g = [[] for _ in range(n)]
    for _ in range(n - 1):
        s, y = GMI()
        g[s].append(y)
        g[y].append(s)

    order, size = [], [1] * n
    par, pos = [-1] * n, [0] * n

    # @bootstrap
    # def dfs(u: int, p=-1):
    #     par[u], pos[u] = p, len(order)
    #     order.append(u)
    #     for v in g[u]:
    #         if v == p:
    #             continue
    #         yield dfs(v, u)
    #         size[u] += size[v]
    #     yield None

    # dfs(0)

    stk = [0]
    while stk:
        u = stk.pop()
        pos[u] = len(order)
        order.append(u)
        for v in g[u]:
            if v == par[u]:
                continue
            par[v] = u
            stk.append(v)

    for u in reversed(order):
        if u:
            size[par[u]] += size[u]

    # debug(order, size, par, pos, sep="\n")

    bit = BIT(n)
    dif = [0] * (n + 1)

    for w in range(n):
        for s in g[w]:
            if s != par[w]:
                x = bit.sum_range(pos[s], pos[s] + size[s])
                dif[0] += x
                dif[n] -= x

                dif[pos[s]] -= x
                dif[pos[s] + size[s]] += x
        x = w - bit.sum_range(pos[w], pos[w] + size[w])
        dif[pos[w]] += x
        dif[pos[w] + size[w]] -= x
        bit.add(pos[w], 1)

    res, cur = [0] * n, 0
    for u in range(n):
        cur += dif[u]
        res[order[u]] = cur
    print(*res)


class BIT:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def build(self, list):
        self.tree[1:] = list.copy()
        for i in range(self.size + 1):
            j = i + (i & (-i))
            if j < self.size + 1:
                self.tree[j] += self.tree[i]

    def sum(self, i):
        # return sum(arr[0: i])
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def sum_range(self, l, r):
        # sum(arr[l: r]) -> return self.sum(r) - self.sum(l)
        s = 0
        while l < r:
            s += self.tree[r]
            r -= r & (-r)
        while r < l:
            s -= self.tree[l]
            l -= l & (-l)
        return s

    def add(self, i, x):
        # arr[i] += 1
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def __getitem__(self, i):
        # return arr[i]
        return self.sum_range(i, i + 1)

    def __repr__(self):
        return "BIT({0})".format([self[i] for i in range(self.size)])

    def __setitem__(self, i, x):
        # arr[i] = x
        self.add(i, x - self[i])

    def bisect(self, x):
        # 总和大于等于x的位置的index
        le = 0
        ri = 1 << (self.size.bit_length() - 1)
        while ri > 0:
            if le + ri <= self.size and self.tree[le + ri] < x:
                x -= self.tree[le + ri]
                le += ri
            ri >>= 1
        return le + 1

    # def __repr__(self):
    #     return f'arr: {list(self)}\npsum: {list(self.sum(i) for i in range(self.size + 1))}\ntree: {self.tree}'


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
