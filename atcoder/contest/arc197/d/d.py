standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 1
output_mode = 1

MOD = 998244353


FAC = [1] * 401
for i in range(2, 401):
    FAC[i] = FAC[i - 1] * i % MOD


def rowBlocks(mat):
    repId, reps, sz, bel = {}, [], [], []
    for i, row in enumerate(mat):
        key = tuple(row)
        if key not in repId:
            bid = len(reps)
            repId[key] = bid
            reps.append(i)
            sz.append(0)
        bid = repId[key]
        bel.append(bid)
        sz[bid] += 1
    return reps, sz, bel


def buildU(mat, reps):
    m = len(reps)
    U = [0] * m
    for i, ri in enumerate(reps):
        mask = 0
        for j, rj in enumerate(reps):
            if mat[ri][rj] == 0:
                mask |= 1 << j
        U[i] = mask
    return U


def findParent(U, root):
    m = len(U)
    par = [-1] * m
    for c in range(m):
        if c == root:
            continue
        best, bestCnt, tie = -1, -1, False
        for p in range(m):
            if p == c:
                continue
            if U[p] & ~U[c] == 0 and U[p] != U[c]:
                cnt = U[p].bit_count()
                if cnt > bestCnt:
                    best, bestCnt, tie = p, cnt, False
                elif cnt == bestCnt:
                    tie = True
        if best == -1 or tie:
            return None
        par[c] = best
    return par


def treeOk(par, root):
    m = len(par)
    g = [[] for _ in range(m)]
    for v, p in enumerate(par):
        if p != -1:
            g[p].append(v)
    vis = [False] * m
    stk = [root]
    vis[root] = True
    while stk:
        v = stk.pop()
        for w in g[v]:
            if vis[w]:
                return False
            vis[w] = True
            stk.append(w)
    return all(vis)


def ancTest(i, j, par):
    while j != -1:
        if i == j:
            return True
        j = par[j]
    return False


def matMatch(mat, reps, par):
    m = len(reps)
    for i in range(m):
        for j in range(m):
            need = 1 if (ancTest(i, j, par) or ancTest(j, i, par)) else 0
            if need != mat[reps[i]][reps[j]]:
                return False
    return True


def countWays(sz, root):
    res = FAC[sz[root] - 1]
    for k, s in enumerate(sz):
        if k != root:
            res = res * FAC[s] % MOD
    return res


def solve():
    n = II()
    A = [LII() for _ in range(n)]

    if any(A[0][j] == 0 for j in range(n)):
        return 0

    reps, sz, bel = rowBlocks(A)
    root = bel[0]

    U = buildU(A, reps)
    par = findParent(U, root)
    if par is None or not treeOk(par, root) or not matMatch(A, reps, par):
        return 0
    else:
        return countWays(sz, root)


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
