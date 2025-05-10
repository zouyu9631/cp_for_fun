standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

# 虚树

class AuxiliaryTree:
    def __init__(self, n, edge, root=0):
        self.n = n
        self.edge = edge
        """
        eular: dfs时节点的访问顺序
        first: 每个节点在欧拉序列中第一次出现的位置
        """
        self.eular = [-1] * (2 * n - 1)
        self.first = [-1] * n
        self.depth = [-1] * n
        self.lgs = [0] * (2 * n)
        for i in range(2, 2 * n):
            self.lgs[i] = self.lgs[i >> 1] + 1
        self.st = []
        self.G = [[] for i in range(n)]

        self.dfs(root)
        self.construct_sparse_table()

    def dfs(self, root):
        stc = [root]
        self.depth[root] = 0
        num = 0
        while stc:
            v = stc.pop()
            if v >= 0:
                self.eular[num] = v
                self.first[v] = num
                num += 1
                for u in self.edge[v][::-1]:
                    if self.depth[u] == -1:
                        self.depth[u] = self.depth[v] + 1
                        stc.append(~v)
                        stc.append(u)
            else:
                self.eular[num] = ~v
                num += 1

    def construct_sparse_table(self):
        self.st.append(self.eular)
        sz = 1
        while 2 * sz <= 2 * self.n - 1:
            prev = self.st[-1]
            nxt = [0] * (2 * self.n - 2 * sz)
            for j in range(2 * self.n - 2 * sz):
                v = prev[j]
                u = prev[j + sz]
                if self.depth[v] <= self.depth[u]:
                    nxt[j] = v
                else:
                    nxt[j] = u
            self.st.append(nxt)
            sz *= 2

    def lca(self, u, v):
        x = self.first[u]
        y = self.first[v]
        if x > y:
            x, y = y, x
        d = self.lgs[y - x + 1]
        return (
            self.st[d][x]
            if self.depth[self.st[d][x]] <= self.depth[self.st[d][y - (1 << d) + 1]]
            else self.st[d][y - (1 << d) + 1]
        )

    def dist(self, u, v):
        return self.depth[u] + self.depth[v] - 2 * self.depth[self.lca(u, v)]

    def query(self, vs):
        """
        vs: 虚树中包含的顶点集合
        self.G: 虚树
        return: 虚树的根 (sz中所有顶点的lca)
        query 后 vs 里还可能包含中间节点
        """

        k = len(vs)
        if k == 0:
            return -1
        vs.sort(key=self.first.__getitem__)
        stc = [vs[0]]
        self.G[vs[0]] = []

        for i in range(k - 1):
            w = self.lca(vs[i], vs[i + 1])
            if w != vs[i]:
                last = stc.pop()
                while stc and self.depth[w] < self.depth[stc[-1]]:
                    self.G[stc[-1]].append(last)
                    last = stc.pop()

                if not stc or stc[-1] != w:
                    stc.append(w)
                    vs.append(w)
                    self.G[w] = [last]
                else:
                    self.G[w].append(last)
            stc.append(vs[i + 1])
            self.G[vs[i + 1]] = []

        for i in range(len(stc) - 1):
            self.G[stc[i]].append(stc[i + 1])

        return stc[0]

def solve():
    N = II()
    G = GRAPH(N)
    A = list(map(int, input().split()))

    ATree = AuxiliaryTree(N, G)

    C = [[] for _ in range(N + 1)]
    for i in range(N):
        C[A[i]].append(i)

    cnt = [0] * N
    parent = [-1] * N
    res = 0

    for color, group in enumerate(C):
        if not group:
            continue
        n = len(group)
        root = ATree.query(group)

        order = [root]
        stack = [root]
        while stack:
            u = stack.pop()
            cnt[u] = 0
            for v in ATree.G[u]:
                parent[v] = u
                order.append(v)
                stack.append(v)

        for u in order[::-1]:
            for v in ATree.G[u]:
                res += cnt[v] * (n - cnt[v]) * ATree.dist(u, v)
                cnt[u] += cnt[v]
            if A[u] == color:
                cnt[u] += 1

    print(res)


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
