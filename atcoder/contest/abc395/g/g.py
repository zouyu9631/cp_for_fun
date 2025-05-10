standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)

DIR4 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
DIR8 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]


def solve():
    n, k = MII()
    INF = 10**18
    # 读入矩阵 C (0-indexed)
    C = [LII() for _ in range(n)]
    Q = II()
    queries = []
    for _ in range(Q):
        s, t = GMI()
        queries.append((s, t))
    
    # Floyd–Warshall 求全点最短路，得到 d[i][j]
    d = [row[:] for row in C]
    for kk in range(n):
        for i in range(n):
            dik = d[i][kk]
            for j in range(n):
                nd = dik + d[kk][j]
                if nd < d[i][j]:
                    d[i][j] = nd

    # Steiner DP 针对必选集合 S0 = {0,1,...,k-1}
    m = 1 << k
    dp = [[INF]*n for _ in range(m)]
    for i in range(k):
        mask = 1 << i
        for v in range(n):
            dp[mask][v] = d[v][i]
    for mask in range(1, m):
        sub = (mask - 1) & mask
        while sub:
            oth = mask ^ sub
            for v in range(n):
                cand = dp[sub][v] + dp[oth][v]
                if cand < dp[mask][v]:
                    dp[mask][v] = cand
            sub = (sub - 1) & mask
        # 松弛：对所有 u, v 用 d[u][v] 放松
        for u in range(n):
            du = dp[mask][u]
            for v in range(n):
                nd = du + d[u][v]
                if nd < dp[mask][v]:
                    dp[mask][v] = nd
    full = m - 1  # 全部必选点 mask

    # 计算两挂点 DP: F2[u][v] 表示 S0 连通且在 u,v 两挂点上的最优费用
    F2 = [[INF]*n for _ in range(n)]
    for v in range(n):
        F2[v][v] = dp[full][v]
    # 枚举 S0 非平凡拆分: mask 非空且不全
    for mask in range(1, full):
        mask2 = full ^ mask
        for u in range(n):
            a = dp[mask][u]
            if a == INF: 
                continue
            for v in range(n):
                b = dp[mask2][v]
                if b == INF:
                    continue
                cand = a + b + d[u][v]
                if cand < F2[u][v]:
                    F2[u][v] = cand

    # 预处理额外顶点答案:
    # 查询中 s,t 均来自 [k, n-1] (0-indexed)
    # ans[s][t] = min_{u,v} { F2[u][v] + min( d[u,s]+d[v,t], d[u,t]+d[v,s] ) }
    ans = [[INF]*n for _ in range(n)]
    for s in range(k, n):
        for t in range(s+1, n):
            best = INF
            for u in range(n):
                ds_u = d[u][s]
                dt_u = d[u][t]
                for v in range(n):
                    cand = F2[u][v] + ds_u + d[v][t]
                    if cand < best:
                        best = cand
                    cand2 = F2[u][v] + dt_u + d[v][s]
                    if cand2 < best:
                        best = cand2
            ans[s][t] = ans[t][s] = best

    out = []
    for s, t in queries:
        # 输入 s,t 为 1-indexed，且保证 s,t 均在 [k+1, n]
        # 转换为 0-indexed
        s0, t0 = s - 1, t - 1
        if s0 > t0:
            s0, t0 = t0, s0
        out.append(str(ans[s0][t0]))
    sys.stdout.write("\n".join(out))



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
