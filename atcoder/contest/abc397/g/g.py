standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0

from bisect import bisect_left
from atcoder.maxflow import MFGraph


def solve():
    N, M, K = MII()
    u = [0] * M
    v = [0] * M
    for i in range(M):
        u[i], v[i] = GMI()

    inf = 1000  # 足够大的数

    def check(x):
        # 对 x==0 单独处理（永远可行，因为每条路径至少有 0 条1边）
        if x == 0:
            return True
        total = N * x + 2  # 每个顶点有 x 个层节点，加上 S 和 T
        S = N * x
        T = S + 1
        g = MFGraph(total)
        # 对于每个顶点 i，构造层内边以及与 S、T 的边
        for i in range(N):
            # 从 i 的第 0 层到 S
            g.add_edge(i * x + 0, S, inf)
            # 从 T 到 i 的最后一层
            g.add_edge(T, i * x + (x - 1), inf)
            # 层内传递：从第 j+1 层到第 j 层，容量 inf
            for j in range(x - 1):
                g.add_edge(i * x + j + 1, i * x + j, inf)
        # 加入额外边：起点和终点
        g.add_edge(0 * x + 0, T, inf)
        g.add_edge(S, (N - 1) * x + (x - 1), inf)
        # 对于每条原图边 (u,v)，添加两类边
        for i in range(M):
            # 第一类：对 j=0...x-1，添加从 v 的 j 层到 u 的 j 层，容量 1
            for j in range(x):
                g.add_edge(v[i] * x + j, u[i] * x + j, 1)
            # 第二类：对 j=0...x-2，添加从 v 的 j+1 层到 u 的 j 层，容量 inf
            for j in range(x - 1):
                g.add_edge(v[i] * x + j + 1, u[i] * x + j, inf)
        f = g.flow(S, T)
        return f <= K

    # 二分答案：答案 x 的范围为 0 ~ N-1
    
    # lo = 0
    # hi = N - 1
    # while lo < hi:
    #     mid = (lo + hi + 1) // 2
    #     if check(mid):
    #         lo = mid
    #     else:
    #         hi = mid - 1
    # print(lo)
    
    print(bisect_left(range(N), True, key=lambda x: not check(x)) - 1)


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
