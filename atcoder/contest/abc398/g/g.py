standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)

DIR4 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
DIR8 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]


def solve():
    N, M = MII()

    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = MII()
        graph[u].append(v)
        graph[v].append(u)

    # 使用 BFS 计算每个连通分量的二分染色信息
    color = [-1] * (N + 1)  # -1 表示未染色，颜色为 0/1
    from collections import deque

    # 初始化参数：
    # x: 各分量内可补边数之和
    # ee, oo, eo: 连通分量的分类计数
    # iso: 孤立点数
    x = 0
    ee = 0
    oo = 0
    eo = 0
    iso = 0

    for i in range(1, N + 1):
        if color[i] != -1:
            continue
        dq = deque()
        dq.append(i)
        color[i] = 0
        comp = []  # 存储当前连通分量的顶点
        edge_cnt = 0  # 记录边的访问次数（每条边会被访问两次）
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in graph[u]:
                edge_cnt += 1
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    dq.append(v)
        # 每条边计数两次
        m_comp = edge_cnt // 2
        size = len(comp)
        if size == 1:
            iso += 1
            continue
        # 统计二分组中各自顶点数
        cnt0 = sum(1 for u in comp if color[u] == 0)
        cnt1 = size - cnt0
        # 在该连通分量内，最大二分图边数为 cnt0*cnt1，
        # 因此还能添加的边数为 cnt0*cnt1 - m_comp
        extra = cnt0 * cnt1 - m_comp
        x += extra
        # 分类：判断两侧顶点数奇偶性
        if (cnt0 & 1) == 0 and (cnt1 & 1) == 0:
            ee += 1
        elif (cnt0 & 1) and (cnt1 & 1):
            oo += 1
        else:
            eo += 1

    # 最终只需关心各参数的模2情况（x 和 oo、iso/2）
    x_mod = x & 1
    oo_mod = oo & 1

    out = []
    if N & 1:  # N 为奇数
        # 结果取决于 (oo + x) mod 2
        if (oo_mod + x_mod) & 1:
            out.append("Aoki")
        else:
            out.append("Takahashi")
    else:
        # N 为偶数
        if eo == 0:
            # 此时结果取决于 (iso/2 + x) mod 2，注意 iso 必为偶数
            if ((iso // 2) + x_mod) & 1:
                out.append("Aoki")
            else:
                out.append("Takahashi")
        elif 1 <= eo <= 2:
            out.append("Aoki")
        else:  # eo >= 3
            if (oo_mod + x_mod) & 1:
                out.append("Aoki")
            else:
                out.append("Takahashi")
    print(*out)


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
