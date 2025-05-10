standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

from atcoder.lazysegtree import LazySegTree


def solve():
    N = II()
    A = LII()
    P = [0] * N
    s = {}
    c = 0
    for i in range(N):
        if A[i] not in s:
            s[A[i]] = 1
            c += 1
        P[i] = c
    S = [0] * N
    s = {}
    c = 0
    for i in range(N - 1, -1, -1):
        if A[i] not in s:
            s[A[i]] = 1
            c += 1
        S[i] = c

    dp = P[: N - 2]
    op = lambda a, b: a if a > b else b
    e = -109
    mapping = lambda f, x: x + f
    composition = lambda f, g: f + g
    id_ = 0
    seg = LazySegTree(op, e, mapping, composition, id_, dp)
    last = {}
    ans = -109
    for j in range(1, N - 1):
        x = A[j]
        L = last.get(x, 0)
        l = max(1, L + 1)
        if l <= j:
            seg.apply(l - 1, j, 1)
        last[x] = j
        cand = seg.prod(0, j) + S[j + 1]
        if cand > ans:
            ans = cand
    print(ans)


def solve_brute():
    N = II()
    A = LII()

    L = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        L[i] = len(seen)

    R = [0] * N
    seen = set()
    for i in range(N - 1, -1, -1):
        seen.add(A[i])
        R[i] = len(seen)
    res = 0
    j = 1  # j为中间段右端点下标（中间段为 A[i+1...j]），保证 j<=N-2
    mid_freq = [0] * (N + 1)  # 使用数组替换字典，因 A[i] 范围在 [1, N]
    mid_dist = 0  # 当前中间段不同数

    # 对于 i，要求中间段从 i+1 至 j
    for i in range(0, N - 2):
        # 确保 j 在合法区间内
        if j < i + 1:
            j = i + 1
            mid_freq = [0] * (N + 1)
            mid_dist = 0
        # 对于初始 i+1位置，如果窗口空则先加入 A[i+1]（当 j==i+1）
        if j == i + 1 and j <= N - 2:
            x = A[j]
            mid_freq[x] += 1
            if mid_freq[x] == 1:
                mid_dist += 1
        # 尝试向右延伸 j，贪心延伸只要能使 (mid_dist + R[j+1]) 不减
        while j < N - 2:
            # 当前值
            cur_val = mid_dist + R[j + 1]
            # 预知延伸后的值
            next_elem = A[j + 1]
            cnt = mid_freq[next_elem]
            next_dist = mid_dist + (1 if cnt == 0 else 0)
            next_val = next_dist + R[j + 2]
            # 如果延伸后得分不低，则延伸
            if next_val >= cur_val:
                j += 1
                mid_freq[next_elem] += 1
                if cnt == 0:
                    mid_dist += 1
            else:
                break
        total = L[i] + mid_dist + R[j + 1]
        if total > res:
            res = total
        # 移动左指针：把 A[i+1] 从中间段移除
        rem = A[i + 1]
        mid_freq[rem] -= 1
        if mid_freq[rem] == 0:
            mid_dist -= 1
        # 注意：不回退 j，全局 j 单调不回退
    print(res)


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
                self.buffer.seek(0, 2), self.write(b), self.buffer.seek(ptr)
            self.newlines = 0
            return self.buffer.read()

        def readline(self):
            while self.newlines == 0:
                b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                self.newlines = b.count(b"\n") + (not b)
                ptr = self.buffer.tell()
                self.buffer.seek(0, 2), self.write(b), self.buffer.seek(ptr)
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
