standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 1, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No


def solve():
    N = II()
    edge = []
    for _ in range(N - 1):
        u, v = GMI()
        edge.append((u, v))

    G = [[] for _ in range(N)]
    for u, v in edge:
        G[u].append(v)
        G[v].append(u)

    col = [-1] * N

    @bootstrap
    def dfs(u, c):
        col[u] = c
        for v in G[u]:
            if col[v] == -1:
                yield dfs(v, 1 - c)
        yield None

    dfs(0, 0)

    cnt0 = sum(1 for c in col if c == 0)
    cnt1 = N - cnt0

    m = cnt0 * cnt1 - (N - 1)

    if m % 2 == 1:
        turn = "first"
        print("First", flush=True)
    else:
        turn = "second"
        print("Second", flush=True)

    cand = []
    for i in range(N):
        for j in range(i + 1, N):
            if col[i] != col[j]:
                cand.append((i, j))

    used = [[False] * N for _ in range(N)]

    for u, v in edge:
        used[u][v] = used[v][u] = True

    def get_move():
        for u, v in cand:
            if not used[u][v]:
                return u, v
        return None

    if turn == "first":
        while True:
            mv = get_move()
            if mv is None:
                break
            u, v = mv

            used[u][v] = used[v][u] = True

            print(u + 1, v + 1, flush=True)

            ou, ov = GMI()
            if ou == -2 and ov == -2:
                break
            used[ou][ov] = used[ov][ou] = True
    else:
        while True:
            ou, ov = GMI()
            if ou == -2 and ov == -2:
                break
            used[ou][ov] = used[ov][ou] = True

            mv = get_move()
            if mv is None:
                break
            u, v = mv
            used[u][v] = used[v][u] = True
            print(u + 1, v + 1, flush=True)

    # color = [-1] * N
    # dq = deque()
    # color[0] = 0
    # dq.append(0)
    # while dq:
    #     cur = dq.popleft()
    #     for nx in G[cur]:
    #         if color[nx] == -1:
    #             color[nx] = color[cur] ^ 1
    #             dq.append(nx)
    # A = [i for i in range(N) if color[i] == 0]
    # B = [i for i in range(N) if color[i] == 1]

    # twoset_edge = set()
    # for u, v in edge:
    #     if color[u] == 0:
    #         twoset_edge.add((u, v))
    #     else:
    #         twoset_edge.add((v, u))

    # candis = set()
    # for a in A:
    #     for b in B:
    #         if (a, b) not in twoset_edge:
    #             candis.add((a, b))

    # if len(candis) % 2 == 1:
    #     print("First", flush=True)
    #     x, y = move = next(iter(candis))
    #     print(x + 1, y + 1, flush=True)
    #     candis.remove(move)
    # else:
    #     print("Second", flush=True)

    # while True:
    #     x, y = MII()
    #     if x == -1 and y == -1:
    #         return
    #     move = (x - 1, y - 1)
    #     move2 = (y - 1, x - 1)
    #     if move in candis:
    #         candis.remove(move)
    #     elif move2 in candis:
    #         candis.remove(move2)
    #     else:
    #         print("Wrong", move, move2, candis, flush=True)
    #         return

    #     if candis:
    #         x, y = next(iter(candis))
    #     else:
    #         print("Wrong no move", flush=True)
    #         return
    #     print(x + 1, y + 1, flush=True)
    #     candis.remove(move)


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


# endregion

if __name__ == "__main__":
    main()
