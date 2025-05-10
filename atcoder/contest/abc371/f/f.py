standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

from sortedcontainers import SortedDict


def solve():
    N = II()
    sd = SortedDict()
    X = [-1] * N
    for i, x in enumerate(LII()):
        sd[i] = X[i] = x - i

    res = 0

    for _ in range(II()):
        T, G = MII()
        T -= 1
        G -= T

        index = sd.bisect_right(T) - 1
        key = sd.keys()[index]
        x = sd[key]

        if x < G:
            sd[T] = x
            it_index = sd.bisect_left(T)
            x_current = x
            keys_to_delete = []
            index_iter = it_index + 1
            while True:
                if index_iter >= len(sd):
                    res += (N - T) * (G - x_current)
                    x_current = G
                    break
                nxt_key = sd.keys()[index_iter]
                nxt_val = sd[nxt_key]
                if nxt_val >= G:
                    res += (nxt_key - T) * (G - x_current)
                    x_current = G
                    break
                else:
                    res += (nxt_key - T) * (nxt_val - x_current)
                    x_current = nxt_val
                    keys_to_delete.append(nxt_key)
                    index_iter += 1
            for k in keys_to_delete:
                del sd[k]
            sd[T] = x_current
        else:
            if T + 1 < N:
                next_index = sd.bisect_right(T)
                if next_index >= len(sd) or sd.keys()[next_index] != T + 1:
                    sd[T + 1] = x
            p = key
            it_index = index
            x_current = x
            keys_to_delete = []
            while True:
                if it_index == 0:
                    res += (T + 1) * (x_current - G)
                    x_current = G
                    break
                prev_index = it_index - 1
                prev_key = sd.keys()[prev_index]
                prev_val = sd[prev_key]
                if prev_val <= G:
                    res += (T + 1 - p) * (x_current - G)
                    x_current = G
                    break
                else:
                    res += (T + 1 - p) * (x_current - prev_val)
                    x_current = prev_val
                    keys_to_delete.append(prev_key)
                    p = prev_key
                    it_index = prev_index
            for k in keys_to_delete:
                del sd[k]
            del sd[key]
            sd[p] = x_current

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
