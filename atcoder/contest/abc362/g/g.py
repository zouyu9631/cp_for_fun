from bisect import bisect_left, bisect_right
from typing import List
from atcoder.string import suffix_array

standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No


class Trie:
    def __init__(self):
        self.trie = [None, [0] * 26]
        self.val = [None, 0]  # note some value

    def insert(self, s):
        p = 1
        for x in s:
            if not self.trie[p][x]:
                self.trie[p][x] = len(self.trie)
                self.trie.append([0] * 26)
                self.val.append(0)  # initail node value
            p = self.trie[p][x]
            # self.val[p] += 1    # update node value, += 1 here means the number of words

    def delete(self, s):
        p = 1
        for x in s:
            if not self.trie[p][x]:
                break
            p = self.trie[p][x]
            self.val[p] -= 1


class ACM:
    def __init__(self, N: int, k=26) -> None:
        self.trie = [[0] * N for _ in range(k)]
        self.val = [None] * N
        self.idx = 0

    def insert(self, s: str, val) -> None:
        p = 0
        for c in map(lambda x: ord(x) - 97, s):
            if not self.trie[c][p]:
                self.idx += 1
                self.trie[c][p] = self.idx
            p = self.trie[c][p]
        self.val[p] = val


def solve():
    S = I()
    Q = II()

    acm = ACM()
    for _ in range(Q):
        acm.insert(I())

    acm.build()
    N = len(S)
    cnt = [0] * Q
    cur = acm.root
    for i in range(N):
        c = ord(S[i]) - ord("a")
        while cur != acm.root and c not in cur.next:
            cur = cur.fail
        if c in cur.next:
            cur = cur.next[c]
        for j in cur.accepts:
            cnt[j] += 1


def solve_SA():
    S = I()
    SA = suffix_array(S)
    for _ in range(II()):
        T = I()
        M = len(T)
        left = bisect_left(SA, T, key=lambda x: S[x : x + M])
        right = bisect_right(SA, T, key=lambda x: S[x : x + M])
        print(right - left)


def lookupAllinSA(s: str, sa: List[int], t: str) -> List[int]:
    n, m = len(s), len(t)
    if n < m or m == 0:
        return []

    left = bisect_left(sa, t, key=lambda x: s[x : x + m])
    right = bisect_right(sa, t, key=lambda x: s[x : x + m])
    return sa[left:right]


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
