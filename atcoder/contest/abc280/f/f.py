from collections import defaultdict
from heapq import heappop, heappush
import sys
import io
import os


# region IO
BUFSIZE = 8192


class FastIO(io.IOBase):
    newlines = 0

    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.buffer = io.BytesIO()
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


class IOWrapper(io.IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


def print(*args, **kwargs):
    """Prints the values to a stream, or to sys.stdout by default."""
    sep, file = kwargs.pop("sep", " "), kwargs.pop("file", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop("end", "\n"))
    if kwargs.pop("flush", False):
        file.flush()


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)


input = lambda: sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

class UnionSet:
    def __init__(self, n: int):
        self.parent = [*range(n)]
        self.rank = [1] * n

    def __len__(self):
        return sum([x == self.parent[x] for x in range(len(self.parent))])

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def check(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def union(self, x, y):
        x_0 = self.find(x)
        y_0 = self.find(y)
        if x_0 != y_0:
            if self.rank[x_0] < self.rank[y_0]:
                self.parent[x_0] = y_0
            elif self.rank[x_0] > self.rank[y_0]:
                self.parent[y_0] = x_0
            else:
                self.rank[x_0] += 1
                self.parent[y_0] = x_0


def solve_my():
    n, m, q = read_int_tuple()
    us = UnionSet(n)
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = read_int_tuple()
        u -= 1; v -= 1
        g[u].append((v, w))
        g[v].append((u, -w))
        us.union(u, v)

    # print(g)
    # print([us.find(x) for x in range(n)])

    group = defaultdict(list)
    for u in range(n):
        group[us.find(u)].append(u)
    
    dist = [inf] * n
    
    for p, dots in group.items():
        hp = [(0, p)]
        dist[p] = 0
        flag = False
        while hp:
            t, u = heappop(hp)
            for v, w in g[u]:
                nt = t + w
                if dist[v] == inf:
                    dist[v] = nt
                    heappush(hp, (nt, v))
                elif dist[v] != nt:
                    dist[p] = -inf
                    flag = True
                    break
            if flag: break
    # print(dist)
    for _ in range(q):
        u, v = read_int_tuple()
        u -= 1; v -= 1
        if us.check(u, v):
            p = us.find(u)
            if dist[p] == -inf:
                print('inf')
            else:
                print(dist[v] - dist[u])
        else:
            print('nan')


def solve():
    n, m, q = read_int_tuple()
    fa = list(range(n))
    dis = [0] * n
    mark = [False] * n
    
    sys.setrecursionlimit(max(n+1, 1000))
    
    def find(x):
        if fa[x] == x:
            return x
        else:
            old = fa[x]
            fa[x] = find(old)
            dis[x] += dis[old]
            return fa[x]

    for _ in range(m):
        u, v, w = read_int_tuple()
        u -= 1; v -= 1
        fu, fv = find(u), find(v)
        if fu == fv:
            if dis[v] + w != dis[u]:
                mark[fu] = True
        else:
            fa[fu] = fv
            dis[fu] = dis[v] + w - dis[u]
            if mark[fu]: mark[fv] = True
    
    for _ in range(q):
        u, v = read_int_tuple()
        u -= 1; v -= 1
        fu, fv = find(u), find(v)
        if fu == fv:
            if mark[fu]:
                print('inf')
            else:
                print(dis[u] - dis[v])
        else:
            print('nan')

T = 1#read_int()
for t in range(T):
    solve()