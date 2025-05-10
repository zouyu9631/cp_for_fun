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
    return map(int, input().split())


def read_int():
    return int(input())

read_str = input




# endregion

sys.setrecursionlimit(10 ** 9)
from collections import defaultdict, deque
class HopCroftKarp:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.G = [[] for _ in range(n)]
        self.RG = [[] for _ in range(m)]
        self.match_l = [-1] * n
        self.match_r = [-1] * m
        self.used = [0] * n
        self.time_stamp = 0

    def add_edge(self, u, v):
        self.G[u].append(v)

    def _build_argument_path(self):
        queue = deque()
        self.dist = [-1] * self.n
        for i in range(self.n):
            if self.match_l[i] == -1:
                queue.append(i)
                self.dist[i] = 0
        while queue:
            a = queue.popleft()
            for b in self.G[a]:
                c = self.match_r[b]
                if c >= 0 and self.dist[c] == -1:
                    self.dist[c] = self.dist[a] + 1
                    queue.append(c)

    def _find_min_dist_argument_path(self, a):
        self.used[a] = self.time_stamp
        for b in self.G[a]:
            c = self.match_r[b]
            if c < 0 or (self.used[c] != self.time_stamp and self.dist[c] == self.dist[
                a] + 1 and self._find_min_dist_argument_path(c)):
                self.match_r[b] = a
                self.match_l[a] = b
                return True
        return False

    def max_matching(self):
        while 1:
            self._build_argument_path()
            self.time_stamp += 1
            flow = 0
            for i in range(self.n):
                if self.match_l[i] == -1:
                    flow += self._find_min_dist_argument_path(i)
            if flow == 0:
                break
        ret = []
        for i in range(self.n):
            if self.match_l[i] >= 0:
                ret.append((i, self.match_l[i]))
        return ret


# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60
dirs2 = ((-1, 0), (0, -1))

"""
矩形网格 二分图匹配
把每个'2'格子拆成两个点 入点和出点 '?'格子只作为出点
"""

def solve():
    H, W = read_int_tuple()
    G = [input() for _ in range(H)]
    
    hck = HopCroftKarp(H * W, H * W)
    cnt = 0
    
    for i, row in enumerate(G):
        for j, ch in enumerate(row):
            if ch == '1': continue
            
            if ch == '2':
                cnt += 1
                for u, v in dirs2:
                    x, y = i + u, j + v
                    if not (0 <= x < H and 0 <= y < W): continue
                    if G[x][y] == '2':
                        hck.add_edge(i * W + j, x * W + y)
                        hck.add_edge(x * W + y, i * W + j)
                    elif G[x][y] == '?':
                        hck.add_edge(i * W + j, x * W + y)
                    
            else:   # ch == '?
                for u, v in dirs2:
                    x, y = i + u, j + v
                    if not (0 <= x < H and 0 <= y < W): continue
                    if G[x][y] == '2':
                        hck.add_edge(x * W + y, i * W + j)
     
    res = hck.max_matching()
    print(['No', 'Yes'][len(res) == cnt])


T = 1#read_int()
for t in range(T):
    solve()