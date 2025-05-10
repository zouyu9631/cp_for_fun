import heapq
from types import GeneratorType
from math import inf
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


def input(): return sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

MOD = 998244353  # 1000000007, 998244353

if 'AW' in os.environ.get('COMPUTERNAME', ''):
    test_no = 1
    f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    def input():
        return f.readline().rstrip("\r\n")


def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
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


def solve(n, limit, g):
    res = [None] * n

    @bootstrap
    def dfs(u, f):

        for v, w in g[u]:
            if v == f:
                continue
            yield dfs(v, u)

        tr = 0
        book = []
        for v, w in g[u]:
            if v == f:
                continue
            tr += res[v][0]
            if res[v][1] + w >= res[v][0]:
                book.append(res[v][1] + w - res[v][0])

        res[u] = [tr, tr]

        if limit[u] == 0:
            res[u][1] = -inf
        else:
            choices = heapq.nlargest(limit[u], book)
            sc = sum(choices)
            res[u][0] += sc
            res[u][1] += sc
            if len(choices) == limit[u]:
                res[u][1] -= choices[-1]

        yield None

    dfs(0, 0)

    return res[0][0]


for _ in range(1):
    n = read_int()
    limit = read_int_list()
    g = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v, w = read_int_tuple()
        g[u - 1].append((v - 1, w))
        g[v - 1].append((u - 1, w))
    print(solve(n, limit, g))
