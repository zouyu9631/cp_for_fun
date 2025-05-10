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

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

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

    def union(self, x, y):  # rank by count
        x_0 = self.find(x)
        y_0 = self.find(y)
        if x_0 != y_0:
            if self.rank[x_0] < self.rank[y_0]:
                self.rank[y_0] += self.rank[x_0]
                self.parent[x_0] = y_0
            else:
                self.rank[x_0] += self.rank[y_0]
                self.parent[y_0] = x_0


n, m, e = read_int_tuple()
edges = []
for _ in range(e):
    u, v = read_int_tuple()
    edges.append((u, v))

res = []

queries = [read_int() - 1 for _ in range(read_int())]

# print(edges, queries)
used = set(range(e))
for i in queries:
    used.remove(i)

us = UnionSet(n + m + 1)
for u in range(n + 1, n + m + 1):
    us.union(0, u)

for i in used:
    u, v = edges[i]
    us.union(u, v)

for i in reversed(queries):
    res.append(us.rank[us.find(0)])
    u, v = edges[i]
    us.union(u, v)

res.reverse()
# print(*[x - m - 1 for x in res])
for x in res:
    print(x - m - 1)