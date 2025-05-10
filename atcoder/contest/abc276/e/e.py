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
#     test_no = 2
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

    def union(self, x, y):  # rank by deep
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

def solve():
    n, m = read_int_tuple()
    g = [input() for _ in range(n)]
    us = UnionSet(n * m)
    S = (-1, -1)
    dirs4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for i in range(n):
        for j in range(m):
            if g[i][j] == 'S':
                S = (i, j)
            elif g[i][j] == '.':
                if i and g[i - 1][j] == '.':
                    us.union(i * m + j, (i - 1) * m + j)
                if j and g[i][j - 1] == '.':
                    us.union(i * m + j, i * m + j - 1)
    
    res = set()
    c = 0
    for x in range(4):
        si, sj = S[0] + dirs4[x][0], S[1]+ dirs4[x][1]
        if not (0 <= si < n and 0 <= sj < m): continue
        res.add(us.find(si * m + sj))
        c += 1
    return len(res) < c

    

T = 1#read_int()
for t in range(T):
    print(['No', 'Yes'][solve()])