from collections import deque
import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60


def solve():
    n, m = read_int_tuple()
    g = [list(input()) for _ in range(n)]
    
    for i in range(n):
        j = 0
        while j < m:
            j += 1
            if g[i][j - 1] == '>':
                while j < m and g[i][j] == '.':
                    g[i][j] = 'X'
                    j += 1
        j = m - 1
        while j >= 0:
            j -= 1
            if g[i][j + 1] == '<':
                while j >= 0 and g[i][j] == '.':
                    g[i][j] = 'X'
                    j -= 1
    
    for j in range(m):
        i = 0
        while i < n:
            i += 1
            if g[i - 1][j] == 'v':
                while i < n and g[i][j] in '.X':
                    g[i][j] = 'X'
                    i += 1
        i = n - 1
        while i >= 0:
            i -= 1
            if g[i + 1][j] == '^':
                while i >= 0 and g[i][j] in '.X':
                    g[i][j] = 'X'
                    i -= 1
    
    # for R in g:
    #     print(R)
    
    for i in range(n):
        for j in range(m):
            if g[i][j] == 'S':
                sx, sy = i, j
            elif g[i][j] == 'G':
                gx, gy = i, j
    
    q = deque([(sx, sy)])
    dist = [[inf] * m for _ in range(n)]
    dist[sx][sy] = 0
    
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return dist[x][y]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and g[nx][ny] in '.G' and dist[nx][ny] == inf:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))

    return -1


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1
    for t in range(T):
        print(solve())
        # print('YES' if solve() else 'NO')
        # print('Yes' if solve() else 'No')


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


def read_ints_minus_one():
    return [int(x) - 1 for x in input().split()]


def read_int_tuple():
    return map(int, input().split())

def read_encode_str(d=97):  # 'a': 97; 'A': 65
    return [ord(x) - d for x in input()]

def read_graph(n: int, m: int, d=1):
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        g[u - d].append(v - d)
        g[v - d].append(u - d)
    return g

def read_grid(m: int):
    return [input() for _ in range(m)]

def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()