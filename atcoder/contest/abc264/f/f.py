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
#     test_no = 2
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion


h, w = read_int_tuple()
R = read_int_list()
C = read_int_list()

grid = [input() for _ in range(h)]
inf = 1 << 63

f = [[inf] * 4 for _ in range(w)]
f[0] = [0, R[0], C[0], C[0] + R[0]]

for i in range(h):
    for j in range(w - 1):
        if grid[i][j] == grid[i][j + 1]:
            f[j+1][0] = min(f[j+1][0], f[j][0])
            f[j+1][1] = min(f[j+1][1], f[j][1])
            f[j+1][2] = min(f[j+1][2], f[j][2]+C[j+1])
            f[j+1][3] = min(f[j+1][3], f[j][3]+C[j+1])
        else:
            f[j+1][0] = min(f[j+1][0], f[j][2])
            f[j+1][1] = min(f[j+1][1], f[j][3])
            f[j+1][2] = min(f[j+1][2], f[j][0]+C[j+1])
            f[j+1][3] = min(f[j+1][3], f[j][1]+C[j+1])
    if i == h - 1:
        break
    
    c = R[i+1]
    for j in range(w):
        if grid[i][j] == grid[i+1][j]:
            f[j] = [f[j][0],f[j][1]+c,f[j][2],f[j][3]+c]
        else:
            f[j] = [f[j][1],f[j][0]+c,f[j][3],f[j][2]+c]

# print(f[-1])
print(min(f[-1]))

#         for u in range(2):
#             for v in range(2):
#                 if i + 1 < h:   # 往下走
#                     a, b = grid[i][j] ^ u ^ v, grid[i + 1][j] ^ v
#                     if a == b:
#                         if g[j][0][v] > f[j][u][v]:
#                             g[j][0][v] = f[j][u][v]
#                     else:
#                         if g[j][1][v] > f[j][u][v] + row_cost[i + 1]:
#                             g[j][1][v] = f[j][u][v] + row_cost[i + 1]

#                 if j + 1 < w:
#                     a, b = grid[i][j] ^ u ^ v, grid[i][j + 1] ^ u
#                     if a == b:
#                         if f[j + 1][u][0] > f[j][u][v]:
#                             f[j + 1][u][0] = f[j][u][v]
#                     else:
#                         if f[j + 1][u][1] > f[j][u][v] + col_cost[j + 1]:
#                             f[j + 1][u][1] = f[j][u][v] + col_cost[j + 1]

#     if i == h - 1:
#         res = f[-1]
#     f = g

# print(min(x for row in res for x in row))
