from bisect import bisect_left
from collections import defaultdict
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

MOD = 998244353  # 1000000007


def solve():
    n, m, sx, sy = read_int_tuple()
    row = defaultdict(lambda: [0, m + 1])
    col = defaultdict(lambda: [0, n + 1])
    for _ in range(read_int()):
        x, y = read_int_tuple()
        row[x].append(y)
        col[y].append(x)
    for rl in row.values():
        rl.sort()
    for cl in col.values():
        cl.sort()

    # print(row)
    # print(col)
    for _ in range(read_int()):
        d, l = input().split()
        ln = int(l)
        if d in ('L', 'R'):
            rl = row[sx]
            i = bisect_left(rl, sy)
            left = rl[i - 1] + 1
            right = rl[i] - 1
            if d == 'L':
                sy -= ln
            else:
                sy += ln
            if sy < left:
                sy = left
            if sy > right:
                sy = right

        else:   # 'U', 'D'
            cl = col[sy]
            i = bisect_left(cl, sx)
            # print(_, cl, i)
            up, down = cl[i - 1] + 1, cl[i] - 1
            if d == 'U':
                sx -= ln
            else:
                sx += ln
            if sx < up:
                sx = up
            if sx > down:
                sx = down

        print(sx, sy)


T = 1  # read_int()
for t in range(T):
    solve()

# 五星红旗迎风飘扬，胜利歌声多么响亮
