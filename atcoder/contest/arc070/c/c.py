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

from heapq import heappop, heappush, heappushpop

INF = 1 << 60

class SlopeTrick:
    def __init__(self):
        self.L = [INF]
        self.R = [INF]
        self.shift_L = 0
        self.shift_R = 0
        self.min_f = 0

    def add_l(self, a):
        # add max(a-x, 0)
        r0 = self.R[0] + self.shift_R
        self.min_f += max(0, a - r0)
        x = heappushpop(self.R, a - self.shift_R) + self.shift_R
        heappush(self.L, -(x - self.shift_L))

    def add_r(self, a):
        # add max(x-a, 0)
        l0 = -self.L[0] + self.shift_L
        self.min_f += max(0, l0 - a)
        x = -heappushpop(self.L, -(a - self.shift_L)) + self.shift_L
        heappush(self.R, x - self.shift_R)

    def add_abs(self, a):
        # add |x-a|
        self.add_l(a)
        self.add_r(a)

    def clear_right(self):
        self.R = [INF]

    def shift_right(self, x: int):
        self.shift_R += x


n = read_int()

st = SlopeTrick()
pl = pr = 0

for i in range(n):
    l, r = read_int_tuple()
    st.shift_L -= r - l
    st.shift_R += pr - pl
    st.add_abs(l)
    pl, pr = l, r

print(st.min_f)
