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

EPS = 1e-9
from math import *

class Vector():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __mul__(self, other):
        ''' DOT mul'''
        return self.__class__(self.x * other, self.y * other)

    def __truediv__(self, other):
        if type(other) is Vector:
            assert False
        else:
            return self.__class__(self.x / other, self.y / other)

    def __matmul__(self, other):
        ''' CROSS mul ; > 0 means self is on the other's right'''
        assert isinstance(other, Vector)
        return self.x * other.y - self.y * other.x

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __lt__(self, other):
        return self.x < other.x or self.x == other.x and self.y < other.y

    def dist2(self):
        return self.x * self.x + self.y * self.y

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    @staticmethod
    def cross_mul(x, y, z):
        # x -> y -> z
        return (y - x) @ (z - x)

    def __abs__(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def __repr__(self):
        return f'{self.__class__.__name__}: ({self.x}, {self.y})'

    
points = [Vector(*read_int_tuple()) for _ in range(4)]
# print(points)
for i in range(4):
    a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
    if Vector.cross_mul(a, b, c) <= 0:
        print('No')
        exit()
print('Yes')