from collections import deque
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

# region slope trick

INF = 1 << 60
from heapq import *


class SlopeTrick:
    def __init__(self):
        self.L = []  # 斜率小于0的减少点的坐标，用负数存
        self.R = []  # 斜率大于0的增长点的坐标
        self.mi = 0  # 最小值
        self.la = 0  # L offset
        self.ra = 0  # R offset

    def add_const(self, a):
        self.mi += a

    def add_l(self, a):
        """ add max(a - x, 0) \\\\__ """
        if self.R:
            r0 = self.R[0] + self.ra
            # self.mi += max(a - r0, 0)
            if a > r0:
                self.mi += a - r0
        heappush(self.L, -(heappushpop(self.R, a - self.ra) + self.ra - self.la))

    def add_r(self, a):
        """ add max(x - a, 0) __/ """
        if self.L:
            l0 = -self.L[0] + self.la
            # self.mi += max(l0 - a, 0)
            if l0 > a:
                self.mi += l0 - a
        heappush(self.R, -(heappushpop(self.L, -(a - self.la))) + self.la - self.ra)

    def add_abs(self, a):
        """ add |x-a| \\\\/ """
        self.add_l(a)
        self.add_r(a)

    def slide(self, a, b=None):
        """ f(x) = min({f(y) | x-b <= y <= x-a }) """
        if b is None: b = a
        assert a <= b

        if a == -INF:
            self.L = []
        else:
            self.la += a

        if b == INF:
            self.R = []
        else:
            self.ra += b

    @property
    def min(self):
        return self.mi

    def f(self, x):
        """ 当前函数在x位置的值 O(n) """
        s = self.mi
        for r in self.R:
            # s += max(x - (r + self.ra), 0)
            tmp = x - (r + self.ra)
            if tmp > 0:
                s += tmp
        for l in self.L:
            # s += max((-l + self.la) - x, 0)
            tmp = (-l + self.la) - x
            if tmp > 0:
                s += tmp

        return s

    @staticmethod
    def merge(a: 'SlopeTrick', b: 'SlopeTrick') -> 'SlopeTrick':
        """ f(x) += g(x) """
        if len(b) > len(a):
            a, b = b, a
        for x in b.R:
            a.add_r(x + b.ra)
        for x in b.L:
            a.add_l(-x + b.la)
        a.mi += b.mi
        return a

    def __len__(self):
        return len(self.L) + len(self.R)

    def __repr__(self) -> str:
        c, s = 3, f'{self.__class__.__name__}(min: {self.mi}; slide: {self.la}, {self.ra})'

        l_pos = nlargest(c, (-x for x in set(self.L)))[::-1]
        r_pos = nsmallest(c, set(self.R))

        vals = []
        if len(l_pos) < len(set(self.L)):
            vals.append('...')
        vals.extend((x + self.la, self.f(x + self.la)) for x in l_pos)
        vals.append('___')
        vals.extend((x + self.ra, self.f(x + self.ra)) for x in r_pos)
        if len(r_pos) < len(set(self.R)):
            vals.append('...')

        return s + '[' + ', '.join(map(str, vals)) + ']'

    def debug(self, l=-3, r=4):
        print("mi, L, R =", self.mi, self.L, self.la, self.R, self.ra)
        print([(x, self.f(x)) for x in range(l, r)])


# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)

"""
跟CF1229F类似
"""

def solve():
    n = read_int()
    A = list(read_int_tuple())
    T = list(read_int_tuple())    

    st = SlopeTrick()
    st.L, st.R = [0] * n, [0] * n

    for a, t in zip(A, T):
        st.slide(-INF, a - t)
        st.add_abs(0)
    print(st.f(0))

T = 1#read_int()
for t in range(T):
    solve()