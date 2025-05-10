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

from heapq import heappop, heappush
from typing import Tuple

INF = 1 << 60

class SlopeTrick:
    __slots__ = ("_min_f", "_pq_l", "_pq_r", "add_l", "add_r")

    def __init__(self):
        self.add_l = 0  # 左侧第一个拐点的位置 -> \_/
        self.add_r = 0  # 右侧第一个拐点的位置 \_/ <-
        self._pq_l = []  # 大根堆
        self._pq_r = []  # 小根堆
        self._min_f = 0

    def query(self) -> Tuple[int, int, int]:
        """返回 `f(x)的最小值, f(x)取得最小值时x的最小值和x的最大值`"""
        return self._min_f, self._top_l(), self._top_r()

    def add_all(self, a: int) -> None:
        """f(x) += a"""
        self._min_f += a

    def add_a_minus_x(self, a: int) -> None:
        """
        ```
        add \\__
        f(x) += max(a - x, 0)
        ```
        """
        tmp = a - self._top_r()
        if tmp > 0:
            self._min_f += tmp
        self._push_r(a)
        self._push_l(self._pop_r())

    def add_x_minus_a(self, a: int) -> None:
        """
        ```
        add __/
        f(x) += max(x - a, 0)
        ```
        """
        tmp = self._top_l() - a
        if tmp > 0:
            self._min_f += tmp
        self._push_l(a)
        self._push_r(self._pop_l())

    def add_abs(self, a: int) -> None:
        """
        ```
        add \\/
        f(x) += abs(x - a)
        ```
        """
        self.add_a_minus_x(a)
        self.add_x_minus_a(a)

    def clear_right(self) -> None:
        """
        取前缀最小值.
        ```
        \\/ -> \\_
        f_{new} (x) = min f(y) (y <= x)
        ```
        """
        while self._pq_r:
            self._pq_r.pop()

    def clear_left(self) -> None:
        """
        取后缀最小值.
        ```
        \\/ -> _/
        f_{new} (x) = min f(y) (y >= x)
        ```
        """
        while self._pq_l:
            self._pq_l.pop()

    def shift(self, a: int, b: int) -> None:
        """
        ```
        \\/ -> \\_/
        f_{new} (x) = min f(y) (x-b <= y <= x-a)
        ```
        """
        assert a <= b
        self.add_l += a
        self.add_r += b

    def translate(self, a: int) -> None:
        """
        函数向右平移a
        ```
        \\/. -> .\\/
        f_{new} (x) = f(x - a)
        ```
        """
        self.shift(a, a)

    def get_destructive(self, x: int) -> int:
        """
        y = f(x), f(x) broken
        会破坏f内部左右两边的堆.
        """
        res = self._min_f
        while self._pq_l:
            tmp = self._pop_l() - x
            if tmp > 0:
                res += tmp
        while self._pq_r:
            tmp = x - self._pop_r()
            if tmp > 0:
                res += tmp
        return res

    def merge_destructive(self, st: "SlopeTrick"):
        """
        f(x) += g(x), g(x) broken
        会破坏g(x)的左右两边的堆.
        """
        if len(st) > len(self):
            st._pq_l, self._pq_l = self._pq_l, st._pq_l
            st._pq_r, self._pq_r = self._pq_r, st._pq_r
            st.add_l, self.add_l = self.add_l, st.add_l
            st.add_r, self.add_r = self.add_r, st.add_r
            st._min_f, self._min_f = self._min_f, st._min_f
        while st._pq_r:
            self.add_x_minus_a(st._pop_r())
        while st._pq_l:
            self.add_a_minus_x(st._pop_l())
        self._min_f += st._min_f

    def _push_r(self, a: int) -> None:
        heappush(self._pq_r, a - self.add_r)

    def _top_r(self) -> int:
        if not self._pq_r:
            return INF
        return self._pq_r[0] + self.add_r

    def _pop_r(self) -> int:
        val = self._top_r()
        if self._pq_r:
            heappop(self._pq_r)
        return val

    def _push_l(self, a: int) -> None:
        heappush(self._pq_l, -a + self.add_l)

    def _top_l(self) -> int:
        if not self._pq_l:
            return -INF
        return -self._pq_l[0] + self.add_l

    def _pop_l(self) -> int:
        val = self._top_l()
        if self._pq_l:
            heappop(self._pq_l)
        return val

    def _size(self) -> int:
        return len(self._pq_l) + len(self._pq_r)

    def __len__(self) -> int:
        return self._size()


# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353

"""
斜率优化模板
每次添加一条斜率为1或者-1的直线
"""

def solve():
    q = read_int()
    st = SlopeTrick()
    
    for _ in range(q):
        t, *args = read_int_tuple()
        if t == 1:
            a, b = args
            st.add_abs(a)
            st.add_all(b)
        else:
            res = st.query()
            print(*res[1::-1])

T = 1#read_int()
for t in range(T):
    solve()