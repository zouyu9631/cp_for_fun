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
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
最大值线段树、最小值线段树
"""

class Max_SegmentTree():
    __slots__ = ['n', 'e', 'log', 'size', 'data']

    def __init__(self, n):
        self.n = n
        self.e = -(1 << 61)
        self.log = (n - 1).bit_length()
        self.size = 1 << self.log
        self.data = [self.e] * (2 * self.size)

    def _update(self, k):
        self.data[k] = self.data[2 * k] if self.data[2 * k] > self.data[2 * k + 1] else self.data[2 * k + 1]


    def build(self, arr):
        # assert len(arr) <= self.n
        for i in range(self.n):
            self.data[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self._update(i)

    def set(self, p, x):
        # assert 0 <= p < self.n
        p += self.size
        self.data[p] = x
        for i in range(self.log):
            p >>= 1
            self._update(p)

    def get(self, p):
        # assert 0 <= p < self.n
        return self.data[p + self.size]

    def prod(self, l, r):
        # assert 0 <= l <= r <= self.n
        sml = smr = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                sml = sml if sml > self.data[l] else self.data[l]
                l += 1
            if r & 1:
                r -= 1
                sml = sml if sml > self.data[r] else self.data[r]

            l >>= 1
            r >>= 1
        return sml if sml > smr else smr


    def all_prod(self):
        return self.data[1]
    

class Min_SegmentTree():
    __slots__ = ['n', 'e', 'log', 'size', 'data']

    def __init__(self, n):
        self.n = n
        self.e = 1 << 61
        self.log = (n - 1).bit_length()
        self.size = 1 << self.log
        self.data = [self.e] * (2 * self.size)

    def _update(self, k):
        self.data[k] = self.data[2 * k] if self.data[2 * k] < self.data[2 * k + 1] else self.data[2 * k + 1]


    def build(self, arr):
        # assert len(arr) <= self.n
        for i in range(self.n):
            self.data[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self._update(i)

    def set(self, p, x):
        # assert 0 <= p < self.n
        p += self.size
        self.data[p] = x
        for i in range(self.log):
            p >>= 1
            self._update(p)

    def get(self, p):
        # assert 0 <= p < self.n
        return self.data[p + self.size]

    def prod(self, l, r):
        # assert 0 <= l <= r <= self.n
        sml = smr = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                sml = sml if sml < self.data[l] else self.data[l]
                l += 1
            if r & 1:
                r -= 1
                sml = sml if sml < self.data[r] else self.data[r]

            l >>= 1
            r >>= 1
        return sml if sml < smr else smr


    def all_prod(self):
        return self.data[1]


def solve():
    n = read_int()
    P = [x - 1 for x in read_int_tuple()]
    res = []
    
    maxst, minst = Max_SegmentTree(n), Min_SegmentTree(n)
    
    for i, pi in enumerate(P):
        tr = min(pi + i - maxst.prod(0, pi), -pi + i + minst.prod(pi + 1, n))
        res.append(tr)
        maxst.set(pi, pi + i)
        minst.set(pi, pi - i)
    
    maxst, minst = Max_SegmentTree(n), Min_SegmentTree(n)
    
    for i in range(n - 1, -1, -1):
        pi = P[i]
        tr = min(pi - i - maxst.prod(0, pi), -pi - i + minst.prod(pi + 1, n))
        res[i] = min(res[i], tr)
        maxst.set(pi, pi - i)
        minst.set(pi, pi + i)
    
    print(*res)


T = 1#read_int()
for t in range(T):
    solve()