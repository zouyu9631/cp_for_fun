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

class SegmentTree():
    __slots__ = ['n', 'oper', 'e', 'log', 'size', 'data']

    def __init__(self, n, oper, e):
        self.n = n
        self.oper = oper
        self.e = e
        self.log = (n - 1).bit_length()
        self.size = 1 << self.log
        self.data = [e] * (2 * self.size)

    def _update(self, k):
        self.data[k] = self.oper(self.data[2 * k], self.data[2 * k + 1])

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
                sml = self.oper(sml, self.data[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.oper(self.data[r], smr)
            l >>= 1
            r >>= 1
        return self.oper(sml, smr)

    def all_prod(self):
        return self.data[1]

    def max_right(self, l, f):
        # assert 0 <= l <= self.n
        # assert f(self.)
        if l == self.n: return self.n
        l += self.size
        sm = self.e
        while True:
            while l % 2 == 0: l >>= 1
            if not f(self.oper(sm, self.data[l])):
                while l < self.size:
                    l = 2 * l
                    if f(self.oper(sm, self.data[l])):
                        sm = self.oper(sm, self.data[l])
                        l += 1
                return l - self.size
            sm = self.oper(sm, self.data[l])
            l += 1
            if (l & -l) == l: break
        return self.n

    def min_left(self, r, f):
        # assert 0 <= r <= self.n
        # assert f(self.)
        if r == 0: return 0
        r += self.size
        sm = self.e
        while True:
            r -= 1
            while r > 1 and (r % 2): r >>= 1
            if not f(self.oper(self.data[r], sm)):
                while r < self.size:
                    r = 2 * r + 1
                    if f(self.oper(self.data[r], sm)):
                        sm = self.oper(self.data[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.oper(self.data[r], sm)
            if (r & -r) == r: break
        return 0



# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

"""
讨论平均数并且有个目标值时，对每一项减去目标值来做记录和判断
"""

def solve():
    n, b, q = read_int_tuple()
    A = read_int_list()
    
    OP = lambda x, y: (x[0] + y[0], max(x[1], x[0] + y[1]))
    
    E = (0, -inf)   # 区间和, 区间内前缀和最大值

    st = SegmentTree(n, OP, E)
    st.build([(x - b, x - b) for x in A])
    
    for _ in range(q):
        c, x = read_int_tuple()
        st.set(c - 1, (x - b, x - b))
        i = st.max_right(0, lambda t: t[1] < 0)
        
        if i < n: i += 1
        res = st.prod(0, i)
        ts = res[0] / i + b
        
        # if i == n:
        #     res = st.all_prod()
        #     ts = res[0] / n + b
        # else:
        #     res = st.prod(0, i + 1)
        #     ts = res[0] / (i + 1) + b

        print(ts)

T = 1#read_int()
for t in range(T):
    solve()