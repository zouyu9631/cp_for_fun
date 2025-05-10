from operator import add
import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

from atcoder.lazysegtree import LazySegTree
from atcoder.segtree import SegTree

def solve():
    N, D, W = read_int_tuple()
    T, X = [], []
    hiT = hiX = 0
    for _ in range(N):
        t, x = read_int_tuple()
        T.append(t)
        X.append(x)
        hiT = max(hiT, t)
        hiX = max(hiX, x)

    event = [[] for _ in range(hiT + 2)]
    for t, x in zip(T, X):
        event[max(0, t - D + 1)].append((max(0, x - W + 1), x, 1))
        event[t + 1].append((max(0, x - W + 1), x, -1))

    def op(a, b):
        return (a[0] + b[0], max(a[1], a[0] + b[1]))
    
    e = (0, 0)
    
    st = SegTree(op, e, hiX + 2)
    cnt = [0] * (hiX + 2)
    
    print(event)
    
    res = 0
    for t in range(hiT + 2):
        for l, r, p in event[t]:
            cnt[l] += p
            st.set(l, (cnt[l], max(0, cnt[l])))
            
            cnt[r + 1] -= p
            st.set(r + 1, (cnt[r + 1], max(0, cnt[r + 1])))
        print(t, cnt)
        res = max(res, st.all_prod()[1])
    
    print(res)

def solve_lst():
    N, D, W = read_int_tuple()
    T, X = [], []
    hiT = hiX = 1
    for _ in range(N):
        t, x = read_int_tuple()
        T.append(t)
        X.append(x)
        hiT = max(hiT, t + 1)
        hiX = max(hiX, x + 1)

    A = [[] for _ in range(hiX)]
    for t, x in zip(T, X):
        A[x].append(t)

    lst = LazySegTree(max, 0, add, add, 0, hiT)
    
    res = 0
    for x in range(hiX):
        for t in A[x]:
            lst.apply(max(0, t - D), t, 1)
        if x >= W:
            for t in A[x - W]:
                lst.apply(max(0, t - D), t, -1)
        res = max(res, lst.all_prod())
    
    print(res)


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
        solve()
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