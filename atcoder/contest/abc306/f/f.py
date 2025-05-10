import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

class BIT:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def build(self, list):
        self.tree[1:] = list.copy()
        for i in range(self.size + 1):
            j = i + (i & (-i))
            if j < self.size + 1:
                self.tree[j] += self.tree[i]

    def sum(self, i):
        # return sum(arr[0: i])
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def sum_range(self, l, r):
        # sum(arr[l: r]) -> return self.sum(r) - self.sum(l)
        s = 0
        while l < r:
            s += self.tree[r]
            r -= r & (-r)
        while r < l:
            s -= self.tree[l]
            l -= l & (-l)
        return s

    def add(self, i, x):
        # arr[i] += 1
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

def solve():
    n, m = read_int_tuple()
    xs, M = [], []
    for _ in range(n):
        A = read_int_list()
        A.sort()
        M.append(A)
        xs.extend(A)
    
    mp = {x: i for i, x in enumerate(sorted(xs))}
    bit = BIT(n * m)
    res = 0
    for R in M[::-1]:
        for x in sorted(R, reverse=True):
            res += bit.sum(mp[x])
            bit.add(mp[x], 1)   
    s = sum(range(m))
    N = n * (n - 1) // 2
    # ts = cur = 0
    
    # while cur < s * n:
    #     ts += cur
    #     cur += s
    
    ts = (n - 1) * s * n // 2
    # print(res, N * m, ts, s, s * n)
    print(res + N * m + ts)

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