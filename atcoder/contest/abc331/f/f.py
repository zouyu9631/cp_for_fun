import sys
import io
import os

from atcoder.segtree import SegTree

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

from random import randrange

HMOD = 2147483647   # 2 ** 61 - 1, 1000000007
BT = 1
# BASE = [randrange(HMOD) for _ in range(BT)]
BASE = 131  # 13331

# e = ((0, 0, 1),) * BT
e = (0, 0, 1)


def OP(a, b):
    # return tuple(
    #     (
    #         (a[i][0] * b[i][2] + b[i][0]) % HMOD,
    #         (a[i][1] + b[i][1] * a[i][2]) % HMOD,
    #         a[i][2] * b[i][2] % HMOD,
    #     )
    #     for i in range(BT)
    # )
    return (
        (a[0] * b[2] + b[0]) % HMOD,
        (a[1] + b[1] * a[2]) % HMOD,
        a[2] * b[2] % HMOD,
    )


def gen(ch: str):
    x = ord(ch) - 97
    # return tuple((x, x, BASE[i]) for i in range(BT))
    return (x, x, BASE)


def solve():
    n, q = read_int_tuple()
    s = input()

    seg = SegTree(OP, e, list(map(gen, s)))

    for _ in range(q):
        o, a, b = input().split()
        if o == "1":
            seg.set(int(a) - 1, gen(b))
        else:
            res = seg.prod(int(a) - 1, int(b))
            # if all(res[i][0] == res[i][1] for i in range(BT)):
            if res[0] == res[1]:
                print("Yes")
            else:
                print("No")


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

input = lambda: sys.stdin.readline().rstrip("\r\n")


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
