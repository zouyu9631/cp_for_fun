import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

# 带合并状态的并查集

class UnionSet:
    def __init__(self, n: int):
        self.parent = [*range(n)]
        self.rank = [1] * n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def check(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def union(self, x, y):
        u = self.find(x)
        v = self.find(y)
        self.rank[v] += self.rank[u]
        self.parent[u] = v

def solve():
    n = read_int()
    
    inverse = [1] * (n + 1)

    for i in range(2, n + 1):
        inverse[i] = ((-inverse[MOD % i] * (MOD // i)) % MOD)
    
    us = UnionSet(n * 2)
    us.rank[n:] = [0] * n
    edges = [[] for _ in range(n * 2)]
    for t in range(n, n + n - 1):
        u, v = read_ints_minus_one()
        fu, fv = us.find(u), us.find(v)
        su, sv = us.rank[fu], us.rank[fv]
        # invtot = pow(su + sv, -1, MOD)
        invtot = inverse[su + sv]
        edges[t].append((fu, su * invtot % MOD))
        edges[t].append((fv, sv * invtot % MOD))
        us.union(u, t)
        us.union(v, t)
    
    V = [0] * (n + n)

    for t in range(n + n - 2, n - 1, -1):
        for u, value in edges[t]:
            V[u] = (V[t] + value) % MOD
    print(*V[:n])
    
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