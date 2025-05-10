import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60


class UnionSet:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return "\n".join(f"{r}: {m}" for r, m in self.all_group_members().items())


def solve():
    n, m = read_int_tuple()
    A = read_ints_minus_one()

    if A[0] > A[-1]:
        for _ in range(m):
            input()
        return 0

    ds = UnionSet(n)
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = read_ints_minus_one()
        if A[u] == A[v]:
            ds.union(u, v)
        elif A[u] > A[v]:
            g[v].append(u)
        else:
            g[u].append(v)

    dp = [-inf] * n
    dp[ds.find(0)] = 1

    book = [[] for _ in range(A[-1] + 1)]
    for u, x in enumerate(A):
        if x <= A[-1]:
            book[x].append(u)

    for x in range(A[0], A[-1] + 1):
        for u in book[x]:
            p = ds.find(u)
            for v in g[u]:
                q = ds.find(v)
                dp[q] = max(dp[q], dp[p] + 1)

    # for u, _ in sorted(enumerate(A), key=itemgetter(1)):
    #     for v in g[u]:
    #         if A[v] > A[u]:
    #             dp[ds.find(v)] = max(dp[ds.find(v)], dp[ds.find(u)] + 1)

    return max(dp[ds.find(n - 1)], 0)


def main():
    # region local test
    # if "AW" in os.environ.get("COMPUTERNAME", ""):
    #     test_no = 3
    #     f = open(os.path.dirname(__file__) + f"\\in{test_no}.txt", "r")

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1
    # T = read_int()
    for t in range(T):
        # solve()
        print(solve())
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


def read_graph_with_weight(n: int, m: int, d=1):
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        g[u - d].append((v - d, w))
        g[v - d].append((u - d, w))
    return g


def read_grid(m: int):
    return [input() for _ in range(m)]


def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()
