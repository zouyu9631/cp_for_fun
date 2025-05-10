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


input = lambda: sys.stdin.readline().rstrip("\r\n")


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return map(int, input().split())


def read_int():
    return int(input())


read_str = input


# endregion

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
树的重心/质心
检查每个孩子是否满足数量要求 最多只能有一个不满足
    不满足 就交换当前的父子关系 再从新的父亲考虑
子树的调整不影响上层 路径要求也能满足
"""


def solve():
    n = read_int()
    g = [set() for _ in range(n)]
    for _ in range(n - 1):
        u, v = read_int_tuple()
        u -= 1
        v -= 1
        g[u].add(v)
        g[v].add(u)

    root = 0
    parent = [-1] * n  # 记录父亲结点
    order = [root]  # 记录dfs序
    stack = [root]
    while stack:
        from_node = stack.pop()
        for to_node in g[from_node]:
            if to_node == parent[from_node]:
                continue
            parent[to_node] = from_node
            order.append(to_node)
            stack.append(to_node)

    cnt = [1] * n  # 子树大小
    for v in order[:0:-1]:
        cnt[parent[v]] += cnt[v]

    stack = [root]
    while stack:
        u, bigchild = stack.pop(), -1
        for v in g[u]:
            if v != parent[u] and cnt[v] * 2 > cnt[u]:
                bigchild = v
                break

        if bigchild != -1:
            g[u].discard(parent[u])
            parent[bigchild] = parent[u]
            parent[u] = bigchild

            cnt[u] -= cnt[bigchild]
            cnt[bigchild] += cnt[u]

            stack.append(bigchild)
        else:
            for v in g[u]:
                if v != parent[u]:
                    stack.append(v)

    print(*[-1 if x == -1 else (x + 1) for x in parent])


T = 1  # read_int()
for t in range(T):
    solve()
