from itertools import count
import sys
import io
import os

class SparseTable:
    def __init__(self, A, merge_func):
        N = len(A)

        self.merge_func = merge_func

        self.lg = [0] * (N + 1)
        for i in range(2, N + 1):
            self.lg[i] = self.lg[i >> 1] + 1
        self.pow_2 = [pow(2, i) for i in range(20)]

        self.table = [None] * (self.lg[N] + 1)
        st0 = self.table[0] = [a for a in A]
        b = 1
        for i in range(self.lg[N]):
            st0 = self.table[i + 1] = [self.merge_func(u, v) for u, v in zip(st0, st0[b:])]
            b <<= 1

    def query(self, s, t):
        b = t - s + 1
        m = self.lg[b]
        return self.merge_func(self.table[m][s], self.table[m][t - self.pow_2[m] + 1])

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

    def __getitem__(self, i):
        # return arr[i]
        return self.sum_range(i, i + 1)

    def  __repr__(self):
        return 'BIT({0})'.format([self[i] for i in range(self.size)])

    def __setitem__(self, i, x):
        # arr[i] = x
        self.add(i, x - self[i])

"""
动态树边权 求两点加权距离
欧拉回路
    配合st表 计算LCA
    配合BIT  计算两点距离
"""

def solve():
    n = read_int()
    E = dict()
    g = [[] for _ in range(n)]
    for i in range(n - 1):
        u, v, w = read_int_tuple()
        u -= 1; v -= 1
        if u > v: u, v = v, u
        E[(u << 20) | v] = (i, w)
        g[u].append(v)
        g[v].append(u)
        
    p = [count() for _ in range(n)]
    euler_path = []
    depth = [-1] * n  # depth[u]: u在以root为根的树中的深度，depth[root] = 0
    stack = [0]
    while stack:
        u = stack[-1]
        euler_path.append(u)
        i = next(p[u])
        if i < len(g[u]) and len(stack) > 1 and g[u][i] == stack[-2]:  # 如果儿子其实是父亲的话就跳过
            i = next(p[u])
        if i < len(g[u]):  # 还有儿子
            stack.append(g[u][i])
        else:  # 儿子都遍历结束
            stack.pop()
            depth[u] = len(stack)
    # euler_depth = [depth[u] for u in euler_path]
    st_table = SparseTable([(depth[u], u) for u in euler_path], min)
    pos = [-1] * n
    for i, u in enumerate(euler_path):
        pos[u] = i


    def get_lca(u: int, v: int) -> int:
        ui, vi = pos[u], pos[v]
        if ui > vi:
            ui, vi = vi, ui
        return st_table.query(ui, vi)[1]

    
    A, m = [], len(euler_path)
    e2e = [[] for _ in range(n - 1)]
    for i in range(m - 1):
        u, v = euler_path[i], euler_path[i + 1]
        ou, ov = u, v
        if u > v: u, v = v, u
        ei, w = E[(u << 20) | v]
        A.append(w if depth[ou] < depth[ov] else -w)
        e2e[ei].append(i)
    
    bit = BIT(len(A))
    bit.build(A)

    for _ in range(read_int()):
        t, x, y = read_int_tuple()
        if t == 1:
            ei, nw = x - 1, y
            in_i, out_i = e2e[ei]
            bit[in_i], bit[out_i] = nw, -nw
            
        else:   # t == 2
            u, v = x - 1, y - 1
            ca = get_lca(u, v)
            ui, vi, ci = pos[u], pos[v], pos[ca]
            res = bit.sum(ui) + bit.sum(vi) - 2 * bit.sum(ci)
            print(res)


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion
    
    T = 1#read_int()
    for t in range(T):
        solve()
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


def read_int_tuple():
    return map(int, input().split())


def read_int():
    return int(input())

read_str = input


# endregion


if __name__ == "__main__":
    main()