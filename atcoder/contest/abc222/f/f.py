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

class Rerooting():
    """https://atcoder.jp/contests/dp/submissions/22766939"""
    __slots__ = ("adjList", "_n", "_decrement", "_root", "_parent", "_order")

    def __init__(self, n: int, decrement: int = 1):
        """n是顶点数， decrement是1-index 或者 0-indexed，默认1-index"""
        self._n = n
        self.adjList = [[] for _ in range(n)]
        self._root = None  # 最初遍历时的根
        self._decrement = decrement

    def add_edge(self, u: int, v: int):
        """添加一条无向边"""
        u -= self._decrement
        v -= self._decrement
        self.adjList[u].append(v)
        self.adjList[v].append(u)

    def rerooting(self, op, merge, e, root=-1):
        """
        <概要>
        1. 以root为根，先用dfs求出一次树结构，计算出树的dfs序：self.order
        2. dp1[u]记录以u为根的子树值，dp2[u]记录u往father指的子树的值
           由于树结构固定，dp1可以自下而上推算，dp2暂存father指向的除了自己外的子树的值
        3. 换根时的dp： 
            dp2[u]还缺dp2[father]的信息，所以：dp2[u] = op(merge(dp2[u], dp2[fa]), u, fa)
            dp1[u] = res = merge(dp1[u], dp2[u])
        复杂度 O(|V|) (V是树的顶点数)
        参考： https://qiita.com/keymoon/items/2a52f1b0fb7ef67fb89e
        """
        # step1
        if root == -1:
            root = 0
        else:
            root -= self._decrement
        assert 0 <= root < self._n
        self._parent = parent = [-1] * self._n  # 记录父亲结点
        self._root = order = [root]  # 记录dfs序，深度单调增加
        stack = [root]
        while stack:
            from_node = stack.pop()
            for to_node in self.adjList[from_node]:
                if to_node == parent[from_node]:
                    continue
                parent[to_node] = from_node
                order.append(to_node)
                stack.append(to_node)
        # step2
        dp1 = list(map(e, range(self._n)))
        dp2 = list(e(i) for i in range(self._n))
        for from_node in order[::-1]:
            t = e(from_node)
            for to_node in self.adjList[from_node]:
                if parent[from_node] == to_node:
                    continue
                dp2[to_node] = t
                t = merge(t, op(dp1[to_node], from_node, to_node))
            t = e(from_node)
            for to_node in self.adjList[from_node][::-1]:
                if parent[from_node] == to_node:
                    continue
                dp2[to_node] = merge(t, dp2[to_node])
                t = merge(t, op(dp1[to_node], from_node, to_node))
            dp1[from_node] = t
        # step3
        for new_root in order[1:]:  # 把最终结果放入dp1，res[u] = merge(dp1[u], dp2[u])
            par = parent[new_root]
            dp2[new_root] = op(merge(dp2[new_root], dp2[par]), new_root, par)
            dp1[new_root] = merge(dp1[new_root], dp2[new_root])
        return dp1

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
换根DP
额外开个EC字典，记录 u - v 边的权重。
"""

def solve():
    n = read_int()
    RT = Rerooting(n, 0)
    EC = dict()
    for _ in range(n - 1):
        u, v, w = read_int_tuple()
        u -= 1; v -= 1
        RT.add_edge(u, v)
        EC[u, v] = EC[v, u] = w
    D = read_int_list()
    
    e = lambda _: 0
    merge = max
    def op(to_dp_val, from_node, to_node):
        return max(to_dp_val, D[to_node]) + EC[from_node, to_node]

    res = RT.rerooting(op, merge, e)
    print(*res, sep='\n')

    
def solve_hand():
    n = read_int()
    g = [[] for _ in range(n)]
    EC = dict()
    for _ in range(n - 1):
        u, v, w = read_int_tuple()
        u -= 1; v -= 1
        g[u].append(v)
        g[v].append(u)
        EC[u, v] = EC[v, u] = w
    D = read_int_list()

    # step1
    root = 0
    parent = [-1] * n  # 记录父亲结点
    order = [root]  # 记录dfs序，深度单调增加
    stack = [root]
    while stack:
        from_node = stack.pop()
        for to_node in g[from_node]:
            if to_node == parent[from_node]:
                continue
            parent[to_node] = from_node
            order.append(to_node)
            stack.append(to_node)
    
    # step2
    e = lambda _: 0
    merge = max
    def op(to_val, from_node, to_node):
        return max(to_val, D[to_node]) + EC[from_node, to_node]
    
    dp1 = list(map(e, range(n)))
    dp2 = list(map(e, range(n)))
    
    for from_node in order[::-1]:
        t = e(from_node)
        for to_node in g[from_node]:
            if parent[from_node] == to_node:
                continue
            dp2[to_node] = t
            t = merge(t, op(dp1[to_node], from_node, to_node))
        t = e(from_node)
        for to_node in g[from_node][::-1]:
            if parent[from_node] == to_node:
                continue
            dp2[to_node] = merge(t, dp2[to_node])
            t = merge(t, op(dp1[to_node], from_node, to_node))
        dp1[from_node] = t

    # step3
    for new_root in order[1:]:  # 把最终结果放入dp1，res[u] = merge(dp1[u], dp2[u])
        par = parent[new_root]
        dp2[new_root] = op(merge(dp2[new_root], dp2[par]), new_root, par)
        dp1[new_root] = merge(dp1[new_root], dp2[new_root])
    
    print(*dp1, sep='\n')

T = 1#read_int()
for t in range(T):
    solve()