from collections import deque
from itertools import product
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


def input(): return sys.stdin.readline().rstrip('\r\n')
def read_int_list(): return list(map(int, input().split()))
def read_int_tuple(): return tuple(map(int, input().split()))
def read_int(): return int(input())

# endregion


# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')
#     def input(): return f.readline().rstrip("\r\n")

class MaxFlow:
    inf = 10**18
    class E:
        def __init__(self, to, cap):
            self.to = to
            self.cap = cap
            self.rev = None

    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        graph = self.graph
        edge = self.E(to, cap)
        edge2 = self.E(fr, 0)
        edge.rev = edge2
        edge2.rev = edge
        graph[fr].append(edge)
        graph[to].append(edge2)

    def bfs(self, s, t):
        level = self.level = [self.n]*self.n
        q = deque([s])
        level[s] = 0
        while q:
            now = q.popleft()
            lw = level[now]+1
            for e in self.graph[now]:
                if e.cap and level[e.to] > lw:
                    level[e.to] = lw
                    if e.to == t:
                        return True
                    q.append(e.to)
        return False

    def dfs(self, s, t, up):
        graph = self.graph
        it = self.it
        level = self.level

        st = deque([t])
        while st:
            v = st[-1]
            if v == s:
                st.pop()
                flow = up
                for w in st:
                    e = graph[w][it[w]].rev
                    flow = min(flow, e.cap)
                for w in st:
                    e = graph[w][it[w]]
                    e.cap += flow
                    e.rev.cap -= flow
                return flow
            lv = level[v]-1
            while it[v] < len(graph[v]):
                e = graph[v][it[v]]
                re = e.rev
                if re.cap == 0 or lv != level[e.to]:
                    it[v] += 1
                    continue
                st.append(e.to)
                break
            if it[v] == len(graph[v]):
                st.pop()
                level[v] = self.n

        return 0

    def flow(self, s, t, flow_limit=inf):
        flow = 0
        while flow < flow_limit and self.bfs(s, t):
            self.it = [0]*self.n
            while flow < flow_limit:
                f = self.dfs(s, t, flow_limit-flow)
                if f == 0:
                    break
                flow += f
        return flow

    def min_cut(self, s):
        visited = [False]*self.n
        q = deque([s])
        while q:
            v = q.pop()
            visited[v] = True
            for e in self.graph[v]:
                if e.cap and not visited[e.to]:
                    q.append(e.to)
        return visited

from collections import deque
from itertools import product
from typing import List, Tuple
class Dinic:
    """
    Usage:
       mf = Dinic(n)
    -> mf.add_link(from, to, capacity)
    -> mf.max_flow(source, target)
    """

    def __init__(self, n: int):
        self.n = n
        self.links: List[List[List[int]]] = [[] for _ in range(n)]
        # if exists an edge (v→u, capacity)...
        #   links[v] = [ [ capacity, u, index of rev-edge in links[u], is_original_edge ], ]

    def add_link(self, from_: int, to: int, capacity: int) -> None:
        print(from_, to, capacity)
        self.links[from_].append([capacity, to, len(self.links[to]), 1])
        self.links[to].append([0, from_, len(self.links[from_]) - 1, 0])

    def bfs(self, s: int) -> List[int]:
        depth = [-1] * self.n
        depth[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for cap, to, rev, _ in self.links[v]:
                if cap > 0 and depth[to] < 0:
                    depth[to] = depth[v] + 1
                    q.append(to)
        return depth

    def dfs(self, s: int, t: int, depth: List[int], progress: List[int], link_counts: List[int]) -> int:
        links = self.links
        stack = [s]

        while stack:
            v = stack[-1]
            if v == t:
                break
            for i in range(progress[v], link_counts[v]):
                progress[v] = i
                cap, to, rev, _ = links[v][i]
                if cap == 0 or depth[v] >= depth[to] or progress[to] >= link_counts[to]:
                    continue
                stack.append(to)
                break
            else:
                progress[v] += 1
                stack.pop()
        else:
            return 0

        f = 1 << 60
        fwd_links = []
        bwd_links = []
        for v in stack[:-1]:
            cap, to, rev, _ = link = links[v][progress[v]]
            f = min(f, cap)
            fwd_links.append(link)
            bwd_links.append(links[to][rev])

        for link in fwd_links:
            link[0] -= f

        for link in bwd_links:
            link[0] += f

        return f

    def max_flow(self, s: int, t: int) -> int:
        link_counts = list(map(len, self.links))
        flow = 0
        while True:
            depth = self.bfs(s)
            if depth[t] < 0:
                break
            progress = [0] * self.n
            current_flow = self.dfs(s, t, depth, progress, link_counts)
            while current_flow > 0:
                flow += current_flow
                current_flow = self.dfs(s, t, depth, progress, link_counts)
        return flow

    def cut_edges(self, s: int) -> List[Tuple[int, int]]:
        """ max_flow之后，返回最小割的边 """
        q = [s]
        reachable = [0] * self.n
        reachable[s] = 1
        while q:
            v = q.pop()
            for cap, u, li, _ in self.links[v]:
                if cap == 0 or reachable[u]:
                    continue
                reachable[u] = 1
                q.append(u)
        edges = []
        for v in range(self.n):
            if reachable[v] == 0:
                continue
            for cap, u, li, orig in self.links[v]:
                if orig == 1 and reachable[u] == 0:
                    edges.append((v, u))
        return edges

# region https://atcoder.jp/contests/abc259/editorial/4298


# def solve(m, n, nums):
#     S, T = m + n, m + n + 1
#     res = 0
#     row_neg, col_neg = [0] * m, [0] * n
#     mf = Dinic(T + 1)

#     for i in range(m):
#         for j in range(n):
#             if nums[i][j] > 0:
#                 mf.add_link(i, m + j, nums[i][j])
#                 res += nums[i][j]
#             elif nums[i][j] < 0:
#                 mf.add_link(m + j, i, MaxFlow.inf)
#                 # mf.add_edge(i, m + j, MaxFlow.inf)
#                 row_neg[i] -= nums[i][j]
#                 col_neg[j] -= nums[i][j]

#     for i in range(m):
#         if row_neg[i]:
#             mf.add_link(S, i, row_neg[i])

#     for j in range(n):
#         if col_neg[j]:
#             mf.add_link(m + j, T, col_neg[j])
    
#     res -= mf.max_flow(S, T)
#     print(mf.cut_edges(S))
#     return res
# endregion

# region https://atcoder.jp/contests/abc259/submissions/33163832
# def solve(m, n, nums):
#     rows = [(i, rs) for i, rs in enumerate(map(sum, nums)) if rs > 0]
#     cols = [(j, cs) for j, cs in enumerate(map(sum, zip(*nums))) if cs > 0]

#     nr, nc = len(rows), len(cols)
#     s, t = nr + nc, nr + nc + 1

#     mf, res = MaxFlow(t + 1), 0

#     for ri, (_, rs) in enumerate(rows):
#         mf.add_edge(s, ri, rs)
#         res += rs
#     for ci, (_, cs) in enumerate(cols, nr):
#         mf.add_edge(ci, t, cs)
#         res += cs
#     for (ri, (i, _)), (ci, (j, _)) in product(enumerate(rows), enumerate(cols, nr)):
#         if nums[i][j] > 0:
#             mf.add_edge(ri, ci, nums[i][j])
#         elif nums[i][j] < 0:
#             mf.add_edge(ri, ci, inf)

#     return res - mf.flow(s, t)
# endregion

# region https://zhuanlan.zhihu.com/p/539701972 ???

# def solve(m, n, nums):
#     s, t = m + n, m + n + 1
#     mf, res = MaxFlow(t + 1), 0

#     for i, row in enumerate(nums):
#         rs = sum(x for x in row if x > 0)
#         if rs:
#             mf.add_edge(s, i, rs)

#         for j, x in enumerate(row, m):
#             if x > 0:
#                 res += x
#                 mf.add_edge(i, j, x)
#             elif x < 0:
#                 mf.add_edge(j, i, 1 << 40)

#     for j, col in enumerate(zip(*nums), m):
#         cs = sum(x for x in col if x > 0)
#         if cs:
#             mf.add_edge(j, t, cs)
#     print(s, t, mf.flow(s, t))
#     return res - mf.flow(s, t)
# endregion

# region https://atcoder.jp/contests/abc259/submissions/33104155
def solve(m, n, nums):
    s = m + n
    t = s + 1
    res = 0
    mf = MaxFlow(t + 1)
    
    for i, rs in enumerate(map(sum, nums)):
        if rs > 0:
            res += rs
            mf.add_edge(s, i, rs)
            for j, x in enumerate(nums[i], m):
                if x < 0:
                    mf.add_edge(i, j, MaxFlow.inf)
                elif x > 0:
                    mf.add_edge(i, j, x)
    
    for j, cs in enumerate(map(sum, zip(*nums)), m):
        if cs > 0:
            res += cs
            mf.add_edge(j, t, cs)

    fl = mf.flow(s, t)
    # print(res, fl)
    return res - fl
    
# endregion

for _ in range(1):
    m, n = read_int_tuple()
    nums = [read_int_list() for _ in range(m)]
    print(solve(m, n, nums))
