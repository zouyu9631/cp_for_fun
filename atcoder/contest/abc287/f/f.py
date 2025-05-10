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
树上背包 即便合并复杂度是n^2 整个算法均摊下来还是n^2
"""

def solve():
    n = read_int()
    g = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = read_int_tuple()
        u -= 1; v -= 1
        g[u].append(v)
        g[v].append(u)

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
    
    CS, DCS = [0, 1], [1, 0]
    
    # step2
    
    dp = [None for _ in range(n)]
    
    def merge(fdp, sdp):
        pass
    
    for from_node in order[::-1]:
        dp[from_node] = [CS.copy(), DCS.copy()]

        for to_node in g[from_node]:
            if parent[from_node] == to_node:
                continue
            # merge(dp[from_node], dp[to_node])
            fdp, sdp = dp[from_node], dp[to_node]

            fcs, fdcs = fdp
            scs, sdcs = sdp
            
            # 不选根
            m = len(fdcs) + len(sdcs) - 1
            dtmp = [0] * m

            for ac, av in enumerate(fdcs):
                for bc, bv in enumerate(sdcs):
                    if ac + bc >= m: break
                    dtmp[ac + bc] += av * bv % MOD
                    dtmp[ac + bc] %= MOD
                for bc, bv in enumerate(scs):
                    if ac + bc >= m: break
                    dtmp[ac + bc] += av * bv % MOD
                    dtmp[ac + bc] %= MOD

            
            # 选根
            m = len(fcs) + len(scs) - 1
            tmp = [0] * m

            for ac, av in enumerate(fcs):
                for bc, bv in enumerate(scs):
                    if ac + bc - 1 >= m: break
                    tmp[ac + bc - 1] += av * bv % MOD
                    tmp[ac + bc - 1] %= MOD
                for bc, bv in enumerate(sdcs):
                    if ac + bc >= m: break
                    tmp[ac + bc] += av * bv % MOD
                    tmp[ac + bc] %= MOD
            
            dp[from_node][0], dp[from_node][1] = tmp, dtmp
    
    # print(dp[root])
    for x in range(1, n + 1):
        print((dp[root][0][x] + dp[root][1][x]) % MOD)
            

    # # step3
    # for new_root in order[1:]:  # 把最终结果放入dp1，res[u] = merge(dp1[u], dp2[u])
    #     par = parent[new_root]
    #     dp2[new_root] = op(merge(dp2[new_root], dp2[par]), new_root, par)
    #     dp1[new_root] = merge(dp1[new_root], dp2[new_root])
    
    # print(*dp1, sep='\n')

T = 1#read_int()
for t in range(T):
    solve()