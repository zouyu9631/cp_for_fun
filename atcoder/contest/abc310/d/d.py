import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

# DFS 枚举 N 个元素 分成 T 个集合/组的方案

def solve_mask():
    N, T, M = read_int_tuple()
    G = [0] * N
    for _ in range(M):
        u, v = read_int_tuple()
        u -= 1; v -= 1
        G[u] |= 1 << v
        G[v] |= 1 << u
    
    # S = set()
    res = 0
    A = [0] * T
    
    def dfs(u=0):
        if u == N:
            if A[-1]:
                # S.add(tuple(sorted(A)))
                nonlocal res
                res += 1
            return

        for i in range(T):
            if G[u] & A[i]: continue
            A[i] ^= 1 << u
            dfs(u + 1)
            A[i] ^= 1 << u
            if A[i] == 0: break

    dfs()
    
    # print(len(S))
    print(res)

def solve_dfs():
    N, T, M = read_int_tuple()
    G = [0] * N
    for _ in range(M):
        u, v = read_int_tuple()
        G[v - 1] |= 1 << (u - 1)

    def dfs(u: int, A: list):
        if u == N:
            return len(A) == T
        res = 0
        for i in range(len(A)):
            if G[u] & A[i]: continue
            A[i] ^= 1 << u
            res += dfs(u + 1, A)
            A[i] ^= 1 << u
        
        if len(A) < T:
            A.append(1 << u)
            res += dfs(u + 1, A)
            A.pop()
        
        return res

    print(dfs(0, []))

def solve_dp():
    N, T, M = read_int_tuple()
    G = [0] * N
    for _ in range(M):
        u, v = read_int_tuple()
        G[v - 1] |= 1 << (u - 1)
    
    available = [False] * (1 << N)
    
    for mask in range(1, 1 << N):
        for u in range(N - 1, 0, -1):
            if mask & (1 << u) and G[u] & mask:
                break
        else:
            available[mask] = True

    dp = [[0] * (1 << N) for _ in range(T + 1)]
    dp[0][0] = 1
    for mask in range(1 << N):
        # cur = ori = (mask + 1) | mask
        cur = ori = mask
        while cur < 1 << N:
            if cur > mask and available[mask ^ cur]:
                for t in range(T):
                    dp[t + 1][cur] += dp[t][mask]
            cur += 1
            cur |= ori
    
    from math import factorial
    print(dp[-1][-1] // factorial(T))
                

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
        solve_dp()
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