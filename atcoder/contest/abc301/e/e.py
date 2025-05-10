from collections import deque
import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

"""
状态压缩dp
"""

def popcount(x):
    x -= (x >> 1) & 0x55555555
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f
    x += x >> 8
    x += x >> 16
    return x & 0x3f

def solve():
    H, W, T = read_int_tuple()
    G, candis = [], []
    for i in range(H):
        s = read_str()
        for j, c in enumerate(s):
            G.append(c == '#')
            p = i * (W + 1) + j
            if c == 'S':
                sp = p
            elif c == 'G':
                gp = p
            elif c == 'o':
                candis.append(p)
        G.append(True)
    H, W = H + 1, W + 1
    G.extend([True] * W)
    
    def dist_map(p):
        dist = [T + 1] * (H * W)
        dist[p] = 0
        q = deque([p])
        while q:
            cur = q.popleft()
            for nxt in (cur - 1, cur + 1, cur - W, cur + W):
                if not G[nxt] and dist[nxt] > dist[cur] + 1:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist
    
    sdist = dist_map(sp)
    
    if sdist[gp] > T:
        print(-1)
        return
    
    gdist = dist_map(gp)
    
    cdists = []
    for p in candis:
        dd = dist_map(p)
        cdists.append([dd[t] for t in candis])
    
    n = len(candis)
    M = 1 << n
    dp = [[T + 1] * M for _ in range(n)]
    for t, p in enumerate(candis):
        dp[t][1 << t] = sdist[p]
    
    for mask in range(1, M):
        cm = popcount(mask)
        for t in range(n):
            if dp[t][mask] > T: continue
            for c in range(n):
                if mask & (1 << c): continue
                nmsk = mask | (1 << c)
                dp[c][nmsk] = min(dp[c][nmsk], dp[t][mask] + cdists[t][c])

    res = 0
    for mask in range(1, M):
        cm = popcount(mask)
        if res < cm and any(dp[t][mask] + gdist[p] <= T for t, p in enumerate(candis)):
            res = cm
    print(res)

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