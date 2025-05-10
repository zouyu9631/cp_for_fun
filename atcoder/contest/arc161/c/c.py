from collections import deque
import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60


def solve():
    n = read_int()
    g = read_graph(n, n - 1)
    if n == 2:
        print(read_str()[::-1])
        return
    cur = [1 if ch == 'B' else -1 for ch in input()]
    pre = [0] * n

    deg = [len(x) for x in g]
    q = deque([i for i in range(n) if deg[i] == 1])

    while q:
        u, ts = q.popleft(), 0
        deg[u] = 0
        for v in g[u]:
            if pre[v]:
                ts += 1 if pre[v] == cur[u] else -1
            
            if deg[v] == 0:
                
                pre[v] = cur[u]
            elif pre[v] != cur[u]:
                print(-1)
                return
            
            deg[v] -= 1
            if deg[v] == 1:
                nq.append(v)

    
    print(pre)
    
    fas = {g[u][0] for u in range(n)}
    for u in fas:
        mc, cnt = cur[u], 0
        for v in g[u]:
            if pre[v] == 0:
                if cnt < mc:
                    pre[v] = 1
                else:
                    pre[v] = -1
            cnt += pre[v]
    print(''.join('B' if x == 1 else 'W' for x in pre))
        


def main():
    # region local test
    if 'AW' in os.environ.get('COMPUTERNAME', ''):
        test_no = 1
        f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

        global input
        input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = read_int()
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