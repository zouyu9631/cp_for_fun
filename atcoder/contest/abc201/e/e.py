from collections import deque
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
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 2
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 1000000007

def solve():
    n = read_int()

    g = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v, w = read_int_tuple()
        u -= 1; v -= 1
        g[u].append((v, w))
        g[v].append((u, w))

    B = 60
    
    book = [0] * B
    
    # BFS
    val = [-1] * n
    val[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v, w in g[u]:
            if val[v] >= 0: continue
            val[v] = val[u] ^ w
            q.append(v)
            for i in range(B):
                book[i] += (val[v] >> i) & 1
    
    
    # # DFS
    # seen = [False] * n
    # stack = [(0, 0)]
    # cur = 0
    # while stack:
    #     u, fw = stack.pop()
        
    #     if u >= 0:
    #         seen[u] = True
    #         stack.append((~u, fw))
    #         cur ^= fw
            
    #         for i in range(B):
    #             book[i] += (cur >> i) & 1
            
    #         for nxt in g[u]:
    #             if seen[nxt[0]]: continue
    #             stack.append(nxt)
    #     else:
    #         cur ^= fw
    
    res = 0
    for i, c1 in enumerate(book):
        res += (1 << i) % MOD * c1 * (n - c1) % MOD
        res %= MOD
    print(res)

T = 1#read_int()
for t in range(T):
    solve()