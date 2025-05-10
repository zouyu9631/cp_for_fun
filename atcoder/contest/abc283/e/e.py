from collections import defaultdict
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
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
位运算，某一位跟左边和右边的值都不同，这一位被称为被孤立。
isolated(x) 返回x中所有被孤立的位，用1标记。例如：
x = int('1101011101', 2)
bin(x), bin(isolate(x)) -> '0b1101011101', '0b11100011'

isolated计算一行内被孤立的位；上下两行被孤立的位可以直接用xor ^ 计算

https://atcoder.jp/contests/abc283/submissions/37664199
"""

def solve():
    H, W = read_int_tuple()
    A = [int(input().replace(' ', ''), 2) for _ in range(H)]
    LEFT = 1 << W - 1
    MASK = (1 << W) - 1
    
    isolated = lambda x: ((x ^ (x >> 1)) | LEFT) & ((x ^ (x << 1)) | 1)
    
    iso = isolated(A[0])
    cur = {(iso, A[0]): 0, (iso, A[0] ^ MASK): 1}
    
    for i in range(1, H):
        a = A[i]
        iso = isolated(a)
        nxt = defaultdict(lambda: inf)
        for (pso, pa), pcost in cur.items():
            for na, cost in ((a, 0), (a ^ MASK, 1)):
                if not pso & (pa ^ na): # 放入 now_a 后，pre_a不再有被孤立的位
                    t = (iso & (pa ^ na), na)
                    nxt[t] = min(nxt[t], pcost + cost)
        cur = nxt
    
    res = min((cost for (pso, _), cost in cur.items() if not pso), default=inf)
    print(res if res < inf else -1)

def solve_my():
    H, W = read_int_tuple()
    dt = (-1, )
    G = [dt * (W + 2)] * 2
    for _ in range(H):
        G.append((-1, ) + read_int_tuple() + (-1, ))
    G.append(dt * (W + 2))
    G.append(dt * (W + 2))

    def judge(i, a, b, c):
        
        for j in range(1, W + 1):
            t = G[i][j] ^ b
            if t in (G[i - 1][j] ^ a, G[i + 1][j] ^ c, G[i][j - 1] ^ b, G[i][j + 1] ^ b):
                continue
            else:
                return False
        
        return True
    
    cur = list(range(4))
    for i in range(2, H + 4):
        nxt = [inf] * 4
        for mask, cost in enumerate(cur):
            a, b = mask >> 1, mask & 1
            for c in range(2):
                if judge(i - 1, a, b, c):
                    t = b + b + c
                    nxt[t] = min(nxt[t], cost + c)
        cur = nxt
    res = min(cur)
    print(res if res < inf else -1)


T = 1#read_int()
for t in range(T):
    solve()