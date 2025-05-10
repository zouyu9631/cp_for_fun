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
讲平方和拆分后方便递推∑(x + 1) ^ 2 = ∑x ^ 2 + 2 * ∑x + ∑1
"""

def solve():
    H, W = read_int_tuple()
    g = ['X' + input() for _ in range(H)]
    g.insert(0, 'X' * (W + 1))
    S = [[0] * (W + 1) for _ in range(H + 1)]
    S2 = [[0] * (W + 1) for _ in range(H + 1)]
    C = [[0] * (W + 1) for _ in range(H + 1)]
    C[0][1] = 1
    
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            
            C[i][j] = (C[i - 1][j] + C[i][j - 1]) % MOD
            
            if g[i][j] == 'X':
                S[i][j] = (S[i - 1][j] + S[i][j - 1]) % MOD
                S2[i][j] = (S2[i - 1][j] + S2[i][j - 1]) % MOD
                continue
            
            if g[i - 1][j] == 'X':
                S[i][j] += S[i - 1][j]
                S2[i][j] += S2[i - 1][j]
            else:
                S[i][j] += (S[i - 1][j] + C[i - 1][j]) % MOD
                S2[i][j] += (S2[i - 1][j] + 2 * S[i - 1][j] + C[i - 1][j]) % MOD
            
            if g[i][j - 1] == 'X':
                S[i][j] += S[i][j - 1]
                S2[i][j] += S2[i][j - 1]
            else:
                S[i][j] += (S[i][j - 1] + C[i][j - 1]) % MOD
                S2[i][j] += (S2[i][j - 1] + 2 * S[i][j - 1] + C[i][j - 1]) % MOD
            
            S[i][j] %= MOD
            S2[i][j] %= MOD
    
    print(S2[-1][-1])
        
    


T = 1#read_int()
for t in range(T):
    solve()