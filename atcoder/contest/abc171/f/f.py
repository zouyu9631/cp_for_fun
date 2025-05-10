
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

"""
计数
确定一种生成方法，然后方便计算合法数量。
"""

MOD = 1000000007
inf = 1 << 60

def solve():
    k = read_int()
    p25, p26 = [1] * (k + 1), [1] * (k + 1)
    for i in range(k):
        p25[i + 1] = p25[i] * 25 % MOD
        p26[i + 1] = p26[i] * 26 % MOD
    # s = input()
    n = len(input())
    
    class FactorialUtils:
        __slots__ = ('fac', 'ifac')
    
        def __init__(self, n: int) -> None:
            assert(n < MOD)
            self.fac = [1] * (n + 1)
            self.ifac = [1] * (n + 1)
            for i in range(2, n + 1): self.fac[i] = self.fac[i - 1] * i % MOD
            self.ifac[n] = pow(self.fac[n], MOD - 2, MOD)
            for i in range(n, 1, -1): self.ifac[i - 1] = self.ifac[i] * i % MOD
    
        def choose(self, n: int, r: int) -> int:
            if r < 0 or r > n: return 0
            return (self.fac[n] * self.ifac[n - r] % MOD) * self.ifac[r] % MOD
    
        def multichoose(self, u: int, k: int) -> int:
            if k < 0 or k > (u + k - 1): return 0
            return (self.fac[u + k - 1] * self.ifac[u - 1] % MOD) * self.ifac[k] % MOD
    
        def permutation(self, n: int, r: int) -> int:
            if r < 0 or r > n: return 0
            return self.fac[n] * self.ifac[n - r] % MOD

    comb = FactorialUtils(n + k).choose
    
    
    
    res = 0
    p26.reverse()
    for ln, (a, b) in enumerate(zip(p25, p26), n):
        res += a * b * comb(ln - 1, n - 1)
        res %= MOD
    print(res)

T = 1#read_int()
for t in range(T):
    solve()