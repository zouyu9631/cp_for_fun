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

inf = 1 << 60


def solve():
    n, m = read_int_tuple()
    C = [[0] * (n + 5) for _ in range((n + 5))]
    for i in range(n + 2):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % m
    P2 = [1]
    for _ in range(n * n): P2.append(2 * P2[-1] % m)
    
    D = [[0] * (n + 5) for _ in range((n + 5))]
    for i in range(n + 1):
        t = P2[i] - 1
        D[i][0] = 1
        for j in range(1, n + 1):
            D[i][j] = D[i][j - 1] * t % m
    
    dp = [[0] * (n + 5) for _ in range((n + 5))]
    dp[1][1] = 1
    for i in range(1, n):
        for j in range(1, i + 1):
            for k in range(1, n - i + 1):
                if i + k <= n:
                    dp[i + k][k] += C[n - i - 1][k] * D[j][k] % m * P2[k * (k - 1) // 2] % m * dp[i][j] % m
                    dp[i + k][k] %= m

    res = 0
    for j in range(1, n + 1):
        res = (res + dp[n - 1][j] * (P2[j] - 1)) % m
    print(res)
        

T = 1#read_int()
for t in range(T):
    solve()