import sys
import io
import os


MOD = 998244353  # 1000000007


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     def input():
    #         return f.readline().rstrip("\r\n")

    #     def read_int_list():
    #         return list(map(int, input().split()))

    #     def read_int_tuple():
    #         return tuple(map(int, input().split()))

    #     def read_int():
    #         return int(input())
    # endregion

    # for _ in range(read_int()):
    n, m, k = read_int_tuple()
    cnt = [0] * n
    for _ in range(m):
        u, v = read_int_tuple()
        cnt[u - 1] += 1
        cnt[v - 1] += 1
    
    odd = sum(cnt[u] & 1 for u in range(n))
    even = n - odd
    # print(odd, even)
    
    res = 0
    for c in range(0, k + 1, 2):
        res += cmb(odd, c) * cmb(even, k - c) % MOD
        res %= MOD
    
    print(res)
    
def cmb(n, r):
    if r < 0 or r > n: return 0
    return (g1[n] * g2[r] % MOD) * g2[n - r] % MOD

N = 200000
g1 = [1] * (N + 1)
g2 = [1] * (N + 1)
inverse = [1] * (N + 1)

for i in range(2, N + 1):
    g1[i] = ((g1[i - 1] * i) % MOD)
    inverse[i] = ((-inverse[MOD % i] * (MOD // i)) % MOD)
    g2[i] = ((g2[i - 1] * inverse[i]) % MOD)
inverse[0] = 0 


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


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

if __name__ == "__main__":
    main()
