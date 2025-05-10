import sys
import io
import os
MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

class Combi:
    __slots__ = ('fac', 'ifac')

    def __init__(self, n: int) -> None:
        assert (n < MOD)
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



def solve():
    n, m, k = read_int_tuple()

    cnt = [0] * (m + 1)
    for x in read_int_tuple():
        cnt[x] += 1
        
    z0 = cnt[0]
    cnt[0] = 0
        
    cnk = Combi(z0).choose
    
    res = lower = 0
    
    for x in range(m):
        lower += cnt[x]
        for locnt in range(lower, min(k, z0 + lower + 1)):
            clo = locnt - lower
            # if clo > z0: break
            res += cnk(z0, clo) * pow(x, clo, MOD) * pow(m - x, z0 - clo, MOD) % MOD
            res %= MOD
        

    invm = pow(m, (MOD - 2) * z0, MOD)
    # res *= pow(invm, z0, MOD)
    # for _ in range(z0):
    #     res *= invm
    #     res %= MOD
    print(res * invm % MOD)
    


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion
    
    T = 1#read_int()
    for t in range(T):
        solve()
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


def read_int_tuple():
    return map(int, input().split())


def read_int():
    return int(input())

read_str = input


# endregion


if __name__ == "__main__":
    main()