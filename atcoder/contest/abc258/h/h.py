import sys, io, os

# region IO
from functools import reduce

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


def read_int_list(): return list(map(int, input().split()))


def read_int_tuple(): return tuple(map(int, input().split()))


def read_int(): return int(input())


# endregion

MOD = 998244353


def matrixPow(init, n: int):
    get_next_status = lambda s, b: [
        [sum(s[0][k] * M[b][k][j] for k in range(2)) % MOD for j in range(2)]]
    return reduce(get_next_status, filter(lambda b: n & (1 << b), range(n.bit_length())), init)


power_mat = lambda M: [[sum(M[i][k] * M[k][j] for k in range(2)) % MOD for j in range(2)] for i in range(2)]
M = [[[1, 1], [1, 0]]]
for _ in range(61): M.append(power_mat(M[-1]))

n, s = read_int_tuple()
ban = [0] + read_int_list() + [s]
res = [[1, 0]]
for i in range(n + 1):
    d = ban[i + 1] - ban[i] - 1
    res = matrixPow(res, d)
    res[0].reverse()
print(res[0][1])

# main()

# cProfile.run("main()")
