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


input = lambda: sys.stdin.readline().rstrip("\r\n")


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

proot, ppoly, order = 10279, 92191, 65535
base = (
    1,
    10279,
    14635,
    32768,
    8445,
    19741,
    56906,
    2583,
    13412,
    58281,
    28045,
    13500,
    43297,
    41331,
    3772,
    3689,
)
exp, log = [0] * 262300, [0] * 65536

# F_{2^16}
exp[0] = 1
for i in range(1, order):
    exp[i] = exp[i - 1] << 1
    if exp[i] > order:
        exp[i] ^= ppoly

pre = [0] * (order + 1)
for b in range(16):
    ist, ien = 1 << b, 1 << (b + 1)
    for i in range(ist, ien):
        pre[i] = pre[i - ist] ^ base[b]
for i in range(order):
    exp[i] = pre[exp[i]]
    log[exp[i]] = i

ie = 2 * order + 30
for i in range(order, ie):
    exp[i] = exp[i - order]
log[0] = ie + 1


# F_{2^32}
def prod_32(i, j):
    iu, il = i >> 16, i & 65535
    ju, jl = j >> 16, j & 65535
    l = exp[log[il] + log[jl]]
    ul = exp[log[iu ^ il] + log[ju ^ jl]]
    uq = exp[log[iu] + log[ju] + 3]
    return ((ul ^ l) << 16) ^ uq ^ l


def Hprod_32(i, j):
    iu, il = i >> 16, i & 65535
    ju, jl = j >> 16, j & 65535
    l = exp[log[il] + log[jl]]
    ul = exp[log[iu ^ il] + log[ju ^ jl]]
    uq = exp[log[iu] + log[ju] + 3]
    ku, kl = ul ^ l, uq ^ l
    return (exp[log[ku ^ kl] + 3] << 16) ^ exp[log[ku] + 6]


# F_{2^64}
def nim_prod(i, j):
    iu, il = i[0], i[1]
    ju, jl = j[0], j[1]
    l = prod_32(il, jl)
    ul = prod_32(iu ^ il, ju ^ jl)
    uq = Hprod_32(iu, ju)
    return (ul ^ l, uq ^ l)

def nim_inv(x):
    

def nim_sum(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


def solve():
    n = read_int()
    A = read_int_list()

    n, m = read_int_tuple()
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = read_int_tuple()
        u -= 1
        v -= 1
        g[u].append(v)
        g[v].append(u)

    grid = [read_int_list() for _ in range(n)]
    print(*grid)

    print(["No", "Yes"][check()])


T = 1  # read_int()
for t in range(T):
    solve()
