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

class BIT:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def build(self, list):
        self.tree[1:] = list.copy()
        for i in range(self.size + 1):
            j = i + (i & (-i))
            if j < self.size + 1:
                self.tree[j] += self.tree[i]

    def sum(self, i):
        # return sum(arr[0: i])
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def sum_range(self, l, r=-1):
        if r == -1: r = self.size
        # sum(arr[l: r]) -> return self.sum(r) - self.sum(l)
        s = 0
        while l < r:
            s += self.tree[r]
            r -= r & (-r)
        while r < l:
            s -= self.tree[l]
            l -= l & (-l)
        return s

    def add(self, i, x):
        # arr[i] += 1
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def __getitem__(self, i):
        # return arr[i]
        return self.sum_range(i, i + 1)

    def  __repr__(self):
        return 'BIT({0})'.format([self[i] for i in range(self.size)])

def solve():
    n = read_int()
    A = read_int_list()
    hi = max(A) + 1
    bit_cnt = BIT(hi)
    bit_sum = BIT(hi)
    val = 0
    for c, x in enumerate(A, 1):
        bit_cnt.add(x, 1)
        bit_sum.add(x, x)
        
        # print(bit_cnt)
        # print(bit_sum)
        
        lo_cnt = bit_cnt.sum(x)
        x_cnt = bit_cnt.sum(x + 1) - lo_cnt
        # hi_sum = bit_sum.sum(hi) - bit_sum.sum(x + 1)
        hi_sum = bit_sum.sum_range(x+1)
        
        val += x * lo_cnt * 2
        val += x * (x_cnt * 2 - 1)
        val += hi_sum * 2
        
        val %= MOD
        
        dec = pow(c * c, MOD - 2, MOD)
        
        # print(val, dec) # val / dec
        print(val * dec % MOD)
    

T = 1#read_int()
for t in range(T):
    solve()