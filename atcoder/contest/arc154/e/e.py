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

MOD = 998244353  # 1000000007
inf = 1 << 60
INV2 = (MOD + 1) >> 1

"""
题意
    交换 倒置 A[i:j+1] 计算所有可能的f(A)的和
    
观察1
    f(Q) = sum(i * (i - x) for i, x in enumerate(Q, 1))
    设 Q[i] = x 有left_lo个j Q[j] < Q[i]; 有left_hi个j Q[j] > Q[i] 其中j < i
                有right_lo个j Q[i] > Q[j]; 有right_hi个j Q[i] < Q[j] 其中i < j 
            那么 left_lo + left_hi = i - 1; right_lo + right_hi = n - i
                 left_lo + right_lo = x - 1; left_hi + right_hi = n - x
    i 对结果的贡献其实是 i * left_hi - i * right_lo
            能够算出 left_hi - right_lo = i - x
            所以i的总贡献是 i * (i - x)

观察2
    对于i 某回合的所有操作中
            要么不涉及i
            要么换到j 并且跟换到 n - j + 1 的数量相同 (换到 j 的次数是 min(i, n - i + 1, j, n - j + 1))
                意思是只要变了 那么最终位置的数量 就是关于 (n + 1) / 2 对称的

推导
    ∑f(Q) = ∑ ∑ (i * (i - x)) = ∑ ∑ (i * i - i * x) = T * ∑ i^2 - ∑ (x * ∑ i)
    ∑ i 可以分为 i 没变过: i * 没变的次数
                i 变过: (n + 1) * 变过的次数 / 2

    T 是k次操作后 所有的可能数量

                 
"""

def solve():
    n, k = read_int_tuple()
    T = pow(n * (n + 1) // 2, k, MOD)
    res = T * n * (n + 1) * (n * 2 + 1) // 6 % MOD

    A = read_int_list()
    
    tmp = 0

    for i, x in enumerate(A, 1):
        move = i * (n - i + 1)
        stay = n * (n + 1) // 2 - move
        stay_k = pow(stay, k, MOD)
        move_k = (T - stay_k) % MOD
        
        total = x * (i * stay_k + (n + 1) * INV2 * move_k) % MOD
        tmp += total
        
        tmp %= MOD
    
    print((res - tmp) % MOD)
    
    
    

T = 1#read_int()
for t in range(T):
    solve()