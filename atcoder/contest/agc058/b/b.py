from itertools import pairwise, permutations
from random import randint
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


def input(): return sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

# region local test
if 'AW' in os.environ.get('COMPUTERNAME', ''):
    test_no = 2
    f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    def input():
        return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007

n = read_int()
nums = read_int_list()

stack = []
left, right = [0] * n, [n] * n
for i, x in enumerate(nums):
    while stack and nums[stack[-1]] < x:
        right[stack[-1]] = i
        stack.pop()
    if stack:
        left[i] = stack[-1] + 1
    stack.append(i)

print(nums)
for i in range(n):
    print(f'{i}: [{left[i]}, {right[i]})')

dp = [[0] * (n + 1) for _ in range(n + 1)]
dp[0][0] = 1
for i in range(1, n + 1):
    for j in range(left[i - 1] + 1, right[i - 1] + 1):
        dp[i][j] = sum(dp[i - 1][:j + 1])
print(dp)

dp = [1] + [0] * n
for i in range(n):
    for j in range(left[i], right[i]):
        dp[j + 1] += dp[j]
        dp[j + 1] %= MOD
    print(dp, [b - a if b > a else 0 for a, b in pairwise(dp)])
print(dp[n])


def brute():
    book = set()
    for ln in range(n):
        for t in permutations(range(n - 1), ln):
            tmp = nums.copy()
            for i in t:
                hi = tmp[i] if tmp[i] > tmp[i + 1] else tmp[i + 1]
                tmp[i] = tmp[i + 1] = hi
            print(t, tmp)
            book.add(tuple(tmp))
    # for _ in range(100):
    #     tmp = nums.copy()
    #     for _ in range(12):
    #         i = randint(0, n - 2)
    #         hi = tmp[i] if tmp[i] > tmp[i + 1] else tmp[i + 1]
    #         tmp[i] = tmp[i + 1] = hi
    #         if tuple(tmp) not in book:
    #             print(tmp)
    return sorted(book)

# res = brute()
# print(len(res))
# print(res)
