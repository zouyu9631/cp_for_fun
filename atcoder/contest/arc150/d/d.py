from collections import defaultdict
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
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

"""
DFS模板
手写栈模拟dfs还是快
"""

MOD = 998244353  # 1000000007

n = read_int()
g = [[] for _ in range(n + 1)]
for v, u in enumerate(read_int_tuple(), 2):
    g[u].append(v)

# depth_cnt = defaultdict(int)

# from types import GeneratorType
# def bootstrap(f, stack=[]):
#     def wrappedfunc(*args, **kwargs):
#         if stack: return f(*args, **kwargs)
#         to = f(*args, **kwargs)
#         while True:
#             if type(to) is GeneratorType:
#                 stack.append(to)
#                 to = next(to)
#             else:
#                 stack.pop()
#                 if not stack: break
#                 to = stack[-1].send(to)
#         return to
#     return wrappedfunc

# @bootstrap
# def dfs(u, d):
#     depth_cnt[d] += 1
#     for v in g[u]:
#         yield dfs(v, d + 1)
#     yield None
# dfs(1, 1)

depth = [0] + [1] * n
stack = [1]
depth_cnt = defaultdict(int)
depth_cnt[1] = 1
while stack:
    u = stack.pop()
    for v in g[u]:
        depth[v] = depth[u] + 1
        depth_cnt[depth[v]] += 1
        stack.append(v)

N = max(depth_cnt)
inverse = [1] * (N + 1)
acc = [0, 1]

for i in range(2, N + 1):
    inverse[i] = ((-inverse[MOD % i] * (MOD // i)) % MOD)
    acc.append((acc[-1] + inverse[i]) % MOD)


# acc = [0]
# for x in range(1, max(depth_cnt) + 1):
#     acc.append((acc[-1] + pow(x, MOD - 2, MOD)) % MOD)

res = 0
for x, t in depth_cnt.items():
    res += t * acc[x] % MOD
    res %= MOD

print(res)