import sys
import io
import os
from typing import List


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
下一个更大的数，各种解法可参考 LC2454

这题A是一个排列，用从小到大加入的方法，可以不用单调栈也能实现O(n)。
"""

MOD = 998244353  # 1000000007
inf = 1 << 60

def solve():
    n = read_int()
    pos, pre, nxt = [-1] * n, list(range(-1, n - 1)), list(range(1, n + 1))
    for i, x in enumerate(read_int_tuple()):
        pos[x - 1] = i
    # print(pos, pre, nxt)
    res = 0
    for x, i in enumerate(pos):
        p1 = pre[i]
        p2 = pre[p1] if p1 >= 0 else p1
        s1 = nxt[i]
        s2 = nxt[s1] if s1 < n else s1
        
        res += (x + 1) * (s2 - s1) * (i - p1)
        res += (x + 1) * (s1 - i) * (p1 - p2)
        
        if p1 >= 0:
            nxt[p1] = s1
        if s1 < n:
            pre[s1] = p1
        # nxt[p1], pre[s1] = s1, p1
        
    print(res)
    

def secondGreaterElement(n, A: List[int], rev=False):
    t = -1 if rev else n
    sk0, sk1, fst_bigger, scd_bigger, tmp = [], [], [t] * n, [t] * n, []
    for i, x in enumerate(A):
        while sk1 and A[sk1[-1]] <= x:
            scd_bigger[sk1.pop()] = i if not rev else n - 1 - i
        while sk0 and A[sk0[-1]] <= x:
            fst_bigger[sk0[-1]] = i if not rev else n - 1 - i
            tmp.append(sk0.pop())
        while tmp:
            sk1.append(tmp.pop())
        sk0.append(i)
    
    if rev:
        fst_bigger.reverse()
        scd_bigger.reverse()
    
    return fst_bigger, scd_bigger 

def solve_stk():
    n = read_int()
    A = read_int_list()

    fst_nxt_bigger, scd_nxt_bigger = secondGreaterElement(n, A)
    fst_pre_bigger, scd_pre_bigger = secondGreaterElement(n, A[::-1], True)
    
    # print(fst_nxt_bigger, scd_nxt_bigger)
    # print(fst_pre_bigger, scd_pre_bigger)
    res = 0
    for i, (x, p1, p2, s1, s2) in enumerate(zip(A, fst_pre_bigger, scd_pre_bigger, fst_nxt_bigger, scd_nxt_bigger)):
        res += x * (s2 - s1) * (i - p1)
        res += x * (s1 - i) * (p1 - p2)
    
    print(res)
    

T = 1#read_int()
for t in range(T):
    solve()