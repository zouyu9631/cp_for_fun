from bisect import bisect_left
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
if 'AW' in os.environ.get('COMPUTERNAME', ''):
    test_no = 1
    f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    def input():
        return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
从高到低，对每一位做判断：
    如果都是0或者都是1，可以使结果为0；
    但某一位既有1也有0的时候，虽然选择一部分可以去掉另一部分，但当下还是不确定选择哪一部分；
    比赛中用cur存储当前可能要做的所有需要判断的组合
    DFS可以简化写法
    *去重排序后，直接在区间里bisect更快，用栈模拟递归*，递归结束条件也可以优化
"""

def solve():
    _ = read_int()
    A, res = sorted(set(read_int_tuple())), inf
    n = len(A)
    stk = [(A[-1].bit_length() - 1, 0, n, 0)]
    while stk:
        bi, lf, rt, va = stk.pop()
        if lf + 1 == rt or bi == -1:
            res = min(res, va)
            continue
        
        if (A[lf] ^ A[rt - 1]) >> bi & 1: # 这一位，在范围内有0也有1
            md = bisect_left(A, A[rt - 1] >> bi << bi, lf, rt)
            vb = 1 << bi | va
            stk.append((bi - 1, lf, md, vb))
            stk.append((bi - 1, md, rt, vb))
        else:
            stk.append((bi - 1, lf, rt, va))
    
    print(res)

def solve_recur():
    _ = read_int()
    A = set(read_int_tuple())
    
    def dfs(bi, A):
        if bi < 0: return 0
        B = [[], []]
        for x in A:
            B[x >> bi & 1].append(x)

        for b in range(2):  # 这一位可以取0
            if not B[b]: return dfs(bi - 1, B[1 - b])

        return 1 << bi | min(dfs(bi - 1, S) for S in B)
    
    print(dfs(max(A).bit_length(), A))

def solve_old():
    _ = read_int()
    cur = [set(read_int_tuple())]
    res = 0
    for i in range(max(cur[0]).bit_length(), -1, -1):
        nxt = []
        bx = 1 << i
        for S in cur:
            tmp = [set(), set()]
            for x in S:
                tmp[(x >> i) & 1].add(x)
            if bx:
                if tmp[0] and tmp[1]:
                    nxt.extend(tmp)
                else:
                    bx = 0
                    if tmp[0]:
                        nxt = [tmp[0]]
                    else:
                        nxt = [tmp[1]]
            else:
                if tmp[0] and tmp[1]:
                    pass
                else:
                    if tmp[0]:
                        nxt.append(tmp[0])
                    else:
                        nxt.append(tmp[1])
            
        res |= bx
        cur = nxt
                    
    print(res)

T = 1#read_int()
for t in range(T):
    solve()