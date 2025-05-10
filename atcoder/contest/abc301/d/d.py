from functools import lru_cache
import sys
import io
import os

from math import inf

class F:
    def __init__(self, x, ptn: str):
        num = list(map(int, bin(x)[2:]))
        ptn = list(ptn)
        self.dn = m = max(len(num), len(ptn))
        self.num = [0] * (m - len(num)) + num
        self.ptn = ['0'] * (m - len(ptn)) + ptn

    @lru_cache(None)
    def calc(self, i=0, is_limit=True) -> int:
        if i == self.dn:
            return 0
        cm = 1 << (self.dn - i - 1)

        if self.ptn[i] != '?':
            ipn = int(self.ptn[i])
            if is_limit and ipn > self.num[i]:
                return -inf
            return ipn * cm + self.calc(i + 1, is_limit and ipn == self.num[i])
        else:
            if is_limit:
                if self.num[i]:
                    r1 = cm + self.calc(i + 1, True)
                    r0 = self.calc(i + 1, False)
                    return max(r1, r0)
                else:
                    return self.calc(i + 1, True)
            else:
                r1 = cm + self.calc(i + 1, False)
                r0 = self.calc(i + 1, False)
                return max(r1, r0)
                

def solve_dp():
    s = input()
    x = read_int()
    c = F(x, s)
    res = c.calc()
    return res if res > -inf else -1

"""
类似二分 枚举当前?能否为1（把后面的?都变成0），能的话这里就是1，否则就是0
"""

def solve():
    s = list(input())
    n = len(s)
    x = read_int()
    for i in range(n):
        if s[i] != '?': continue
        t = s[:]
        t[i] = '1'
        for j in range(i + 1, n):
            if t[j] == '?': t[j] = '0'
        y = int(''.join(t), 2)
        if y <= x:
            s[i] = '1'
        else:
            s[i] = '0'
    res = int(''.join(s), 2)
    return res if res <= x else -1
    
    
    # s = list(input())
    # s = ['0'] * (61 - len(s)) + s
    # s.reverse()
    # x = read_int()
    # res = 0
    # is_limit = True
    # for b in range(60, -1, -1):
    #     t = s[b]
    #     p = (x >> b) & 1
    #     res *= 2
    #     if is_limit and t == '?':
    #         t = p
    #         if p == 1:
    #             flag = False
    #             for i in range(b-1, -1, -1):
    #                 if s[i] == '?' or int(s[i]) < (x >> i) & 1:
    #                     break
    #                 if int(s[i]) > (x >> i) & 1:
    #                     flag = True
    #                     break                    
    #             if flag:
    #                 t = 0

    #     else:        
    #         t = int(t) if t != '?' else 1
    #     res += t
    #     if is_limit:
    #         if t > p:
    #             return -1
    #         elif t < p:
    #             is_limit = False
    
    # return res

def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1
    for t in range(T):
        print(solve())
        # print('YES' if solve() else 'NO')
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


def read_ints_minus_one():
    return [int(x) - 1 for x in input().split()]


def read_int_tuple():
    return map(int, input().split())


def read_graph(n: int, m: int, d=1):
    g = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        g[u - d].append(v - d)
        g[v - d].append(u - d)
    return g

def read_grid(m: int):
    return [input() for _ in range(m)]

def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()