import sys
import io
import os

"""
数位DP + KMP 参考：
https://atcoder.jp/contests/abc295/submissions/40042014

简易写法：
既然每次匹配s都计分，那就考虑固定s在x的位置时，有多少满足条件即可
"""

def calc(x: int, ss: str) -> int:
    res = 0
    s, sx = int(ss), str(x)
    lx, ls = len(sx), len(ss)
    
    """
    split sx to pre + mid + xxx
    'mid' have the same length with s
    enumerate the length of 'xxx' -> ln
    """
    
    for ln in range(lx - ls + 1 - (ss[0] == '0')):
        mid = int(sx[lx - ln - ls: lx - ln]) # to match s
        
        # for pre == sx[lx - ln - ls]
        if mid == s:
            res += int('0' + sx[lx - ln:]) + 1
        elif mid > s:
            res += pow(10, ln)
        
        # for pre < sx[lx - ln - ls]
        pre = int('0' + sx[:lx - ln - ls]) - (ss[0] == '0')
        res += pre * pow(10, ln)
    
    return res

def solve():
    s, l, r = input().split()
    print(calc(int(r), s) - calc(int(l) - 1, s))


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion
    
    T = read_int()
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