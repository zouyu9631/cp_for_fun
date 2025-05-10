import sys
import io
import os


"""
基环内向树 森林 求所有环内点的个数
两种写法：平扫记录和拓扑排序
"""


def solve_straight():
    n = read_int()
    A = [x - 1 for x in read_int_tuple()]
    seen, res = [False] * n, 0
    for x in range(n):
        if seen[x]: continue
        d = dict()
        while not seen[x]:
            seen[x] = True
            d[x] = len(d)
            x = A[x]
        if x in d:
            res += len(d) - d[x]
    print(res)

def solve_pre(n, A: list):
    ind = [0] * n
    for x in A:
        ind[x] += 1
    
    q = [x for x in range(n) if ind[x] == 0]
    while q:
        x = q.pop()
        y = A[x]
        ind[y] -= 1
        if ind[y] == 0:
            q.append(y)
    return sum(ind)

def solve():
    n = read_int()
    A = [x - 1 for x in read_int_tuple()]
    ind = [0] * n
    for x in A:
        ind[x] += 1
    
    q = [x for x in range(n) if ind[x] == 0]
    while q:
        x = q.pop()
        y = A[x]
        ind[y] -= 1
        if ind[y] == 0:
            q.append(y)
    return sum(ind)

def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion
    
    # T = 1#read_int()
    # for t in range(T):
    #     solve()
    #     # print('Yes' if solve() else 'No')
    # def iterate_tokens():
    #     for line in sys.stdin:
    #         for word in line.split():
    #             yield word
    # tokens = iterate_tokens()
    # n = int(next(tokens))
    # A = [int(next(tokens)) - 1 for _ in range(n)]
    # print(solve(n, A))
    print(solve())


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

def read_ints_minus_one():
    return [int(x) - 1 for x in input().split()]

def read_int():
    return int(input())

read_str = input


# endregion


if __name__ == "__main__":
    main()