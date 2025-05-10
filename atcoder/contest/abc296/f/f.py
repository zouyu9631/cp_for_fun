import sys
import io
import os

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

    def sum_range(self, l, r):
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

    def __setitem__(self, i, x):
        # arr[i] = x
        self.add(i, x - self[i])

"""
群论 三阶置换可以拆成两个二阶置换
    对于一个n阶置换 n为奇数时 一定可以拆成偶数个二阶置换 即可成功
                   n为偶数时 需要另一个也是偶数的置换群拼一起才能成功
"""

def solve():
    n = read_int()
    A = [x - 1 for x in read_int_tuple()]
    B = [x - 1 for x in read_int_tuple()]
    cA, cB = [0] * n, [0] * n
    for x in A:
        cA[x] += 1
    for x in B:
        cB[x] += 1
    if cA != cB:
        return False
    if 0 in cA:
        return True
    
    seen, res = [False] * n, 0
    for i in range(n):
        x, cnt = A[i], 0
        while not seen[x]:
            seen[x] = True
            cnt += 1
            x = A[B[x]]
            
        res += cnt and cnt & 1 == 0
    
    return res & 1 == 0
    
    # bit, cnt = BIT(n), 0
    # for x in map(B.__getitem__, sorted(range(n), key=A.__getitem__)):
    #     cnt ^= bit.sum_range(x + 1, n) & 1
    #     bit.add(x, 1)
    
    # return cnt == 0
    

def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion
    
    T = 1#read_int()
    for t in range(T):
        # solve()
        print('Yes' if solve() else 'No')


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