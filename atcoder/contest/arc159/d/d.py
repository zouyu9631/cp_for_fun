import sys
import io
import os

inf = 1 << 60

class Max_SegmentTree():
    __slots__ = ['n', 'e', 'log', 'size', 'data']

    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.log = (n - 1).bit_length()
        self.size = 1 << self.log
        self.data = [e] * (2 * self.size)

    def build(self, arr):
        # assert len(arr) <= self.n
        for i in range(self.n):
            self.data[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self.data[i] = self.data[2 * i] if self.data[2 * i] > self.data[2 * i + 1] else self.data[2 * i + 1]

    def set(self, p, x):
        # assert 0 <= p < self.n
        p += self.size
        self.data[p] = x
        for i in range(self.log):
            p >>= 1
            self.data[p] = self.data[2 * p] if self.data[2 * p] > self.data[2 * p + 1] else self.data[2 * p + 1]

    def get(self, p):
        # assert 0 <= p < self.n
        return self.data[p + self.size]

    def prod(self, l, r):
        # assert 0 <= l <= r <= self.n
        sml = smr = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                if sml < self.data[l]:
                    sml = self.data[l]

                l += 1
            if r & 1:
                r -= 1
                if sml < self.data[r]:
                    sml = self.data[r]
            l >>= 1
            r >>= 1

        return sml if sml > smr else smr

    def all_prod(self):
        return self.data[1]
    
    def __repr__(self) -> str:
        return str(self.data[self.size:self.size + self.n])

    def max_right(self, l, f):
        if l == self.n: return self.n
        l += self.size
        sm = self.e
        while True:
            while l % 2 == 0: l >>= 1
            tmp = f(sm if sm > self.data[l] else self.data[l])

            if not tmp:
                while l < self.size:
                    l = 2 * l
                    tmp = f(sm if sm > self.data[l] else self.data[l])

                    if tmp:
                        if sm < self.data[l]:
                            sm = self.data[l]
                        l += 1
                return l - self.size
            if sm < self.data[l]:
                sm = self.data[l]

            l += 1
            if (l & -l) == l: break
        return self.n

class Min_SegmentTree():
    __slots__ = ['n', 'e', 'log', 'size', 'data']

    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.log = (n - 1).bit_length()
        self.size = 1 << self.log
        self.data = [e] * (2 * self.size)

    def build(self, arr):
        # assert len(arr) <= self.n
        for i in range(self.n):
            self.data[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self.data[i] = self.data[2 * i] if self.data[2 * i] < self.data[2 * i + 1] else self.data[2 * i + 1]

    def set(self, p, x):
        # assert 0 <= p < self.n
        p += self.size
        self.data[p] = x
        for i in range(self.log):
            p >>= 1
            self.data[p] = self.data[2 * p] if self.data[2 * p] < self.data[2 * p + 1] else self.data[2 * p + 1]

    def get(self, p):
        # assert 0 <= p < self.n
        return self.data[p + self.size]

    def prod(self, l, r):
        # assert 0 <= l <= r <= self.n
        sml = smr = self.e
        l += self.size
        r += self.size
        while l < r:
            if l & 1:
                if sml > self.data[l]:
                    sml = self.data[l]
                l += 1
            if r & 1:
                r -= 1
                if sml > self.data[r]:
                    sml = self.data[r]
            l >>= 1
            r >>= 1
        return sml if sml < smr else smr

    def all_prod(self):
        return self.data[1]
    
    def __repr__(self) -> str:
        return str(self.data[self.size:self.size + self.n])

    def max_right(self, l, f):
        if l == self.n: return self.n
        l += self.size
        sm = self.e
        while True:
            while l % 2 == 0: l >>= 1
            tmp = f(sm if sm < self.data[l] else self.data[l])
            if not tmp:
                while l < self.size:
                    l = 2 * l
                    tmp = f(sm if sm < self.data[l] else self.data[l])
                    if tmp:
                        if sm > self.data[l]:
                            sm = self.data[l]

                        l += 1
                return l - self.size
            
            if sm > self.data[l]:
                sm = self.data[l]
            # sm = self.oper(sm, self.data[l])
            l += 1
            if (l & -l) == l: break
        return self.n

"""
Max线段树和Min线段树

区间添加的LIS
每来一段l和r 就更新r点的LIS[r]最大值 它可能是 LIS[l] + (r - l) 或者 r - min(blank[l: r])
"""

def solve():
    n = read_int()
    sg = []
    xs = set()
    for _ in range(n):
        l, r = read_int_tuple()
        sg.append((l, r + 1))
        xs.add(l)
        xs.add(r + 1)

    xs = sorted(xs)
    idx = {x: i for i, x in enumerate(xs)}

    blank_seg = Min_SegmentTree(len(xs), inf)
    blank_seg.build(xs)
    lis_seg = Max_SegmentTree(len(xs), 0)

    for l, r in sg:
        li, ri = idx[l], idx[r]

        blank = blank_seg.prod(li, ri)
        pre = lis_seg.prod(0, li)

        lis = max(lis_seg.get(ri), pre + r - l, r - blank)
        blank_seg.set(ri, r - lis)
        lis_seg.set(ri, lis)

    print(lis_seg.all_prod())


def main():
    # region local test
    # if 'AW' in os.environ.get('COMPUTERNAME', ''):
    #     test_no = 1
    #     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    #     global input
    #     input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = 1  #read_int()
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


def read_int():
    return int(input())


read_str = input

# endregion

if __name__ == "__main__":
    main()