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
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

class UnionSet:
    def __init__(self, n: int):
        self.parent = [*range(n)]

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x


    def union(self, x, y):  # rank by deep
        x_0 = self.find(x)
        y_0 = self.find(y)
        self.parent[y_0] = x_0

"""
需要改写的并查集
操作后，生成新的盒子和球，做个对应
"""

def solve():
    n, q = read_int_tuple()
    
    h = n + 1
    D = n + q
    k = D + n + 1
    us = UnionSet(2 * D + 1)
    
    id2box = list(range(n + 1))
    box2id = list(range(D + 1))
    
    for x in range(1, n + 1):
        us.union(x, D + x)

    for _ in range(q):
        O = read_int_tuple()
        if O[0] == 1:   # put y into x
            x, y = O[1], O[2]
            bx, by = id2box[x], id2box[y]
            us.union(bx, by)
            id2box[y] = h
            box2id[h] = y
            h += 1

        elif O[0] == 2: # add k into x
            x = O[1]
            x = id2box[x]
            us.union(x, k)
            k += 1
        else:   # O[0] == 3 query id: x
            x = D + O[1]
            b = us.find(x)
            
            print(box2id[us.find(x)])

T = 1#read_int()
for t in range(T):
    solve()