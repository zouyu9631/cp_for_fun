from itertools import combinations
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


h1, w1 = read_int_tuple()
M1 = [read_int_list() for _ in range(h1)]
h2, w2 = read_int_tuple()
M2 = [read_int_list() for _ in range(h2)]

if any(all(M1[i1][j1] == M2[i2][j2]
           for i1, i2 in zip(row_ids, range(h2))
           for j1, j2 in zip(col_ids, range(w2)))
       for row_ids in combinations(range(h1), h2)
        for col_ids in combinations(range(w1), w2)):
    print('Yes')
else:
    print('No')

# for row_ids in combinations(range(h1), h2):
#     for col_ids in combinations(range(w1), w2):
#         if all(M1[i1][j1] == M2[i2][j2]
#                for i1, i2 in zip(row_ids, range(h2))
#                for j1, j2 in zip(col_ids, range(w2))):
#             print('Yes')
#             exit()
# print('No')
