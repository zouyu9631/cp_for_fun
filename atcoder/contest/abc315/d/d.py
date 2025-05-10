from collections import defaultdict
import sys
import io
import os

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

def solve():
    n, m = read_int_tuple()
    rows = [[0] * 26 for _ in range(n)]
    cols = [[0] * 26 for _ in range(m)]
    ridx, cidx = set(range(n)), set(range(m))
    row_cnt, col_cnt = [0] * n, [0] * m
    
    G = [read_encode_str() for _ in range(n)]
    
    for i, R in enumerate(G):
        for j, x in enumerate(R):
            rows[i][x] += 1
            if rows[i][x] == 1:
                row_cnt[i] += 1
            cols[j][x] += 1
            if cols[j][x] == 1:
                col_cnt[j] += 1
    
    while True:
        dr, dc = [], []
        for i in ridx:
            if row_cnt[i] == 1 and m > 1:
                dr.append(i)
        for j in cidx:
            if col_cnt[j] == 1 and n > 1:
                dc.append(j)
        
        if not dr and not dc:
            return n * m
        
        for i in dr:
            n -= 1
            for j in cidx:
                if (x := G[i][j]) == -1:
                    continue
                
                rows[i][x] -= 1
                if rows[i][x] == 0:
                    row_cnt[i] -= 1
                cols[j][x] -= 1
                if cols[j][x] == 0:
                    col_cnt[j] -= 1
                    
                G[i][j] = -1
        
        for j in dc:
            m -= 1
            for i in ridx:
                if (x := G[i][j]) == -1:
                    continue
                
                rows[i][x] -= 1
                if rows[i][x] == 0:
                    row_cnt[i] -= 1
                cols[j][x] -= 1
                if cols[j][x] == 0:
                    col_cnt[j] -= 1
                    
                G[i][j] = -1
             
        for i in dr:
            ridx.remove(i)
        for j in dc:
            cidx.remove(j)
        
    

def main():
    # region local test
    if 'AW' in os.environ.get('COMPUTERNAME', ''):
        test_no = 1
        f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

        global input
        input = lambda: f.readline().rstrip("\r\n")
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

def read_encode_str(d=97):  # 'a': 97; 'A': 65
    return [ord(x) - d for x in input()]

read_str = input

# endregion

if __name__ == "__main__":
    main()