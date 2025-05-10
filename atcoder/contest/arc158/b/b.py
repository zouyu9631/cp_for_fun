from bisect import bisect_left
import sys
import io
import os


MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

def calc(a: int, b: int, c: int) -> float:
    return (a + b + c) / (a * b * c)
 
def solve():
    n = read_int()
    pos, neg = [], []
    for x in read_int_tuple():
        if x > 0:
            pos.append(x)
        else:
            neg.append(-x)
            
    pos.sort()
    neg.sort()
    hi, lo = -inf, inf
    
    if len(pos) >= 3:
        t = calc(*pos[-3:])
        hi = max(hi, t)
        lo = min(lo, t)
        t = calc(*pos[:3])
        hi = max(hi, t)
        lo = min(lo, t)
    if len(neg) >= 3:
        t = calc(*neg[-3:])
        hi = max(hi, t)
        lo = min(lo, t)
        t = calc(*neg[:3])
        hi = max(hi, t)
        lo = min(lo, t)
    
    if len(neg) >= 2:
        for a in pos:
            t = calc(a, -neg[0], -neg[1])
            hi = max(hi, t)
            lo = min(lo, t)
            t = calc(a, -neg[-1], -neg[-2])
            hi = max(hi, t)
            lo = min(lo, t)
            
            # idx = bisect_left(neg, a)
            # start = max(0, idx - 2)
            # for i in range(start, min(idx + 1, len(neg) - 1)):
            #     b, c = neg[i], neg[i + 1]
            #     t = calc(a, -b, -c)
            #     hi = max(hi, t)
            #     lo = min(lo, t)
 
    if len(pos) >= 2:
        for a in neg:
            t = calc(a, -pos[0], -pos[1])
            hi = max(hi, t)
            lo = min(lo, t)
            t = calc(a, -pos[-1], -pos[-2])
            hi = max(hi, t)
            lo = min(lo, t)
            
            # idx = bisect_left(pos, a)
            # start = max(0, idx - 2)
            # for i in range(start, min(idx + 1, len(pos) - 1)):
            #     b, c = pos[i], pos[i + 1]
            #     t = calc(a, -b, -c)
            #     hi = max(hi, t)
            #     lo = min(lo, t)
    
    print(f'{lo:.15f}')
    print(f'{hi:.15f}')

def solve_brute():
    n = read_int()
    pos, neg = [], []
    for x in read_int_tuple():
        if x > 0:
            pos.append(x)
        else:
            neg.append(x)
    pos.sort()
    neg.sort()
    hi, lo = -inf, inf
    
    A = []
    if len(pos) > 6:
        A.extend(pos[:3])
        A.extend(pos[-3:])
    else:
        A.extend(pos)
    
    if len(neg) > 6:
        A.extend(neg[:3])
        A.extend(neg[-3:])
    else:
        A.extend(neg)
    
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                t = calc(A[i], A[j], A[k])
                lo = min(lo, t)
                hi = max(hi, t)
    
    print(lo)
    print(hi)
    
    return

    



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