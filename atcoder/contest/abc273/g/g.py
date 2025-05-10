from collections import defaultdict
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
#     test_no = 3
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007


def solve():
    n = read_int()

    row_cnt, col_cnt = [0, 0], [0, 0]
    for x in read_int_tuple():
        if x: row_cnt[x - 1] += 1
    for x in read_int_tuple():
        if x: col_cnt[x - 1] += 1
    
    if row_cnt[0] + 2 * row_cnt[1] != col_cnt[0] + 2 * col_cnt[1]:
        return 0
    rem = col_cnt[0] + 2 * col_cnt[1]
    
    dp = {col_cnt[1]: 1}
    for _ in range(row_cnt[0]): # 每行1个
        np = defaultdict(int)
        for c2, v in dp.items():
            c1 = rem - c2 * 2
            if c2:
                np[c2 - 1] += v * c2
                np[c2 - 1] %= MOD
            if c1:
                np[c2] += v * c1
                np[c2] %= MOD
        dp = np
        rem -= 1
        
    for _ in range(row_cnt[1]): # 每行2个
        np = defaultdict(int)
        for c2, v in dp.items():
            c1 = rem - c2 * 2
            if c2:  # c2里选1个，直接减2
                np[c2 - 1] += v * c2
                np[c2 - 1] %= MOD
            if c2 > 1:  # c2里选2个，各减1
                np[c2 - 2] += v * (c2 * (c2 - 1) // 2)
                np[c2 - 2] %= MOD
            if c2 and c1:   # c2， c1各选1个
                np[c2 - 1] += v * c2 * c1
                np[c2 - 1]
            if c1 > 1:  # c1里选2个，各减1
                np[c2] += v * (c1 * (c1 - 1) // 2)
                np[c2] %= MOD
        dp = np
        rem -= 2
    
    return dp[0]

    # mem TLE 21 pass    
    # mem = dict()
    # @bootstrap
    # def calc(t, rc, cc):    # total, row 1-cnt, col 1-cnt
    #     if (t, rc, cc) in mem:
    #         res = mem[t, rc, cc]
    #     elif t == rc == cc:
    #         res = yield fact(t)
    #     elif rc > cc:
    #         res = yield calc(t, cc, rc)
    #     else:
    #         rt, ct = (t - rc) >> 1, (t - cc) >> 1 # row 2-cnt, col 2-cnt
    #         res = 0
    #         if cc >= 2:
    #             tmp = yield calc(t - 2, rc, cc - 2)
    #             res += comb(cc, 2) * tmp
    #             res %= MOD
    #         if ct >= 2:
    #             tmp = yield calc(t - 2, rc, cc + 2)
    #             res += comb(ct, 2) * tmp
    #             res %= MOD
    #         if cc and ct:
    #             tmp = yield calc(t - 2, rc, cc)
    #             res += cc * ct * tmp
    #             res %= MOD
    #         if ct:
    #             tmp = yield calc(t - 2, rc, cc)
    #             res += ct * tmp
    #             res %= MOD
    #     mem[t, rc, cc] = res
    #     yield res
        

    # return calc(row_cnt[0], col_cnt[0])

T = 1#read_int()
for t in range(T):
    print(solve())