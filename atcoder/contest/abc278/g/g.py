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

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
交互题
优化结论的博弈论 SG NIM

segs 线段标记：-1: 已经删除，其他指向线段最左端
heads 线段最左端: 线段长度
"""

def solve():
    N, L, R = read_int_tuple()
    if L < R or (N + L) & 1 == 0:
        # 搞掉中间，两边对称博弈
        if (N + L) & 1: L += 1
        print('First', flush=True)
        print(f'{(N - L) // 2 + 1} {L}', flush=True)
        while True:
            x, ln = read_int_tuple()
            if x == 0: break
            print(f'{N + 2 - x - ln} {ln}', flush=True)
    else:  # L == R and (N + L) & 1 == 1
        # 预处理SG
        SG = [0] * (N + 1)
        # strategy = [[-1] * (N + 1) for _ in range(N + 1)]    # (sg, ln) -> left_ln
        strategy = [-1] * ((N + 1) * (N + 1))   # (sg * N + ln) -> left_ln
        for ln in range(L, N + 1):
            book = set()
            for left_len in range((ln - L + 3) // 2):
                right_len = ln - L - left_len
                book.add(SG[left_len] ^ SG[right_len])
            SG[ln] = next(mex for mex in range(N + 1) if mex not in book)
            
            for left_len in range((ln - L + 3) // 2):
                right_len = ln - L - left_len
                sg = SG[left_len] ^ SG[right_len] ^ SG[ln]
                strategy[sg * N + ln] = left_len
            
        # for ln in range(N + 1): print(ln, SG[ln], strategy[SG[ln] * N + ln])
        # print(SG, flush=True)

        # Run
        segs = [1] * (N + 1)
        heads = {1: N}
        
        def remove_seg(x, sg):
            st = segs[x]
            sl = heads.pop(st)
            sg ^= SG[sl]
            
            if x + L < st + sl:
                heads[x + L] = pl = st + sl - (x + L)
                segs[x + L: st + sl] = [x + L] * pl
                sg ^= SG[pl]
            
            if st < x:
                heads[st] = pl = x - st
                sg ^= SG[pl]
            
            segs[x: x + L] = [-1] * L

            return sg
        
        if SG[N]:
            print('First', flush=True)
            print(f'{strategy[SG[N] * N + N] + 1} {L}', flush=True)
            cur_sg = remove_seg(strategy[SG[N] * N + N] + 1, SG[N])
        else:
            print('Second', flush=True)

        # print(segs, flush=True)
        # print(heads, flush=True)
        # print(strategy, flush=True)

        while True:
            x, _ = read_int_tuple()
            if x == 0: break
            cur_sg = remove_seg(x, 0)
            # 找一条线段操作，把cur_sg变回0
            for x, ln in heads.items():
                if strategy[cur_sg * N + ln] >= 0:
                    mx = x + strategy[cur_sg * N + ln]
                    cur_sg = remove_seg(mx, cur_sg)
                    assert cur_sg == 0
                    print(f'{mx} {L}', flush=True)
                    break


T = 1  #read_int()
for t in range(T):
    solve()