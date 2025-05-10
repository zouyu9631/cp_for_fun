from collections import defaultdict
from heapq import heappop, heappush
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

# sys.setrecursionlimit(200000)

"""
区间DP 数轴从0点开始，左右探索。https://atcoder.jp/contests/abc273/editorial/5046
如果目标为负数，则所有坐标都取相反数让目标为正数。
"""

inf = 1 << 60

def solve():
    n, T = read_int_tuple()
    fc = 1
    if T < 0:
        fc, T = -1, -T

    XS = [[0], [0, T]]  # 0: 负轴， 1: 正轴
    K = dict()
    for w, k in zip(read_int_tuple(), read_int_tuple()):
        w *= fc
        k *= fc
        
        if 0 < w < T and w < k:
            return -1
        
        if 0 < k < w or w < k < 0:
            continue
        
        K[w] = k
        
        if w < T:
            XS[w > 0].append(w)
        if k < T:
            XS[k > 0].append(k)

    XS[0].sort(reverse=True)
    XS[1].sort()
    
    negln, posln = map(len, XS)
    N = negln * posln
    dp0, dp1 = [inf] * N, [inf] * N
    dp0[0] = dp1[0] = 0
    
    for cur in range(1, N):
        li, ri = divmod(cur, posln)
        lx, rx = XS[0][li], XS[1][ri]
        if li and (lx not in K or lx < K[lx] <= rx):
            dp0[cur] = min(dp0[cur - posln] + XS[0][li - 1] - lx, dp1[cur - posln] + rx - lx)
        
        if ri and (rx not in K or lx <= K[rx] < rx):
            dp1[cur] = min(dp0[cur - 1] + rx - lx, dp1[cur - 1] + rx - XS[1][ri - 1])
        
        if rx == T:
            res = min(dp0[cur], dp1[cur])
            if res < inf:
                return res
    
    return -1

    
    # # interval-DP 424 ms
    # N = n * 2 + 2
    # XS = [0, T]
    # K = dict()
    # for w, k in zip(read_int_tuple(), read_int_tuple()):  # walls: -1, keys: 1
    #     K[w * fc] = k * fc
    #     XS.append(w * fc)
    #     XS.append(k * fc)
    # XS.sort()
    # P0 = XS.index(0)
    # dp = [inf] * (N * N * 2) # (left, right, side): dist ; i: (left * N + right) * 2 + side
    # ori = (P0 * N + P0) * 2
    # dp[ori] = dp[ori + 1] = 0
    # q = [ori]
    
    # res = inf
    # while q:
    #     nq = []
    #     for pos in q:
    #         dist = dp[pos]
    #         (left, right), side = divmod(pos >> 1, N), pos & 1
    #         lx, rx = XS[left], XS[right]
    #         cur = rx if side else lx
            
    #         left_pos = ((left - 1) * N + right) * 2
    #         if left and (XS[left - 1] not in K or XS[left] <= K[XS[left - 1]] <= XS[right]) and dp[left_pos] > dist + cur - XS[left - 1]:
    #             dp[left_pos] = dist + cur - XS[left - 1]
    #             nq.append(left_pos)
            
    #         right_pos = (left * N + right + 1) * 2 + 1
    #         if right + 1 < N and (XS[right + 1] not in K or XS[left] <= K[XS[right + 1]] <= XS[right]) and dp[right_pos] > dist + XS[right + 1] - cur:
    #             dp[right_pos] = dist + XS[right + 1] - cur
    #             nq.append(right_pos)
    #             if XS[right + 1] == T and res > dp[right_pos]:
    #                 res = dp[right_pos]
        
    #     if res < inf: return res
    #     q = nq
        
    # return -1

    # # 递推 1082 ms
    # dp = {(P0, P0, 0): 0}   # (left, right, side): dist
    
    # while dp:
    #     book = defaultdict(lambda: inf)
    #     for (left, right, side), dist in dp.items():
    #         if XS[right] == T:
    #             return dist

    #         lx, rx = XS[left], XS[right]
    #         cur = rx if side else lx
    #         if left and (XS[left - 1] not in K or XS[left] <= K[XS[left - 1]] <= XS[right]) and book[left - 1, right, 0] > dist + cur - XS[left - 1]:
    #             book[left - 1, right, 0] = dist + cur - XS[left - 1]
    #         if right + 1 < N and (XS[right + 1] not in K or XS[left] <= K[XS[right + 1]] <= XS[right]) and book[left, right + 1, 1] > dist + XS[right + 1] - cur:
    #             book[left, right + 1, 1] = dist + XS[right + 1] - cur
        
    #     dp = book
    
    # return -1
    
    # # dij TLE 2
    # q = [(0, P0, P0, 0)]
    # book = defaultdict(lambda: inf)
    # book[P0, P0, 0] = 0

    # while q:
    #     dist, left, right, side = heappop(q)
    #     if XS[right] == T:
    #         return dist
        
    #     lx, rx = XS[left], XS[right]
    #     cur = rx if side else lx
    #     if left and (XS[left - 1] not in K or XS[left] <= K[XS[left - 1]] <= XS[right]) and book[left - 1, right, 0] > dist + cur - XS[left - 1]:
    #         book[left - 1, right, 0] = dist + cur - XS[left - 1]
    #         heappush(q, (dist + cur - XS[left - 1], left - 1, right, 0))
    #     if right + 1 < N and (XS[right + 1] not in K or XS[left] <= K[XS[right + 1]] <= XS[right]) and book[left, right + 1, 1] > dist + XS[right + 1] - cur:
    #         book[left, right + 1, 1] = dist + XS[right + 1] - cur
    #         heappush(q, (dist + XS[right + 1] - cur, left, right + 1, 1))
        
    # return -1

    # # mem search TLE 4
    # book = dict()
    # @bootstrap
    # def calc(left, right, side):
    #     if (left, right, side) in book:
    #         yield book[left, right, side]
    #     if XS[right] == T:
    #         book[left, right, side] = yield 0
    #     lx, rx = XS[left], XS[right]
    #     cur = rx if side else lx
    #     if cur in K and (K[cur] < lx or K[cur] > rx):
    #         book[left, right, side] = yield inf
        
    #     res = inf
    #     if left:
    #         tr = yield calc(left - 1, right, 0)
    #         res = abs(cur - XS[left - 1]) + tr
    #     if right < N - 1:
    #         tr = yield calc(left, right + 1, 1)
    #         res = min(res, abs(cur - XS[right + 1]) + tr)
    #     yield res

    # res = calc(P0, P0, 0)
    # print(res if res < inf else -1)


T = 1  #read_int()
for t in range(T):
    print(solve())