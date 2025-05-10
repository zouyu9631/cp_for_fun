standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 1
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No


def solve():
    N = II()
    A = LII()

    points = []

    for x, y in enumerate(A):
        points.append((x, y, y))

    srs = StaticRectangleSum(points)

    res = 0
    for _ in range(II()):
        a, b, c = MII()
        L, R, X = a ^ res, b ^ res, c ^ res
        res = srs.query(L - 1, R, 1, X + 1)
        print(res)


def main():
    # region local test
    if read_from_file and "AW" in os.environ.get("COMPUTERNAME", ""):
        test_no = read_from_file
        f = open(os.path.dirname(__file__) + f"\\in{test_no}.txt", "r")
        global input
        input = lambda: f.readline().rstrip("\r\n")
    # endregion

    T = II() if multi_test else 1
    for t in range(T):
        if output_mode == 0:
            solve()
        elif output_mode == 1:
            print(solve())
        elif output_mode == 2:
            print("YES" if solve() else "NO")
        elif output_mode == 3:
            print("Yes" if solve() else "No")


from bisect import bisect_left
from typing import List, Tuple


class StaticRectangleSum:
    __slots__ = "_wm", "_xs"

    def __init__(self, points: List[Tuple[int, int, int]]):
        """静态二维矩形区域计数 预先添加可能出现的点

        Args:
            points (List[Tuple[int, int, int]]): x, y, weight
        """
        points = sorted(points, key=itemgetter(0))
        self._xs, ys, ws = zip(*points)
        self._wm = WaveletMatrix(ys, ws)

    def query(self, x1: int, x2: int, y1: int, y2: int) -> int:
        """求矩形x1<=x<x2,y1<=y<y2的权值和 注意是左闭右开"""
        leftX, rightX = bisect_left(self._xs, x1), bisect_left(self._xs, x2)
        return self._wm.range_sum(leftX, rightX, y1, y2)


class BitVector:
    # reference: https://tiramister.net/blog/posts/bitvector/
    TABLE = bytes([
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
    ])

    def __init__(self, N):
        self.cnum = (N + 255) >> 8

        self.bit = bytearray(self.cnum << 5)
        self.chunk = [0] * (self.cnum + 1)
        self.blocks = bytearray(self.cnum << 5)

        self.built = False

    def set(self, pos):
        self.bit[pos >> 3] |= 1 << (pos & 7)

    def access(self, pos):
        return self.bit[pos >> 3] >> (pos & 7) & 1

    def popcount(self, num):
        return self.TABLE[num]

    def build(self):
        for i in range(self.cnum):
            k = i << 5
            for j in range(31):
                self.blocks[k + 1] = self.blocks[k] + self.popcount(self.bit[k])
                k += 1
            self.chunk[i + 1] = (
                self.chunk[i] + self.blocks[k] + self.popcount(self.bit[k])
            )
        self.built = True

    def rank(self, pos):
        assert self.built
        cpos, tmp = pos >> 8, pos & 255
        bpos, offset = tmp >> 3, tmp & 7

        i = cpos << 5 | bpos
        rest = self.bit[i] & ((1 << offset) - 1)
        return self.chunk[cpos] + self.blocks[i] + self.popcount(rest)

    def select(self, num):
        """return minimum i that satisfies rank(i) = num"""
        assert self.built
        if num == 0:
            return 0
        if self.rank(self.N) < num:
            return -1

        l = -1
        r = self.N
        while r - l > 1:
            c = (l + r) >> 1
            if self.rank(c) >= num:
                r = c
            else:
                l = c
        return r


class WaveletMatrix:
    # reference: https://miti-7.hatenablog.com/entry/2018/04/28/152259
    def __init__(self, A: List[int], weight: List[int] = None):
        self.nums = sorted(set(A))
        self.idx = {a: i for i, a in enumerate(self.nums)}
        self.A = [self.idx[a] for a in A]

        self.digit = (len(self.nums) - 1).bit_length()
        self.B = [None] * self.digit
        self.offset = [None] * self.digit

        self.weight = weight
        if self.weight:
            self.S = [
                [0 for _ in range(len(self.A) + 1)] for _ in range(self.digit + 1)
            ]
            for i, a in enumerate(self.A):
                self.S[self.digit][i + 1] = self.S[self.digit][i] + self.weight[i]

        if self.weight:
            T = list(zip(self.A, self.weight))
            for k in range(self.digit)[::-1]:
                self.B[k] = BitVector(len(T) + 1)
                zeros = []
                ones = []
                for i, (a, w) in enumerate(T):
                    if a >> k & 1:
                        self.B[k].set(i)
                        ones.append((a, w))
                    else:
                        zeros.append((a, w))
                self.B[k].build()
                self.offset[k] = len(zeros)
                T = zeros + ones
                for i, (a, w) in enumerate(T):
                    self.S[k][i + 1] = self.S[k][i] + w
        else:
            T = self.A
            for k in range(self.digit)[::-1]:
                self.B[k] = BitVector(len(T) + 1)
                zeros = []
                ones = []
                for i, a in enumerate(T):
                    if a >> k & 1:
                        self.B[k].set(i)
                        ones.append(a)
                    else:
                        zeros.append(a)
                self.B[k].build()
                self.offset[k] = len(zeros)
                T = zeros + ones

    def access(self, i: int):
        """return i-th value"""
        ret = 0
        cur = i
        for k in range(self.digit)[::-1]:
            if self.B[k].access(cur):
                ret |= 1 << k
                cur = self.B[k].rank(cur) + self.offset[k]
            else:
                cur -= self.B[k].rank(cur)
        return self.nums[ret]

    def rank(self, l: int, r: int, x: int):
        """return the number of x's in [l, r) range"""
        x = self.idx.get(x)
        if x is None:
            return 0
        for k in range(self.digit)[::-1]:
            if x >> k & 1:
                l = self.B[k].rank(l) + self.offset[k]
                r = self.B[k].rank(r) + self.offset[k]
            else:
                l -= self.B[k].rank(l)
                r -= self.B[k].rank(r)
        return r - l

    def quantile(self, l: int, r: int, n: int):
        """return n-th (0-indexed) smallest value in [l, r) range"""
        # assert 0 <= n < r - l
        ret = 0
        for k in range(self.digit)[::-1]:
            rank_l = self.B[k].rank(l)
            rank_r = self.B[k].rank(r)
            ones = rank_r - rank_l
            zeros = r - l - ones
            if zeros <= n:
                ret |= 1 << k
                l = rank_l + self.offset[k]
                r = rank_r + self.offset[k]
                n -= zeros
            else:
                l -= rank_l
                r -= rank_r
        return self.nums[ret]

    def rquantile(self, l: int, r: int, n: int):
        """return n-th (0-indeed) largest value in [l, r) range"""
        return self.quantile(l, r, r - l - 1 - n)

    def range_freq(self, l: int, r: int, lower: int, upper: int):
        """return the number of values s.t. lower <= x < upper"""
        if lower >= upper:
            return 0
        if lower + 1 == upper:
            return self.rank(l, r, lower)
        return self._range_freq_upper(l, r, upper) - self._range_freq_upper(l, r, lower)

    def prev_value(self, l: int, r: int, upper: int):
        """return maximum x s.t. x < upper in [l, r) range if exist, otherwise None"""
        cnt = self._range_freq_upper(l, r, upper)
        if cnt == 0:
            return None
        return self.quantile(l, r, cnt - 1)

    def next_value(self, l: int, r: int, lower: int):
        """return minimum x s.t. x >= lower in [l, r) range if exist, otherwise None"""
        cnt = self._range_freq_upper(l, r, lower)
        if cnt == r - l:
            return None
        return self.quantile(l, r, cnt)

    def range_sum(self, l: int, r: int, lower: int, upper: int):
        """return sum of values s.t. lower <= x < upper in [l, r) range
        must be constructed with weight
        """
        assert self.weight
        return self._range_sum_upper(l, r, upper) - self._range_sum_upper(l, r, lower)

    def range_least_sum(self, l: int, r: int, n: int):
        """return sum of least n (**1-indexed**) values in [l, r) range
        must be constructed with weight
        """
        assert self.weight
        assert 1 <= n <= r - l
        if self.digit == 0:
            return self.nums[0] * n
        ret = 0
        for k in range(self.digit)[::-1]:
            rank_l = self.B[k].rank(l)
            rank_r = self.B[k].rank(r)
            ones = rank_r - rank_l
            zeros = r - l - ones
            if zeros <= n:
                ret += self.S[k][r - rank_r] - self.S[k][l - rank_l]
                l = rank_l + self.offset[k]
                r = rank_r + self.offset[k]
                n -= zeros
                if n == 0:
                    return ret
            else:
                l -= rank_l
                r -= rank_r
        ret += self.S[0][l + n] - self.S[0][l]
        return ret

    def _range_freq_upper(self, l: int, r: int, upper: int):
        """return the number of values s.t. x < upper in [l, r) range"""
        if l >= r:
            return 0
        if upper > self.nums[-1]:
            return r - l
        if upper <= self.nums[0]:
            return 0
        upper = bisect_left(self.nums, upper)
        ret = 0
        for k in range(self.digit)[::-1]:
            rank_l = self.B[k].rank(l)
            rank_r = self.B[k].rank(r)
            ones = rank_r - rank_l
            zeros = r - l - ones
            if upper >> k & 1:
                ret += zeros
                l = rank_l + self.offset[k]
                r = rank_r + self.offset[k]
            else:
                l -= rank_l
                r -= rank_r
        return ret

    def _range_sum_upper(self, l: int, r: int, upper: int):
        """return sum of values s.t. x < upper in [l, r) range"""
        if l >= r:
            return 0
        if upper > self.nums[-1]:
            return self.S[self.digit][r] - self.S[self.digit][l]
        if upper <= self.nums[0]:
            return 0
        upper = bisect_left(self.nums, upper)
        ret = 0
        for k in range(self.digit)[::-1]:
            rank_l = self.B[k].rank(l)
            rank_r = self.B[k].rank(r)
            ones = rank_r - rank_l
            zero = r - l - ones
            if upper >> k & 1:
                ret += self.S[k][r - rank_r] - self.S[k][l - rank_l]
                l = rank_l + self.offset[k]
                r = rank_r + self.offset[k]
            else:
                l -= rank_l
                r -= rank_r
        return ret


# region
if standard_input:
    import os, sys, math

    input = lambda: sys.stdin.readline().strip()

    inf = math.inf

    def I():
        return input()

    def II():
        return int(input())

    def MII():
        return map(int, input().split())

    def LI():
        return list(input().split())

    def LII():
        return list(map(int, input().split()))

    def LFI():
        return list(map(float, input().split()))

    def GMI():
        return map(lambda x: int(x) - 1, input().split())

    def LGMI():
        return list(map(lambda x: int(x) - 1, input().split()))


if packages:
    from io import BytesIO, IOBase
    import math
    import random
    import bisect
    import typing
    from collections import Counter, defaultdict, deque
    from copy import deepcopy
    from functools import cmp_to_key, lru_cache, reduce
    from heapq import *
    from itertools import accumulate, combinations, permutations, count, product
    from operator import add, iand, ior, itemgetter, mul, xor
    from string import ascii_lowercase, ascii_uppercase, ascii_letters
    from typing import *

    BUFSIZE = 4096

if output_together:

    class FastIO(IOBase):
        newlines = 0

        def __init__(self, file):
            self._fd = file.fileno()
            self.buffer = BytesIO()
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

    class IOWrapper(IOBase):
        def __init__(self, file):
            self.buffer = FastIO(file)
            self.flush = self.buffer.flush
            self.writable = self.buffer.writable
            self.write = lambda s: self.buffer.write(s.encode("ascii"))
            self.read = lambda: self.buffer.read().decode("ascii")
            self.readline = lambda: self.buffer.readline().decode("ascii")

    sys.stdout = IOWrapper(sys.stdout)

if dfs_tag:
    from types import GeneratorType

    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if type(to) is GeneratorType:
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to

        return wrappedfunc


if int_hashing:
    RANDOM = random.getrandbits(20)

    class Wrapper(int):
        def __init__(self, x):
            int.__init__(x)

        def __hash__(self):
            return super(Wrapper, self).__hash__() ^ RANDOM


if True:

    def debug(*args, **kwargs):
        print("\033[92m", end="")
        print(*args, **kwargs)
        print("\033[0m", end="")


# endregion

if __name__ == "__main__":
    main()
