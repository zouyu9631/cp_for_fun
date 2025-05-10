standard_input, packages, output_together = 1, 1, 1
dfs_tag, int_hashing, read_from_file = 0, 0, 0
multi_test = 0
output_mode = 0  # 0: in solve; 1: one res; 2: YES/NO; 3: Yes/No

def overlap_length(start1, end1, start2, end2):
    """计算两个区间的重叠长度"""
    return max(0, min(end1, end2) - max(start1, start2))

def overlap_length3(start1, end1, start2, end2, start3, end3):
    """计算三个区间的重叠长度"""
    return max(0, min(end1, end2, end3) - max(start1, start2, start3))

def calc(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    # 计算每个立方体的边界
    cube1 = [(a1, a1+7), (b1, b1+7), (c1, c1+7)]
    cube2 = [(a2, a2+7), (b2, b2+7), (c2, c2+7)]
    cube3 = [(a3, a3+7), (b3, b3+7), (c3, c3+7)]
    
    # 计算两两立方体和三个立方体的重叠体积
    overlap_2 = 0  # 两个立方体的重叠体积
    for i in range(3):
        for j in range(i+1, 3):
            cube_i = [cube1, cube2, cube3][i]
            cube_j = [cube1, cube2, cube3][j]
            overlap_len = 1
            for k in range(3):
                overlap_len *= overlap_length(cube_i[k][0], cube_i[k][1], cube_j[k][0], cube_j[k][1])
            overlap_2 += overlap_len
    
    # 计算三个立方体的重叠体积
    overlap_len3 = 1
    for i in range(3):
        overlap_len3 *= overlap_length3(cube1[i][0], cube1[i][1], cube2[i][0], cube2[i][1], cube3[i][0], cube3[i][1])
        
    # 计算单独的体积
    volume_single = 3 * (7**3) - 2 * overlap_2 + 3 * overlap_len3
    
    # 计算两两重叠的体积
    volume_double = overlap_2 - 3 * overlap_len3
    
    # 三个立方体共有的体积已经直接计算
    volume_triple = overlap_len3
    
    return volume_single, volume_double, volume_triple


def solve():
    V1, V2, V3 = MII()
    if 343 * 3 != V1 + 2 * V2 + 3 * V3:
        print('No')
        return
    
    a1 = b1 = c1 = 0
    for a2 in range(-7, 7):
        for a3 in range(-7, 7):
            for b2 in range(-7, 7):
                for b3 in range(-7, 7):
                    for c2 in range(-7, 7):
                        for c3 in range(-7, 7):
                            v1, v2, v3 = calc(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                            if v1 == V1 and v2 == V2 and v3 == V3:
                                print('Yes')
                                print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                return

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

    def GRAPH(n: int, m=-1):
        if m == -1:
            m = n - 1
        g = [[] for _ in range(n)]
        for _ in range(m):
            u, v = GMI()
            g[u].append(v)
            g[v].append(u)
        return g


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
