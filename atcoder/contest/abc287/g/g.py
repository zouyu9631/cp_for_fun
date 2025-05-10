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
    return map(int, input().split())


def read_int():
    return int(input())

read_str = input




# endregion
class SegmentTree():
    __slots__ = ['n', 'oper', 'e', 'log', 'size', 'data']

    def __init__(self, n, oper, e):
        self.n = n
        self.oper = oper
        self.e = e
        self.log = (n - 1).bit_length()
        self.size = 1 << self.log
        self.data = [e] * (2 * self.size)

    def _update(self, k):
        self.data[k] = self.oper(self.data[2 * k], self.data[2 * k + 1])

    def build(self, arr):
        # assert len(arr) <= self.n
        for i in range(self.n):
            self.data[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self._update(i)

    def set(self, p, x):
        # assert 0 <= p < self.n
        p += self.size
        self.data[p] = x
        for i in range(self.log):
            p >>= 1
            self._update(p)

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
                sml = self.oper(sml, self.data[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.oper(self.data[r], smr)
            l >>= 1
            r >>= 1
        return self.oper(sml, smr)

    def all_prod(self):
        return self.data[1]

    def max_right(self, l, f):
        # assert 0 <= l <= self.n
        # assert f(self.)
        if l == self.n: return self.n
        l += self.size
        sm = self.e
        while True:
            while l % 2 == 0: l >>= 1
            if not f(self.oper(sm, self.data[l])):
                while l < self.size:
                    l = 2 * l
                    if f(self.oper(sm, self.data[l])):
                        sm = self.oper(sm, self.data[l])
                        l += 1
                return l - self.size
            sm = self.oper(sm, self.data[l])
            l += 1
            if (l & -l) == l: break
        return self.n

    def min_left(self, r, f):
        # assert 0 <= r <= self.n
        # assert f(self.)
        if r == 0: return 0
        r += self.size
        sm = self.e
        while True:
            r -= 1
            while r > 1 and (r % 2): r >>= 1
            if not f(self.oper(self.data[r], sm)):
                while r < self.size:
                    r = 2 * r + 1
                    if f(self.oper(self.data[r], sm)):
                        sm = self.oper(self.data[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.oper(self.data[r], sm)
            if (r & -r) == r: break
        return 0

class BIT:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def build(self, list):
        self.tree[1:] = list.copy()
        for i in range(self.size + 1):
            j = i + (i & (-i))
            if j < self.size + 1:
                self.tree[j] += self.tree[i]

    def sum(self, i):
        # return sum(arr[0: i])
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def sum_range(self, l, r):
        # sum(arr[l: r]) -> return self.sum(r) - self.sum(l)
        s = 0
        while l < r:
            s += self.tree[r]
            r -= r & (-r)
        while r < l:
            s -= self.tree[l]
            l -= l & (-l)
        return s

    def add(self, i, x):
        # arr[i] += 1
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def __getitem__(self, i):
        # return arr[i]
        return self.sum_range(i, i + 1)

    def  __repr__(self):
        return 'BIT({0})'.format([self[i] for i in range(self.size)])

    def __setitem__(self, i, x):
        # arr[i] = x
        self.add(i, x - self[i])

    def bisect(self, x):
        # 总和大于等于x的位置的index
        le = 0
        ri = 1 << (self.size.bit_length() - 1)
        while ri > 0:
            if le + ri <= self.size and self.tree[le + ri] < x:
                x -= self.tree[le + ri]
                le += ri
            ri >>= 1
        return le + 1

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007   INV2 = (MOD + 1) >> 1 # pow(2, MOD - 2, MOD)
inf = 1 << 60

"""
线段树 树状数组 把查询里修改的值也离散化掉 就不用再修改时考虑大小问题
"""

def solve():
    n = read_int()
    book = []
    vals = set()
    for _ in range(n):
        s, c = read_int_tuple()
        vals.add(s)
        book.append((s, c))

    Opers = []
    q = read_int()
    for _ in range(q):
        ops = tuple(read_int_tuple())
        Opers.append(ops)
        
        if ops[0] == 1:
            vals.add(ops[2])
    
    vals = sorted(vals)
    inv = {s: i for i, s in enumerate(vals)}

    N = len(vals)

    bit_cnt, bit_tot = BIT(N), BIT(N)
    
    cnt = tot = 0
    
    def modify(s, c):
        nonlocal cnt, tot
        
        i = inv[s]
        bit_cnt.add(i, c)
        bit_tot.add(i, s * c)
        cnt += c
        tot += s * c
    
    
    for s, c in book:
        modify(s, c)
    
    for op, *args in Opers:
        if op == 3:
            y = cnt - args[0]

            if y < 0:
                print(-1)
            else:
                i = bit_cnt.bisect(y) - 1
                dif = y - bit_cnt.sum(i)
                if dif:
                    p = bit_tot[i] // bit_cnt[i]
                    print(tot - bit_tot.sum(i) - dif * p)
                else:
                    print(tot - bit_tot.sum(i))

        elif op == 2:
            i, c = args
            i -= 1
            s, old_c = book[i]
            
            modify(s, c - old_c)
            book[i] = (s, c)

        elif op == 1:
            i, s = args
            i -= 1
            old_s, c = book[i]
            
            modify(old_s, -c)
            modify(s, c)
            book[i] = (s, c)

    return 

    """ 线段树写法 """
    e = (0, 0)  # 总分， 数量
    def merge(x, y):
        return (x[0] + y[0], x[1] + y[1])
    
    st = SegmentTree(N, merge, e)
    A = [e] * N
    
    for s, c in book:
        A[inv[s]] = merge(A[inv[s]], (s * c, c))
    
    st.build(A)

    # print(A)
    # print(st.data)
    
    for op, *args in Opers:
        if op == 3:
            x = args[0]
            i = st.min_left(N, lambda seg: seg[1] < x)
            if i == 0:
                print(-1)
            else:
                seg = st.prod(i - 1, N)
                s, c = st.get(i - 1)
                p = s // c
                res = seg[0] - (seg[1] - x) * p
                print(res)
            
            # print(st.data)
            # print(i, x)
        elif op == 2:
            i, c = args
            i -= 1
            s, old_c = book[i]
            idx = inv[s]

            cnt = st.get(idx)[1]
            cnt += (c - old_c)
            st.set(idx, (s * cnt, cnt))
            
            book[i] = (s, c)

        elif op == 1:
            i, s = args
            i -= 1
            old_s, c = book[i]
            idx = inv[old_s]
            
            cnt = st.get(idx)[1]
            cnt += (0 - c)
            st.set(idx, (old_s * cnt, cnt))
            
            idx = inv[s]
            
            cnt = st.get(idx)[1]
            cnt += c
            st.set(idx, (s * cnt, cnt))

            book[i] = (s, c)

    # print(st.data)



T = 1#read_int()
for t in range(T):
    solve()