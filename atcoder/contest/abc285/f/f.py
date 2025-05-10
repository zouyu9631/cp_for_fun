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

# region SortedList
class SL:
    def __init__(self, iterable=[], _load=30):
        """Initialize sorted list instance."""
        values = sorted(iterable)
        self._len = _len = len(values)
        self._load = _load
        self._lists = _lists = [values[i:i + _load] for i in range(0, _len, _load)]
        self._list_lens = [len(_list) for _list in _lists]
        self._mins = [_list[0] for _list in _lists]
        self._fen_tree = []
        self._rebuild = True

    def _fen_build(self):
        """Build a fenwick tree instance."""
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for i in range(len(_fen_tree)):
            if i | i + 1 < len(_fen_tree):
                _fen_tree[i | i + 1] += _fen_tree[i]
        self._rebuild = False

    def _fen_update(self, index, value):
        """Update `fen_tree[index] += value`."""
        if not self._rebuild:
            _fen_tree = self._fen_tree
            while index < len(_fen_tree):
                _fen_tree[index] += value
                index |= index + 1

    def _fen_query(self, end):
        """Return `sum(_fen_tree[:end])`."""
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        x = 0
        while end:
            x += _fen_tree[end - 1]
            end &= end - 1
        return x

    def _fen_findkth(self, k):
        """Return a pair of (the largest `idx` such that `sum(_fen_tree[:idx]) <= k`, `k - sum(_fen_tree[:idx])`)."""
        _list_lens = self._list_lens
        if k < _list_lens[0]:
            return 0, k
        if k >= self._len - _list_lens[-1]:
            return len(_list_lens) - 1, k + _list_lens[-1] - self._len
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        idx = -1
        for d in reversed(range(len(_fen_tree).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(_fen_tree) and k >= _fen_tree[right_idx]:
                idx = right_idx
                k -= _fen_tree[idx]
        return idx + 1, k

    def _delete(self, pos, idx):
        """Delete value at the given `(pos, idx)`."""
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len -= 1
        self._fen_update(pos, -1)
        del _lists[pos][idx]
        _list_lens[pos] -= 1

        if _list_lens[pos]:
            _mins[pos] = _lists[pos][0]
        else:
            del _lists[pos]
            del _list_lens[pos]
            del _mins[pos]
            self._rebuild = True

    def _loc_left(self, value):
        """Return an index pair that corresponds to the first position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        lo, pos = -1, len(_lists) - 1
        while lo + 1 < pos:
            mi = (lo + pos) >> 1
            if value <= _mins[mi]:
                pos = mi
            else:
                lo = mi

        if pos and value <= _lists[pos - 1][-1]:
            pos -= 1

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value <= _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def _loc_right(self, value):
        """Return an index pair that corresponds to the last position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        pos, hi = 0, len(_lists)
        while pos + 1 < hi:
            mi = (pos + hi) >> 1
            if value < _mins[mi]:
                hi = mi
            else:
                pos = mi

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value < _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def add(self, value):
        """Add `value` to sorted list."""
        _load = self._load
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len += 1
        if _lists:
            pos, idx = self._loc_right(value)
            self._fen_update(pos, 1)
            _list = _lists[pos]
            _list.insert(idx, value)
            _list_lens[pos] += 1
            _mins[pos] = _list[0]
            if _load + _load < len(_list):
                _lists.insert(pos + 1, _list[_load:])
                _list_lens.insert(pos + 1, len(_list) - _load)
                _mins.insert(pos + 1, _list[_load])
                _list_lens[pos] = _load
                del _list[_load:]
                self._rebuild = True
        else:
            _lists.append([value])
            _mins.append(value)
            _list_lens.append(1)
            self._rebuild = True

    def discard(self, value):
        """Remove `value` from sorted list if it is a member."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_right(value)
            if idx and _lists[pos][idx - 1] == value:
                self._delete(pos, idx - 1)

    def remove(self, value):
        """Remove `value` from sorted list; `value` must be a member."""
        _len = self._len
        self.discard(value)
        if _len == self._len:
            raise ValueError('{0!r} not in list'.format(value))

    def pop(self, index=-1):
        """Remove and return value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        value = self._lists[pos][idx]
        self._delete(pos, idx)
        return value

    def bisect_left(self, value):
        """Return the first index to insert `value` in the sorted list."""
        pos, idx = self._loc_left(value)
        return self._fen_query(pos) + idx

    def bisect_right(self, value):
        """Return the last index to insert `value` in the sorted list."""
        pos, idx = self._loc_right(value)
        return self._fen_query(pos) + idx

    def count(self, value):
        """Return number of occurrences of `value` in the sorted list."""
        return self.bisect_right(value) - self.bisect_left(value)

    def __len__(self):
        """Return the size of the sorted list."""
        return self._len

    def __getitem__(self, index):
        """Lookup value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        return self._lists[pos][idx]

    def __delitem__(self, index):
        """Remove value at `index` from sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        self._delete(pos, idx)

    def __contains__(self, value):
        """Return true if `value` is an element of the sorted list."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_left(value)
            return idx < len(_lists[pos]) and _lists[pos][idx] == value
        return False

    def __iter__(self):
        """Return an iterator over the sorted list."""
        return (value for _list in self._lists for value in _list)

    def __reversed__(self):
        """Return a reverse iterator over the sorted list."""
        return (value for _list in reversed(self._lists) for value in reversed(_list))

    def __repr__(self):
        """Return string representation of sorted list."""
        return 'SortedList({0})'.format(list(self))
# endregion

from typing import Sequence
class BitOrSegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """
    def __init__(self, init_val: Sequence[int]):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.num: int = 1 << (n - 1).bit_length()
        # 配列の値を葉にセット
        self.tree: Sequence[int] = ([0] * self.num) + init_val + ([0] * (self.num - n))
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.tree[i << 1] | self.tree[(i << 1) + 1]
 
    def update(self, k: int, x: int):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            x |= self.tree[k ^ 1]
            self.tree[k >> 1] = x
            k >>= 1
 
    def query(self, l: int, r: int):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        res = 0 #self.ide_ele
 
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res |= self.tree[l]
                l += 1
            if r & 1:
                res |= self.tree[r - 1]
            l >>= 1
            r >>= 1
        return res

class BoolOrSegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """
    def __init__(self, init_val: Sequence[bool]):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.num: int = 1 << (n - 1).bit_length()
        # 配列の値を葉にセット
        self.tree: Sequence[bool] = ([False] * self.num) + init_val + ([False] * (self.num - n))
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = (self.tree[i << 1] or self.tree[(i << 1) + 1])
 
    def update(self, k: int, x: int):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        if self.tree[k] == x:
            return

        self.tree[k] = x
        while k > 1:
            x = (x or self.tree[k ^ 1])
            self.tree[k >> 1] = x
            k >>= 1
 
    def query(self, l: int, r: int):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                if self.tree[l]:
                    return True
                l += 1
            if r & 1:
                if self.tree[r - 1]:
                    return True
            l >>= 1
            r >>= 1
        return False

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 0
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
传统线段树的思路：
合并两个线段，如果合并后线段非「非单调减」，则用False表示该线段
然后判断条件 1 当前线段必须「非单调减」 2 去掉两头中间包含的字符需要跟原串该字符数量相等

SortedList思路：
记录并维护S串中变化点的位置 每次查询时最多在线段中查找26次，确认上面的条件是否满足

线段树 位运算优化思路：
bool_or_segtree 记录查询线段内是否有下降 违反条件1
bit_or_segtree 记录查询线段内的所有字符 检查条件2 线段外不应该有同样的字符

mask_range
"""

class Solution:
    def __init__(self, n: int, s: str) -> None:
        self.n = n
        s = self.s = [26] + [x - 97 for x in s.encode()] + [26]
        self._down_seg = BoolOrSegTree([s[i] > s[i + 1] for i in range(n + 1)])
        self._mask_seg = BitOrSegTree([1 << x for x in s])
    
    def update(self, i: int, ch: str) -> None:
        s, c = self.s, ord(ch) - 97
        if s[i] == c:
            return
        
        s[i] = c
        
        self._mask_seg.update(i, 1 << c)
        self._down_seg.update(i - 1, s[i - 1] > s[i])
        self._down_seg.update(i, s[i] > s[i + 1])
    
    @staticmethod
    def _range_mask(l: int, r: int) -> int:
        # return a mask, mask[i] is 1 if and only if i in range(l, r)
        return (-(1 << l)) & ((1 << r) - 1)
        
    def query(self, l: int, r: int) -> bool:
        s = self.s
        if self._down_seg.query(l, r):
            return False
        
        # mask = (-(1 << (1 + s[l]))) & ((1 << s[r]) - 1)
        mask = self._range_mask(s[l] + 1, s[r])
        
        left_mask = self._mask_seg.query(1, l)
        if left_mask & mask:
            return False
        
        right_mask = self._mask_seg.query(r + 1, self.n + 1)
        if right_mask & mask:
            return False
        
        return True
        

def solve():
    n = read_int()
    s = input()
    Sol = Solution(n, s)

    for _ in range(read_int()):
        t, a, b = input().split()
        if t == '1':
            Sol.update(int(a), b)
        else:
            res = Sol.query(int(a), int(b))
            print('Yes' if res else 'No')


def solve_with_SL():
    n = read_int()
    S = [x - 97 for x in input().encode()] + [26]
    sl = SL([i for i in range(n + 1) if S[i] != S[i - 1]])
    cnt = [0] * 27
    for x in S: cnt[x] += 1
    
    # print(sl)
    # print(cnt)
    
    for _ in range(read_int()):
        ops = input().split()
        if int(ops[0]) == 1:    # update
            i, x = int(ops[1]) - 1, ord(ops[2]) - 97
            
            if S[i - 1] != S[i]:
                if S[i - 1] == x:
                    sl.remove(i)
            else:
                if S[i - 1] != x:
                    sl.add(i)
            
            if S[i] != S[i + 1]:
                if S[i + 1] == x:
                    sl.remove(i + 1)
            else:
                if S[i + 1] != x:
                    sl.add(i + 1)

            cnt[S[i]] -= 1
            cnt[x] += 1

            S[i] = x
            
        else:   # query
            l, r = map(int, ops[1:])
            l -= 1
            m = sl[sl.bisect_right(l)]
            flag = True
            
            while flag and m < r:
                if S[l] > S[m]:
                    flag = False
                elif any(cnt[x] for x in range(S[l] + 1, S[m])):
                    flag = False
                else:
                    l = m
                    m = sl[sl.bisect_right(l)]
                    if m < r and m - l < cnt[S[l]]:
                        flag = False
            print(['No', 'Yes'][flag])


def solve_with_segmenttree():
    n = read_int()
    S = input()
    
    e = [0] * 26
    
    def merge(a, b):
        if a is False or b is False:
            return False
        res = [0] * 26
        flag = False
        for i, (x, y) in enumerate(zip(a, b)):
            if x and flag:
                return False
            res[i] = x + y

            if y: flag = True
        
        return res
    
    # A, B = list(range(4)[::-1]), list(range(4))
    # print(merge(A, B))
    # print(merge(B, A))
    
    
    st = SegmentTree(n, merge, e)
    
    def gen(ch: str):
        res = [0] * 26
        res[ord(ch) - 97] += 1
        return res
    
    st.build([gen(ch) for ch in S])
    
    cur = [0] * 26
    for ch in S:
        x = ord(ch) - 97
        cur[x] += 1
        
    # print(st.data)

    for _ in range(read_int()):
        ops = input().split()
        if int(ops[0]) == 1:    # update
            i, x = int(ops[1]) - 1, ord(ops[2]) - 97
            data = st.get(i)
            if data[x] == 1:
                continue

            data = data.copy()
            y = data.index(1)
            data[y] = 0
            data[x] = 1
            st.set(i, data)
            
            cur[y] -= 1
            cur[x] += 1
            
        else:   # query
            l, r = map(int, ops[1:])
            l -= 1
            data = st.prod(l, r)
            if data is False:
                print('No')
                continue
            start = next(i for i in range(26) if data[i]) + 1
            end = next(i for i in range(26)[::-1] if data[i])
            # print(start, end)

            # print(data, l, r)
            # print(cur)

            
            print(['No', 'Yes'][all(data[i] == cur[i] for i in range(start + 1, end))])
    
    # print(st.data)
            

T = 1#read_int()
for t in range(T):
    solve()