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
from bisect import bisect_left, bisect_right
class SortedSet:
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def __init__(self, A=[]):
        "Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
        A = list(A)
        if not all(A[i] < A[i + 1] for i in range(len(A) - 1)):
            A = sorted(set(A))
        self.__build(A)
        return

    def __build(self, A=None):
        if A is None:
            A = list(self)

        self.N = N = len(A)
        K = 1
        while self.BUCKET_RATIO * K * K < N:
            K += 1

        self.list = [A[N * i // K: N * (i + 1) // K] for i in range(K)]

    def __iter__(self):
        for A in self.list:
            for a in A:
                yield a

    def __reversed__(self):
        for A in reversed(self.list):
            for a in reversed(A):
                yield a

    def __len__(self):
        return self.N

    def __bool__(self):
        return bool(self.N)

    def __str__(self):
        string = str(list(self))
        return "{" + string[1:-1] + "}"

    def __repr__(self):
        return "Sorted Set: " + str(self)

    def __find_bucket(self, x):
        "Find the bucket which should contain x. self must not be empty."
        for A in self.list:
            if x <= A[-1]:
                return A
        else:
            return A

    def __contains__(self, x):
        if self.N == 0:
            return False

        A = self.__find_bucket(x)
        i = bisect_left(A, x)
        return i != len(A) and A[i] == x

    def add(self, x):
        "Add an element and return True if added False if existed. / O(√N)"
        if self.N == 0:
            self.list = [[x]]
            self.N += 1
            return True

        A = self.__find_bucket(x)
        i = bisect_left(A, x)

        if i != len(A) and A[i] == x:
            return False

        A.insert(i, x)
        self.N += 1

        if len(A) > len(self.list) * self.REBUILD_RATIO:
            self.__build()
        return True

    def discard(self, x):
        "Remove an element and return True if removed False if not exist. / O(√N)"
        if self.N == 0:
            return False

        A = self.__find_bucket(x)
        i = bisect_left(A, x)

        if not (i != len(A) and A[i] == x):
            return False

        A.pop(i)
        self.N -= 1

        if len(A) == 0:
            self.__build()

        return True

    def remove(self, x):
        if not self.discard(x):
            raise KeyError(x)

    # === get, pop

    def __getitem__(self, index):
        "Return the x-th element, or IndexError if it doesn't exist."
        if index < 0:
            index += self.N
            if index < 0:
                raise IndexError("index out of range")

        for A in self.list:
            if index < len(A):
                return A[index]
            index -= len(A)
        else:
            raise IndexError("index out of range")

    def get_min(self):
        if self.N == 0:
            raise ValueError("This is empty set.")

        return self.list[0][0]

    def pop_min(self):
        if self.N == 0:
            raise ValueError("This is empty set.")

        A = self.list[0]
        value = A.pop(0)
        self.N -= 1

        if len(A) == 0:
            self.__build()

        return value

    def get_max(self):
        if self.N == 0:
            return ValueError("This is empty set.")

        return self.list[-1][-1]

    def pop_max(self):
        if self.N == 0:
            raise ValueError("This is empty set.")

        A = self.list[-1]
        value = A.pop(-1)
        self.N -= 1

        if len(A) == 0:
            self.__build()

        return value

    # === previous, next

    def previous(self, value, mode=False):
        """
        get the maxmium x in S which x < value
        mode: True x <= value
        """

        if self.N == 0:
            return None

        if mode:
            for A in reversed(self.list):
                if A[0] <= value:
                    return A[bisect_right(A, value) - 1]
        else:
            for A in reversed(self.list):
                if A[0] < value:
                    return A[bisect_left(A, value) - 1]

    def next(self, value, mode=False):
        """
        get the minimum x in S which value < x
        mode: True value <= x
        """

        if self.N == 0:
            return None

        if mode:
            for A in self.list:
                if A[-1] >= value:
                    return A[bisect_left(A, value)]
        else:
            for A in self.list:
                if A[-1] > value:
                    return A[bisect_right(A, value)]

    # === count
    def less_count(self, value, equal=False):
        """
        return the number of x in S which x < value
        equal=True: a <= value
        """

        count = 0
        if equal:
            for A in self.list:
                if A[-1] > value:
                    return count + bisect_right(A, value)
                count += len(A)
        else:
            for A in self.list:
                if A[-1] >= value:
                    return count + bisect_left(A, value)
                count += len(A)
        return count

    def more_count(self, value, equal=False):
        """
        return the number of x in S which x > value
        equal=True: a <>= value
        """

        return self.N - self.less_count(value, not equal)

    # ===
    def is_upper_bound(self, x, equal=True):
        if self.N:
            a = self.list[-1][-1]
            return (a < x) or (bool(equal) and a == x)
        else:
            return True

    def is_lower_bound(self, x, equal=True):
        if self.N:
            a = self.list[0][0]
            return (x < a) or (bool(equal) and a == x)
        else:
            return True

    # === index
    def index(self, value):
        index = 0
        for A in self.list:
            if A[-1] > value:
                i = bisect_left(A, value)
                if A[i] == value:
                    return index + i
                else:
                    raise ValueError("{} is not in Set".format(value))
            index += len(A)
        raise ValueError("{} is not in Set".format(value))

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
拆开绝对值公式后 分析可得 1 就是插入两个点 2 就是找范围里距离最近的已有的点的距离
SortedSet
"""

def solve():
    q, a, b = read_int_tuple()
    ss = SortedSet([-inf, a - b, a + b, inf])
    for _ in range(q):
        t, a, b = read_int_tuple()
        if t == 1:
            ss.add(a - b)
            ss.add(a + b)
        else:
            left = ss.next(a, True)
            right = ss.next(b, True)
            if left < right:
                print(0)
            else:
                left -= 1
                print(min(a - ss.previous(a), right - b))
            

T = 1#read_int()
for t in range(T):
    solve()