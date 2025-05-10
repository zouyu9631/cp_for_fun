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

# region Convolution

_fft_mod = 998244353
_fft_imag = 911660635
_fft_iimag = 86583718
_fft_rate2 = (911660635, 509520358, 369330050, 332049552, 983190778, 123842337, 238493703, 975955924, 603855026, 856644456, 131300601,
              842657263, 730768835, 942482514, 806263778, 151565301, 510815449, 503497456, 743006876, 741047443, 56250497, 867605899)
_fft_irate2 = (86583718, 372528824, 373294451, 645684063, 112220581, 692852209, 155456985, 797128860, 90816748, 860285882, 927414960,
               354738543, 109331171, 293255632, 535113200, 308540755, 121186627, 608385704, 438932459, 359477183, 824071951, 103369235)
_fft_rate3 = (372528824, 337190230, 454590761, 816400692, 578227951, 180142363, 83780245, 6597683, 70046822, 623238099,
              183021267, 402682409, 631680428, 344509872, 689220186, 365017329, 774342554, 729444058, 102986190, 128751033, 395565204)
_fft_irate3 = (509520358, 929031873, 170256584, 839780419, 282974284, 395914482, 444904435, 72135471, 638914820, 66769500,
               771127074, 985925487, 262319669, 262341272, 625870173, 768022760, 859816005, 914661783, 430819711, 272774365, 530924681)
 
 
def _butterfly(a):
    n = len(a)
    h = (n - 1).bit_length()
    len_ = 0
    while len_ < h:
        if h - len_ == 1:
            p = 1 << (h - len_ - 1)
            rot = 1
            for s in range(1 << len_):
                offset = s << (h - len_)
                for i in range(p):
                    l = a[i + offset]
                    r = a[i + offset + p] * rot % _fft_mod
                    a[i + offset] = (l + r) % _fft_mod
                    a[i + offset + p] = (l - r) % _fft_mod
                if s + 1 != (1 << len_):
                    rot *= _fft_rate2[(~s & -~s).bit_length() - 1]
                    rot %= _fft_mod
            len_ += 1
        else:
            p = 1 << (h - len_ - 2)
            rot = 1
            for s in range(1 << len_):
                rot2 = rot * rot % _fft_mod
                rot3 = rot2 * rot % _fft_mod
                offset = s << (h - len_)
                for i in range(p):
                    a0 = a[i + offset]
                    a1 = a[i + offset + p] * rot
                    a2 = a[i + offset + p * 2] * rot2
                    a3 = a[i + offset + p * 3] * rot3
                    a1na3imag = (a1 - a3) % _fft_mod * _fft_imag
                    a[i + offset] = (a0 + a2 + a1 + a3) % _fft_mod
                    a[i + offset + p] = (a0 + a2 - a1 - a3) % _fft_mod
                    a[i + offset + p * 2] = (a0 - a2 + a1na3imag) % _fft_mod
                    a[i + offset + p * 3] = (a0 - a2 - a1na3imag) % _fft_mod
                if s + 1 != (1 << len_):
                    rot *= _fft_rate3[(~s & -~s).bit_length() - 1]
                    rot %= _fft_mod
            len_ += 2
 
 
def _butterfly_inv(a):
    n = len(a)
    h = (n - 1).bit_length()
    len_ = h
    while len_:
        if len_ == 1:
            p = 1 << (h - len_)
            irot = 1
            for s in range(1 << (len_ - 1)):
                offset = s << (h - len_ + 1)
                for i in range(p):
                    l = a[i + offset]
                    r = a[i + offset + p]
                    a[i + offset] = (l + r) % _fft_mod
                    a[i + offset + p] = (l - r) * irot % _fft_mod
                if s + 1 != (1 << (len_ - 1)):
                    irot *= _fft_irate2[(~s & -~s).bit_length() - 1]
                    irot %= _fft_mod
            len_ -= 1
        else:
            p = 1 << (h - len_)
            irot = 1
            for s in range(1 << (len_ - 2)):
                irot2 = irot * irot % _fft_mod
                irot3 = irot2 * irot % _fft_mod
                offset = s << (h - len_ + 2)
                for i in range(p):
                    a0 = a[i + offset]
                    a1 = a[i + offset + p]
                    a2 = a[i + offset + p * 2]
                    a3 = a[i + offset + p * 3]
                    a2na3iimag = (a2 - a3) * _fft_iimag % _fft_mod
                    a[i + offset] = (a0 + a1 + a2 + a3) % _fft_mod
                    a[i + offset + p] = (a0 - a1 +
                                         a2na3iimag) * irot % _fft_mod
                    a[i + offset + p * 2] = (a0 + a1 -
                                             a2 - a3) * irot2 % _fft_mod
                    a[i + offset + p * 3] = (a0 - a1 -
                                             a2na3iimag) * irot3 % _fft_mod
                if s + 1 != (1 << (len_ - 1)):
                    irot *= _fft_irate3[(~s & -~s).bit_length() - 1]
                    irot %= _fft_mod
            len_ -= 2
 
 
def _convolution_naive(a, b):
    n = len(a)
    m = len(b)
    ans = [0] * (n + m - 1)
    if n < m:
        for j in range(m):
            for i in range(n):
                ans[i + j] = (ans[i + j] + a[i] * b[j]) % _fft_mod
    else:
        for i in range(n):
            for j in range(m):
                ans[i + j] = (ans[i + j] + a[i] * b[j]) % _fft_mod
    return ans
 
 
def _convolution_fft(a, b):
    a = a.copy()
    b = b.copy()
    n = len(a)
    m = len(b)
    z = 1 << (n + m - 2).bit_length()
    a += [0] * (z - n)
    _butterfly(a)
    b += [0] * (z - m)
    _butterfly(b)
    for i in range(z):
        a[i] = a[i] * b[i] % _fft_mod
    _butterfly_inv(a)
    a = a[:n + m - 1]
    iz = pow(z, _fft_mod - 2, _fft_mod)
    for i in range(n + m - 1):
        a[i] = a[i] * iz % _fft_mod
    return a
 
 
def _convolution_square(a):
    a = a.copy()
    n = len(a)
    z = 1 << (2 * n - 2).bit_length()
    a += [0] * (z - n)
    _butterfly(a)
    for i in range(z):
        a[i] = a[i] * a[i] % _fft_mod
    _butterfly_inv(a)
    a = a[:2 * n - 1]
    iz = pow(z, _fft_mod - 2, _fft_mod)
    for i in range(2 * n - 1):
        a[i] = a[i] * iz % _fft_mod
    return a
 
 
def convolution(a, b):
    """It calculates (+, x) convolution in mod 998244353. 
    Given two arrays a[0], a[1], ..., a[n - 1] and b[0], b[1], ..., b[m - 1], 
    it calculates the array c of length n + m - 1, defined by
 
    >   c[i] = sum(a[j] * b[i - j] for j in range(i + 1)) % 998244353.
 
    It returns an empty list if at least one of a and b are empty.
 
    Constraints
    -----------
 
    >   len(a) + len(b) <= 8388609
 
    Complexity
    ----------
 
    >   O(n log n), where n = len(a) + len(b).
    """
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return []
    if min(n, m) <= 0:
        return _convolution_naive(a, b)
    if a is b:
        return _convolution_square(a)
    return _convolution_fft(a, b)

# endregion


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
卷积例题
a | b = a + b - (a & b)
单位运算 & 相当于 *

参考 https://atcoder.jp/contests/abc291/submissions/39243280
"""

M = 5

def solve():
    n = read_int()
    A = read_int_list()
    B = read_int_list()
    B.reverse()

    res = [0] * n
    for bi in range(M):
        X = [(a >> bi) & 1 for i, a in enumerate(A)]
        Y = [(a >> bi) & 1 for i, a in enumerate(B)]
        cnt = X.count(1) + Y.count(1)

        conv = convolution(X, Y)
        for i in range(n, n + n - 1):
            conv[i - n] += conv[i]
        

        v = 1 << bi
        for i in range(n):
            res[i] += (cnt - conv[i]) * v

    print(max(res))
    

T = 1#read_int()
for t in range(T):
    solve()