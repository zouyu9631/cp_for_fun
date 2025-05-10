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

# region local test
if 'AW' in os.environ.get('COMPUTERNAME', ''):
    test_no = 1
    f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

    def input():
        return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60

"""
数论 离散对数 大步小步 BSGS minimum x: a ^ x = b (mod=MOD)

本题要解决 10 ^ x == 1 (mod k) 还有另外一个思路
"""

from math import gcd
def DiscreteLogarithm(a,b,MOD,off=0):
    """
    return minimum x s.t. a^x=b (mod=MOD)
    off: off以上最小值
    """
    if b >= MOD or b < 0:
        return -1
    if a == 0:
        if MOD == 1:
            return 0
        if b == 1:
            return 0
        if b == 0:
            return 1
        return -1
    p = 3
    tmp =a-1
    cnt = 0
    primes = []
    counts = []
    ps = 0
    while tmp & 1:
        tmp >>= 1
        cnt += 1
    if cnt:
        primes.append(2)
        counts.append(cnt)
        ps += 1
    tmp += 1
    while tmp != 1:
        cnt = 0
        while tmp % p == 0:
            tmp //= p
            cnt += 1
        if cnt:
            primes.append(p)
            counts.append(cnt)
            ps += 1
        p += 2
        if tmp != 1 and p * p > a:
            primes.append(tmp)
            counts.append(1)
            ps += 1
            break
    tail = 0
    mp = MOD
    for i in range(ps):
        f = 0
        while mp % primes[i] == 0:
            mp //= primes[i]
            f += 1
        if tail < (f + counts[i] - 1) // counts[i]:
            tail = (f + counts[i] - 1) // counts[i]
    z = 1
    for i in range(tail):
        if z == b:
            if i>off:
                return i
        z =z*a%MOD
    if b % gcd(z,MOD):
        return -1
    p = 3
    u = mp
    tmp = mp - 1
    if tmp & 1:
        u >>= 1
        while tmp & 1:
            tmp >>= 1
    tmp += 1
    while tmp != 1:
        if tmp % p == 0:
            u //= p
            u *= p - 1
            while tmp % p == 0:
                tmp //= p
        p += 2
        if tmp != 1 and p * p > mp:
            u //= tmp
            u *= tmp - 1
            break
    p = 1
    loop = u
    while p * p <= u:
        if u % p == 0:
            if z * pow(a,p,MOD) % MOD == z:
                loop = p
                break
            ip = u // p
            if z * pow(a,ip,MOD) % MOD == z:
                loop = ip
        p += 1
    l, r = 0, loop+1
    sq = (loop+1) >> 1
    while r - l > 1:
        if sq * sq <= loop:
            l = sq
        else:
            r = sq
        sq = (l + r) >> 1
    if sq * sq < loop:
        sq += 1
    h = pow(pow(a,loop-1,MOD),sq,MOD)
    d = {}
    f = z
    for i in range(sq):
        d[f] = i
        f =f*a%MOD
    g = b
    for i in range(sq+1):
        if g in d:
            if i*sq+d[g]+tail>off:
                return i*sq+d[g]+tail
        g =g*h%MOD
    return -1

def solve():
    k = 9 * read_int()
    if k & 1 == 0: k >>= 1
    print(DiscreteLogarithm(10, 1, k))

def isPrimeMR(n):
    """大质数判断"""
    d = n - 1
    d = d // (d & -d)
    L = [2, 3, 5, 7, 11, 13, 17]
    for a in L:
        t = d
        y = pow(a, t, n)
        if y == 1: continue
        while y != n - 1:
            y = (y * y) % n
            if y == 1 or t == n - 1: return 0
            t <<= 1
    return 1
 
from math import gcd
def findFactorRho(n):
    """大数找约数"""
    m = 1 << n.bit_length() // 8
    for c in range(1, 99):
        f = lambda x: (x * x + c) % n
        y, r, q, g = 2, 1, 1, 1
        while g == 1:
            x = y
            for i in range(r):
                y = f(y)
            k = 0
            while k < r and g == 1:
                ys = y
                for i in range(min(m, r - k)):
                    y = f(y)
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += m
            r <<= 1
        if g == n:
            g = 1
            while g == 1:
                ys = f(ys)
                g = gcd(abs(x - ys), n)
        if g < n:
            if isPrimeMR(g):
                return g
            elif isPrimeMR(n // g):
                return n // g
            return findFactorRho(g)
 
def primeFactor(n):
    """分解质因数，每个质数的数量"""
    i = 2
    ret = {}
    rhoFlg = 0
    while i * i <= n:
        k = 0
        while n % i == 0:
            n //= i
            k += 1
        if k: ret[i] = k
        i += 1 + i % 2
        if i == 101 and n >= 2 ** 20:
            while n > 1:
                if isPrimeMR(n):
                    ret[n], n = 1, 1
                else:
                    rhoFlg = 1
                    j = findFactorRho(n)
                    k = 0
                    while n % j == 0:
                        n //= j
                        k += 1
                    ret[j] = k
 
    if n > 1: ret[n] = 1
    if rhoFlg: ret = {x: ret[x] for x in sorted(ret)}
    return ret
 
def divisors(n):
    """所有n的约数"""
    res = [1]
    prime = primeFactor(n)
    for p in prime:
        newres = []
        for d in res:
            for j in range(prime[p] + 1):
                newres.append(d * pow(p, j))
        res = newres
    res.sort()
    return res
 
def euler_phi(n):
    res = n
    for x in range(2, n + 1):
        if x * x > n:
            break
        if n % x == 0:
            res = res // x * (x - 1)
            while n % x == 0:
                n //= x
    if n != 1:
        res = res // n * (n - 1)
    return res
 
 
# for a ^ x == 1 mod k
def solve_alter():
    k = 9 * read_int()
    if k & 1 == 0: k >>= 1
    phi = euler_phi(k)
    # print(k, phi, divisors(phi))
    for p in divisors(phi):
        if pow(10, p, k) == 1:
            print(p)
            return
    print(-1)

T = read_int()
for t in range(T):
    solve()