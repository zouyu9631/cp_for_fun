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

class LazySegmentTree():
    __slots__ = ['n', 'log', 'size', 'd', 'lz', 'e', 'op', 'mapping', 'composition', 'identity']
    # def _update(self, k): self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])
    def _all_apply(self, k, f):
        self.d[k]=self.mapping(f,self.d[k])
        if (k<self.size):self.lz[k]=self.composition(f,self.lz[k])
    def _push(self, k):
        if self.lz[k] == self.identity: return
        self._all_apply(2 * k, self.lz[k])
        self._all_apply(2 * k + 1, self.lz[k])
        self.lz[k]=self.identity
    def __init__(self,V,OP,E,MAPPING,COMPOSITION,ID):

        self.n=len(V)
        self.log=(self.n-1).bit_length()
        self.size=1<<self.log
        self.d=[E for i in range(2*self.size)]
        self.lz=[ID for i in range(self.size)]
        self.e=E
        self.op=OP
        self.mapping=MAPPING
        self.composition=COMPOSITION
        self.identity=ID
        for i in range(self.n):self.d[self.size+i]=V[i]
        for k in range(self.size-1,0,-1): self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])
    def set(self,p,x):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self._push(p >> i)
        self.d[p]=x
        for i in range(1,self.log+1):
            k = p >> i
            self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])
    def get(self,p):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self._push(p >> i)
        return self.d[p]
    def prod(self,l,r):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return self.e
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self._push(l >> i)
            if (((r>>i)<<i)!=r):self._push(r >> i)
        sml,smr=self.e,self.e
        while(l<r):
            if l&1:
                sml=self.op(sml,self.d[l])
                l+=1
            if r&1:
                r-=1
                smr=self.op(self.d[r],smr)
            l>>=1
            r>>=1
        return self.op(sml,smr)
    def all_prod(self):return self.d[1]
    def apply_point(self,p,f):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self._push(p >> i)
        self.d[p]=self.mapping(f,self.d[p])
        for i in range(1,self.log+1):
            k = p >> i
            self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])
    def apply(self,l,r,f):
        assert 0<=l and l<=r and r<=self.n
        if l==r:return
        l+=self.size
        r+=self.size
        for i in range(self.log,0,-1):
            if (((l>>i)<<i)!=l):self._push(l >> i)
            if (((r>>i)<<i)!=r):self._push((r - 1) >> i)
        l2,r2=l,r
        while(l<r):
            if (l&1):
                self._all_apply(l, f)
                l+=1
            if (r&1):
                r-=1
                self._all_apply(r, f)
            l>>=1
            r>>=1
        l,r=l2,r2
        for i in range(1,self.log+1):
            if (((l>>i)<<i)!=l):
                k = l >> i
                self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])
            if (((r>>i)<<i)!=r):
                k = (r - 1) >> i
                self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])


# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion

MOD = 998244353  # 1000000007
inf = 1 << 60


def solve():
    n, q = read_int_tuple()
    
    T = [1] * n
    AT = [1] * n
    for i in range(1, n):
        T[i] = T[i - 1] * 10 % MOD
        AT[i] = (AT[i - 1] + T[i]) % MOD
    # T.reverse()
    # AT.reverse()
    
    B = 18
    M = (1 << B) - 1
    
    E = 0
    
    def OP(x, y):
        xv, xn = x >> B, x & M
        yv, yn = y >> B, y & M
        return ((xv * T[yn] + yv) % MOD) << B | (xn + yn)

    
    def MAPP(d, x):
        xn = x & M
        return (AT[xn - 1] * d % MOD) << B | xn
        
        
    def COMP(fn, gn):
        return fn if fn >= 0 else gn
        
    ID = -1
        
    
    lst = LazySegmentTree([1 << B | 1 for _ in range(n)], OP, E, MAPP, COMP, ID)
    # print(lst.all_prod() >> B)
    
    
    for _ in range(q):
        l, r, d = read_int_tuple()
        lst.apply(l - 1, r, d)
        print(lst.all_prod() >> B)
        

T = 1#read_int()
for t in range(T):
    solve()