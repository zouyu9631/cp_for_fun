import cProfile
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


def input(): return sys.stdin.readline().rstrip('\r\n')


def read_int_list():
    return list(map(int, input().split()))


def read_int_tuple():
    return tuple(map(int, input().split()))


def read_int():
    return int(input())


# endregion

class LazySegmentTree():
    __slots__ = ['n', 'log', 'size', 'd', 'lz', 'e', 'op', 'mapping', 'composition', 'identity']
    def _update(self, k):self.d[k]=self.op(self.d[2 * k], self.d[2 * k + 1])
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
        for i in range(self.size-1,0,-1):self._update(i)
    def set(self,p,x):
        assert 0<=p and p<self.n
        p+=self.size
        for i in range(self.log,0,-1):self._push(p >> i)
        self.d[p]=x
        for i in range(1,self.log+1):self._update(p >> i)
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
        for i in range(1,self.log+1):self._update(p >> i)
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
            if (((l>>i)<<i)!=l):self._update(l >> i)
            if (((r>>i)<<i)!=r):self._update((r - 1) >> i)
    def max_right(self,l,g):
        assert 0<=l and l<=self.n
        assert g(self.e)
        if l==self.n:return self.n
        l+=self.size
        for i in range(self.log,0,-1):self._push(l >> i)
        sm=self.e
        while(1):
            while(i%2==0):l>>=1
            if not(g(self.op(sm,self.d[l]))):
                while(l<self.size):
                    self._push(l)
                    l=(2*l)
                    if (g(self.op(sm,self.d[l]))):
                        sm=self.op(sm,self.d[l])
                        l+=1
                return l-self.size
            sm=self.op(sm,self.d[l])
            l+=1
            if (l&-l)==l:break
        return self.n
    def min_left(self,r,g):
        assert (0<=r and r<=self.n)
        assert g(self.e)
        if r==0:return 0
        r+=self.size
        for i in range(self.log,0,-1):self._push((r - 1) >> i)
        sm=self.e
        while(1):
            r-=1
            while(r>1 and (r%2)):r>>=1
            if not(g(self.op(self.d[r],sm))):
                while(r<self.size):
                    self._push(r)
                    r=(2*r+1)
                    if g(self.op(self.d[r],sm)):
                        sm=self.op(self.d[r],sm)
                        r-=1
                return r+1-self.size
            sm=self.op(self.d[r],sm)
            if (r&-r)==r:break
        return 0


# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 4
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion


n, q = read_int_tuple()
A = read_int_list()

L = 17
M = (1 << L) - 1
IJ = [[1, 0], [2, 0], [2, 1]]
# FL = [0, 2, 4]
FM = 3  # 0b11

E = [0, 0, 0]   # (1-0-rev_0cnt, 2-0-rev_1cnt, 2-1-rev_2cnt)

def OP(x, y):
    x1 = x[1] & M
    x2 = x[2] & M
    y0 = y[0] & M
    y1 = y[1] & M
    return [x[0] + y[0] + (x1 * y0 << L), x[1] + y[1] + (x2 * y0 << L), x[2] + y[2] + (x2 * y1 << L)]

def MAP(fn, x):
    p0, p1, p2 = fn & 3, fn >> 2 & 3, fn >> 4
    x0, x1, x2 = x[0] & M, x[1] & M, x[2] & M
    r10, r20, r21 = x[0] >> L, x[1] >> L, x[2] >> L

    res = [0] * 3
    res[p0] += x0
    res[p1] += x1
    res[p2] += x2
    if p1 > p0:
        res[p1 + p0 - 1] += r10 << L
    elif p1 < p0:
        res[p1 + p0 - 1] += (x0 * x1 - r10) << L
    if p2 > p0:
        res[p2 + p0 - 1] += r20 << L
    elif p2 < p0:
        res[p2 + p0 - 1] += (x0 * x2 - r20) << L
    if p2 > p1:
        res[p2 + p1 - 1] += r21 << L
    elif p2 < p1:
        res[p2 + p1 - 1] += (x1 * x2 - r21) << L

    return res

ID = 36 # 0b100100

def COMP(fn, gn):
    p = (fn & 3, fn >> 2 & 3, fn >> 4)
    q = (gn & 3, gn >> 2 & 3, gn >> 4)
    return p[q[0]] | p[q[1]] << 2 | p[q[2]] << 4


lst = LazySegmentTree([[1 if i == x else 0 for i in range(3)] for x in A], OP, E, MAP, COMP, ID)
for _ in range(q):
    ops = read_int_tuple()
    if ops[0] == 1:
        _, l, r = ops
        res = lst.prod(l - 1, r)
        print(sum(res) >> L)
    else:
        # _, l, r, *fx = ops
        # lst.apply(l - 1, r, sum(fx[i] << FL[i] for i in range(3)))
        _, l, r, z, o, t = ops
        lst.apply(l - 1, r, z | o << 2 | t << 4)