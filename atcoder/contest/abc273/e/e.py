from types import GeneratorType
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


def input(): return sys.stdin.readline().rstrip('\r\n')


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


def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
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


def solve_old():
    q = read_int()
    query = [['S', 0]]
    res = [-1] + [-1] * q
    stk = [-1]

    save_pos = defaultdict(int)
    edges = defaultdict(list)

    for i in range(1, q + 1):
        tmp = input()
        t = tmp[0]
        if t == 'D':
            query.append([t])
        else:
            x = int(tmp.split()[1])
            query.append([t, x])
            if t == 'S':
                save_pos[x] = i
            elif t == 'L':
                edges[save_pos[x]].append(i)

    seen = [False] + [False] * q

    @bootstrap
    def dfs(i: int, flag):
        if i <= q and not seen[i]:
            ops = query[i]

            if len(ops) == 1:  # 'D'
                seen[i] = True
                tmp = None
                if stk[-1] != -1:
                    tmp = stk.pop()
                res[i] = stk[-1]
                yield dfs(i + 1, False)
                if tmp is not None:
                    stk.append(tmp)
            else:
                t, x = ops
                if t == 'A':
                    seen[i] = True
                    stk.append(x)
                    res[i] = x
                    yield dfs(i + 1, False)
                    stk.pop()
                elif t == 'S':
                    seen[i] = True
                    res[i] = stk[-1]
                    for j in edges[i]:
                        yield dfs(j, True)
                    yield dfs(i + 1, False)
                elif flag:   # 'L'
                    seen[i] = True
                    res[i] = stk[-1]
                    yield dfs(i + 1, False)
                    
            
        yield None

    dfs(0, False)
    
    print(*res[1:])

def solve():
    q = read_int()
    notebook = dict()
    tree = [-1] * (q + 1)
    fatr = [0] * (q + 1)
    cur, nxt, res = 0, 1, [0] * q
    
    for i in range(q):
        tmp = input()
        t = tmp[0]
        if t == 'D':
            cur = fatr[cur]
        else:
            x = int(tmp.split()[1])
            if t == 'S':
                notebook[x] = cur
            elif t == 'L':
                cur = notebook.get(x, 0)
            else:   # 'ADD'
                tree[nxt] = x
                fatr[nxt] = cur
                cur = nxt
                nxt += 1
        
        res[i] = tree[cur]
    
    print(*res)

T = 1  # read_int()
for t in range(T):
    solve()
