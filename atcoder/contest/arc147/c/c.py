import sys


# region IO
import sys
input = sys.stdin.readline



# endregion

# region local test
# if 'AW' in os.environ.get('COMPUTERNAME', ''):
#     test_no = 1
#     f = open(os.path.dirname(__file__) + f'\\in{test_no}.txt', 'r')

#     def input():
#         return f.readline().rstrip("\r\n")
# endregion



# n = read_int()
# L, R = [], []
# for _ in range(n):
#     l, r = read_int_tuple()
#     heappush(L, -l)
#     heappush(R, r)

# cur, res = n - 1, 0
# while -L[0] > R[0]:
#     l, r = -heappop(L), heappop(R)
#     res += cur * (l - r)
#     cur -= 2
# print(res)

# n = read_int()
# L, R = [], []
# for _ in range(n):
#     l, r = read_int_tuple()
#     L.append(l)
#     R.append(r)
# L.sort(reverse=True)
# R.sort()

# res = 0
# for i in range(n):
#     if L[i] <= R[i]:
#         break
#     res += (n - 2 * i - 1) * (L[i] - R[i])
# print(res)
# print(sum(max(0, l - r) * (n - 2 * i - 1) for i, (l, r) in enumerate(zip(L, R))))

# n = int(input())
# L, R = [], []
# for _ in range(n):
#     l, r = map(int, input().split())
#     L.append(l)
#     R.append(r)
# L.sort(reverse=True)
# R.sort()

# res, cur = 0, n - 1
# for l, r in zip(L, R):
#     if l <= r:
#         break
#     res += cur * (l - r)
#     cur -= 2
# print(res)

n = int(input())
rg = [tuple(map(int, input().split())) for _ in range(n)]

def dist(x):
    res = 0
    for l, r in rg:
        if r <= x:
            res += x - r
        elif x <= l:
            res += l - x
    return res

def tersect(left, right, func):
    # res = inf
    while left <= right:
        lmid = left + (right - left) // 3
        rmid = right - (right - left) // 3
        fl, fr = func(lmid), func(rmid)
        if fl < fr:
            res, right = fl, rmid - 1
        else:
            res, left = fr, lmid + 1
        # res = min(res, fl, fr)
    return right, res

x, rx = tersect(1, 10000000, dist)

pos = []
for l, r in rg:
    if r <= x:
        pos.append(r)
    elif x <= l:
        pos.append(l)
    else:
        pos.append(x)

pos.sort()

print(sum((n - 1 - i - i) * (pos[n - 1 - i] - pos[i]) for i in range(n // 2)))

# res = 0
# for i in range(n // 2):
#     res += (n - 1 - i - i) * (pos[n - 1 - i] - pos[i])
# print(res)

# res, l, r = 0, 0, n - 1
# while l < r:
#     res += (r - l) * (pos[r] - pos[l])
#     l += 1
#     r -= 1
# print(res)

# res = pre = 0
# for i, p in enumerate(pos):
#     res += i * p - pre
#     pre += p
# print(res)