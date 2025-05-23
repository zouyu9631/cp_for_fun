import sys, os
sys.setrecursionlimit(1 << 25)
read = sys.stdin.buffer.readline
MOD  = 998244353

# ---------- fast factorial / inv ----------
N_MAX = 250_000 + 5
fact  = [1] * (N_MAX)
invf  = [1] * (N_MAX)
for i in range(1, N_MAX):
    fact[i] = fact[i-1] * i % MOD
invf[-1] = pow(fact[-1], MOD-2, MOD)
for i in range(N_MAX-1, 0, -1):
    invf[i-1] = invf[i] * i % MOD
inv_int = [0]*(N_MAX)
inv_int[1] = 1
for i in range(2, N_MAX):
    inv_int[i] = MOD - MOD//i * inv_int[MOD%i] % MOD   # 1/i

# ---------- Hilbert order ----------
def hilbert(x:int, y:int, pow2:int=1<<18, rot:int=0)->int:
    if pow2 == 1:
        return 0
    h   = (x < (pow2>>1)) ^ (y < (pow2>>1)) ^ rot
    seg = ((y >= (pow2>>1))<<1 | (x >= (pow2>>1))) ^ rot
    nx, ny = (x, y) if h else (y, x)
    return (seg * (pow2*pow2>>2) +
            hilbert(nx & (pow2//2-1), ny & (pow2//2-1), pow2>>1, rot^h))

# ---------- main ----------
def main() -> None:
    N, Q = map(int, read().split())
    A = [0]+list(map(int, read().split()))           # 1-indexed

    queries = []
    for qi in range(Q):
        L, R, X = map(int, read().split())
        queries.append((hilbert(L, R), L, R, X, qi))
    queries.sort()

    # —— value-block information ——
    VSZ   = 800                         # √N 级
    VBLK  = (N+VSZ)//VSZ + 1
    cnt       = [0]*(N+1)
    cnt_blk   = [0]*VBLK
    prod_blk  = [1]*VBLK

    def upd(v:int, delta:int)->None:
        b   = v//VSZ
        c0  = cnt[v]
        c1  = c0 + delta
        cnt[v] = c1
        cnt_blk[b] += delta
        prod_blk[b] = prod_blk[b]*fact[c0]%MOD*invf[c1]%MOD

    curL, curR = 1, 0
    tot  = 0            # 当前区间总元素个数
    ans  = 1            # 当前 multinomial 值
    res  = [0]*Q

    for _, L, R, X, qi in queries:
        # ---- expand / shrink 到目标区间 ----
        while curL > L:
            curL -= 1
            v = A[curL]
            ans = ans * (tot+1) % MOD * inv_int[cnt[v]+1] % MOD
            upd(v, 1);  tot += 1
        while curR < R:
            curR += 1
            v = A[curR]
            ans = ans * (tot+1) % MOD * inv_int[cnt[v]+1] % MOD
            upd(v, 1);  tot += 1
        while curL < L:
            v = A[curL]
            ans = ans * cnt[v] % MOD * inv_int[tot] % MOD
            upd(v, -1); tot -= 1
            curL += 1
        while curR > R:
            v = A[curR]
            ans = ans * cnt[v] % MOD * inv_int[tot] % MOD
            upd(v, -1); tot -= 1
            curR -= 1

        # ---- 计算 <X 的部分 ----
        if X <= 1:
            res[qi] = 1               # 删除后为空
            continue
        lim   = X-1
        b_lim = lim//VSZ
        s  = 0
        pr = 1
        for b in range(b_lim):              # 整块
            s  += cnt_blk[b]
            pr  = pr * prod_blk[b] % MOD
        start = b_lim*VSZ
        for v in range(start, lim+1):       # 残块
            c = cnt[v]
            s += c
            pr = pr*invf[c]%MOD

        res[qi] = fact[s]*pr % MOD

    sys.stdout.write('\n'.join(map(str, res)))

if __name__ == "__main__":
    main()