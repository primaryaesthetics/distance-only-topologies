"""The fed cycle C_k with additions free: a comparison program, the pairwise game, and wall crossings.

C_k has source 0, arcs 0>i of weight b_i and cycle arcs i>i+1 (mod k) of weight c_i. With
P_i = c_1 + ... + c_{i-1}, W = c_1 + ... + c_k and a_j = b_j - P_j, vertex i is entered directly
(self-served) iff a_i < a_j for every j < i and a_i < a_j + W for every j > i.

Part 1 checks the program: a prefix-minimum scan of a (k - 1 comparisons, each b_r + c_r + ... + c_{i-1}
against b_i), then a binary search over the p prefix records for the first record r with
b_r < b_m + c_m + ... + c_k + c_1 + ... + c_{r-1}, m the last record (ceil(log2 p) comparisons).
It is run on random weights against brute-force distances and the worst comparison count is printed.

Part 2 solves the pairwise game exactly, both query sets up to k = 6 and the window set at k = 7:
queries a_i - a_j ? tW, strict answers, the program wins when every vertex's entry is implied. A set of answers is a digraph with edge (i, j, t) for a_i - a_j < tW;
it is consistent iff every cycle has total t >= 1, and a_i - a_j < tW is implied iff the shortest
i -> j path has length <= t. Query set L=1: positions of the unrolled sequence at distance < k (the
window comparisons, all legal path-register queries). L=2: distance < 2k, which contains every query
with t in {-1, 0, 1}.

Part 3 counts, for every wall a_i - a_j = tW (|t| <= 1), the cells whose interior it crosses.
"""
import math
import random
import sys
import time
from fractions import Fraction as F

# ---------------------------------------------------------------- part 1


def walk(c, j, i, k):
    s = 0
    t = j
    while t != i:
        s += c[t]
        t = (t + 1) % k
    return s


def program(b, c):
    k = len(b)
    n = 0
    records = [0]
    cur = 0
    for i in range(1, k):
        n += 1
        if b[i] < b[cur] + walk(c, cur, i, k):
            cur = i
            records.append(i)
    m = cur
    lo, hi = 0, len(records) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        r = records[mid]
        n += 1
        if b[r] < b[m] + walk(c, m, r, k):
            hi = mid
        else:
            lo = mid + 1
    served = set(records[lo:])
    entry = []
    for i in range(k):
        j = i
        while j not in served:
            j = (j - 1) % k
        entry.append(j)
    return entry, n


def sample(k, kind, rng):
    if kind == "int-ties":
        return [rng.randint(1, 4) for _ in range(k)], [rng.randint(1, 3) for _ in range(k)]
    if kind == "int-wide":
        return [rng.randint(1, 50) for _ in range(k)], [rng.randint(1, 20) for _ in range(k)]
    if kind == "float":
        return [F(rng.random()) for _ in range(k)], [F(rng.random()) for _ in range(k)]
    if kind == "mixed":
        def sc():
            return F(rng.random()) * F(10) ** rng.randint(-6, 6)
        return [sc() for _ in range(k)], [sc() for _ in range(k)]
    # "decreasing": every a_j is a prefix record, the worst case of the binary search
    c = [F(rng.random()) for _ in range(k)]
    P = [sum(c[:j]) for j in range(k)]
    W = sum(c)
    a = sorted((F(rng.random()) * 2 * W for _ in range(k)), reverse=True)
    base = max(0, -min(a[j] + P[j] for j in range(k))) + 1
    return [a[j] + P[j] + base for j in range(k)], c


def part1(N):
    rng = random.Random(20261011)
    print("# part 1: program on random weights (exact rational arithmetic)")
    print("# k | samples | wrong | worst comparisons | k-1+ceil(log2 k)")
    for k in range(3, 11):
        worst = bad = tot = 0
        for kind in ["int-ties", "int-wide", "float", "mixed", "decreasing"]:
            for _ in range(N):
                b, c = sample(k, kind, rng)
                entry, n = program(b, c)
                truth = [min(b[j] + walk(c, j, i, k) for j in range(k)) for i in range(k)]
                got = [b[entry[i]] + walk(c, entry[i], i, k) for i in range(k)]
                bad += got != truth
                worst = max(worst, n)
                tot += 1
        print(f"{k} | {tot} | {bad} | {worst} | {k - 1 + math.ceil(math.log2(k))}", flush=True)


# ---------------------------------------------------------------- part 2

INF = 10 ** 6


def init(k):
    d = [INF] * (k * k)
    for u in range(k):
        d[u * k + u] = 0
    return tuple(d)


def add(d, k, i, j, t):
    if d[j * k + i] < INF and d[j * k + i] + t < 1:
        return None
    n = list(d)
    for u in range(k):
        dui = d[u * k + i]
        if dui >= INF:
            continue
        for v in range(k):
            djv = d[j * k + v]
            if djv >= INF:
                continue
            if dui + t + djv < n[u * k + v]:
                n[u * k + v] = dui + t + djv
    return tuple(n)


def cells_of(k):
    def off(i, j):
        return 1 if j > i else 0
    out = []
    for mask in range(1, 1 << k):
        cert = []
        for i in range(k):
            j = i
            while not mask >> j & 1:
                j = (j - 1) % k
            cert += [(j, l, off(i, l) - off(i, j)) for l in range(k) if l != j]
        out.append(cert)
    return out


def queries_of(k, L):
    out = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            for t in range(-L, L + 1):
                if abs(i - j + t * k) >= L * k:
                    continue
                key = min((i, j, t), (j, i, -t))
                if key not in out:
                    out.append(key)
    return out


def with_cert(d, k, cert):
    for (i, j, t) in cert:
        if d[i * k + j] <= t:
            continue
        d = add(d, k, i, j, t)
        if d is None:
            return None
    return d


def game(k, L):
    cells = cells_of(k)
    qs = queries_of(k, L)

    def consistent(d, idx):
        return [c for c in idx if with_cert(d, k, cells[c]) is not None]

    def rot(d):
        # relabel vertex u as u - 1; a_0 = a'_{k-1} + W
        n = [INF] * (k * k)
        for u in range(k):
            for v in range(k):
                x = d[u * k + v]
                if u == v:
                    n[((u - 1) % k) * k + (u - 1) % k] = 0
                    continue
                if x >= INF:
                    continue
                n[((u - 1) % k) * k + (v - 1) % k] = x - (u == 0) + (v == 0)
        return tuple(n)

    def rotcell(c, r):
        m = c + 1
        return (((m >> r) | (m << (k - r))) & ((1 << k) - 1)) - 1

    memo = {}

    def win(d0, idx, depth):
        best, br, cur = d0, 0, d0
        for r in range(1, k):
            cur = rot(cur)
            if cur < best:
                best, br = cur, r
        d = best
        if br:
            idx = [rotcell(c, br) for c in idx]
        lo, hi = memo.get(d, (-1, INF))
        if depth >= hi:
            return True
        if depth <= lo:
            return False
        cs = consistent(d, idx)
        if len(cs) == 1:
            res = True
        elif depth == 0 or len(cs) > 1 << depth:
            res = False
        else:
            opts = []
            for (i, j, t) in qs:
                if d[i * k + j] <= t or d[j * k + i] <= -t:
                    continue
                d1 = add(d, k, i, j, t)
                d2 = add(d, k, j, i, -t)
                if d1 is None or d2 is None:
                    continue
                c1 = consistent(d1, cs)
                c2 = consistent(d2, cs)
                if max(len(c1), len(c2)) > 1 << (depth - 1):
                    continue
                opts.append((max(len(c1), len(c2)), d1, d2, c1, c2))
            opts.sort(key=lambda o: o[0])
            res = any(win(d1, c1, depth - 1) and win(d2, c2, depth - 1)
                      for _, d1, d2, c1, c2 in opts)
        if res:
            hi = min(hi, depth)
        else:
            lo = max(lo, depth)
        memo[d] = (lo, hi)
        return res

    allc = list(range(len(cells)))
    for depth in range(k, 4 * k):
        if win(init(k), allc, depth):
            return depth, len(qs), len(memo)


def part2(kmax):
    print("# part 2: pairwise game, exact value under strict answers")
    print("# k | L | queries | value | states | k-1+ceil(log2 k) | seconds")
    rows = [(k, L) for k in range(2, kmax + 1) for L in (1, 2)] + [(kmax + 1, 1)]
    for k, L in rows:
        t0 = time.time()
        v, nq, ns = game(k, L)
        print(f"{k} | {L} | {nq} | {v} | {ns} | {k - 1 + math.ceil(math.log2(k))} | "
              f"{time.time() - t0:.1f}", flush=True)


# ---------------------------------------------------------------- part 3


def part3(kmax):
    print("# part 3: cells crossed by each wall a_i - a_j = tW, |t| <= 1 (minimum over walls)")
    print("# k | walls | min crossings | 2^(k-2)-1")
    for k in range(3, kmax + 1):
        cells = cells_of(k)
        d0 = init(k)
        base = [with_cert(d0, k, c) for c in cells]
        mn = None
        nw = 0
        for i in range(k):
            for j in range(i + 1, k):
                for t in (-1, 0, 1):
                    nw += 1
                    cnt = 0
                    for d in base:
                        if (d[i * k + j] > t and d[j * k + i] > -t
                                and add(d, k, i, j, t) is not None
                                and add(d, k, j, i, -t) is not None):
                            cnt += 1
                    mn = cnt if mn is None else min(mn, cnt)
        print(f"{k} | {nw} | {mn} | {2 ** (k - 2) - 1}", flush=True)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    part1(N)
    part2(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    part3(9)
