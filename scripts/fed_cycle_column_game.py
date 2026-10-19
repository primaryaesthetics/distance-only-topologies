"""The fed cycle C_k: a family of k^2 weightings on which every pairwise comparison a_i - a_j ? tW
tests one window end only, and the column game it reduces to.

Notation as in scripts/fed_cycle_free.py: P_i = c_1 + ... + c_{i-1}, W = c_1 + ... + c_k,
a_j = b_j - P_j, and the unrolled sequence u_{j + qk} = a_j - qW, so that a_i - a_j ? tW compares
u_i with u_{j - tk}.

The family. For a window end e in 1..k and a threshold index x0 in 1..k, the window positions
p = e - k + x (x = 1..k) carry u_p = -x*delta for x < k and u_e = -(k-1)*delta - 1, and
W = 1 + (k - x0 - 1/2)*delta, delta = 1/(100 k^2). The self-served set is the residues of the
positions e - k + x0, ..., e.

Part 1 checks, for k = 2..12, that the self-served set is that cyclic interval and that a comparison
of positions P < P' (D = P' - P) answers "later larger" exactly when D < k, P = e mod k and x0 > D,
for every pair i != j and every t in -3..3; and that the k(k-1)+1 cells are distinct.
Part 2 solves the identification game on these k^2 points with the pairwise queries (t in -1, 0, 1):
the least depth of a tree whose leaves hold points of one cell.
Part 3 solves the column game (the same game written on column heights) by dynamic programming.
Part 4 compares the capacity bound with k - 1 + ceil(log2 k).
Part 5 repeats part 2 with every 0/1 query sum_A a - sum_B a ? tW, |A| = |B| = 2, added, and counts
how many of those queries split the points across two window ends.
"""
import itertools
import math
import sys
import time
from fractions import Fraction as F
from functools import lru_cache


def point(k, e, x0):
    d = F(1, 100 * k * k)
    v = {x: -x * d for x in range(1, k)}
    v[k] = v[k - 1] - 1
    W = 1 + (k - x0 - F(1, 2)) * d
    a = {r: (v[r - e + k] if r <= e else v[r - e] - W) for r in range(1, k + 1)}
    return a, W


def self_served(k, a, W):
    return frozenset(i for i in range(1, k + 1)
                     if all(a[i] < a[j] for j in range(1, i))
                     and all(a[i] < a[j] + W for j in range(i + 1, k + 1)))


def interval(k, e, x0):
    return frozenset((p - 1) % k + 1 for p in range(e - k + x0, e + 1))


def part1(kmax):
    print("# part 1: cells and the one-window-end rule")
    print("# k | points | comparisons checked | cell mismatches | rule mismatches | distinct cells | k(k-1)+1")
    for k in range(2, kmax + 1):
        n = bad_cell = bad_rule = 0
        cells = set()
        for e in range(1, k + 1):
            for x0 in range(1, k + 1):
                a, W = point(k, e, x0)
                S = self_served(k, a, W)
                cells.add(S)
                bad_cell += S != interval(k, e, x0)
                for i in range(1, k + 1):
                    for j in range(1, k + 1):
                        if i == j:
                            continue
                        for t in range(-3, 4):
                            n += 1
                            P1, P2 = i, j - t * k
                            u1, u2 = a[i], a[j] + t * W
                            if u1 == u2:
                                bad_rule += 1
                                continue
                            (P, uP), (Pp, uPp) = sorted([(P1, u1), (P2, u2)])
                            pred = (Pp - P) < k and (P - e) % k == 0 and x0 > Pp - P
                            bad_rule += (uPp > uP) != pred
        print(f"{k} | {k * k} | {n} | {bad_cell} | {bad_rule} | {len(cells)} | {k * (k - 1) + 1}", flush=True)


def ident(k, queries):
    cands = [(e, x0) for e in range(1, k + 1) for x0 in range(2, k + 1)] + [(1, 1)]
    pts = [point(k, e, x0) for e, x0 in cands]
    sets = set()
    for q in queries:
        s = frozenset(n for n, (a, W) in enumerate(pts) if q(a, W))
        if 0 < len(s) < len(cands):
            sets.add(s)
    sets = list(sets)

    @lru_cache(maxsize=None)
    def V(S):
        if len(S) <= 1:
            return 0
        lb = math.ceil(math.log2(len(S)))
        best = 10 ** 9
        for Q in sets:
            A, B = S & Q, S - Q
            if A and B:
                best = min(best, 1 + max(V(A), V(B)))
                if best == lb:
                    break
        return best
    return V(frozenset(range(len(cands)))), len(sets), cands, pts


def pairwise(k):
    return [(lambda a, W, i=i, j=j, t=t: a[i] - a[j] < t * W)
            for i in range(1, k + 1) for j in range(1, k + 1) if i != j for t in (-1, 0, 1)]


def two_vs_two(k):
    out = []
    for A in itertools.combinations(range(1, k + 1), 2):
        for B in itertools.combinations([r for r in range(1, k + 1) if r not in A], 2):
            for t in (-1, 0, 1):
                out.append(lambda a, W, A=A, B=B, t=t: sum(a[r] for r in A) - sum(a[r] for r in B) < t * W)
    return out


def part2(kmax):
    print("# part 2: identification game on the k^2 points, pairwise queries, t in {-1,0,1}")
    print("# k | distinct splits | value | k-1+ceil(log2 k) | seconds")
    for k in range(2, kmax + 1):
        t0 = time.time()
        v, ns, _, _ = ident(k, pairwise(k))
        print(f"{k} | {ns} | {v} | {k - 1 + math.ceil(math.log2(k))} | {time.time() - t0:.1f}", flush=True)


def part3(kmax):
    print("# part 3: column game by dynamic programming (columns of heights h_c, query (c, D): e = c and x0 > D)")
    print("# k | value | k-1+ceil(log2 k) | states | seconds")

    def loc(n):
        return math.ceil(math.log2(n)) if n > 1 else 0
    for k in range(2, kmax + 1):
        t0 = time.time()

        @lru_cache(maxsize=None)
        def V(h):
            if sum(x - 1 for x in h) == 0:
                return 0
            best = 10 ** 9
            for n, hc in enumerate(h):
                if n and h[n - 1] == hc:
                    continue
                for D in range(1, hc):
                    hd = tuple(sorted([x for x in h[:n] + (D,) + h[n + 1:] if x >= 2], reverse=True))
                    best = min(best, 1 + max(loc(hc - D), V(hd)))
            return best
        v = V(tuple([k] * k))
        print(f"{k} | {v} | {k - 1 + math.ceil(math.log2(k))} | {V.cache_info().currsize} | {time.time() - t0:.1f}", flush=True)


def part4(kmax):
    print("# part 4: least V >= k with sum_{i=1..V} min(k-1, 2^(V-i)) >= k(k-1), against k-1+ceil(log2 k)")
    bad = []
    for k in range(2, kmax + 1):
        V = k
        while sum(min(k - 1, 2 ** r) if r < 64 else k - 1 for r in range(V)) < k * (k - 1):
            V += 1
        if V != k - 1 + math.ceil(math.log2(k)):
            bad.append(k)
    print(f"k = 2..{kmax}: exceptions {bad}")


def part5(kmax):
    print("# part 5: identification game with the 2-vs-2 0/1 queries added")
    print("# k | 2-vs-2 queries | constant | one side inside one window end | across window ends | value | ceil(log2(k(k-1)+1)) | seconds")
    for k in range(3, kmax + 1):
        t0 = time.time()
        v, _, cands, pts = ident(k, pairwise(k) + two_vs_two(k))
        const = local = multi = 0
        for q in two_vs_two(k):
            yes = [c for c, (a, W) in zip(cands, pts) if q(a, W)]
            no = [c for c, (a, W) in zip(cands, pts) if not q(a, W)]
            if not yes or not no:
                const += 1
                continue

            def one_end(side):
                return (1, 1) not in side and len(set(c[0] for c in side)) == 1
            if one_end(yes) or one_end(no):
                local += 1
            else:
                multi += 1
        print(f"{k} | {const + local + multi} | {const} | {local} | {multi} | {v} | "
              f"{math.ceil(math.log2(k * (k - 1) + 1))} | {time.time() - t0:.1f}", flush=True)


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k, e, x0 = 8, 8, 4
    a, W = point(k, e, x0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ps = list(range(-k + 1, 2 * k + 1))
    us = [float(a[(p - 1) % k + 1]) - ((p - 1) // k) * float(W) for p in ps]
    ax[0].plot(ps, us, "o-", ms=4, color="0.4")
    S = interval(k, e, x0)
    rec = [(p, u) for p, u in zip(ps, us) if (p - 1) % k + 1 in S]
    ax[0].plot([p for p, _ in rec], [u for _, u in rec], "o", ms=7, mfc="none", color="C3",
               label="self-served residues")
    for c in range(-1, 2):
        ax[0].axvline(e + c * k + 0.5, color="C0", lw=0.6, ls=":")
    ax[0].axhline(us[ps.index(e)] + float(W), color="C2", lw=0.8, ls="--", label="u_e + W")
    ax[0].set_xlabel("position p of the unrolled sequence")
    ax[0].set_ylabel("u_p")
    ax[0].set_title(f"k = {k}, e = {e}, x0 = {x0}: a tight cluster, one drop of about W per period", fontsize=9)
    ax[0].legend(fontsize=8)
    for k2, col in [(9, "C1"), (12, "C4")]:
        V = k2 - 1 + math.ceil(math.log2(k2))
        env = [min(k2 - 1, 2 ** (V - i)) for i in range(1, V + 1)]
        ax[1].step(range(1, V + 1), env, where="mid", color=col, label=f"k = {k2}: min(k-1, 2^(V-i)), V = {V}")
        ax[1].axhline((k2 * (k2 - 1)) / V, color=col, lw=0.6, ls=":")
    ax[1].set_xlabel("spine index i")
    ax[1].set_ylabel("largest block that can leave the spine at i")
    ax[1].set_title("capacity envelope; dotted: k(k-1)/V, the mean block the spine must carry", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"# plot written to {path}")


if __name__ == "__main__":
    part1(12)
    part2(6)
    part3(10)
    part4(1024)
    part5(6)
    if len(sys.argv) > 1:
        plot(sys.argv[1])
