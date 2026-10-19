"""The fed cycle C_k with additions free and 0/1-form queries: an exact lower-bound certificate.

C_k has source 0, arcs 0>j of weight b_j and cycle arcs j>j+1 (mod k) of weight c_j, 2k arcs in all.
A 0/1-form query compares two sums of arc weights, so it is the sign of q.(b, c) for a vector q in
{-1, 0, 1}^{2k}; there are (3^{2k} - 1)/2 of them up to sign, 29524 at k = 5 and 265720 at k = 6.

A certificate is a finite set of points in gap coordinates g_1..g_k (g_r = a_r - a_{r+1} for r < k,
g_k = a_k - a_1 + W, in the notation of scripts/fed_cycle_free.py: P_j = c_1 + ... + c_{j-1},
W = c_1 + ... + c_k, a_j = b_j - P_j). Each point is turned into an explicit positive weighting,
in exact rational arithmetic:

    a_1 = 0,  a_{r+1} = a_r - g_r,  W = g_1 + ... + g_k  (required > 0),
    c_j = delta * r_j for j < k (r_j fixed distinct rationals),  c_k = W - (c_1 + ... + c_{k-1}),
    b_j = a_j + P_j + Lambda,

with delta tiny and Lambda huge for the point (chosen below; nothing relies on the choice, since
everything that matters is checked on the weighting itself). The weighting is scaled to integers.
For every point the script then

  * computes the shortest-path tree by Bellman-Ford and asserts that no vertex has two shortest
    entries, so the point lies in the interior of one cell (the set of directly entered vertices),
    and that this cell agrees with the closed form of scripts/fed_cycle_free.py;
  * evaluates every query q in {-1, 0, 1}^{2k} exactly and asserts that none vanishes.

Every query then splits the point set into two sides. The script keeps the distinct nontrivial
bipartitions and decides, by exhaustive memoised search over subsets, the least depth d of a decision
tree over these bipartitions whose every leaf holds points of one cell. No tree of depth d - 1 means:
every linear decision tree with {-1, 0, 1} queries that decides the cell of C_k (under strict or under
full answers) has depth at least d, since restricted to these points, at which no query vanishes and
no shortest path ties, it is such a tree.

Usage:
    fed_cycle_zero_one.py cert <file> <max_depth>         points from a certificate file
    fed_cycle_zero_one.py sample <k> <n> <seed> <max_depth>  n random points (a positive control)

A certificate file holds a line `k <k>` and then one point per line, k numbers, each read as an
exact rational (decimal or p/q); lines starting with # are comments. Sampled points at which some
query vanishes are dropped and counted.
"""
import itertools
import math
import random
import sys
import time
from fractions import Fraction as F

R = [F(3, 7), F(5, 11), F(2, 13), F(7, 17), F(11, 19), F(13, 23), F(17, 29), F(19, 31)]


def weighting(g):
    """Gap coordinates -> (a, W, integer weights b_1..b_k, c_1..c_k)."""
    k = len(g)
    a = [F(0)]
    for r in range(k - 1):
        a.append(a[-1] - g[r])
    W = sum(g)
    assert W > 0, "W must be positive"
    # The least nonzero |sum_j beta_j a_j + tW| over beta in {-1, 0, 1}^k with sum 0, t in {-1, 0, 1}:
    # the leading-order queries. delta is taken far below it, Lambda far above every |a_j| and W.
    least = W
    for beta in itertools.product((-1, 0, 1), repeat=k):
        if sum(beta) != 0:
            continue
        base = sum(x * y for x, y in zip(beta, a))
        for t in (-1, 0, 1):
            v = abs(base + t * W)
            if v and v < least:
                least = v
    r = R[:k - 1]
    delta = least / (4000 * k * (1 + sum(r)))
    c = [delta * x for x in r] + [W - delta * sum(r)]
    P = [F(0)]
    for j in range(k - 1):
        P.append(P[-1] + c[j])
    Lam = 8000 * k * (max(abs(x) for x in a) + W) + 1
    b = [a[j] + P[j] + Lam for j in range(k)]
    w = b + c
    assert all(x > 0 for x in w), "weights must be positive"
    den = 1
    for x in w:
        den = den * x.denominator // math.gcd(den, x.denominator)
    return a, W, [int(x * den) for x in w]


def cell_bellman_ford(w, k):
    """Bitmask of directly entered vertices; asserts no vertex has two shortest entries."""
    arcs = [(0, j, w[j - 1]) for j in range(1, k + 1)] + \
           [(j, j % k + 1, w[k + j - 1]) for j in range(1, k + 1)]
    d = [0] + [None] * k
    for _ in range(k + 1):
        for u, v, x in arcs:
            if d[u] is not None and (d[v] is None or d[u] + x < d[v]):
                d[v] = d[u] + x
    cell = 0
    for j in range(1, k + 1):
        pred = (j - 2) % k + 1
        direct, around = w[j - 1], d[pred] + w[k + pred - 1]
        assert direct != around, "two shortest entries"
        assert d[j] == min(direct, around)
        if direct < around:
            cell |= 1 << (j - 1)
    return cell


def cell_closed_form(a, W):
    k = len(a)
    return sum(1 << i for i in range(k)
               if all(a[i] < a[j] for j in range(i)) and all(a[i] < a[j] + W for j in range(i + 1, k)))


def query_signs(w):
    """Values q.w for every q in {-1, 0, 1}^n with last nonzero coordinate +1 (one of q, -q), in a
    fixed order: index sum_j (q_j + 1) 3^j, upper half of the full table."""
    vals = [0]
    for x in w:
        vals = [v - x for v in vals] + vals + [v + x for v in vals]
    return vals[(len(vals) - 1) // 2 + 1:]


def build(points, k, drop_vanishing):
    labels, rows, kept, vanishing, dropped = [], [], [], 0, 0
    for g in points:
        a, W, w = weighting(g)
        cell = cell_bellman_ford(w, k)
        assert cell == cell_closed_form(a, W), "Bellman-Ford cell differs from the closed form"
        vals = query_signs(w)
        zeros = vals.count(0)
        if zeros:
            if drop_vanishing:
                dropped += 1
                continue
            vanishing += zeros
        labels.append(cell)
        rows.append(bytes(49 if v < 0 else 48 for v in vals))  # b'1' where the query is negative
        kept.append(g)
    N = len(labels)
    full = (1 << N) - 1
    masks, constant = set(), 0
    for col in zip(*rows):
        m = int(bytes(reversed(col)), 2)  # bit p is point p
        if m == 0 or m == full:
            constant += 1
        else:
            masks.add(min(m, full ^ m))
    return labels, sorted(masks), len(rows[0]) if rows else 0, vanishing, constant, dropped


class Search:
    """Least depth of a tree over the bipartitions whose leaves hold one cell each."""

    def __init__(self, labels, masks):
        self.masks = masks
        cells = sorted(set(labels))
        self.cell_masks = [sum(1 << i for i, x in enumerate(labels) if x == c) for c in cells]
        self.memo = {}  # S -> (largest depth known infeasible, least depth known feasible)
        self.nodes = 0

    def ncells(self, S):
        return sum(1 for m in self.cell_masks if m & S)

    def feasible(self, S, d):
        c = self.ncells(S)
        if c <= 1:
            return True
        if c > 1 << d:
            return False
        lo, hi = self.memo.get(S, (-1, None))
        if d <= lo:
            return False
        if hi is not None and d >= hi:
            return True
        self.nodes += 1
        half = 1 << (d - 1)
        seen = set()
        ok = False
        for q in self.masks:
            A = S & q
            if A == 0 or A == S:
                continue
            B = S ^ A
            key = min(A, B)
            if key in seen:
                continue
            seen.add(key)
            if self.ncells(A) > half or self.ncells(B) > half:
                continue
            if self.feasible(A, d - 1) and self.feasible(B, d - 1):
                ok = True
                break
        if ok:
            self.memo[S] = (lo, d if hi is None else min(hi, d))
        else:
            self.memo[S] = (max(lo, d), hi)
        return ok

    def tree(self, S, d):
        """A tree of depth <= d on S (feasible(S, d) must hold), as nested (mask, yes, no) or a leaf."""
        if self.ncells(S) <= 1:
            return S
        for q in self.masks:
            A = S & q
            if A and A != S and self.feasible(A, d - 1) and self.feasible(S ^ A, d - 1):
                return (q, self.tree(A, d - 1), self.tree(S ^ A, d - 1))
        raise AssertionError("no split found")


def check_tree(node, S, labels):
    """Independent walk: every leaf pure; returns (depth, leaves)."""
    if not isinstance(node, tuple):
        assert node == S and len({labels[i] for i in range(len(labels)) if S >> i & 1}) <= 1
        return 0, 1
    q, yes, no = node
    assert q in MASKS
    d1, l1 = check_tree(yes, S & q, labels)
    d2, l2 = check_tree(no, S & ~q, labels)
    return 1 + max(d1, d2), l1 + l2


MASKS = set()


def read_certificate(path):
    k, points = None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("k "):
            k = int(line.split()[1])
            continue
        row = [F(x) for x in line.split()]
        assert k is not None and len(row) == k, "malformed certificate line"
        points.append(row)
    return k, points


def sample_points(k, n, seed):
    rng = random.Random(seed)
    points = []
    while len(points) < n:
        a = [F(rng.randint(-10**6, 10**6), 10**6) * F(10) ** rng.randint(-3, 3) for _ in range(k)]
        W = F(rng.randint(1, 10**6), 10**6) * F(10) ** rng.randint(-3, 3)
        points.append([a[r] - a[r + 1] for r in range(k - 1)] + [a[k - 1] - a[0] + W])
    return points


def main(argv):
    t0 = time.time()
    if argv[0] == "cert":
        k, points = read_certificate(argv[1])
        max_depth = int(argv[2])
        print(f"# certificate {argv[1]}: k = {k}, {len(points)} points")
        drop = False
    elif argv[0] == "sample":
        k, n, seed, max_depth = map(int, argv[1:5])
        points = sample_points(k, n, seed)
        print(f"# random sample: k = {k}, {n} points, seed {seed}")
        drop = True
    else:
        print(__doc__)
        return 2
    labels, masks, nq, vanishing, constant, dropped = build(points, k, drop)
    MASKS.update(masks)
    print(f"points kept: {len(labels)}  (dropped for a vanishing query: {dropped})")
    print(f"distinct cells: {len(set(labels))} of {2 ** k - 1}")
    print(f"queries up to sign: {nq}")
    print(f"vanishing (query, point) pairs: {vanishing}")
    print(f"queries constant on the points: {constant}")
    print(f"distinct nontrivial bipartitions: {len(masks)}")
    print(f"exact build: {time.time() - t0:.1f} s", flush=True)
    search = Search(labels, masks)
    full = (1 << len(labels)) - 1
    least = None
    for d in range(max(0, math.ceil(math.log2(len(set(labels))))), max_depth + 1):
        t1 = time.time()
        ok = search.feasible(full, d)
        print(f"depth {d}: {'feasible' if ok else 'INFEASIBLE'}  "
              f"({search.nodes} search nodes, {time.time() - t1:.1f} s)", flush=True)
        if ok:
            least = d
            depth, leaves = check_tree(search.tree(full, d), full, labels)
            print(f"tree found and checked: depth {depth}, {leaves} leaves, every leaf one cell")
            break
    if least is None:
        print(f"least depth: above {max_depth}")
    else:
        print(f"least depth: {least}")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.exit(main(sys.argv[1:]))
