#!/usr/bin/env python3
"""How many selections the four-rule program needs at scale, and what passes cost instead.

The four-rule program of `theory/game/rule_programs.py`, run on concrete
weights and counted in comparisons and additions:

  R0  arcs (u, v) with v dominating u are deleted (dominator tree by the
      iterative algorithm of Cooper, Harvey and Kennedy);
  R1  strongly connected components are processed in one topological order,
      each with its own heap;
  R2  a labelled vertex all of whose remaining in-arcs come from final
      vertices is final with no selection (`free`);
  R3  otherwise the least label of the earliest unfinished component's heap is
      extracted (`sel`), charged ceil(log2(h + 1)) comparisons for the h
      labelled non-final vertices of the component, the size of a
      decrease-key heap (the stale entries of the lazy heap used here are not
      charged).

A relaxation into a non-final vertex costs one addition, and one comparison
when the head already holds a label. `lf` is the same count with R2 and R3
only. The distances of both are checked against a plain binary-heap Dijkstra.

Beside it, the alternating-pass program: least source arc by comparisons, then
passes over the vertices in a fixed order, forward and backward alternately,
relaxing every in-arc of each vertex (the least-weight one of parallel arcs),
one addition and one comparison each, until a pass changes nothing (at most
64). Two orders: depth-first and breadth-first from the least entry,
restarted from the other source out-neighbours when it is exhausted. `delta`
is the largest and mean number of order-reversing arcs on the shortest-path
tree path of a vertex, which bounds the passes by 1 + delta.

Families, each at n - 1 = 2^6 .. 2^14 non-source vertices: the cycle fed at
every vertex, the bidirected path fed at every vertex, the bidirected path
fed at its two ends, the bidirected grid fed at every vertex (g x g, or
g x 2g at odd powers), and a random digraph fed at every vertex with three
out-arcs per vertex. With --ny GRAPH, a DIMACS graph from its vertex 1.

    python ref/rule_programs/scale_probe.py [--ny data/graphs/USA-road-d.NY.gr]
                                            [--plot out.png]
"""
from __future__ import annotations

import heapq
import math
import random
import sys
from collections import deque

INF = float("inf")


# ----------------------------------------------------------------- families
def cycle(k, rng):
    arcs = [(0, i, rng.random() * 10) for i in range(1, k + 1)]
    arcs += [(i, i % k + 1, rng.random()) for i in range(1, k + 1)]
    return k + 1, arcs


def bipath(k, rng, feeds=None):
    arcs = [(0, i, rng.random() * 10) for i in (feeds or range(1, k + 1))]
    for i in range(1, k):
        arcs.append((i, i + 1, rng.random()))
        arcs.append((i + 1, i, rng.random()))
    return k + 1, arcs


def bigrid(rows, cols, rng):
    idx = lambda r, c: 1 + r * cols + c  # noqa: E731
    arcs = [(0, idx(r, c), rng.random() * 10) for r in range(rows) for c in range(cols)]
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 < rows and c2 < cols:
                    arcs.append((idx(r, c), idx(r2, c2), rng.random()))
                    arcs.append((idx(r2, c2), idx(r, c), rng.random()))
    return rows * cols + 1, arcs


def rand3(k, rng):
    arcs = [(0, i, rng.random() * 10) for i in range(1, k + 1)]
    for i in range(1, k + 1):
        for _ in range(3):
            j = rng.randint(1, k)
            if j != i:
                arcs.append((i, j, rng.random()))
    return k + 1, arcs


def family(name, p, rng):
    k = 2 ** p
    if name == "fed cycle":
        return cycle(k, rng)
    if name == "fed bidirected path":
        return bipath(k, rng)
    if name == "bidirected path, two feeds":
        return bipath(k, rng, feeds=[1, k])
    if name == "fed bidirected grid":
        g = 2 ** (p // 2)
        return bigrid(g, g * (2 if p % 2 else 1), rng)
    if name == "fed random 3-out":
        return rand3(k, rng)
    raise ValueError(name)


FAMILIES = ("fed cycle", "fed bidirected path", "bidirected path, two feeds",
            "fed bidirected grid", "fed random 3-out")


def read_dimacs(path):
    n = 0
    arcs = []
    with open(path) as f:
        for line in f:
            if line.startswith("p"):
                n = int(line.split()[2])
            elif line.startswith("a"):
                _, u, v, w = line.split()
                arcs.append((int(u) - 1, int(v) - 1, float(w)))
    return n, arcs


# ----------------------------------------------------------------- structure
def dijkstra(n, arcs, src=0):
    outs = [[] for _ in range(n)]
    for u, v, w in arcs:
        outs[u].append((v, w))
    d = [INF] * n
    d[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > d[u]:
            continue
        for v, w in outs[u]:
            if du + w < d[v]:
                d[v] = du + w
                heapq.heappush(heap, (d[v], v))
    return d


def dominator_intervals(n, outs, inn, src):
    """Pre/post numbers of the dominator tree, so that v dominates u iff
    pre[v] <= pre[u] and post[u] <= post[v]."""
    order = []
    seen = [False] * n
    seen[src] = True
    stack = [(src, iter(outs[src]))]
    while stack:
        u, it = stack[-1]
        for v, _ in it:
            if not seen[v]:
                seen[v] = True
                stack.append((v, iter(outs[v])))
                break
        else:
            order.append(u)
            stack.pop()
    rpo = order[::-1]
    num = [-1] * n
    for i, v in enumerate(rpo):
        num[v] = i
    idom = [-1] * n
    idom[src] = src

    def meet(a, b):
        while a != b:
            while num[a] > num[b]:
                a = idom[a]
            while num[b] > num[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            new = -1
            for u in inn[v]:
                if num[u] < 0 or idom[u] < 0:
                    continue
                new = u if new < 0 else meet(u, new)
            if new != idom[v]:
                idom[v] = new
                changed = True
    kids = [[] for _ in range(n)]
    for v in rpo[1:]:
        kids[idom[v]].append(v)
    pre = [-1] * n
    post = [-1] * n
    clock = 0
    stack = [(src, 0)]
    while stack:
        u, i = stack.pop()
        if i == 0:
            pre[u] = clock
            clock += 1
        if i < len(kids[u]):
            stack.append((u, i + 1))
            stack.append((kids[u][i], 0))
        else:
            post[u] = clock
            clock += 1
    return pre, post


def components(n, outs):
    """Tarjan's strongly connected components, iteratively; returns the
    component of each vertex numbered in topological order (sources first)."""
    index = [-1] * n
    low = [0] * n
    on = [False] * n
    comp = [-1] * n
    st = []
    counter = 0
    found = 0
    for root in range(n):
        if index[root] >= 0:
            continue
        work = [(root, 0)]
        while work:
            u, i = work.pop()
            if i == 0:
                index[u] = low[u] = counter
                counter += 1
                st.append(u)
                on[u] = True
            recurse = False
            while i < len(outs[u]):
                v = outs[u][i][0]
                i += 1
                if index[v] < 0:
                    work.append((u, i))
                    work.append((v, 0))
                    recurse = True
                    break
                if on[v]:
                    low[u] = min(low[u], index[v])
            if recurse:
                continue
            if low[u] == index[u]:
                while True:
                    x = st.pop()
                    on[x] = False
                    comp[x] = found
                    if x == u:
                        break
                found += 1
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[u])
    # Tarjan emits components in reverse topological order.
    return [found - 1 - c for c in comp], found


# ----------------------------------------------------------------- programs
def rule_program(n, arcs, src=0, prune=True, condense=True):
    outs = [[] for _ in range(n)]
    inn = [[] for _ in range(n)]
    for u, v, w in arcs:
        outs[u].append((v, w))
        inn[v].append(u)
    if prune:
        pre, post = dominator_intervals(n, outs, inn, src)
        kept = [(u, v, w) for u, v, w in arcs
                if u == src or pre[u] < 0 or not (pre[v] <= pre[u] and post[u] <= post[v])]
        outs = [[] for _ in range(n)]
        inn = [[] for _ in range(n)]
        for u, v, w in kept:
            outs[u].append((v, w))
            inn[v].append(u)
    else:
        kept = arcs
    if condense:
        comp, ncomp = components(n, outs)
    else:
        comp, ncomp = [0] * n, 1
    left = [0] * ncomp
    for v in range(n):
        left[comp[v]] += 1
    heaps = [[] for _ in range(ncomp)]
    d = [INF] * n
    final = [False] * n
    unfinal_in = [len(inn[v]) for v in range(n)]
    live = [0] * ncomp
    ready = deque()
    adds = cmps = sel = free = 0

    def settle(u):
        nonlocal adds, cmps
        final[u] = True
        left[comp[u]] -= 1
        if u != src:
            live[comp[u]] -= 1
        for v, w in outs[u]:
            unfinal_in[v] -= 1
            if final[v]:
                continue
            if u == src:
                nd = w
            else:
                adds += 1
                nd = d[u] + w
            if d[v] < INF:
                cmps += 1
            else:
                live[comp[v]] += 1
            if nd < d[v]:
                d[v] = nd
                heapq.heappush(heaps[comp[v]], (nd, v))
            if unfinal_in[v] == 0:
                ready.append(v)

    d[src] = 0.0
    settle(src)
    cur = 0
    done = 1
    while done < n:
        if ready:
            v = ready.popleft()
            if final[v] or d[v] == INF:
                continue
            free += 1
        else:
            while cur < ncomp and left[cur] == 0:
                cur += 1
            if cur == ncomp:
                break
            h = heaps[cur]
            while h and final[h[0][1]]:
                heapq.heappop(h)
            if not h:
                break
            sel += 1
            cmps += max(1, math.ceil(math.log2(live[cur] + 1)))
            _, v = heapq.heappop(h)
        settle(v)
        done += 1
    return d, dict(adds=adds, cmps=cmps, sel=sel, free=free, ops=adds + cmps,
                   pruned=len(arcs) - len(kept), components=ncomp)


def entry_order(n, outs, src, entry, bfs):
    """Depth- or breadth-first order from the least entry over the non-source
    arcs; when that is exhausted, from the next source out-neighbour in the
    order the source arcs are listed, then the unreached vertices by index."""
    order = []
    seen = [False] * n
    seen[src] = True
    starts = [entry] + [v for v, _ in outs[src] if v != entry]
    for start in starts:
        if seen[start]:
            continue
        if bfs:
            q = deque([start])
            seen[start] = True
            while q:
                u = q.popleft()
                order.append(u)
                for v, _ in outs[u]:
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
        else:
            stack = [start]
            while stack:
                u = stack.pop()
                if seen[u]:
                    continue
                seen[u] = True
                order.append(u)
                for v, _ in reversed(outs[u]):
                    if not seen[v]:
                        stack.append(v)
    order += [v for v in range(n) if not seen[v]]
    return order


def passes(n, arcs, d_true, bfs, src=0, max_passes=64):
    best_w: dict[tuple[int, int], float] = {}
    for u, v, w in arcs:
        if (u, v) not in best_w or w < best_w[(u, v)]:
            best_w[(u, v)] = w
    outs = [[] for _ in range(n)]
    inn = [[] for _ in range(n)]
    for (u, v), w in best_w.items():
        outs[u].append((v, w))
        inn[v].append((u, w))
    d = [INF] * n
    d[src] = 0.0
    cmps = 0
    entry = None
    for v, w in outs[src]:
        if entry is None:
            entry = (w, v)
        else:
            cmps += 1
            if w < entry[0]:
                entry = (w, v)
        d[v] = w
    order = entry_order(n, outs, src, entry[1], bfs)
    pos = {v: i for i, v in enumerate(order)}
    adds = 0
    count = 0
    changed = True
    fwd = True
    while changed and count < max_passes:
        changed = False
        count += 1
        for v in (order if fwd else order[::-1]):
            for u, w in inn[v]:
                if u == src or d[u] == INF:
                    continue
                adds += 1
                cmps += 1
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
                    changed = True
        fwd = not fwd
    ok = all(abs(d[v] - d_true[v]) <= 1e-9 * max(1.0, d_true[v]) for v in range(n) if d_true[v] < INF)
    parent = [-1] * n
    for v in range(n):
        if v == src or d_true[v] == INF:
            continue
        for u, w in inn[v]:
            if abs(d_true[u] + w - d_true[v]) <= 1e-9 * max(1.0, d_true[v]):
                parent[v] = u
                break
    delta = [0] * n
    for v in sorted(range(n), key=lambda x: d_true[x]):
        p = parent[v]
        if p <= 0 or p == src:
            continue
        delta[v] = delta[p] + (1 if pos[p] > pos[v] else 0)
    return dict(passes=count, ops=adds + cmps, ok=ok, converged=not changed,
                delta_max=max(delta), delta_mean=sum(delta) / max(1, n - 1))


# ----------------------------------------------------------------- report
HEADER = ("family | n m | four-rule sel free sel/n ops/m pruned comps | lf sel/n ops/m | "
          "dfs passes ops/m ok delta max mean | bfs passes ops/m ok delta max mean")


def report(name, n, arcs, src=0):
    m = len(arcs)
    d_ref = dijkstra(n, arcs, src)
    d4, a = rule_program(n, arcs, src)
    dl, b = rule_program(n, arcs, src, prune=False, condense=False)
    tol = lambda x, y: abs(x - y) <= 1e-9 * max(1.0, abs(y)) if y < INF else x == INF  # noqa: E731
    if not all(tol(d4[v], d_ref[v]) for v in range(n)) or not all(tol(dl[v], d_ref[v]) for v in range(n)):
        raise AssertionError(f"{name}: rule-program distances differ from Dijkstra's")
    cells = [f"{name} | {n} {m} | {a['sel']} {a['free']} {a['sel'] / n:.4f} {a['ops'] / m:.2f} "
             f"{a['pruned']} {a['components']} | {b['sel'] / n:.4f} {b['ops'] / m:.2f}"]
    for bfs in (False, True):
        y = passes(n, arcs, d_ref, bfs, src)
        cells.append(f"{y['passes']}{'' if y['converged'] else '+'} {y['ops'] / m:.2f} {y['ok']} "
                     f"{y['delta_max']} {y['delta_mean']:.2f}")
    print(" | ".join(cells), flush=True)
    return a["sel"] / n


def plot(out, series, ny):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for (name, pts), colour in zip(series.items(), colours):
        xs = [x - 1 for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=4, label=name)
    if ny is not None:
        ax.plot([ny[0] - 1], [ny[1]], marker="D", markersize=8, color="#6250d6", linestyle="none",
                label="USA-road-d.NY")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("non-source vertices")
    ax.set_ylabel("sel / n")
    ax.set_title("Selections per vertex of the four-rule program")
    ax.grid(color="#e0e0dc", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main(argv):
    rng = random.Random(1)
    print(HEADER)
    series: dict[str, list[tuple[int, float]]] = {}
    for name in FAMILIES:
        for p in range(6, 15):
            n, arcs = family(name, p, rng)
            series.setdefault(name, []).append((n, report(name, n, arcs)))
    ny = None
    if "--ny" in argv:
        path = argv[argv.index("--ny") + 1]
        n, arcs = read_dimacs(path)
        ny = (n, report(path.replace("\\", "/").rsplit("/", 1)[-1], n, arcs))
    if "--plot" in argv:
        plot(argv[argv.index("--plot") + 1], series, ny)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
