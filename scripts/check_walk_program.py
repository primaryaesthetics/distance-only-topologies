#!/usr/bin/env python3
"""Check the walk program on cycles fed from the source at some vertices.

The topology: a cycle `1 -> 2 -> ... -> k -> 1` with arc weights `c_i` on
`i -> i+1`, and source arcs `b_i` into the vertices of a set `F`, `|F| = f`.
The walk program `W(k, f)`:

  1. find `s`, the vertex of `F` with the least source arc, in `f - 1`
     comparisons, and set `d(s) = b_s`;
  2. walk once around the cycle from `s`: at each vertex `i != s` form
     `d(i-1) + c_{i-1}` with one addition and, when `i` is in `F`, compare it
     with `b_i` and keep the smaller.

Every held form is a source arc followed by consecutive cycle arcs, so the
program is a path-register program, and on every branch it makes
`(f - 1) + (k - 1) + (f - 1) = k + 2f - 3` operations.

For every `k` from 2 to `KMAX`, every `f` from 1 to `k` and every feed set
containing vertex 1 (each feed placement up to rotation, some more than
once), the script runs `W` on `TRIALS` random weightings, drawn with the
source arcs and the cycle arcs at independently chosen scales so that every
outcome of the comparisons occurs, and checks that the distances it returns
equal those of Dijkstra's algorithm, that every held form is a sub-path form
and that the operation count is `k + 2f - 3`. It prints one line per
`(k, f)`: the placements, the trials, the operation counts seen, and the
number of distinct comparison-outcome sequences met.

Usage: check_walk_program.py KMAX TRIALS SEED
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys


def dijkstra(k: int, feeds: dict[int, int], c: list[int]) -> list[int]:
    """Distances from the source to vertices 1..k (index 0 unused)."""
    dist = [None] * (k + 1)
    heap = [(w, v) for v, w in feeds.items()]
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if dist[v] is not None:
            continue
        dist[v] = d
        nxt = v % k + 1
        if dist[nxt] is None:
            heapq.heappush(heap, (d + c[v], nxt))
    return dist


def walk(k: int, feeds: dict[int, int], c: list[int]):
    """Runs W; returns distances, operation count and the outcome sequence.

    A held form is (j, length): the source arc into j followed by `length`
    cycle arcs from j. Its value is recomputed from that description, so a
    form that is not a sub-path cannot be produced unnoticed.
    """
    def value(form):
        j, length = form
        total, x = feeds[j], j
        for _ in range(length):
            total += c[x]
            x = x % k + 1
        return total

    ops = 0
    outcomes = []
    order = sorted(feeds)
    s = order[0]
    for j in order[1:]:
        ops += 1
        less = feeds[j] < feeds[s]
        outcomes.append(less)
        if less:
            s = j
    form = {s: (s, 0)}
    prev = s
    for step in range(1, k):
        i = (s - 1 + step) % k + 1
        j, length = form[prev]
        candidate = (j, length + 1)
        ops += 1
        assert value(candidate) == value(form[prev]) + c[prev]
        if i in feeds:
            ops += 1
            less = feeds[i] < value(candidate)
            outcomes.append(less)
            candidate = (i, 0) if less else candidate
        form[i] = candidate
        prev = i
    for i, (j, length) in form.items():
        assert j in feeds and 0 <= length < k
        x = j
        for _ in range(length):
            x = x % k + 1
        assert x == i, "form does not end at its vertex"
    dist = [None] + [value(form[i]) for i in range(1, k + 1)]
    return dist, ops, tuple(outcomes)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    kmax, trials, seed = int(argv[0]), int(argv[1]), int(argv[2])
    rng = random.Random(seed)
    failures = 0
    print("# k f | placements trials | operation counts | k+2f-3 | outcome sequences")
    for k in range(2, kmax + 1):
        for f in range(1, k + 1):
            placements = [(1, *rest) for rest in itertools.combinations(range(2, k + 1), f - 1)]
            counts: set[int] = set()
            sequences: set = set()
            for placed in placements:
                for _ in range(trials):
                    scale_b = 10 ** rng.randint(0, 6)
                    scale_c = 10 ** rng.randint(0, 6)
                    feeds = {v: rng.randint(1, scale_b) for v in placed}
                    c = [0] + [rng.randint(1, scale_c) for _ in range(k)]
                    got, ops, outcomes = walk(k, feeds, c)
                    want = dijkstra(k, feeds, c)
                    counts.add(ops)
                    sequences.add((placed, outcomes))
                    if got != want or ops != k + 2 * f - 3:
                        failures += 1
                        print(f"FAIL k={k} feeds={placed} b={feeds} c={c[1:]} got={got[1:]} want={want[1:]} ops={ops}")
            law = k + 2 * f - 3
            print(
                f"{k} {f} | {len(placements)} {len(placements) * trials} | "
                f"{sorted(counts)} | {law} | {len(sequences)}",
                flush=True,
            )
    print(f"# failures {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
