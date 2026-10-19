#!/usr/bin/env python3
"""Check the feed-and-contract inequality against the game solver.

Let `v` be a non-source vertex whose in-arcs are exactly one source arc
`s>v` and one arc `u>v` with `u` not the source and reachable from the source
without passing through `v`. Contracting `u>v` (every arc
out of `v` leaves `u` instead, loops dropped) and deleting `s>v` gives a rooted
topology `G'` on one vertex fewer, and the strict-adversary value satisfies

    V(G) >= V(G') + 2.

The argument: weight `s>v` at a scale above every other weight and `u>v` at a
scale below; a program for `G` run on such weights is a program for `G'` once
every register is projected onto the remaining coordinates; on every branch
one comparison is decided by the large weight alone — otherwise the leaf cone
is unbounded in that coordinate and `d(v)` is the source arc somewhere on it —
and one addition repeats a projected form, since `d(v)` and `d(u)` differ by
the small weight. Both are free in `G'`.

For every rooted topology on `N` vertices with at most `MMAX` arcs, one per
isomorphism class, and every vertex meeting the hypothesis, the script
computes both values with `unsorted-game solve` and prints them, the
inequality's verdict, and whether it holds with equality.

A contraction can carry parallel arcs, when `u` and `v` both have an arc to
the same vertex; the solver plays such a multigraph with each parallel arc as
its own coordinate, and those instances are counted with the rest and marked
`parallel` in the last column.

Usage: check_contract_lemma.py N MMAX MODE SECONDS
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "target" / "release" / ("unsorted-game.exe" if sys.platform == "win32" else "unsorted-game")


def canon(n: int, arcs: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    best = None
    for p in itertools.permutations(range(1, n)):
        relabel = {0: 0, **{i + 1: p[i] for i in range(n - 1)}}
        key = tuple(sorted((relabel[a], relabel[b]) for a, b in arcs))
        if best is None or key < best:
            best = key
    return best


def reachable(n: int, arcs) -> bool:
    seen, stack = {0}, [0]
    while stack:
        x = stack.pop()
        for a, b in arcs:
            if a == x and b not in seen:
                seen.add(b)
                stack.append(b)
    return len(seen) == n


def text(arcs) -> str:
    return ",".join(f"{a}>{b}" for a, b in arcs)


CACHE: dict = {}


def value(n: int, arcs, mode: str, seconds: int) -> int | None:
    key = canon(n, arcs)
    if key in CACHE:
        return CACHE[key]
    if not key:
        CACHE[key] = 0
        return 0
    out = subprocess.run(
        [str(BINARY), "solve", text(key), f"--mode={mode}", f"--seconds={seconds}"],
        capture_output=True, text=True, check=True,
    ).stdout
    got = re.search(r"value (\S+)", out).group(1)
    CACHE[key] = int(got) if got.isdigit() else None
    return CACHE[key]


def contract(n: int, arcs, v: int, u: int):
    rest = [a for a in arcs if a not in ((0, v), (u, v))]
    rest = [(u if a == v else a, u if b == v else b) for a, b in rest]
    rest = [(a, b) for a, b in rest if a != b]
    relabel = {x: i for i, x in enumerate(y for y in range(n) if y != v)}
    return n - 1, [(relabel[a], relabel[b]) for a, b in rest]


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    n, mmax, mode, seconds = int(argv[0]), int(argv[1]), argv[2], int(argv[3])
    candidates = [(a, b) for a in range(n) for b in range(1, n) if a != b]
    seen = set()
    checked = failures = tight = undecided = parallel = 0
    print(f"# G | v | V(G) | G' | V(G') | verdict   ({mode}, {seconds} s per game)")
    for m in range(1, mmax + 1):
        for arcs in itertools.combinations(candidates, m):
            arcs = list(arcs)
            if not reachable(n, arcs):
                continue
            key = canon(n, arcs)
            if key in seen:
                continue
            seen.add(key)
            arcs = list(key)
            for v in range(1, n):
                into = [a for a in arcs if a[1] == v]
                if len(into) != 2 or (0, v) not in into:
                    continue
                u = next(a for a, _ in into if a != 0)
                n2, arcs2 = contract(n, arcs, v, u)
                if not reachable(n2, arcs2):
                    continue
                multi = len(set(arcs2)) < len(arcs2)
                parallel += multi
                big, small = value(n, arcs, mode, seconds), value(n2, arcs2, mode, seconds)
                if big is None or small is None:
                    undecided += 1
                    print(f"{text(arcs)} | {v} | {big} | {text(arcs2)} | {small} | undecided")
                    continue
                checked += 1
                holds = big >= small + 2
                failures += not holds
                tight += big == small + 2
                verdict = ("tight" if big == small + 2 else "holds") if holds else "FAILS"
                note = " parallel" if multi else ""
                print(f"{text(arcs)} | {v} | {big} | {text(arcs2)} | {small} | {verdict}{note}", flush=True)
    print(
        f"# checked {checked}, failures {failures}, tight {tight}, undecided {undecided}, "
        f"of which with parallel arcs {parallel}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
