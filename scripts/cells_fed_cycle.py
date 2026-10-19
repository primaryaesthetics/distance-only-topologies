#!/usr/bin/env python3
"""Count the strict cells of the fed cycle and the bound they give.

The fed cycle `C_k` has source 0, vertices 1..k, a source arc `0>i` of weight
`b_i` into every vertex and the cycle arcs `i>i+1` (indices mod k) of weight
`c_i`. A cell is an assignment of one strict-minimum path to every vertex that
holds on an open set of weights. A strict-minimum path to `i` enters the cycle
at some vertex `e(i)` and walks forward, and by subpath optimality either
`e(i) = i` or `e(i) = e(i - 1)`; so a cell is the nonempty set
`S = {i : e(i) = i}` of vertices served by their own source arc, there are
`2^k - 1` cells, and the cell `S` needs `k - |S|` additions, one per vertex
outside `S`.

For each `k` the script prints three counts that must agree:

- the cells of the root cone as the game solver builds them by linear
  programming (`unsorted-game frontier ARCS 0`, which prints the cone and stops);
- the cells met by random weightings, each checked to satisfy
  `e(i) in {i, e(i - 1)}`;
- `2^k - 1`;

and the bound the cells give a strict-adversary game in which every leaf lies
in one cell: by Kraft's inequality over the leaves of the comparison tree,
`2^r >= sum_S 2^(k - |S|) = 3^k - 2^k`, so the value is at least
`ceil(log2(3^k - 2^k))`. That count is printed beside the corner
`fwd + (m - n + 1) = 2k - 1` and the upper bound `3k - 3`.

Usage: cells_fed_cycle.py KMAX SAMPLES SEED
"""

from __future__ import annotations

import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "target" / "release" / ("unsorted-game.exe" if sys.platform == "win32" else "unsorted-game")


def arcs(k: int) -> str:
    feeds = [f"0>{i}" for i in range(1, k + 1)]
    ring = [f"{i}>{i % k + 1}" for i in range(1, k + 1)]
    return ",".join(feeds + ring)


def solver_cells(k: int) -> int:
    out = subprocess.run(
        [str(BINARY), "frontier", arcs(k), "0"], capture_output=True, text=True, check=True
    ).stdout
    return int(re.search(r"cells (\d+)", out).group(1))


def entries(b: list[float], c: list[float]) -> tuple[int, ...] | None:
    """The entry vertex of the strict-minimum path to every vertex, or None on a tie."""
    k = len(b)
    e = []
    for i in range(k):
        costs = []
        for j in range(k):
            s, t = b[j], j
            while t != i:
                s += c[t]
                t = (t + 1) % k
            costs.append((s, j))
        costs.sort()
        if costs[1][0] - costs[0][0] < 1e-9:
            return None
        e.append(costs[0][1])
    return tuple(e)


def sampled_cells(k: int, samples: int, rng: random.Random) -> int:
    seen = set()
    for _ in range(samples):
        scale = rng.choice([0.01, 0.1, 1.0, 10.0])
        b = [rng.random() for _ in range(k)]
        c = [rng.random() * scale for _ in range(k)]
        e = entries(b, c)
        if e is None:
            continue
        for i in range(k):
            assert e[i] == i or e[i] == e[i - 1], (b, c, e)
        seen.add(e)
    return len(seen)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    kmax, samples, seed = int(argv[0]), int(argv[1]), int(argv[2])
    rng = random.Random(seed)
    print("k | solver cells | sampled cells | 2^k-1 | 3^k-2^k | Kraft bound | corner 2k-1 | 3k-3")
    for k in range(2, kmax + 1):
        lp = solver_cells(k)
        sampled = sampled_cells(k, samples, rng)
        mass = 3**k - 2**k
        kraft = math.ceil(math.log2(mass))
        print(f"{k} | {lp} | {sampled} | {2**k - 1} | {mass} | {kraft} | {2 * k - 1} | {3 * k - 3}")
        assert lp == 2**k - 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
