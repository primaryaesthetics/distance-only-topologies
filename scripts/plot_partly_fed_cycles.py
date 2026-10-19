#!/usr/bin/env python3
"""Draw the distance above the corner of cycles fed at some of their vertices.

Reads tables printed by `unsorted-game table` and keeps every topology that is
a single cycle through all non-source vertices with source arcs into some of
them: `k` cycle vertices, `f` of them fed. For each it takes the value of
either strict-adversary game where one is decided, or the best lower bound
(`>=v`) where none is, and subtracts the corner `fwd + (m − n + 1) = k − 1 + f`.

The left panel is the `(k, f)` grid with the value written in each decided
cell and the distance above the corner as its colour; the right panel draws
that distance against `k`, one line per `f`, with `f − 2` dashed beside it.
Whether the distance moves with the length of the cycle or with the number of
fed vertices is what the picture is drawn to show.

Prints the table the plot is drawn from.

Usage: plot_partly_fed_cycles.py <out.png> <table-stdout.txt>...
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def cycle_shape(arcs: list[tuple[int, int]]) -> tuple[int, int] | None:
    """(k, f) if the topology is one fed cycle through every non-source vertex."""
    feeds = [b for a, b in arcs if a == 0]
    inner = [(a, b) for a, b in arcs if a != 0]
    verts = {x for a in inner for x in a}
    if not inner or len(inner) != len(verts):
        return None
    succ = dict(inner)
    if len(succ) != len(inner) or set(succ.values()) != verts:
        return None
    start = next(iter(verts))
    x, seen = start, 0
    while True:
        x, seen = succ[x], seen + 1
        if x == start:
            break
    if seen != len(verts) or not set(feeds) <= verts or len(set(feeds)) != len(feeds):
        return None
    return len(verts), len(feeds)


def read(paths: list[str]) -> dict[tuple[int, int], list[tuple[str, int, bool]]]:
    cells: dict[tuple[int, int], list[tuple[str, int, bool]]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            arcs = [tuple(map(int, a.split(">"))) for a in parts[0].split(",")]
            shape = cycle_shape(arcs)
            if shape is None:
                continue
            values = [p.split()[0] for p in parts[2:]]
            exact = [int(v) for v in values if v.isdigit()]
            lower = [int(v[2:]) for v in values if v.startswith(">=")]
            if exact:
                cells.setdefault(shape, []).append((parts[0], min(exact), True))
            elif lower:
                cells.setdefault(shape, []).append((parts[0], max(lower), False))
    return cells


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out, paths = argv[0], argv[1:]
    cells = read(paths)
    print("# k f | arcs | value | corner | above corner | f - 2")
    best: dict[tuple[int, int], tuple[int, bool]] = {}
    for (k, f), rows in sorted(cells.items()):
        for arcs, value, exact in rows:
            corner = k - 1 + f
            mark = "" if exact else ">="
            print(f"{k} {f} | {arcs} | {mark}{value} | {corner} | {mark}{value - corner} | {f - 2}")
            old = best.get((k, f))
            if old is None or (exact and not old[1]):
                best[(k, f)] = (value, exact)

    ks = sorted({k for k, _ in best})
    fs = sorted({f for _, f in best})
    fig, (grid, lines) = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.get_cmap("viridis")
    lo, hi = -1, max(f for f in fs) - 2
    for (k, f), (value, exact) in best.items():
        above = value - (k - 1 + f)
        colour = cmap((above - lo) / max(1, hi - lo))
        grid.add_patch(plt.Rectangle((k - 0.5, f - 0.5), 1, 1, fc=colour if exact else "white",
                                     ec="black", lw=0.8))
        text = f"{value}" if exact else f"≥{value}"
        grid.text(k, f, text, ha="center", va="center", fontsize=10,
                  color="white" if exact and above < (lo + hi) / 2 else "black")
    grid.set_xlim(min(ks) - 0.5, max(ks) + 0.5)
    grid.set_ylim(min(fs) - 0.5, max(fs) + 0.5)
    grid.set_xticks(ks)
    grid.set_yticks(fs)
    grid.set_xlabel("cycle length k")
    grid.set_ylabel("fed vertices f")
    grid.set_title("value; colour: value − corner; white: lower bound")
    grid.set_aspect("equal")

    for f in fs:
        pts = sorted((k, v - (k - 1 + f), e) for (k, ff), (v, e) in best.items() if ff == f)
        xs = [k for k, _, _ in pts]
        ys = [y for _, y, _ in pts]
        line, = lines.plot(xs, ys, marker="o", label=f"f = {f}")
        for k, y, e in pts:
            if not e:
                lines.plot([k], [y], marker="^", color=line.get_color(), ms=9, mfc="white")
        lines.plot([min(ks), max(ks)], [f - 2, f - 2], ls="--", lw=0.8, color=line.get_color())
    lines.set_xlabel("cycle length k")
    lines.set_ylabel("value − corner")
    lines.set_title("value − corner; dashed: f − 2; open: lower bound")
    lines.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
