#!/usr/bin/env python3
"""Draws the values of the fed four-cycle against one near and one far
lexicographic world per cell, for every pair of orders, from the output of
`unsorted-fedcycle --worlds=pairs`.

    fed_cycle_worlds_plot.py PAIRS_STDOUT OUT_PNG

Left: the value for each pair, near order by row and far order by column, an
order listed as positions from s from the smallest facet value up. Right: the
nodes the search expanded for each pair, against its value.
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = []
    for line in open(src, encoding="utf-8"):
        m = re.match(r"# k=4 near:(\S+) far:(\S+) value (\d+)\s+nodes (\d+)", line)
        if m:
            rows.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    nears = sorted({r[0] for r in rows})
    fars = sorted({r[1] for r in rows})
    grid = [[float("nan")] * len(fars) for _ in nears]
    for n, f, v, _ in rows:
        grid[nears.index(n)][fars.index(f)] = v
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    im = a.imshow(grid, cmap="Greys", vmin=7, vmax=9.5)
    for i in range(len(nears)):
        for j in range(len(fars)):
            v = grid[i][j]
            a.text(j, i, "" if v != v else str(v), ha="center", va="center",
                   color="white" if v == 9 else "black", fontsize=11)
    a.set_xticks(range(len(fars)), fars)
    a.set_yticks(range(len(nears)), nears)
    a.set_xlabel("far world: source arcs from b_s up, by position from s")
    a.set_ylabel("near world: facet values from smallest up")
    a.set_title("C_4, value per pair (3k - 3 = 9)")
    fig.colorbar(im, ax=a, shrink=0.7)
    for v, marker in ((8, "o"), (9, "s")):
        pts = [r for r in rows if r[2] == v]
        b.scatter([fars.index(r[1]) + 0.08 * nears.index(r[0]) for r in pts],
                  [r[3] for r in pts], marker=marker, label=f"value {v}",
                  facecolors="none" if v == 8 else "black", edgecolors="black")
    b.set_yscale("log")
    b.set_xticks(range(len(fars)), fars)
    b.set_xlabel("far order (near order offsets the points)")
    b.set_ylabel("search nodes, all budgets")
    b.legend()
    b.set_title("search effort per pair")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"{len(rows)} pairs, {sum(r[2] == 9 for r in rows)} at value 9")
    for n, f, v, _ in sorted(rows):
        print(f"near:{n} far:{f} value {v}")


if __name__ == "__main__":
    main()
