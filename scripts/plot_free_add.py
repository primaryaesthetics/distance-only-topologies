#!/usr/bin/env python3
"""Draw the free-addition comparison cost of every small topology against its size.

Reads one or more tables printed by `unsorted-game enumerate` or
`unsorted-game table` in the free-addition modes (`path-free`,
`generic-free`), in which every holdable form is held from the start and
only comparisons are charged, so the value is the depth of a linear decision
tree that locates the weight vector's cell — one shortest path per label —
with queries drawn from differences of held forms. Beside each value the
picture draws the two bounds the value lives between:

  * `ceil(log2 tau)`, where `tau` is the number of spanning arborescences
    rooted at the source, computed by the matrix-tree theorem as the
    determinant of the in-degree Laplacian with the source's row and column
    removed. Every arborescence is the unique shortest-path tree of some
    weighting (tree arcs at 1, the rest at `n`), so a decision tree needs a
    leaf per arborescence and `tau` is the exact cell count;
  * `m − n + 1`, the comparison count of a program that fixes the
    shortest-path tree one in-arc at a time, and Cai's exact comparison
    coordinate on acyclic topologies (arXiv:2609.04825, Theorem 4.1).

Acyclic and cyclic topologies take different marks. The vertical distance of
a mark above `m − n + 1` is what a cycle costs in comparisons once additions
are free, and whether that distance moves with `m` is the question the
picture is drawn to answer. A second panel follows one family, the cycle fed
from the source at every vertex, where `tau = 2^k − 1` and the walk program
spends `2k − 2`.

Prints the table the plot is drawn from, so a recorded run carries the
numbers as well as the picture.

Usage: plot_free_add.py <out.png> <table-stdout.txt>...
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = ("path-free", "generic-free")


def parse(path: Path) -> list[dict]:
    """Rows of a table; a cell is a value or `>=k` when the search timed out."""
    rows = []
    modes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# arcs"):
            modes = [c.strip().split()[0] for c in line.split("|")[2:]]
            continue
        if line.startswith("# ") and "  mode " in line and "  value " in line:
            # The header of `unsorted-game solve`: one topology, one mode.
            words = line[2:].split()
            arcs, mode, head = words[0], words[words.index("mode") + 1], words[words.index("value") + 1]
            n = 1 + max(int(x) for tok in arcs.split(",") for x in tok.split(">"))
            value = None if head.startswith(">=") else int(head)
            bound = int(head[2:]) if head.startswith(">=") else int(head)
            rows.append({
                "arcs": arcs, "n": n, "m": arcs.count(",") + 1, "dag": acyclic(arcs, n),
                "ds": sum(tok.startswith("0>") for tok in arcs.split(",")), "fwd": None,
                "values": {mode: value}, "bounds": {mode: bound},
            })
            continue
        if not line or line.startswith("#") or "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        arcs, meta, results = cells[0], cells[1], cells[2:]
        n, m, dag, ds, fwd, _law = meta.split()
        values: dict[str, int | None] = {}
        bounds: dict[str, int] = {}
        for mode, cell in zip(modes, results):
            head = cell.split()[0]
            if head.startswith(">="):
                values[mode], bounds[mode] = None, int(head[2:])
            else:
                values[mode] = bounds[mode] = int(head)
        rows.append({
            "arcs": arcs, "n": int(n), "m": int(m), "dag": dag == "1",
            "ds": int(ds), "fwd": int(fwd), "values": values, "bounds": bounds,
        })
    return rows


def acyclic(arcs: str, n: int) -> bool:
    """Whether the topology has no directed cycle, by repeated source removal."""
    pairs = [tuple(int(x) for x in tok.split(">")) for tok in arcs.split(",")]
    indeg = [0] * n
    for _, v in pairs:
        indeg[v] += 1
    stack = [v for v in range(n) if indeg[v] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for a, b in pairs:
            if a == u:
                indeg[b] -= 1
                if indeg[b] == 0:
                    stack.append(b)
    return seen == n


def arborescences(arcs: str, n: int) -> int:
    """Spanning arborescences rooted at vertex 0, by the matrix-tree theorem."""
    size = n - 1
    lap = [[Fraction(0)] * size for _ in range(size)]
    for tok in arcs.split(","):
        u, v = (int(x) for x in tok.split(">"))
        lap[v - 1][v - 1] += 1
        if u != 0:
            lap[u - 1][v - 1] -= 1
    # Gaussian elimination over the rationals; the determinant is exact.
    det = Fraction(1)
    for col in range(size):
        pivot = next((r for r in range(col, size) if lap[r][col] != 0), None)
        if pivot is None:
            return 0
        if pivot != col:
            lap[col], lap[pivot] = lap[pivot], lap[col]
            det = -det
        det *= lap[col][col]
        for r in range(col + 1, size):
            factor = lap[r][col] / lap[col][col]
            if factor:
                lap[r] = [a - factor * b for a, b in zip(lap[r], lap[col])]
    assert det.denominator == 1
    return int(det)


def ceil_log2(k: int) -> int:
    return 0 if k <= 1 else (k - 1).bit_length()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    rows: list[dict] = []
    for path in argv[1:]:
        rows.extend(parse(Path(path)))

    print("arcs | n m dag | tau log2 corner | path-free generic-free | over-corner over-count")
    for row in rows:
        tau = arborescences(row["arcs"], row["n"])
        row["tau"] = tau
        row["count"] = ceil_log2(tau)
        row["corner"] = row["m"] - row["n"] + 1
        pf, gf = row["values"].get("path-free"), row["values"].get("generic-free")
        best = next((v for v in (gf, pf) if v is not None), None)
        row["best"] = best
        def shown(v: int | None, mode: str) -> str:
            if v is not None:
                return str(v)
            return f">={row['bounds'][mode]}" if mode in row["bounds"] else "-"
        over_corner = "-" if best is None else best - row["corner"]
        over_count = "-" if best is None else best - row["count"]
        print(
            f"{row['arcs']} | {row['n']} {row['m']} {int(row['dag'])} | {tau} {row['count']} "
            f"{row['corner']} | {shown(pf, 'path-free')} {shown(gf, 'generic-free')} | "
            f"{over_corner} {over_count}"
        )

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})

    # Left: every topology, value against m, with its two bounds.
    for row in rows:
        if row["best"] is None:
            continue
        x = row["m"] + (0.12 if row["dag"] else -0.12)
        left.plot([x, x], [row["count"], row["corner"]], color="0.8", lw=1, zorder=1)
        left.scatter([x], [row["count"]], marker="_", color="0.5", s=80, zorder=2)
        left.scatter([x], [row["corner"]], marker="_", color="0.2", s=80, zorder=2)
        left.scatter(
            [x], [row["best"]],
            marker="s" if row["dag"] else "o",
            facecolor="white" if row["dag"] else "tab:red",
            edgecolor="tab:blue" if row["dag"] else "tab:red",
            s=36, zorder=3,
        )
    left.set_xlabel("m (arcs)")
    left.set_ylabel("comparisons, additions free")
    left.set_title("Every topology: value (marks) between ceil(log2 tau) (grey) and m − n + 1 (dark)")
    left.grid(alpha=0.3)
    left.scatter([], [], marker="s", facecolor="white", edgecolor="tab:blue", label="acyclic")
    left.scatter([], [], marker="o", color="tab:red", label="cyclic")
    left.legend(loc="upper left")

    # Right: the fed cycle C_k, whichever members the tables contain.
    fed = []
    for row in rows:
        k = row["n"] - 1
        cycle = ",".join(f"0>{i}" for i in range(1, k + 1)) + "," + ",".join(
            f"{i}>{i % k + 1}" for i in range(1, k + 1)
        )
        canonical = ",".join(sorted(cycle.split(","), key=lambda a: tuple(int(x) for x in a.split(">"))))
        mine = ",".join(sorted(row["arcs"].split(","), key=lambda a: tuple(int(x) for x in a.split(">"))))
        if canonical == mine and row["best"] is not None:
            fed.append((k, row))
    fed.sort()
    if fed:
        ks = [k for k, _ in fed]
        right.plot(ks, [2 * k - 2 for k in ks], "--", color="0.3", label="walk program 2k − 2")
        right.plot(ks, [k for k in ks], ":", color="0.2", label="m − n + 1 = k")
        right.plot(ks, [ceil_log2(2 ** k - 1) for k in ks], ":", color="0.6", label="ceil(log2(2^k − 1))")
        right.plot(ks, [row["best"] for _, row in fed], "o-", color="tab:red", label="value")
        right.set_xlabel("k (cycle length)")
        right.set_title("The cycle fed from the source at every vertex")
        right.set_xticks(ks)
        right.grid(alpha=0.3)
        right.legend(loc="upper left")
        print("fed cycle: " + ", ".join(f"k={k}: {row['best']}" for k, row in fed))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
