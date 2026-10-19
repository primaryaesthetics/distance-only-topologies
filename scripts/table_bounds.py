#!/usr/bin/env python3
"""Corner and acyclic-envelope bounds over recorded `unsorted-game table` runs.

Reads the standard output of `unsorted-game table` runs (rows
`arcs | n m dag d_s fwd law | path nodes lp seconds`) and, over the rows the
path-register game decided, prints the largest value minus the topology's
corner `fwd + (m - n + 1)` and the largest value minus the acyclic envelope,
the largest acyclic value with the same `n` and `m` (Theorem 4.1 of Cai,
arXiv:2609.04825, at the least source degree an acyclic topology with those
counts can have; none exists when `m` exceeds `C(n - 1, 2) + n - 1`). A
path-register value is an upper bound on `OPT_DIST`, so both maxima bound
`OPT_DIST` from above as well.

With `--relabel FILE`, also reads arc sets, one per line as the first
`|`-separated field (lines starting with `#` or not starting with a digit are
skipped), and prints for each one every table row that is a relabelling of
its non-source vertices, with that row's value.

Usage: table_bounds.py [--relabel FILE] <table-stdout.txt>...
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path


def parse_arcs(text: str) -> tuple[int, frozenset[tuple[int, int]]]:
    arcs = frozenset(tuple(int(x) for x in a.split(">")) for a in text.split(","))
    n = 1 + max(max(a) for a in arcs)
    return n, arcs


def canonical(n: int, arcs: frozenset[tuple[int, int]]) -> tuple:
    """The least sorted arc list over every relabelling fixing the source 0."""
    best = None
    for perm in permutations(range(1, n)):
        image = (0,) + perm
        key = tuple(sorted((image[u], image[v]) for u, v in arcs))
        if best is None or key < best:
            best = key
    return (n, best)


def envelope(n: int, m: int) -> int | None:
    pairs = (n - 1) * (n - 2) // 2
    if m > pairs + (n - 1):
        return None
    return (m - n + 1) + (m - max(1, m - pairs))


def read_tables(paths: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            n, m, dag, _ds, fwd, _law = cells[1].split()
            head = cells[2].split()[0]
            value = int(head) if head.lstrip("-").isdigit() else None
            row = {"n": int(n), "m": int(m), "dag": dag == "1", "fwd": int(fwd), "value": value}
            kept = rows.get(cells[0])
            if kept is not None and kept["value"] is not None and value is not None:
                assert kept["value"] == value, (cells[0], kept["value"], value)
            if kept is None or kept["value"] is None:
                rows[cells[0]] = row
    return rows


def main(argv: list[str]) -> int:
    relabel = None
    if argv[:1] == ["--relabel"]:
        relabel, argv = Path(argv[1]), argv[2:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    rows = read_tables([Path(p) for p in argv])
    by_n: dict[int, list[tuple[str, dict]]] = {}
    for arcs, r in rows.items():
        by_n.setdefault(r["n"], []).append((arcs, r))
    for n in sorted(by_n):
        group = by_n[n]
        decided = [(a, r) for a, r in group if r["value"] is not None]
        print(f"n={n}: {len(group)} rows, {len(decided)} decided")
        best_corner, best_env = None, None
        no_env = 0
        for arcs, r in sorted(decided):
            corner = r["fwd"] + (r["m"] - n + 1)
            above = r["value"] - corner
            if best_corner is None or above > best_corner[0]:
                best_corner = (above, arcs, r["value"], corner)
            env = envelope(n, r["m"])
            if env is None:
                no_env += 1
                continue
            excess = r["value"] - env
            if best_env is None or excess > best_env[0]:
                best_env = (excess, arcs, r["value"], env)
        counts: dict[int, int] = {}
        for arcs, r in decided:
            above = r["value"] - (r["fwd"] + r["m"] - n + 1)
            counts[above] = counts.get(above, 0) + 1
        print(f"  largest value - corner: {best_corner[0]:+d}  {best_corner[1]}  "
              f"(value {best_corner[2]}, corner {best_corner[3]})")
        print("  value - corner, rows: " + ", ".join(f"{k:+d}: {counts[k]}" for k in sorted(counts)))
        if best_env is not None:
            print(f"  largest value - envelope: {best_env[0]:+d}  {best_env[1]}  "
                  f"(value {best_env[2]}, envelope {best_env[3]})")
        print(f"  decided rows with no acyclic topology of their size: {no_env}")
    if relabel is not None:
        index: dict[tuple, list[str]] = {}
        for arcs, r in rows.items():
            n, s = parse_arcs(arcs)
            index.setdefault(canonical(n, s), []).append(arcs)
        print()
        print(f"relabellings of the arc sets of {relabel.as_posix()} among the table rows")
        found = 0
        for line in relabel.read_text(encoding="utf-8").splitlines():
            if not line[:1].isdigit():
                continue
            arcs = line.split("|")[0].strip()
            n, s = parse_arcs(arcs)
            hits = index.get(canonical(n, s), [])
            for h in hits:
                v = rows[h]["value"]
                print(f"  {arcs}  ~  {h}  value {'-' if v is None else v}")
                found += 1
            if not hits:
                print(f"  {arcs}  ~  none")
        print(f"# arc sets with a relabelling among the rows: {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
