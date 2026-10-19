#!/usr/bin/env python3
"""Check the unfed-vertex inequality against the recorded game table.

Let `v` be a non-source vertex with exactly one in-arc `u>v`, where `u` is
not the source and there is no arc `v>u`. Contracting `u>v` (every arc out
of `v` leaves `u` instead) gives a rooted topology `G/v` on one vertex fewer,
and in either strict game

    V(G) >= V(G/v) + 1.

The argument: weight `u>v` at a scale below every other weight; a program for
`G` run on such weights is a program for `G/v` once every register is
projected onto the remaining coordinates, comparisons that differ only in the
small weight being answered by its sign. At the leaf the held form for `d(v)`
is the held form for `d(u)` plus the small weight, so the two project to the
same form, and the addition that created the later of them is free in `G/v`.

The values are read from the `RECORDED` table of the game solver
(`crates/unsorted-game/src/main.rs`): every rooted topology on at most four
vertices with at most six arcs, one per isomorphism class, with its
path-register and strict-adversary values. A topology whose arcs all leave
the source has value 0; the table omits the one with a single arc. For every
listed topology and every vertex meeting the hypothesis whose contraction
creates no parallel arc, the script prints both values in each game and the
slack `V(G) - V(G/v) - 1`, which must be non-negative.

Usage: check_lemma_u.py PATH_TO_MAIN_RS
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

MODES = ("path", "generic")


def parse(arcs: str) -> list[tuple[int, int]]:
    return [tuple(map(int, a.split(">"))) for a in arcs.split(",")]


def text(arcs) -> str:
    return ",".join(f"{a}>{b}" for a, b in arcs)


def canon(arcs) -> str:
    """The least arc list over relabellings that fix the source."""
    verts = sorted({x for a in arcs for x in a if x != 0})
    best = None
    for perm in itertools.permutations(range(1, len(verts) + 1)):
        relabel = {0: 0, **dict(zip(verts, perm))}
        key = text(sorted((relabel[a], relabel[b]) for a, b in arcs))
        if best is None or key < best:
            best = key
    return best


def read_table(source: str) -> dict[str, tuple[int | None, int | None]]:
    def val(s: str) -> int | None:
        return None if s == "None" else int(s[5:-1])

    rows = re.findall(r'\("([0-9>,]+)", (Some\(\d+\)|None), (Some\(\d+\)|None)\)', source)
    return {arcs: (val(p), val(g)) for arcs, p, g in rows}


def value(table, arcs, mode: int) -> int | None:
    if all(a == 0 for a, _ in arcs):
        return 0
    entry = table.get(canon(arcs))
    return None if entry is None else entry[mode]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    table = read_table(Path(argv[0]).read_text(encoding="utf-8"))
    checked = [0, 0]
    failures = [0, 0]
    tight = [0, 0]
    undecided = [0, 0]
    instances = 0
    print("# G | v | G/v | " + " | ".join(f"{m}: V(G) V(G/v) slack" for m in MODES))
    for key in table:
        arcs = parse(key)
        for v in sorted({x for a in arcs for x in a} - {0}):
            into = [a for a, b in arcs if b == v]
            if len(into) != 1 or into[0] == 0:
                continue
            u = into[0]
            out_v = [b for a, b in arcs if a == v]
            out_u = [b for a, b in arcs if a == u]
            if u in out_v or set(out_v) & set(out_u):
                continue
            small = [(a, b) for a, b in arcs if v not in (a, b)] + [(u, b) for b in out_v]
            instances += 1
            cells = []
            for mode in range(len(MODES)):
                big, less = value(table, arcs, mode), value(table, small, mode)
                if big is None or less is None:
                    undecided[mode] += 1
                    cells.append(f"{big} {less} undecided")
                    continue
                slack = big - less - 1
                checked[mode] += 1
                failures[mode] += slack < 0
                tight[mode] += slack == 0
                cells.append(f"{big} {less} {slack}" + (" FAILS" if slack < 0 else ""))
            print(f"{key} | {v} | {canon(small)} | " + " | ".join(cells))
    print(f"# instances {instances}")
    for mode, name in enumerate(MODES):
        print(
            f"# {name}: checked {checked[mode]}, failures {failures[mode]}, "
            f"tight {tight[mode]}, undecided {undecided[mode]}"
        )
    return 1 if any(failures) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
