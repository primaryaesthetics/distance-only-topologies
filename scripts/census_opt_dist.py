#!/usr/bin/env python3
"""Census of recorded OPT_DIST tables, read from files under experiments/.

    census_opt_dist.py plot FILE...
    census_opt_dist.py coverage N MMAX FILE...

`plot` reads the stdout of `scripts/plot_opt_dist.py`, whose rows give for
each cyclic topology `n`, `m`, `fwd`, the value shown, the acyclic envelope at
the same `n` and `m` (or `none`), the excess over it, the corner
`fwd + (m - n + 1)`, the distance above the corner, and how the value was
obtained (blank for exact, `(strict)` for a strict-adversary value of the
unrestricted program, `(path)` for a path-register value alone). For every
pair `(n, m)` it prints the number of cyclic topologies, how their values
were obtained, the largest and smallest distance above the corner and excess
over the envelope, and the topologies attaining the extremes; then the same
extremes over each `n` and over the whole file.

`coverage` enumerates every rooted topology on `N` vertices with at most
`MMAX` arcs (no arc into the source, no self-loop, every vertex reachable),
one per relabelling of the non-source vertices, and reads values from
enumeration tables of `unsorted-game` or `theory/game/opt_dist.py` (a `# arcs`
header naming one column per mode) and from tables of
`scripts/check_contract_lemma.py` (a `# G | v | V(G) | G' | V(G')` header
naming the mode). Only the modes `path` and `generic` are read, and only
decided values; a cell `>=k` is a search stopped at its budget. For every arc
count it prints how many topologies have a decided path-register value, a
decided unrestricted value, both, and on how many of those the two agree;
then, for every arc count, how many topologies are acyclic and how many have
a cycle, and how many of each have a row in the files, decided or not; and
lists the topologies that lack either value.
"""

from __future__ import annotations

import collections
import itertools
import sys
from pathlib import Path


def canon(n: int, arcs) -> tuple[tuple[int, int], ...]:
    """The least sorted arc tuple over relabellings fixing the source."""
    best = None
    for p in itertools.permutations(range(1, n)):
        relabel = {0: 0, **{i + 1: p[i] for i in range(n - 1)}}
        key = tuple(sorted((relabel[a], relabel[b]) for a, b in arcs))
        if best is None or key < best:
            best = key
    return best


def text(arcs) -> str:
    return ",".join(f"{a}>{b}" for a, b in arcs)


def parse_arcs(field: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in a.split(">")) for a in field.strip().split(",")]


# ---------------------------------------------------------------- plot ----

def plot_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("0>"):
            continue
        p = line.split()
        if p[5] == "none":
            env, excess, corner, above, rest = None, None, int(p[6]), int(p[7]), p[8:]
        else:
            env, excess, corner, above, rest = int(p[5]), int(p[6]), int(p[7]), int(p[8]), p[9:]
        rows.append({
            "arcs": p[0], "n": int(p[1]), "m": int(p[2]), "fwd": int(p[3]),
            "value": int(p[4]), "env": env, "excess": excess, "corner": corner,
            "above": above, "kind": rest[0].strip("()") if rest else "exact",
        })
    return rows


def describe(r: dict) -> str:
    env = "none" if r["env"] is None else r["env"]
    return f"{r['arcs']} {r['kind']} {r['value']} env {env} corner {r['corner']}"


def extremes(rows: list[dict], key: str) -> str:
    present = [r for r in rows if r[key] is not None]
    if not present:
        return f"{key}: none"
    hi = max(r[key] for r in present)
    lo = min(r[key] for r in present)
    at_hi = [describe(r) for r in present if r[key] == hi]
    at_lo = [describe(r) for r in present if r[key] == lo]
    return (f"{key}: max {hi:+d} on {len(at_hi)} ({'; '.join(at_hi[:4])}{'; ...' if len(at_hi) > 4 else ''}), "
            f"min {lo:+d} on {len(at_lo)} ({'; '.join(at_lo[:4])}{'; ...' if len(at_lo) > 4 else ''})")


def report_plot(path: Path) -> None:
    rows = plot_rows(path)
    print(f"== {path.as_posix()}: {len(rows)} cyclic topologies")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["n"], r["m"])].append(r)
    print("n  m  cyclic  exact strict path | above corner: max min | excess over envelope: max min")
    for (n, m), rs in sorted(groups.items()):
        kinds = collections.Counter(r["kind"] for r in rs)
        ab = [r["above"] for r in rs]
        ex = [r["excess"] for r in rs if r["excess"] is not None]
        exs = f"{max(ex):+d} {min(ex):+d}" if ex else "none"
        print(f"{n}  {m}  {len(rs):6d}  {kinds['exact']:5d} {kinds['strict']:6d} {kinds['path']:4d} | "
              f"{max(ab):+d} {min(ab):+d} | {exs}")
    print()
    for (n, m), rs in sorted(groups.items()):
        print(f"n={n} m={m}")
        print(f"  {extremes(rs, 'above')}")
        print(f"  {extremes(rs, 'excess')}")
        hist = sorted(collections.Counter(r["above"] for r in rs).items())
        print(f"  distance above corner, count: {', '.join(f'{a:+d}: {c}' for a, c in hist)}")
    print()
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        by_m = []
        for m in sorted({r["m"] for r in rs}):
            by_m.append(f"m={m} {max(r['above'] for r in rs if r['m'] == m):+d}")
        print(f"n={n}: largest distance above corner by m: {', '.join(by_m)}")
        print(f"  {extremes(rs, 'above')}")
        print(f"  {extremes(rs, 'excess')}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("acyclic topologies off their corner"):
            print(f"{line} (as printed in the file)")
    print("all:")
    print(f"  {extremes(rows, 'above')}")
    print(f"  {extremes(rows, 'excess')}")
    print()


# ------------------------------------------------------------ coverage ----

def rooted_topologies(n: int, mmax: int) -> dict[tuple, int]:
    pairs = [(a, b) for a in range(n) for b in range(1, n) if a != b]
    found: dict[tuple, int] = {}
    for m in range(n - 1, mmax + 1):
        for arcs in itertools.combinations(pairs, m):
            seen, stack = {0}, [0]
            while stack:
                x = stack.pop()
                for a, b in arcs:
                    if a == x and b not in seen:
                        seen.add(b)
                        stack.append(b)
            if len(seen) == n:
                found.setdefault(canon(n, arcs), m)
    return found


def is_acyclic(n: int, arcs) -> bool:
    indeg = [0] * n
    for _, b in arcs:
        indeg[b] += 1
    queue = [v for v in range(n) if indeg[v] == 0]
    seen = 0
    while queue:
        x = queue.pop()
        seen += 1
        for a, b in arcs:
            if a == x:
                indeg[b] -= 1
                if indeg[b] == 0:
                    queue.append(b)
    return seen == n


def read_values(path: Path, n: int, values: dict, listed: set | None = None) -> None:
    """Adds (topology, mode) -> {value: [file]} for decided values on n vertices,
    and to `listed` every topology on n vertices with a row, decided or not."""
    modes: list[str] = []
    contract_mode = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# arcs"):
            modes = [c.strip().split()[0] for c in line.split("|")[2:]]
            continue
        if line.startswith("# G | v | V(G)"):
            contract_mode = line.split("(")[-1].split(",")[0].strip()
            continue
        if not line or line.startswith("#") or "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if contract_mode is not None:
            pairs = [(cells[0], cells[2]), (cells[3], cells[4])]
            entries = [(arcs, contract_mode, cell) for arcs, cell in pairs]
        elif modes:
            entries = [(cells[0], mode, cell) for mode, cell in zip(modes, cells[2:])]
        else:
            continue
        for arcs_text, mode, cell in entries:
            if mode not in ("path", "generic"):
                continue
            arcs = parse_arcs(arcs_text)
            if len(set(arcs)) != len(arcs):
                continue
            if 1 + max(max(a, b) for a, b in arcs) != n:
                continue
            if listed is not None:
                listed.add(canon(n, arcs))
            head = cell.split()[0] if cell.split() else ""
            if not head.isdigit():
                continue
            values[(canon(n, arcs), mode)].setdefault(int(head), []).append(path.parent.name)


def report_coverage(n: int, mmax: int, files: list[Path]) -> None:
    tops = rooted_topologies(n, mmax)
    values: dict = collections.defaultdict(dict)
    listed: set = set()
    for f in files:
        read_values(f, n, values, listed)
    conflicts = [(k, v) for k, v in values.items() if len(v) > 1]
    print(f"== coverage on {n} vertices, at most {mmax} arcs, {len(tops)} rooted topologies")
    print(f"files: {len(files)}; conflicting decided values: {len(conflicts)}")
    for (top, mode), v in conflicts:
        print(f"  CONFLICT {text(top)} {mode} {v}")
    print("m  topologies  path  generic  both  agree")
    totals = [0, 0, 0, 0, 0]
    missing = []
    for m in sorted(set(tops.values())):
        ts = [t for t, mm in tops.items() if mm == m]
        p = [t for t in ts if values.get((t, "path"))]
        g = [t for t in ts if values.get((t, "generic"))]
        both = [t for t in ts if t in p and t in g]
        agree = [t for t in both if set(values[(t, "path")]) == set(values[(t, "generic")])]
        row = [len(ts), len(p), len(g), len(both), len(agree)]
        totals = [a + b for a, b in zip(totals, row)]
        print(f"{m}  {row[0]:10d}  {row[1]:4d}  {row[2]:7d}  {row[3]:4d}  {row[4]:5d}")
        for t in ts:
            if t not in both:
                pv = ",".join(map(str, values.get((t, "path"), {}))) or "-"
                gv = ",".join(map(str, values.get((t, "generic"), {}))) or "-"
                missing.append(f"  m={m} {text(t)}  path {pv}  generic {gv}")
    print(f"all  {totals[0]:8d}  {totals[1]:4d}  {totals[2]:7d}  {totals[3]:4d}  {totals[4]:5d}")
    print("m  acyclic  cyclic  cyclic-with-a-row  acyclic-with-a-row")
    for m in sorted(set(tops.values())):
        ts = [t for t, mm in tops.items() if mm == m]
        acyc = [t for t in ts if is_acyclic(n, t)]
        cyc = [t for t in ts if t not in acyc]
        print(f"{m}  {len(acyc):7d}  {len(cyc):6d}  {sum(t in listed for t in cyc):17d}  "
              f"{sum(t in listed for t in acyc):18d}")
    print("topologies without both decided values:")
    print("\n".join(missing) if missing else "  none")
    print("sources of decided unrestricted values, by arc count:")
    for m in sorted(set(tops.values())):
        src = collections.Counter()
        for t, mm in tops.items():
            if mm == m and values.get((t, "generic")):
                for files_ in values[(t, "generic")].values():
                    for name in sorted(set(files_)):
                        src[name] += 1
        print(f"  m={m}: " + ", ".join(f"{k} {c}" for k, c in sorted(src.items())))
    print()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "plot" and len(argv) > 1:
        for f in argv[1:]:
            report_plot(Path(f))
        return 0
    if argv[0] == "coverage" and len(argv) > 3:
        report_coverage(int(argv[1]), int(argv[2]), [Path(f) for f in argv[3:]])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
