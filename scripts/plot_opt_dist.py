#!/usr/bin/env python3
"""Draw the exact distance-only cost of every small topology against its size.

Reads one or more tables printed by `theory/game/opt_dist.py enumerate` and
draws, for each topology, its exact `OPT_DIST` (Cai, arXiv:2609.04825,
Theorem J.3) against its arc count `m`, acyclic and cyclic topologies in
different marks, with the acyclic law `(m − n + 1) + (m − d_s)` of Theorem 4.1
drawn as the envelope of the DAGs with the same `n` at each `m`: the law at
the least source degree a DAG on `n` vertices with `m` arcs can have, which
is `max(1, m − C(n − 1, 2))`, and which exists only while `m ≤ C(n, 2)`. The
vertical distance of a cyclic mark above the envelope is the excess a cycle
costs, and whether that excess moves with `m` is the question the picture is
drawn to answer. A value of the unrestricted strict game is `OPT_DIST` itself:
a strategy that wins against the strict adversary is a program correct on
every weighting once each tie is broken toward one fixed generic weighting,
so the strict value is both a lower and an upper bound. A topology whose
unrestricted strict game was not decided is drawn hollow, as a triangle at
its path-register value, which is an upper bound on `OPT_DIST`, or as a
circle at the largest budget refuted, a lower bound.

Prints the excess table the plot is drawn from, so a recorded run carries the
numbers as well as the picture.

Usage: plot_opt_dist.py <out.png> <table-stdout.txt>...
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse(path: Path) -> list[dict]:
    """Rows of an enumeration table. The header names the modes present, one
    column each; a cell is a value or `>=k` when the search timed out."""
    rows = []
    modes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# arcs"):
            modes = [c.strip().split()[0] for c in line.split("|")[2:]]
            continue
        if not line or line.startswith("#") or "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        arcs, meta, results = cells[0], cells[1], cells[2:]
        n, m, dag, ds, fwd, law = meta.split()
        values: dict[str, int | None] = {}
        bounds: dict[str, int] = {}
        for mode, cell in zip(modes, results):
            head = cell.split()[0]
            if head.startswith(">="):
                values[mode], bounds[mode] = None, int(head[2:])
            else:
                values[mode] = bounds[mode] = int(head)
        row = {
            "arcs": arcs, "n": int(n), "m": int(m), "dag": dag == "1",
            "ds": int(ds), "fwd": int(fwd), "law": None if law == "-" else int(law),
            "values": values, "bounds": bounds,
        }
        classify(row)
        rows.append(row)
    return rows


def classify(row: dict) -> None:
    """Exact when the full game or the unrestricted strict game finished;
    otherwise an upper bound from the path-register game, or a lower bound
    from a refuted budget. A strict game that timed out at a bound no lower
    than the path-register value has that value, since the strict program
    may hold what the path-register one holds."""
    values, bounds = row["values"], row["bounds"]
    if values.get("generic") is None and values.get("path") is not None             and bounds.get("generic", -1) >= values["path"]:
        values["generic"] = bounds["generic"] = values["path"]
    exact = values.get("full")
    if exact is None:
        exact = values.get("generic")
    shown = None
    kind = "exact"
    if exact is None:
        kind = "path"
        for mode in ("path", "pathfull"):
            if values.get(mode) is not None:
                shown = values[mode]
                break
        if shown is None:
            shown = max(bounds.values())
            kind = "bound"
    row["exact"] = exact
    row["shown"] = exact if exact is not None else shown
    row["kind"] = kind


def merge(rows: list[dict]) -> list[dict]:
    """One row per topology across tables: a topology played in different
    modes by different runs gets the union of its values, and its exactness
    is judged on the union."""
    by_arcs: dict[str, dict] = {}
    for r in rows:
        if r["arcs"] not in by_arcs:
            by_arcs[r["arcs"]] = r
            continue
        kept = by_arcs[r["arcs"]]
        for mode, value in r["values"].items():
            if value is not None or mode not in kept["values"]:
                kept["values"][mode] = value
            kept["bounds"][mode] = max(kept["bounds"].get(mode, -1), r["bounds"][mode])
    for r in by_arcs.values():
        classify(r)
    return list(by_arcs.values())


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    rows = merge([r for p in argv[1:] for r in parse(Path(p))])

    # The envelope by (n, m), from the law at the least source degree; every
    # DAG row present must sit at or below it, and the largest one at it.
    def law_envelope(n: int, m: int) -> int | None:
        pairs = (n - 1) * (n - 2) // 2
        if m > pairs + (n - 1):
            return None
        return (m - n + 1) + (m - max(1, m - pairs))

    seen_dag: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["dag"]:
            key = (r["n"], r["m"])
            seen_dag[key] = max(seen_dag.get(key, 0), r["law"])
    for (n, m), law in seen_dag.items():
        assert law == law_envelope(n, m), (n, m, law, law_envelope(n, m))

    # Two baselines. The envelope is the largest acyclic value at the same
    # `m`. The corner is `fwd + (m - n + 1)`: the topology's own addition
    # minimum (Theorem 3.1) plus the comparison count every acyclic topology
    # pays (Theorem 4.1), which every DAG attains exactly, so the distance
    # above it is what a cycle costs beyond its own coordinate minima.
    print("cyclic topologies against the acyclic envelope at the same m and against fwd + (m - n + 1)")
    print(f"{'arcs':40s} {'n':>2s} {'m':>2s} {'fwd':>3s} {'value':>6s} {'env':>4s} {'excess':>6s} "
          f"{'corner':>6s} {'above':>6s}")
    worst: dict[int, tuple[int, str]] = {}
    worst_corner: dict[int, tuple[int, str]] = {}
    for r in sorted(rows, key=lambda r: (r["m"], r["dag"], r["arcs"])):
        r["corner"] = r["fwd"] + (r["m"] - r["n"] + 1)
        if r["dag"]:
            continue
        value = r["shown"]
        tag = "" if r["kind"] == "exact" else f" ({r['kind']})"
        env = law_envelope(r["n"], r["m"])
        above = value - r["corner"]
        if env is None:
            env_s, excess_s = "none", ""
        else:
            excess = value - env
            env_s, excess_s = str(env), f"{excess:+d}"
            if r["m"] not in worst or excess > worst[r["m"]][0]:
                worst[r["m"]] = (excess, r["arcs"])
        print(f"{r['arcs']:40s} {r['n']:2d} {r['m']:2d} {r['fwd']:3d} {value:6d} "
              f"{env_s:>4s} {excess_s:>6s} {r['corner']:6d} {above:+6d}{tag}")
        if r["m"] not in worst_corner or above > worst_corner[r["m"]][0]:
            worst_corner[r["m"]] = (above, r["arcs"])
    print()
    print("largest excess over the acyclic envelope, by m")
    for m in sorted(worst):
        print(f"  m={m}: {worst[m][0]:+d}  {worst[m][1]}")
    print("largest distance above fwd + (m - n + 1), by m")
    for m in sorted(worst_corner):
        print(f"  m={m}: {worst_corner[m][0]:+d}  {worst_corner[m][1]}")
    dag_off = [r for r in rows if r["dag"] and r["shown"] != r["corner"]]
    print(f"acyclic topologies off their corner: {len(dag_off)}")

    # The panels are stacked so that, at the width of a printed page, the
    # counts beside the marks stay as large as the text around the figure.
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 12))

    # One mark per (arc count, value, kind), its area growing with the number
    # of topologies it stands for and that number written beside it, so a
    # column of many topologies stays legible and countable. Acyclic marks sit
    # left of the arc count, exact cyclic values on it, and path-register
    # upper bounds right of it.
    offset = {"dag": -0.24, "exact": 0.0, "path": 0.24, "bound": 0.24}

    def draw(axis, y_of, labelled: bool) -> None:
        counts: dict = defaultdict(int)
        for r in rows:
            kind = "dag" if r["dag"] else r["kind"]
            counts[(r["m"], y_of(r), kind)] += 1
        for (m, y, kind), c in sorted(counts.items()):
            x = m + offset[kind]
            size = 3.5 + 2.2 * c ** 0.5
            if kind == "dag":
                axis.plot(x, y, "s", color="#4c72b0", ms=size, alpha=0.85)
            elif kind == "exact":
                axis.plot(x, y, "o", color="#c44e52", ms=size, alpha=0.85)
            elif kind == "path":
                axis.plot(x, y, "^", mfc="white", mec="#c44e52", mew=1.4, ms=size)
            else:
                axis.plot(x, y, "o", mfc="white", mec="#c44e52", mew=1.4, ms=size)
            if labelled and c > 1:
                # Counts of neighbouring kinds at one arc count would touch if
                # all sat above their marks: acyclic counts go below, exact
                # counts above, path-register counts to the right.
                if kind == "dag":
                    xy, ha, va = (0, -(size / 2 + 2)), "center", "top"
                elif kind == "exact":
                    xy, ha, va = (0, size / 2 + 2), "center", "bottom"
                else:
                    xy, ha, va = (size / 2 + 2, 0), "left", "center"
                axis.annotate(str(c), (x, y), xytext=xy, textcoords="offset points",
                              ha=ha, va=va, fontsize=12, color="#333333")

    # Values one apart crowd the upper panel, so only the lower one, where the
    # rows are fewer, carries the counts.
    draw(ax2, lambda r: r["shown"] - r["corner"], True)
    ax2.axhline(0, color="#4c72b0", lw=1.2)
    ax2.set_xlabel("arcs $m$")
    ax2.set_ylabel(r"value $-\ (\mathrm{fwd} + m - n + 1)$")
    ax2.set_title("Distance above the corner")
    ax2.grid(alpha=0.3)
    draw(ax, lambda r: r["shown"], False)
    styles = {3: ":", 4: "--", 5: "-"}
    for n in sorted({r["n"] for r in rows}):
        ms = sorted({r["m"] for r in rows if r["n"] == n and law_envelope(n, r["m"]) is not None})
        if ms:
            ax.plot(ms, [law_envelope(n, m) for m in ms], styles.get(n, "-"), color="#4c72b0",
                    lw=1.4, label=f"acyclic envelope, $n = {n}$")
    ax.plot([], [], "s", color="#4c72b0", label="acyclic topology")
    ax.plot([], [], "o", color="#c44e52", label=r"cyclic, $\mathrm{OPT}_{\mathrm{DIST}}$ exact")
    ax.plot([], [], "^", mfc="white", mec="#c44e52", label=r"cyclic, $V^{\mathrm{path}}$ (upper bound)")
    if any(r["kind"] == "bound" for r in rows):
        ax.plot([], [], "o", mfc="white", mec="#c44e52", label="cyclic, refuted budget (lower bound)")
    ax.set_xlabel("arcs $m$")
    ax.set_ylabel("value, charged operations")
    ax.set_title("Value of every topology, by size")
    ax.set_xticks(sorted({r["m"] for r in rows}))
    ax2.set_xticks(sorted({r["m"] for r in rows}))
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
