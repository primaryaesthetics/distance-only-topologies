#!/usr/bin/env python3
"""Exact checks on the fed cycle C_k, with z3 deciding every cone over Q.

C_k has source 0, arcs 0>i of weight b_i and cycle arcs i>i+1 (mod k) of
weight c_i. A transcript is a list of strict rows r.w > 0; its cone is the set
of positive weightings satisfying all of them. The single-entry cell E_s is
the set of weightings at which vertex s alone is reached by its own source
arc; it is cut out by the k - 1 facets

    f_j = b_j - walk_s(j) > 0,   walk_s(j) = b_s + c_s + ... + c_{j-1},  j != s.

Nothing here shares code with the Rust solver `unsorted-fedcycle`: forms are
coefficient tuples, cones are decided by z3, and a leaf is ready when every
vertex has a held form that is at most every path form to it on the whole
cone, which is the game's definition and does not mention cells.

Adversaries, as restrictions of the strict one:
    full       every answer whose cone is nonempty
    single     those after which some single-entry cell still meets the cone
    min-entry  those keeping the most single-entry cells

Subcommands:

    tree FILE K BUDGET [--adversary=A] [--branch] [--expect=ok|fail]
        Plays a strategy file (the output of `unsorted-fedcycle --print`, or
        written by hand: `compare X ? Y`, `-> X > Y`, `add X + Y = Z`, `ready`,
        two spaces of indentation per level, `#` lines ignored). Without
        --branch every answer the adversary may give must be covered; with it
        every answer in the file must be one the adversary may give. Every
        addition must hold its operands and lie below a path, and every leaf
        must be ready at depth at most BUDGET. Exits 0 when the verdict is the
        expected one.

    chain K... [--plot=PNG]
        The branch b_1 > b_2, b_2 > b_3, ..., b_{k-1} > b_k, the walk from k
        (k - 1 additions), and b_{k-1} > walk_k(k-1): 2k - 1 operations. Checks
        it in branch mode against min-entry and against full, and checks that
        four mutants of it are rejected.

    forced-loss K... [--plot=PNG]
        For each k: the most single-entry cells a comparison of two source
        arcs removes whichever way it is answered, at the root and after the
        rows c_i > b_i for every i, each of which is a tie for the min-entry
        adversary; the four-cycle instance c_1 > b_1, c_3 > b_3, b_1 ? b_3; and
        the order of the source arcs these rows force inside every E_s.

    points FILE K
        Reads the `# point E_s w...` lines a `points` run prints and checks
        in integer arithmetic that each weighting lies strictly inside E_s and
        that no two distinct forms below a path tie at it.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from z3 import Real, Solver, sat, unsat

# ---------------------------------------------------------------- forms


class Cycle:
    """Forms and cones of C_k; arc i < k is b_{i+1}, arc k + i is c_{i+1}."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.m = 2 * k
        self.w = [Real(f"w{i}") for i in range(self.m)]
        self.paths = {v: [self.walk(e, v) for e in range(k)] for v in range(k)}
        self.path_set = {p for ps in self.paths.values() for p in ps}
        self.programs = 0

    def walk(self, s: int, j: int) -> tuple[int, ...]:
        """The path entering the cycle at s and ending at j."""
        v = [0] * self.m
        v[s] = 1
        u = s
        while u != j:
            v[self.k + u] = 1
            u = (u + 1) % self.k
        return tuple(v)

    def unit(self, i: int) -> tuple[int, ...]:
        v = [0] * self.m
        v[i] = 1
        return tuple(v)

    def parse(self, text: str) -> tuple[int, ...]:
        v = [0] * self.m
        if text.strip() == "0":
            return tuple(v)
        for term in text.split("+"):
            term = term.strip()
            index = int(term[1:]) - 1
            v[index if term[0] == "b" else self.k + index] += 1
        return tuple(v)

    def show(self, v: tuple[int, ...]) -> str:
        parts = [f"b{i + 1}" for i in range(self.k) if v[i]]
        parts += [f"c{i + 1}" for i in range(self.k) if v[self.k + i]]
        return "+".join(parts) or "0"

    def below_path(self, z: tuple[int, ...]) -> bool:
        return all(c in (0, 1) for c in z) and any(
            all(a <= b for a, b in zip(z, p)) for p in self.path_set
        )

    def feasible(self, rows: list[tuple[int, ...]]) -> bool:
        s = Solver()
        for x in self.w:
            s.add(x > 0)
        for r in rows:
            s.add(sum(c * self.w[i] for i, c in enumerate(r) if c) > 0)
        self.programs += 1
        verdict = s.check()
        assert verdict in (sat, unsat)
        return verdict == sat

    def cell_rows(self, s: int) -> list[tuple[int, ...]]:
        return [diff(self.unit(j), self.walk(s, j)) for j in range(self.k) if j != s]

    def alive(self, tau: list[tuple[int, ...]]) -> list[int]:
        """The single-entry cells meeting the cone of tau, 0-based."""
        return [s for s in range(self.k) if self.feasible(tau + self.cell_rows(s))]

    def ready(self, held: frozenset, tau: list[tuple[int, ...]]) -> bool:
        for v in range(self.k):
            if not any(
                z in held
                and all(not self.feasible(tau + [diff(z, p)]) for p in self.paths[v] if p != z)
                for z in self.paths[v]
            ):
                return False
        return True


def diff(x: tuple[int, ...], y: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(x, y))


# ---------------------------------------------------------------- trees


def parse_tree(text: str, cyc: Cycle):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append(((len(raw) - len(raw.lstrip(" "))) // 2, raw.strip()))
    pos = 0

    def node(depth: int):
        nonlocal pos
        indent, t = lines[pos]
        if indent != depth:
            raise ValueError(f"indentation {indent} where {depth} was expected: {t}")
        pos += 1
        if t.startswith("ready"):
            return ("ready",)
        if t.startswith("add "):
            lhs, z = t[4:].split(" = ")
            x, y = lhs.split(" + ")
            return ("add", cyc.parse(x), cyc.parse(y), cyc.parse(z), node(depth + 1))
        if t.startswith("compare "):
            x, y = (cyc.parse(f) for f in t[8:].split(" ? "))
            kids = {}
            while pos < len(lines) and lines[pos][0] == depth and lines[pos][1].startswith("->"):
                larger = cyc.parse(lines[pos][1][3:].split(" > ")[0])
                pos += 1
                kids[larger] = node(depth + 1)
            return ("cmp", x, y, kids)
        raise ValueError(t)

    tree = node(0)
    if pos != len(lines):
        raise ValueError(f"unparsed lines from {lines[pos][1]!r}")
    return tree


class Player:
    def __init__(self, cyc: Cycle, adversary: str, branch: bool) -> None:
        self.cyc = cyc
        self.adversary = adversary
        self.branch = branch
        self.leaves = 0
        self.depth = 0

    def allowed(self, tau, x, y):
        d = diff(x, y)
        answers = [(big, row) for big, row in ((x, d), (y, diff(y, x))) if self.cyc.feasible(tau + [row])]
        assert answers, "the cone is empty"
        if len(answers) == 2 and self.adversary != "full":
            kept = [len(self.cyc.alive(tau + [row])) for _, row in answers]
            if self.adversary == "single":
                answers = [a for a, n in zip(answers, kept) if n >= 1]
            else:
                answers = [a for a, n in zip(answers, kept) if n == max(kept)]
        return answers

    def play(self, node, held: frozenset, tau, depth: int) -> None:
        if node[0] == "ready":
            self.leaves += 1
            self.depth = max(self.depth, depth)
            if not self.cyc.ready(held, tau):
                raise AssertionError(f"leaf at depth {depth} is not ready")
            return
        if node[0] == "add":
            _, x, y, z, child = node
            if x not in held or y not in held:
                raise AssertionError(f"operand not held at depth {depth}")
            if tuple(a + b for a, b in zip(x, y)) != z:
                raise AssertionError(f"wrong sum at depth {depth}")
            if not self.cyc.below_path(z):
                raise AssertionError(f"{self.cyc.show(z)} lies below no path")
            self.play(child, held | {z}, tau, depth + 1)
            return
        _, x, y, kids = node
        if x not in held or y not in held:
            raise AssertionError(f"compared form not held at depth {depth}")
        answers = dict(self.allowed(tau, x, y))
        if self.branch:
            for big, child in kids.items():
                if big not in answers:
                    raise AssertionError(
                        f"answer {self.cyc.show(big)} larger is not one {self.adversary} may give at depth {depth}"
                    )
                self.play(child, held, tau + [answers[big]], depth + 1)
            return
        for big, row in answers.items():
            if big not in kids:
                raise AssertionError(f"answer {self.cyc.show(big)} larger is not covered at depth {depth}")
            self.play(kids[big], held, tau + [row], depth + 1)


def initial(cyc: Cycle) -> frozenset:
    return frozenset([cyc.unit(i) for i in range(cyc.m)] + [tuple([0] * cyc.m)])


def check(text: str, k: int, budget: int, adversary: str, branch: bool) -> tuple[bool, str]:
    cyc = Cycle(k)
    player = Player(cyc, adversary, branch)
    try:
        player.play(parse_tree(text, cyc), initial(cyc), [], 0)
    except AssertionError as e:
        return False, f"rejected: {e}"
    if player.depth > budget:
        return False, f"rejected: depth {player.depth} exceeds budget {budget}"
    return True, f"accepted: {player.leaves} leaves, all ready, depth at most {player.depth}, {cyc.programs} z3 calls"


def cmd_tree(args: list[str], opts: dict[str, str]) -> int:
    path, k, budget = Path(args[0]), int(args[1]), int(args[2])
    adversary = opts.get("adversary", "min-entry")
    expect = opts.get("expect", "ok")
    ok, report = check(path.read_text(encoding="utf-8"), k, budget, adversary, "branch" in opts)
    mode = "branch" if "branch" in opts else "tree"
    print(f"{path.name} C_{k} budget {budget} {adversary} {mode}: {report}")
    return 0 if ok == (expect == "ok") else 1


# ---------------------------------------------------------------- chain


def chain_branch(k: int) -> list[tuple[str, ...]]:
    """The 2k - 1 branch as steps: ("cmp", x, y, larger) or ("add", x, y, z)."""
    steps = [("cmp", f"b{j}", f"b{j + 1}", f"b{j}") for j in range(1, k)]
    cur = [f"b{k}"]
    for t in range(1, k):
        arc = f"c{(k + t - 2) % k + 1}"
        new = sorted(cur + [arc], key=lambda s: (s[0], int(s[1:])))
        steps.append(("add", "+".join(cur), arc, "+".join(new)))
        cur = new
    steps.append(("cmp", f"b{k - 1}", "+".join(cur), f"b{k - 1}"))
    return steps


def render(steps: list[tuple[str, ...]]) -> str:
    lines, depth = [], 0
    for step in steps:
        pad = "  " * depth
        if step[0] == "cmp":
            _, x, y, big = step
            small = y if big == x else x
            lines += [f"{pad}compare {x} ? {y}", f"{pad}-> {big} > {small}"]
        else:
            _, x, y, z = step
            lines.append(f"{pad}add {x} + {y} = {z}")
        depth += 1
    lines.append("  " * depth + "ready")
    return "\n".join(lines) + "\n"


def mutants(k: int) -> dict[str, list[tuple[str, ...]]]:
    steps = chain_branch(k)
    last = steps[-1]
    walk_short = steps[-2][1]  # walk_k(k - 2): the walk one arc short
    return {
        "final comparison dropped": steps[:-1],
        "final answer reversed": steps[:-1] + [("cmp", last[1], last[2], last[2])],
        "final comparison against the walk one arc short": steps[:-2] + [("cmp", last[1], walk_short, last[1])],
        "first answer reversed": [("cmp", "b1", "b2", "b2")] + steps[1:],
    }


def cmd_chain(args: list[str], opts: dict[str, str]) -> int:
    ks = [int(a) for a in args]
    failures = 0
    for k in ks:
        steps = chain_branch(k)
        text = render(steps)
        print(f"# C_{k}: the chain branch, {len(steps)} operations (2k - 1 = {2 * k - 1})")
        print(text, end="")
        for adversary in ("min-entry", "full"):
            ok, report = check(text, k, 2 * k - 1, adversary, branch=True)
            failures += not ok
            print(f"  {adversary}: {report}")
        for name, bad in mutants(k).items():
            ok, report = check(render(bad), k, 2 * k - 1, "min-entry", branch=True)
            failures += ok
            print(f"  mutant ({name}): {'ESCAPED' if ok else 'caught'}, {report}")
    if "plot" in opts:
        plot_chain(ks, Path(opts["plot"]))
    print(f"# failures {failures}")
    return 1 if failures else 0


def plot_chain(ks: list[int], out: Path) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks), 3.8), squeeze=False)
    for ax, k in zip(axes[0], ks):
        pos = {
            v: (math.cos(math.pi / 2 - 2 * math.pi * (v - 1) / k), math.sin(math.pi / 2 - 2 * math.pi * (v - 1) / k))
            for v in range(1, k + 1)
        }
        for v in range(1, k + 1):
            u = v % k + 1
            ax.annotate("", pos[u], pos[v], arrowprops=dict(arrowstyle="-", color="0.85", lw=6))
        for j in range(1, k):
            (x0, y0), (x1, y1) = pos[j], pos[j + 1]
            ax.annotate(
                "", (x1 * 0.8, y1 * 0.8), (x0 * 0.8, y0 * 0.8),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6),
            )
        (x0, y0), (x1, y1) = pos[k - 1], pos[k]
        ax.annotate(
            "", (x1 * 1.18, y1 * 1.18), (x0 * 1.18, y0 * 1.18),
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6, connectionstyle="arc3,rad=-0.3"),
        )
        for v, (x, y) in pos.items():
            ax.plot(x, y, "o", ms=16, color="tab:red" if v == k else "white", mec="black")
            ax.text(x, y, str(v), ha="center", va="center", fontsize=9, color="white" if v == k else "black")
        ax.set_title(f"C_{k}: {k - 1} source rows, {k - 1} additions,\n1 facet row = {2 * k - 1} (3k - 3 = {3 * k - 3})", fontsize=9)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        "Rows of the chain branch, each drawn from the larger side's entry to the smaller side's.\n"
        "Blue: b_j > b_{j+1}, every one forward from the survivor k (filled). Red: the one facet row "
        "b_{k-1} > walk_k(k-1).\nEach facet telescopes, f_j = (b_j - b_{j+1}) + ... + f_{k-1} + c_j + ... "
        "+ c_{k-2} (the leaf certificate), so one facet row certifies all of E_k.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=150)


# ---------------------------------------------------------------- forced loss


def forced_loss(cyc: Cycle, tau) -> tuple[int, dict]:
    """The most single-entry cells a source-arc comparison removes either way."""
    here = len(cyc.alive(tau))
    worst, table = 0, {}
    for p, q in itertools.combinations(range(cyc.k), 2):
        kept = [cyc.alive(tau + [diff(cyc.unit(p), cyc.unit(q))]), cyc.alive(tau + [diff(cyc.unit(q), cyc.unit(p))])]
        loss = here - max(len(a) for a in kept)
        table[(p, q)] = (loss, kept)
        worst = max(worst, loss)
    return worst, table


def cmd_forced_loss(args: list[str], opts: dict[str, str]) -> int:
    failures = 0
    tables = {}
    for k in (int(a) for a in args):
        cyc = Cycle(k)
        root, _ = forced_loss(cyc, [])
        tau = []
        ties = True
        for i in range(k):
            row = diff(cyc.unit(k + i), cyc.unit(i))  # c_i > b_i
            a, b = len(cyc.alive(tau + [row])), len(cyc.alive(tau + [diff(cyc.unit(i), cyc.unit(k + i))]))
            ties &= a == b == k
            tau.append(row)
        after, table = forced_loss(cyc, tau)
        tables[k] = table
        # Inside E_s with every c_i > b_i, the source arcs increase along the cycle from s.
        ordered = all(
            not cyc.feasible(tau + cyc.cell_rows(s) + [diff(cyc.unit((s + i) % k), cyc.unit((s + j) % k))])
            for s in range(k)
            for i, j in itertools.combinations(range(k), 2)
        )
        print(
            f"C_{k}: forced loss of source-arc comparisons at the root {root}; after c_i > b_i for all i "
            f"(each a min-entry tie keeping all {k} cells: {ties}) {after}; "
            f"inside every E_s the source arcs then increase from s: {ordered}"
        )
        failures += root != 1 or after != k // 2 or not ties or not ordered
    cyc = Cycle(4)
    tau = [diff(cyc.unit(4), cyc.unit(0)), diff(cyc.unit(6), cyc.unit(2))]  # c1 > b1, c3 > b3
    steps = [len(cyc.alive(tau[:1])), len(cyc.alive([diff(cyc.unit(0), cyc.unit(4))]))]
    steps += [len(cyc.alive(tau)), len(cyc.alive(tau[:1] + [diff(cyc.unit(2), cyc.unit(6))]))]
    small = [s + 1 for s in cyc.alive(tau + [diff(cyc.unit(2), cyc.unit(0))])]
    large = [s + 1 for s in cyc.alive(tau + [diff(cyc.unit(0), cyc.unit(2))])]
    print(
        f"C_4 after c1 > b1, c3 > b3 (cells kept by each answer of c1 ? b1: {steps[0]}, {steps[1]}; "
        f"of c3 ? b3: {steps[2]}, {steps[3]}): b1 < b3 keeps E{small}, b1 > b3 keeps E{large}"
    )
    failures += small != [1, 4] or large != [2, 3] or steps != [4, 4, 4, 4]
    if "plot" in opts:
        plot_forced_loss(tables, Path(opts["plot"]))
    print(f"# failures {failures}")
    return 1 if failures else 0


def plot_forced_loss(tables: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ks = sorted(tables)
    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.4))
    for ax, k in zip(np.atleast_1d(axes), ks):
        grid = np.full((k, k), np.nan)
        for (p, q), (loss, _) in tables[k].items():
            grid[p, q] = grid[q, p] = loss
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=max(ks) // 2)
        for (p, q), (loss, _) in tables[k].items():
            for a, b in ((p, q), (q, p)):
                ax.text(b, a, str(loss), ha="center", va="center", color="white", fontsize=8)
        ax.set_xticks(range(k), [f"b{i + 1}" for i in range(k)], fontsize=7)
        ax.set_yticks(range(k), [f"b{i + 1}" for i in range(k)], fontsize=7)
        ax.set_title(f"C_{k}", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="single-entry cells removed either way")
    fig.suptitle(
        "Comparison b_p ? b_q after c_i > b_i for every i: inside each E_s the b's increase from s,\n"
        "so the comparison halves the alive cells (at the root every such comparison removes at most one)",
        fontsize=8,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------- points


def cmd_points(args: list[str], opts: dict[str, str]) -> int:
    path, k = Path(args[0]), int(args[1])
    cyc = Cycle(k)
    forms = [z for z in itertools.product((0, 1), repeat=cyc.m) if cyc.below_path(z)]
    count = outside = ties = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# point E_"):
            continue
        fields = line.split()
        s = int(fields[2][2:]) - 1
        w = [int(x) for x in fields[3:]]
        if len(w) != cyc.m:
            continue
        count += 1
        weigh = lambda v: sum(a * b for a, b in zip(v, w))  # noqa: E731
        outside += sum(1 for row in cyc.cell_rows(s) if weigh(row) <= 0)
        sums = sorted(weigh(z) for z in forms)
        ties += sum(1 for a, b in zip(sums, sums[1:]) if a == b)
    print(f"C_{k}: {count} points, {outside} facet violations, {ties} ties among {len(forms)} forms below a path")
    return 0 if count and not outside and not ties else 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    args = [a for a in rest if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in rest if a.startswith("--"))
    commands = {"tree": cmd_tree, "chain": cmd_chain, "forced-loss": cmd_forced_loss, "points": cmd_points}
    if command not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    return commands[command](args, opts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
