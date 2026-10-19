#!/usr/bin/env python3
"""Checks a refutation certificate of the strict transcript-cone game.

The game (Cai, arXiv:2609.04825, under the strict adversary): a state is a set
H of held forms, non-negative integer vectors over the m arcs, and a
transcript T of strict linear inequalities on the weights, which with w > 0
cut out an open cone K(T). The program starts holding the zero form and the m
unit vectors, with T empty. A move adds two held forms (possibly the same one
twice), or compares two distinct held forms x, y; the adversary answers
sign(x - y) = s with any s for which K(T + {s(x - y) > 0}) has non-empty
interior. The program has won when, for every non-source vertex v, some held
form equals the distance to v on the whole cone. A certificate claims that
the program cannot win within a budget R of moves from the start.

This checker shares no code with the program that writes certificates. It
re-derives the simple paths, the distances, the automorphisms, every legal
move of every state, cone membership of every point, the cells and the
addition counts, and the budget accounting, in exact integer arithmetic.

What it accepts, and why that proves the claim. Each line is a state X with a
budget R_X. Pass one checks every state on its own:

* every point lies strictly inside K(T_X) and strictly inside one cell, the
  set of weights where one path to each vertex is the unique shortest; the
  points lie in distinct cells; the addition count written beside each point
  equals chain_H(c), the least number of additions producing every distance
  form of its cell c from H, recomputed here;
* a leaf (no entries) satisfies a + ceil(log2 N_a) > R_X, where N_a counts its
  points with chain >= a; an inner state satisfies a + ceil(log2 N_a) >= 1,
  which says it is not ready, and R_X >= 1.

Pass two checks that every inner state answers every legal move: for every
pair i <= j of held forms the state with H[i] + H[j] added, and for every
comparison of distinct held forms either the child of the given answer, is
listed (up to an automorphism of the topology) with budget at least R_X - 1;
or the comparison's answer s is shown to change nothing, by multipliers
D > 0, l >= 0 with D s (x - y) - sum_t l_t T_t >= 0 in every coordinate, so
that s (x - y) > 0 holds on all of K(T_X) and the child is X itself.

Soundness, by induction on k: every listed X loses at min(R_X, k).

* The cell bound. A winning tree from a state with b moves left has, for
  every cell meeting its cone, a distinct leaf: follow a point of the cell
  off every hyperplane the tree compares on; the strict answers it gives are
  legal, and a ready leaf meets exactly one cell, since a held form equal to
  the distance on an open cone is the form of a path that is a shortest path
  on all of it, and distinct paths have distinct forms. The leaf of cell c
  holds c's distance forms, so at least chain_H(c) of the at most b moves on
  its branch are additions; additions do not branch, so at most 2^(b - a)
  leaves have a or more additions above them: b >= a + log2 N_a. Hence a
  leaf loses at every budget up to R_X, and an inner state is not ready.
* An inner X at k <= R_X: a winning tree of depth at most k is not a bare
  ready leaf; its first move leads, under the answer the certificate names,
  to a child cone with non-empty interior (the child's point lies in it, or
  it is K(T_X) itself), so the subtree wins the child within k - 1 moves,
  whereas the child loses at k - 1 by the induction, its listed budget being
  at least R_X - 1 >= k - 1.
* An automorphism of the rooted topology maps paths to paths and the whole
  game onto itself, so a state and its image lose at the same budgets.

Restricting chain_H to forms below some distance form loses nothing: every
form a shortest addition sequence produces is used, and forms only grow.

Four steps the argument above uses without saying so:

* The multiplier case is strict. Write D s (x - y) = sum_t l_t T_t + rho with
  rho >= 0. Since x != y, the l_t and rho are not all zero, so s (x - y) . w > 0
  on all of K(T_X), not only >= 0. Legality and readiness depend on the
  transcript only through the cone, so the child is X as a game state.
* Leaves sit at different depths. At most 2^(b - a) leaves have a or more
  additions above them because the strings of comparison answers on their
  branches are prefix-free and at most b - a long (Kraft's inequality);
  comparisons whose answer is forced only reduce the branching.
* The open cone is enough. The cone of the game is {w >= 0 : T w > 0}; the
  checker works with its interior {w > 0 : T w > 0}, on which readiness is the
  same condition, since d_w is continuous in w.
* The addition counts are memoised without the set `below` of forms they may
  use: every `below` passed contains the boxes of the targets, and within
  them the least count does not depend on it.

    python scripts/check_refutation.py CERT [--expect-budget R] [--expect-arcs ARCS]
    python scripts/check_refutation.py --self-test CERT

CERT may be gzip-compressed (a name ending in .gz). `--self-test` checks the
certificate, then checks that five corrupted copies of it are rejected: one
with a comparison entry left out, one with a point moved outside its cone, one
with an addition count lowered, one with its first state left out (and the
count at the end lowered to match), and one claiming every budget one higher.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import struct
import sys
import time
from math import gcd


class Rejected(Exception):
    pass


def need(cond: bool, why: str) -> None:
    if not cond:
        raise Rejected(why)


# ----------------------------------------------------------------- topology


class Topology:
    def __init__(self, text: str):
        arcs = []
        for tok in text.split(","):
            u, v = tok.split(">")
            arcs.append((int(u), int(v)))
        need(len(set(arcs)) == len(arcs), "parallel arcs are not supported")
        need(all(u != v for u, v in arcs), "self-loops are not supported")
        need(all(v != 0 for _, v in arcs), "an arc into the source")
        self.arcs = arcs
        self.m = len(arcs)
        self.n = 1 + max(max(u, v) for u, v in arcs)
        self.paths = self._paths()
        need(all(self.paths[v] for v in range(1, self.n)), "a vertex the source does not reach")
        self.autos = self._automorphisms()

    def _paths(self) -> dict[int, list[tuple[int, ...]]]:
        """Every simple path from the source to each vertex, as a 0/1 form."""
        out: dict[int, list[tuple[int, ...]]] = {v: [] for v in range(1, self.n)}

        def walk(u: int, used: list[int], visited: set[int]) -> None:
            for i, (a, b) in enumerate(self.arcs):
                if a != u or b in visited:
                    continue
                used.append(i)
                form = tuple(1 if k in used else 0 for k in range(self.m))
                out[b].append(form)
                visited.add(b)
                walk(b, used, visited)
                visited.discard(b)
                used.pop()

        walk(0, [], {0})
        for v in out:
            need(len(set(out[v])) == len(out[v]), "two paths with one form")
        return out

    def _automorphisms(self) -> list[tuple[int, ...]]:
        """Each relabelling of the non-source vertices that maps the arc set
        onto itself, as `sigma` with arc i sent to arc sigma[i]."""
        index = {a: i for i, a in enumerate(self.arcs)}
        out = []
        for perm in itertools.permutations(range(1, self.n)):
            pi = (0,) + perm
            sigma = []
            for u, v in self.arcs:
                j = index.get((pi[u], pi[v]))
                if j is None:
                    break
                sigma.append(j)
            else:
                out.append(tuple(sigma))
        return out


def act(sigma: tuple[int, ...], z: tuple[int, ...]) -> tuple[int, ...]:
    """The image of a form or a row under the arc permutation `sigma`."""
    img = [0] * len(z)
    for i, s in enumerate(sigma):
        img[s] = z[i]
    return tuple(img)


def primitive(row: tuple[int, ...]) -> tuple[int, ...]:
    g = 0
    for x in row:
        g = gcd(g, x)
    need(g > 0, "a zero row")
    return tuple(x // g for x in row)


def dot(a, b) -> int:
    return sum(x * y for x, y in zip(a, b))


def ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


# --------------------------------------------------------------------- keys


def enc_form(z: tuple[int, ...]) -> bytes:
    """Two bytes per coefficient; a certificate needing more is rejected."""
    need(all(0 <= x < 1 << 16 for x in z), "a coefficient beyond the key range")
    return struct.pack(f">{len(z)}H", *z)


def enc_row(t: tuple[int, ...]) -> bytes:
    need(all(-(1 << 15) <= x < 1 << 15 for x in t), "a row entry beyond the key range")
    return struct.pack(f">{len(t)}h", *t)


class Keys:
    """The key of a state up to automorphism: the least, over the topology's
    automorphisms, of an injective encoding of its image (sorted forms, then
    sorted primitive rows, with their counts)."""

    def __init__(self, top: Topology):
        self.autos = top.autos

    def images(self, forms, rows):
        """Per automorphism, the encodings of the images of `forms` and `rows`."""
        return [
            ([enc_form(act(s, z)) for z in forms], [enc_row(act(s, t)) for t in rows])
            for s in self.autos
        ]

    @staticmethod
    def join(fs: list[bytes], rs: list[bytes]) -> bytes:
        fs = sorted(fs)
        rs = sorted(set(rs))
        return (
            len(fs).to_bytes(2, "big") + b"".join(fs)
            + len(rs).to_bytes(2, "big") + b"".join(rs)
        )

    def key(self, forms, rows) -> bytes:
        return min(self.join(f, r) for f, r in self.images(forms, rows))


# ---------------------------------------------------------------- the cells


class Cells:
    def __init__(self, top: Topology):
        self.top = top
        self.memo: dict[tuple, int] = {}
        self.reach_memo: dict[tuple, bool] = {}

    def cell_of(self, p: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """The distance form of each vertex at `p`, which must be strict."""
        cell = []
        for v in range(1, self.top.n):
            vals = [(dot(f, p), f) for f in self.top.paths[v]]
            least = min(x for x, _ in vals)
            at = [f for x, f in vals if x == least]
            need(len(at) == 1, "a point on a boundary between cells")
            cell.append(at[0])
        return tuple(cell)

    def chain(self, held: frozenset, targets) -> int:
        """The least number of additions producing every target from `held`."""
        want = frozenset(t for t in targets if t not in held)
        if not want:
            return 0
        below = set()
        for t in want:
            for v in itertools.product(*(range(x + 1) for x in t)):
                below.add(v)
        useful = frozenset(z for z in held if z in below)
        key = (useful, want)
        if key in self.memo:
            return self.memo[key]
        # Every addition of a shortest sequence produces a new form below some
        # target, so no sequence needs more additions than there are such forms.
        cap = len(below - useful)
        d = len(want)
        while not self._reach(useful, want, d, below):
            d += 1
            need(d <= cap, "a distance form no addition sequence produces")
        self.memo[key] = d
        return d

    def _reach(self, held: frozenset, want: frozenset, depth: int, below) -> bool:
        if not want:
            return True
        if len(want) > depth:
            return False
        # The memo is not keyed on `below`: every `below` passed here contains
        # the boxes of `want`, and within them the answer does not depend on it.
        key = (held, want, depth)
        hit = self.reach_memo.get(key)
        if hit is not None:
            return hit
        out = False
        hs = sorted(held)
        for i, x in enumerate(hs):
            for y in hs[i:]:
                z = tuple(a + b for a, b in zip(x, y))
                if z in held or z not in below:
                    continue
                if self._reach(held | {z}, want - {z}, depth - 1, below):
                    out = True
                    break
            if out:
                break
        self.reach_memo[key] = out
        return out


# ------------------------------------------------------------------ parsing


def ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def parse_state(line: str, m: int):
    parts = line.split("|")
    need(len(parts) == 6, "a state line without six fields")
    r = int(parts[0])
    forms = [ints(s) for s in parts[1].split()]
    rows = [ints(s) for s in parts[2].split()]
    points = []
    for s in parts[3].split():
        p, c = s.split(":")
        points.append((ints(p), int(c)))
    a = int(parts[4])
    entries = parts[5].split()
    need(all(len(z) == m and min(z) >= 0 for z in forms), "a malformed form")
    need(len(set(forms)) == len(forms), "a form held twice")
    need(all(len(t) == m for t in rows), "a malformed row")
    need(all(len(p) == m for p, _ in points), "a malformed point")
    need(r >= 0 and a >= 0, "a negative budget or count")
    return r, forms, rows, points, a, entries


def lines(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="ascii") as f:
        for line in f:
            yield line.rstrip("\n")


# ------------------------------------------------------------------- checks


def check(path: str, expect_budget: int | None = None, quiet: bool = False,
          expect_arcs: str | None = None) -> dict:
    t0 = time.time()
    before = digests(path)
    it = lines(path)
    need(next(it) == "unsorted-refutation 1", "not a refutation certificate")
    arcs = next(it)
    need(arcs.startswith("arcs "), "no arcs line")
    top = Topology(arcs[5:])
    if expect_arcs is not None:
        need(top.arcs == Topology(expect_arcs).arcs, "the certificate is for another topology")
    budget_line = next(it)
    need(budget_line.startswith("budget "), "no budget line")
    budget = int(budget_line[7:])
    need(budget >= 0, "a negative budget")
    if expect_budget is not None:
        need(budget == expect_budget, "the certificate refutes another budget")
    m = top.m
    keys = Keys(top)
    cells = Cells(top)
    refuted: dict[bytes, int] = {}
    stats = {"states": 0, "leaves": 0, "inner": 0, "points": 0}
    end = None

    # Pass one: each state on its own, and the table of refuted states.
    for line in it:
        if line.startswith("end "):
            end = int(line[4:])
            break
        r, forms, rows, points, a, entries = parse_state(line, m)
        held = frozenset(forms)
        prim = [primitive(t) for t in rows]
        need(len(points) >= 1, "a state without a point")
        seen = set()
        n_a = 0
        for p, claimed in points:
            need(min(p) > 0, "a point outside the open orthant")
            need(all(dot(t, p) > 0 for t in rows), "a point outside its cone")
            c = cells.cell_of(p)
            need(c not in seen, "two points in one cell")
            seen.add(c)
            ch = cells.chain(held, c)
            need(ch == claimed, "a wrong addition count")
            n_a += ch >= a
        need(n_a >= 1, "no point reaches the addition count")
        bound = a + ceil_log2(n_a)
        if entries:
            need(r >= 1, "an inner state at budget 0")
            need(bound >= 1, "an inner state that may be ready")
            stats["inner"] += 1
        else:
            need(bound > r, "a leaf the cell bound does not refute")
            stats["leaves"] += 1
        stats["points"] += len(points)
        stats["states"] += 1
        k = keys.key(forms, prim)
        if refuted.get(k, -1) < r:
            refuted[k] = r
    need(end is not None and end == stats["states"], "a truncated certificate")
    t1 = time.time()

    # Pass two: every legal move of every inner state is answered.
    it = lines(path)
    for _ in range(3):
        next(it)
    moves = 0
    for line in it:
        if line.startswith("end "):
            break
        r, forms, rows, points, a, entries = parse_state(line, m)
        if not entries:
            continue
        prim = [primitive(t) for t in rows]
        held = set(forms)
        imgs = keys.images(forms, prim)
        sigmas = keys.autos

        def listed(new_form, new_row) -> bool:
            best = None
            for s, (fs, rs) in zip(sigmas, imgs):
                f2 = fs + [enc_form(act(s, new_form))] if new_form is not None else fs
                r2 = rs + [enc_row(act(s, new_row))] if new_row is not None else rs
                k = keys.join(f2, r2)
                if best is None or k < best:
                    best = k
            return refuted.get(best, -1) >= r - 1

        for i in range(len(forms)):
            for j in range(i, len(forms)):
                z = tuple(x + y for x, y in zip(forms[i], forms[j]))
                moves += 1
                need(listed(None if z in held else z, None), "an addition left unanswered")
        answers = {}
        for e in entries:
            f = [int(x) for x in e.split(",")]
            need(len(f) == 3 or len(f) == 4 + len(rows), "a malformed entry")
            i, j, s = f[0], f[1], f[2]
            need(0 <= i < j < len(forms), "an entry naming no comparison")
            need(s in (1, -1), "a sign other than 1 or -1")
            need((i, j) not in answers, "a comparison answered twice")
            answers[(i, j)] = f
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                moves += 1
                f = answers.get((i, j))
                need(f is not None, "a comparison left unanswered")
                s = f[2]
                d = tuple(s * (x - y) for x, y in zip(forms[i], forms[j]))
                if len(f) == 3:
                    need(listed(None, primitive(d)), "the answer to a comparison is not refuted")
                else:
                    den, lam = f[3], f[4:]
                    need(den > 0 and min(lam, default=0) >= 0, "multipliers of the wrong sign")
                    for c in range(m):
                        need(
                            den * d[c] - sum(l * t[c] for l, t in zip(lam, rows)) >= 0,
                            "multipliers that do not imply the answer",
                        )
    t2 = time.time()

    root_forms = [tuple(0 for _ in range(m))] + [
        tuple(1 if k == i else 0 for k in range(m)) for i in range(m)
    ]
    need(refuted.get(keys.key(root_forms, []), -1) >= budget, "the start is not refuted")
    file_sha, text_sha, text_bytes = digests(path)
    need((file_sha, text_sha, text_bytes) == before, "the file changed while it was checked")
    stats.update(
        arcs=arcs[5:], budget=budget, moves=moves, automorphisms=len(top.autos),
        distinct=len(refuted), pass1=round(t1 - t0, 1), pass2=round(t2 - t1, 1),
    )
    if not quiet:
        print(
            f"# {stats['arcs']}  budget {budget} refuted: the program cannot win "
            f"within {budget} moves from the start"
        )
        print(
            f"states {stats['states']} (distinct {stats['distinct']})  leaves {stats['leaves']}  "
            f"inner {stats['inner']}  points {stats['points']}  moves checked {moves}  "
            f"automorphisms {stats['automorphisms']}  pass one {stats['pass1']}s  "
            f"pass two {stats['pass2']}s"
        )
        print(f"file sha256 {file_sha}")
        print(f"text sha256 {text_sha}  text bytes {text_bytes}")
    return stats


def digests(path: str) -> tuple[str, str, int]:
    """The sha256 of the file as stored, and of its text once decompressed."""
    raw = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            raw.update(block)
    text = hashlib.sha256()
    size = 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            text.update(block)
            size += len(block)
    return raw.hexdigest(), text.hexdigest(), size


# ---------------------------------------------------------------- self-test


def corrupt(src: str, dst: str, how: str) -> None:
    """Writes a copy of `src` with one defect of kind `how`."""
    out = []
    done = False
    for line in lines(src):
        parts = line.split("|")
        if not done and len(parts) == 6:
            if how == "drop-entry" and len(parts[5].split()) > 1:
                parts[5] = " ".join(parts[5].split()[1:])
                done = True
            elif how == "move-point" and parts[2]:
                row = ints(parts[2].split()[0])
                neg = [c for c, x in enumerate(row) if x < 0]
                if neg:
                    pts = parts[3].split()
                    p, ch = pts[0].split(":")
                    p = list(ints(p))
                    # Raise one coordinate the row weighs negatively until the
                    # row is no longer positive at the point.
                    c = neg[0]
                    p[c] += dot(row, p) // (-row[c]) + 1
                    pts[0] = ",".join(map(str, p)) + ":" + ch
                    parts[3] = " ".join(pts)
                    done = True
            elif how == "lower-chain":
                pts = parts[3].split()
                for k, s in enumerate(pts):
                    p, ch = s.split(":")
                    if int(ch) > 0:
                        pts[k] = f"{p}:{int(ch) - 1}"
                        parts[3] = " ".join(pts)
                        done = True
                        break
        if how == "drop-state":
            if len(parts) == 6 and not done:
                done = True
                continue
            if line.startswith("end "):
                parts = [f"end {int(line[4:]) - 1}"]
        if how == "raise-budget":
            if len(parts) == 6:
                parts[0] = str(int(parts[0]) + 1)
            elif line.startswith("budget "):
                parts = [f"budget {int(line[7:]) + 1}"]
                done = True
        out.append("|".join(parts))
    need(done, f"no place to apply {how}")
    with open(dst, "w", encoding="ascii") as f:
        f.write("\n".join(out) + "\n")


def self_test(path: str) -> int:
    import os
    import tempfile

    check(path)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for how in ("drop-entry", "move-point", "lower-chain", "drop-state", "raise-budget"):
            dst = os.path.join(tmp, how + ".txt")
            corrupt(path, dst, how)
            try:
                check(dst, quiet=True)
            except Rejected as e:
                print(f"self-test {how}: rejected ({e})")
            else:
                print(f"self-test {how}: ACCEPTED")
                failures += 1
    return failures


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "--self-test":
        if len(argv) != 2:
            print("usage: --self-test CERT", file=sys.stderr)
            return 2
        return 1 if self_test(argv[1]) else 0
    # Only the documented options are accepted, each with a value, so that a
    # misspelt or `=`-joined flag cannot leave the result unbound.
    options = {"--expect-budget": None, "--expect-arcs": None}
    rest = argv[1:]
    while rest:
        flag = rest.pop(0)
        if flag not in options or options[flag] is not None or not rest:
            print(f"unknown, repeated or valueless argument: {flag}", file=sys.stderr)
            return 2
        options[flag] = rest.pop(0)
    expect = None if options["--expect-budget"] is None else int(options["--expect-budget"])
    arcs = options["--expect-arcs"]
    try:
        check(argv[0], expect, expect_arcs=arcs)
    except Rejected as e:
        print(f"rejected: {e}")
        return 1
    except Exception as e:  # malformed input of any other kind is a rejection too
        print(f"rejected: malformed ({type(e).__name__}: {e})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
