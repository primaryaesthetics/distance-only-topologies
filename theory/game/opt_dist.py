#!/usr/bin/env python3
"""Exact distance-only complexity of a small topology, by the transcript-cone game.

Cai, arXiv:2609.04825, Theorem J.3: for a fixed rooted directed topology, the
least worst-case number of charged operations — additions of two held
registers plus three-way comparisons of two held registers — that any
deterministic comparison-addition program needs to hold every distance in a
register is the least budget `K` for which the game below is won from the
initial state.

    state    a set `M` of coefficient vectors in N^m (the forms of the held
             registers) and a transcript `tau` of comparison outcomes, which
             cut the weight orthant to a relatively open cone `P(tau)`
    move     add two forms of `M` (one form may be used twice), or compare two
             distinct forms and win on every outcome the cone leaves feasible
    ready    every label `v != s` has one fixed form in `M` equal to `d_w(s, v)`
             on the whole of `P(tau)`

Every register is homogeneous by the recession normal form (Lemma J.1), so
literal constants are the zero form and nothing else is lost.

Four values are computed, and they bracket `OPT_DIST` from both sides.
`full` is the game as stated, with equality outcomes and the zero form, and
equals `OPT_DIST` by the theorem. `generic` restricts the adversary to strict
outcomes on full-dimensional cones, which can only lower the value, so it is
a lower bound on `OPT_DIST` and is the game the paper's own certificate for
`H_2` is played in. `pathfull` restricts the program instead, to forms that
lie below some path vector coordinatewise — the path-register programs the
paper names as the weaker model — against the unrestricted adversary; that
can only raise the value, so it is an upper bound on `OPT_DIST`. `path`
restricts both, and is the exact value of the weaker model under the strict
adversary. So

    generic  <=  OPT_DIST = full  <=  pathfull,      generic <= path <= pathfull,

and where `generic` meets `pathfull` the exact value is known without playing
`full`, which is the slowest of the four. All are minimax searches with
iterative deepening, a transposition table keyed on the state up to the
topology's automorphisms, a lower bound from the cone's cells and their
addition chains, and cone feasibility and readiness decided exactly by `z3`
over the rationals.

    python theory/game/opt_dist.py validate [--quick] the DAG law and H_2
    python theory/game/opt_dist.py enumerate N MMAX [SECONDS] [--modes=M,M] [--m=K] [--pick=I,I]
                                                     every rooted topology on N
                                                     vertices with at most MMAX
                                                     arcs, up to isomorphism
    python theory/game/opt_dist.py strategy ARCS [--MODE]
                                                     print a winning strategy

An arc list is written `0>1,0>2,1>2`; the source is vertex 0.
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from dataclasses import dataclass, field

from z3 import And, Or, Real, Solver, sat, unsat

LT, EQ, GT = -1, 0, 1


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """A rooted directed graph on vertices `0..n-1`, source `0`, arcs by index."""

    n: int
    arcs: tuple[tuple[int, int], ...]

    @property
    def m(self) -> int:
        return len(self.arcs)

    @staticmethod
    def parse(text: str) -> "Topology":
        arcs = []
        for token in text.split(","):
            u, v = token.split(">")
            arcs.append((int(u), int(v)))
        n = 1 + max(max(u, v) for u, v in arcs)
        return Topology(n, tuple(arcs))

    def __str__(self) -> str:
        return ",".join(f"{u}>{v}" for u, v in self.arcs)

    def is_dag(self) -> bool:
        indeg = [0] * self.n
        for _, v in self.arcs:
            indeg[v] += 1
        stack = [v for v in range(self.n) if indeg[v] == 0]
        seen = 0
        while stack:
            u = stack.pop()
            seen += 1
            for a, b in self.arcs:
                if a == u:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        stack.append(b)
        return seen == self.n

    def source_out(self) -> int:
        return sum(1 for u, _ in self.arcs if u == 0)

    def reachable(self) -> bool:
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for a, b in self.arcs:
                if a == u and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return len(seen) == self.n

    def paths(self) -> dict[int, list[tuple[int, ...]]]:
        """Every simple `s`-`v` path as a 0/1 coefficient vector over the arcs.

        A shortest walk under non-negative weights has a simple representative,
        so these are the only forms a distance can take.
        """
        out: dict[int, list[tuple[int, ...]]] = {v: [] for v in range(1, self.n)}
        m = self.m

        def walk(u: int, used: list[int], onpath: set[int]) -> None:
            for i, (a, b) in enumerate(self.arcs):
                if a != u or b in onpath:
                    continue
                used.append(i)
                vec = [0] * m
                for j in used:
                    vec[j] = 1
                out[b].append(tuple(vec))
                onpath.add(b)
                walk(b, used, onpath)
                onpath.discard(b)
                used.pop()

        walk(0, [], {0})
        return out

    def fwd(self) -> int:
        """`fwd(G, s)` of Theorem 3.1: the largest number of forward non-source
        arcs over rooted vertex orders. Exponential, and fine at these sizes."""
        best = 0
        for order in itertools.permutations(range(1, self.n)):
            pos = {0: 0}
            for i, v in enumerate(order, start=1):
                pos[v] = i
            rooted = all(
                any(pos[u] < pos[v] for u, w in self.arcs if w == v)
                for v in order
            )
            if not rooted:
                continue
            count = sum(1 for u, v in self.arcs if u != 0 and pos[u] < pos[v])
            best = max(best, count)
        return best

    def dag_law(self) -> int:
        """Theorem 4.1's corner, `(m - n + 1) + (q - d_s)`, for a simple DAG."""
        return (self.m - self.n + 1) + (self.m - self.source_out())

    def automorphisms(self) -> list[tuple[int, ...]]:
        """Every relabelling of the non-source vertices that maps the arc set
        onto itself, as the permutation it induces on arc indices. The
        identity is first. Two game states that differ by one of these have
        the same value, which halves or sixths the search on symmetric
        topologies."""
        index = {arc: i for i, arc in enumerate(self.arcs)}
        out = []
        for perm in itertools.permutations(range(1, self.n)):
            relabel = {0: 0}
            relabel.update(zip(range(1, self.n), perm))
            image = []
            for u, v in self.arcs:
                arc = (relabel[u], relabel[v])
                if arc not in index:
                    break
                image.append(index[arc])
            else:
                out.append(tuple(image))
        out.sort(key=lambda sigma: sigma != tuple(range(self.m)))
        return out


def enumerate_topologies(n: int, max_arcs: int) -> list[Topology]:
    """Every reachable simple rooted digraph on `n` vertices with at most
    `max_arcs` arcs, one representative per isomorphism class. Arcs into the
    source and self-loops are excluded: neither lies on a shortest path from
    the source, and Appendix P.1 removes them without changing the value."""
    slots = [(u, v) for u in range(n) for v in range(1, n) if u != v]
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[Topology] = []
    perms = list(itertools.permutations(range(1, n)))
    for k in range(n - 1, max_arcs + 1):
        for chosen in itertools.combinations(slots, k):
            top = Topology(n, tuple(chosen))
            if not top.reachable():
                continue
            canon = None
            for perm in perms:
                relabel = {0: 0}
                relabel.update(zip(range(1, n), perm))
                cand = tuple(sorted((relabel[u], relabel[v]) for u, v in chosen))
                if canon is None or cand < canon:
                    canon = cand
            if canon in seen:
                continue
            seen.add(canon)
            out.append(Topology(n, canon))
    return out


# --------------------------------------------------------------------------
# Cones and readiness
# --------------------------------------------------------------------------

Vec = tuple[int, ...]
Constraint = tuple[Vec, int]
Transcript = frozenset


def normalise(c: list[int]) -> tuple[Vec, int]:
    """A difference vector up to positive scaling and sign, and the sign flip."""
    g = 0
    for x in c:
        g = math.gcd(g, abs(x))
    if g == 0:
        return tuple(c), 1
    c = [x // g for x in c]
    for x in c:
        if x != 0:
            if x < 0:
                return tuple(-y for y in c), -1
            return tuple(c), 1
    return tuple(c), 1


class Oracle:
    """Exact rational decisions about transcript cones, memoised."""

    def __init__(self, top: Topology, generic: bool):
        self.top = top
        self.generic = generic
        self.m = top.m
        self.w = [Real(f"w{i}") for i in range(self.m)]
        self.paths = top.paths()
        self.labels = [v for v in range(1, top.n)]
        self.path_set = {v: set(ps) for v, ps in self.paths.items()}
        self.solver = Solver()
        self.solver.add(*[(w > 0) if generic else (w >= 0) for w in self.w])
        self._forms: dict = {}
        self._constraints: dict = {}
        self._wrong: dict = {}
        self._right: dict = {}
        self._feasible: dict = {}
        self._ready: dict = {}
        self._possible: dict = {}
        self._cells: dict = {}
        self._chain: dict = {}
        self.calls = 0

    def form(self, a: Vec):
        """The `z3` term of a coefficient vector, built once per vector."""
        if a not in self._forms:
            terms = [c * self.w[i] for i, c in enumerate(a) if c]
            self._forms[a] = sum(terms) if terms else 0
        return self._forms[a]

    def constraint(self, c: Vec, sign: int):
        key = (c, sign)
        if key not in self._constraints:
            f = self.form(c)
            self._constraints[key] = f < 0 if sign == LT else f == 0 if sign == EQ else f > 0
        return self._constraints[key]

    def cone(self, tau: Transcript):
        """The transcript's constraints; the orthant is asserted once."""
        return [self.constraint(c, sign) for c, sign in tau]

    def check(self, *assertions) -> bool:
        self.calls += 1
        self.solver.push()
        self.solver.add(*assertions)
        r = self.solver.check()
        self.solver.pop()
        if r == sat:
            return True
        if r == unsat:
            return False
        raise RuntimeError("z3 returned unknown")

    @staticmethod
    def has_equality(tau: Transcript) -> bool:
        return any(sign == EQ for _, sign in tau)

    def candidates(self, tau: Transcript, v: int, z: Vec) -> bool:
        """Whether `z` can equal `d_w(v)` on the cone at all.

        The cone is relatively open in the subspace its equality outcomes cut,
        and on any open subset of that subspace where one path is the unique
        minimum the two linear forms must coincide there. Without an equality
        outcome the subspace is the whole orthant, so `z` must be a path form
        of `v`; with one, a non-path form can agree with a path on the face,
        and the exact test is used.
        """
        return self.has_equality(tau) or z in self.path_set[v]

    def feasible(self, tau: Transcript) -> bool:
        if tau not in self._feasible:
            self._feasible[tau] = self.check(*self.cone(tau))
        return self._feasible[tau]

    def outcomes(self, tau: Transcript, c: Vec) -> list[int]:
        signs = (LT, GT) if self.generic else (LT, EQ, GT)
        return [s for s in signs if self.feasible(tau | {(c, s)})]

    def ready(self, tau: Transcript, v: int, z: Vec) -> bool:
        """`<z, w> = d_w(v)` on the whole cone."""
        if not self.candidates(tau, v, z):
            return False
        if self.generic:
            i = self.labels.index(v)
            return all(cell[i] == z for cell in self.cells(tau))
        key = (tau, v, z)
        if key not in self._ready:
            if (v, z) not in self._wrong:
                zf = self.form(z)
                ps = [self.form(p) for p in self.paths[v]]
                self._wrong[(v, z)] = Or(Or(*[p < zf for p in ps]), And(*[p > zf for p in ps]))
            self._ready[key] = not self.check(*self.cone(tau), self._wrong[(v, z)])
        return self._ready[key]

    def possible(self, tau: Transcript, v: int, z: Vec) -> bool:
        """`<z, w> = d_w(v)` somewhere on the cone. A form that is right
        nowhere on the cone is right nowhere on any of its sub-cones."""
        if not self.candidates(tau, v, z):
            return False
        if self.generic:
            i = self.labels.index(v)
            return any(cell[i] == z for cell in self.cells(tau))
        key = (tau, v, z)
        if key not in self._possible:
            if (v, z) not in self._right:
                zf = self.form(z)
                ps = [self.form(p) for p in self.paths[v]]
                self._right[(v, z)] = And(Or(*[p == zf for p in ps]), And(*[p >= zf for p in ps]))
            self._possible[key] = self.check(*self.cone(tau), self._right[(v, z)])
        return self._possible[key]

    def cells(self, tau: Transcript) -> list[tuple[Vec, ...]]:
        """The full-dimensional cells of the cone: every assignment of one
        path to each label that is the strict minimum on some open subset.
        Backtracking over the labels, each prefix checked for feasibility, so
        an infeasible prefix is never extended. Cached per transcript."""
        if tau not in self._cells:
            found: list[tuple[Vec, ...]] = []
            base = self.cone(tau)

            def extend(i: int, chosen: list[Vec], constraints: list) -> None:
                if i == len(self.labels):
                    found.append(tuple(chosen))
                    return
                v = self.labels[i]
                for p in self.paths[v]:
                    pf = self.form(p)
                    strict = [pf < self.form(q) for q in self.paths[v] if q != p]
                    if not self.check(*base, *constraints, *strict):
                        continue
                    chosen.append(p)
                    extend(i + 1, chosen, constraints + strict)
                    chosen.pop()

            extend(0, [], [])
            self._cells[tau] = found
        return self._cells[tau]

    def chain(self, M: frozenset, targets: frozenset) -> int:
        """The least number of additions that produce every target form from
        the held forms. An intermediate form that is not below some target
        componentwise can never contribute, since coefficients only grow, so
        the search is over such forms only; iterative deepening, memoised."""
        need = frozenset(t for t in targets if t not in M)
        if not need:
            return 0
        key = (M, need)
        if key in self._chain:
            return self._chain[key]

        def below(z: Vec) -> bool:
            return any(all(a <= b for a, b in zip(z, t)) for t in need)

        def reach(held: frozenset, want: frozenset, depth: int) -> bool:
            if not want:
                return True
            if depth == 0:
                return False
            if len(want) > depth:
                return False
            hs = sorted(held)
            for i, x in enumerate(hs):
                for y in hs[i:]:
                    z = vec_add(x, y)
                    if z in held or not below(z):
                        continue
                    if reach(held | {z}, want - {z}, depth - 1):
                        return True
            return False

        useful = frozenset(z for z in M if below(z))
        depth = len(need)
        while not reach(useful, need, depth):
            depth += 1
        self._chain[key] = depth
        return depth


# --------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------


@dataclass
class Move:
    kind: str  # "add" or "cmp"
    x: Vec
    y: Vec
    c: Vec = ()  # the normalised difference, for a comparison
    flip: int = 1


def vec_add(x: Vec, y: Vec) -> Vec:
    return tuple(a + b for a, b in zip(x, y))


@dataclass
class Game:
    top: Topology
    generic: bool
    path_only: bool = False
    oracle: Oracle = field(init=False)
    win_at: dict = field(default_factory=dict)
    lose_at: dict = field(default_factory=dict)
    lb_at: dict = field(default_factory=dict)
    nodes: int = 0
    deadline: float = math.inf

    def __post_init__(self):
        self.oracle = Oracle(self.top, self.generic)
        self.autos = self.top.automorphisms()
        self.path_forms = {p for ps in self.oracle.paths.values() for p in ps}
        self.sub_forms = set()
        for p in self.path_forms:
            idx = [i for i, c in enumerate(p) if c]
            for r in range(len(idx) + 1):
                for sub in itertools.combinations(idx, r):
                    vec = [0] * self.top.m
                    for i in sub:
                        vec[i] = 1
                    self.sub_forms.add(tuple(vec))

    def key(self, M: frozenset, tau: Transcript):
        """The transposition-table key: the least image of the state under
        the topology's automorphisms."""
        if len(self.autos) == 1:
            return (M, tau)
        best = None
        for sigma in self.autos:
            m_img = tuple(sorted(tuple(z[i] for i in sigma) for z in M))
            t_img = []
            for c, sign in tau:
                c2, flip = normalise([c[i] for i in sigma])
                t_img.append((c2, sign * flip))
            cand = (m_img, tuple(sorted(t_img)))
            if best is None or cand < best:
                best = cand
        return best

    def initial(self) -> tuple[frozenset, Transcript]:
        m = self.top.m
        forms = {tuple(0 for _ in range(m))}
        for i in range(m):
            forms.add(tuple(1 if j == i else 0 for j in range(m)))
        return frozenset(forms), frozenset()

    def unready(self, M: frozenset, tau: Transcript) -> list[int]:
        out = []
        for v in self.oracle.labels:
            if not any(self.oracle.ready(tau, v, z) for z in M):
                out.append(v)
        return out

    def missing(self, M: frozenset, tau: Transcript, labels: list[int]) -> int:
        """Labels no held form can serve anywhere on this cone."""
        return sum(
            1 for v in labels if not any(self.oracle.possible(tau, v, z) for z in M)
        )

    def lower_bound(self, M: frozenset, tau: Transcript, labels: list[int]) -> int:
        """Charged operations any continuation needs, against an adversary who
        answers strictly whenever it can. Sound only where the cone is
        full-dimensional, which is every state without an equality outcome.

        Under strict answers every leaf lies in one full-dimensional cell, a
        cell fixes the minimising path of each label, and a leaf's witness is
        one form, so distinct cells need distinct leaves. A leaf in cell `c`
        holds the cell's paths, which takes `chain(c)` additions on its branch,
        leaving at most `r - chain(c)` comparisons, and a binary tree has at
        most `2^d` leaves at comparison depth `d`. So for every `a`,

            r  >=  a + log2 |{ cells with chain >= a }|.
        """
        if self.oracle.has_equality(tau):
            return 1
        key = self.key(M, tau)
        if key not in self.lb_at:
            chains = sorted(
                (self.oracle.chain(M, frozenset(cell)) for cell in self.oracle.cells(tau)),
                reverse=True,
            )
            best = 1
            for i, a in enumerate(chains):
                best = max(best, a + math.ceil(math.log2(i + 1)))
            self.lb_at[key] = best
        return self.lb_at[key]

    def moves(self, M: frozenset, tau: Transcript) -> list[Move]:
        forms = sorted(M)
        adds, cmps = [], []
        for i, x in enumerate(forms):
            for y in forms[i:]:
                z = vec_add(x, y)
                if z in M or (self.path_only and z not in self.sub_forms):
                    continue
                adds.append(Move("add", x, y))
        decided = {k for k, _ in tau}
        for i, x in enumerate(forms):
            if self.generic and not any(x):
                continue
            for y in forms[i + 1:]:
                c, flip = normalise([a - b for a, b in zip(x, y)])
                if not any(c) or c in decided:
                    continue
                cmps.append(Move("cmp", x, y, c, flip))

        def rank(mv: Move) -> tuple:
            if mv.kind == "add":
                z = vec_add(mv.x, mv.y)
                tier = 0 if z in self.path_forms else 1 if z in self.sub_forms else 2
            else:
                both_paths = mv.x in self.path_forms and mv.y in self.path_forms
                both_sub = mv.x in self.sub_forms and mv.y in self.sub_forms
                tier = 0 if both_paths else 1 if both_sub else 2
            return (tier, mv.x, mv.y)

        return sorted(cmps, key=rank) + sorted(adds, key=rank)

    def wins(self, M: frozenset, tau: Transcript, r: int) -> bool:
        key = self.key(M, tau)
        if key in self.win_at and self.win_at[key] <= r:
            return True
        if key in self.lose_at and self.lose_at[key] >= r:
            return False
        self.nodes += 1
        if time.monotonic() > self.deadline:
            raise TimeoutError
        open_labels = self.unready(M, tau)
        if not open_labels:
            self.win_at[key] = 0
            return True
        if r == 0 or r < self.lower_bound(M, tau, open_labels):
            self.lose_at[key] = max(self.lose_at.get(key, 0), r)
            return False
        for mv in self.moves(M, tau):
            if mv.kind == "add":
                if self.wins(M | {vec_add(mv.x, mv.y)}, tau, r - 1):
                    self.win_at[key] = min(self.win_at.get(key, r), r)
                    return True
            else:
                outcomes = self.oracle.outcomes(tau, mv.c)
                if len(outcomes) < 2:
                    continue

                # The adversary answers first with the outcome that leaves the
                # most labels unserved, which is the one most likely to refute.
                def hurt(s: int) -> int:
                    return -self.missing(M, tau | {(mv.c, s)}, open_labels)

                if all(self.wins(M, tau | {(mv.c, s)}, r - 1)
                       for s in sorted(outcomes, key=hurt)):
                    self.win_at[key] = min(self.win_at.get(key, r), r)
                    return True
        self.lose_at[key] = max(self.lose_at.get(key, 0), r)
        return False

    def value(self, cap: int | None = None, budget_seconds: float = math.inf):
        """Least winning budget, by iterative deepening. `None` on timeout."""
        self.deadline = time.monotonic() + budget_seconds
        M, tau = self.initial()
        cap = cap if cap is not None else 2 * self.top.m * (self.top.n - 1)
        try:
            start = 0
            if self.unready(M, tau):
                start = self.lower_bound(M, tau, self.unready(M, tau))
            for r in range(start, cap + 1):
                if self.wins(M, tau, r):
                    return r
        except TimeoutError:
            return None
        return None

    def lower_bound_reached(self) -> int:
        """One more than the largest budget shown losing from the start."""
        return self.lose_at.get(self.key(*self.initial()), -1) + 1

    def strategy(self, M: frozenset, tau: Transcript, r: int, depth: int = 0) -> None:
        """Print one winning strategy from a state known to win within `r`."""
        pad = "  " * depth
        open_labels = self.unready(M, tau)
        if not open_labels:
            parts = []
            for v in self.oracle.labels:
                z = next(z for z in M if self.oracle.ready(tau, v, z))
                parts.append(f"d({v}) = {fmt(self.top, z)}")
            print(f"{pad}ready: " + "; ".join(parts))
            return
        for mv in self.moves(M, tau):
            if mv.kind == "add":
                z = vec_add(mv.x, mv.y)
                if self.wins(M | {z}, tau, r - 1):
                    print(f"{pad}add {fmt(self.top, mv.x)} + {fmt(self.top, mv.y)}")
                    self.strategy(M | {z}, tau, r - 1, depth)
                    return
            else:
                outcomes = self.oracle.outcomes(tau, mv.c)
                if len(outcomes) < 2:
                    continue
                if all(self.wins(M, tau | {(mv.c, s)}, r - 1) for s in outcomes):
                    print(f"{pad}compare {fmt(self.top, mv.x)} ? {fmt(self.top, mv.y)}")
                    for s in outcomes:
                        sym = {LT: "<", EQ: "=", GT: ">"}[s * mv.flip]
                        print(f"{pad}  {sym}:")
                        self.strategy(M, tau | {(mv.c, s)}, r - 1, depth + 2)
                    return
        raise RuntimeError("no winning move at a winning state")


def fmt(top: Topology, z: Vec) -> str:
    if not any(z):
        return "0"
    parts = []
    for i, c in enumerate(z):
        if c:
            u, v = top.arcs[i]
            parts.append(("" if c == 1 else f"{c}*") + f"w({u}>{v})")
    return " + ".join(parts)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def solve(top: Topology, generic: bool, budget_seconds: float, cap: int | None = None,
          path_only: bool = False):
    g = Game(top, generic, path_only)
    t0 = time.monotonic()
    v = g.value(cap=cap, budget_seconds=budget_seconds)
    return v, g, time.monotonic() - t0


def validate(quick: bool = False) -> int:
    """The solver against the two exact values the paper states. `quick`
    stops at four arcs and skips `H_2`, which is what continuous integration
    can afford; the full run is recorded as an experiment."""
    failures = 0
    print("DAG law, Theorem 4.1: OPT_DIST = (m - n + 1) + (m - d_s) on simple DAGs")
    for n, mmax in ((2, 1), (3, 3), (4, 4 if quick else 5)):
        for top in enumerate_topologies(n, mmax):
            if not top.is_dag():
                continue
            v, g, dt = solve(top, generic=False, budget_seconds=600)
            ok = v == top.dag_law()
            failures += not ok
            print(f"  {str(top):24s} n={n} m={top.m} law={top.dag_law()} game={v} "
                  f"nodes={g.nodes} z3={g.oracle.calls} {dt:.1f}s {'ok' if ok else 'MISMATCH'}",
                  flush=True)
    if quick:
        return failures
    print("H_2, Theorem 5.1: C = 4, P = 2, C_2 = 5, so OPT_DIST <= 7; expected game value 7")
    h2 = Topology.parse("0>1,0>2,2>1,1>2,0>3,3>1,1>3")
    v, g, dt = solve(h2, budget_seconds=3600, **MODES["path"])
    print(f"  path value {v} nodes={g.nodes} z3={g.oracle.calls} {dt:.1f}s "
          f"{'ok' if v == 7 else 'MISMATCH'}", flush=True)
    failures += v != 7
    v, g, dt = solve(h2, budget_seconds=3600, **MODES["generic"])
    bound = v if v is not None else g.lower_bound_reached()
    verdict = "ok" if v == 7 else "MISMATCH" if v is not None else "timed out"
    print(f"  generic value {v} (bound reached {bound}) nodes={g.nodes} z3={g.oracle.calls} "
          f"{dt:.1f}s {verdict}", flush=True)
    failures += v is not None and v != 7
    return failures


MODES = {
    "path": dict(generic=True, path_only=True),
    "generic": dict(generic=True, path_only=False),
    "pathfull": dict(generic=False, path_only=True),
    "full": dict(generic=False, path_only=False),
}


def enumerate_cmd(n: int, mmax: int, budget_seconds: float, modes: list[str],
                  only_m: int | None = None, pick: list[int] | None = None) -> None:
    tops = enumerate_topologies(n, mmax)
    if only_m is not None:
        tops = [t for t in tops if t.m == only_m]
    if pick is not None:
        tops = [t for i, t in enumerate(tops) if i in pick]
    print(f"# {len(tops)} rooted topologies on {n} vertices with up to {mmax} arcs"
          + (f", m = {only_m} only" if only_m is not None else "")
          + (f", positions {','.join(map(str, pick))}" if pick is not None else ""))
    print("# arcs | n m dag d_s fwd law | " + " | ".join(f"{mode} nodes z3 seconds" for mode in modes))
    for top in tops:
        cells = []
        for mode in modes:
            v, g, dt = solve(top, budget_seconds=budget_seconds, **MODES[mode])
            cell = f"{v}" if v is not None else f">={g.lower_bound_reached()}"
            cells.append(f"{cell} {g.nodes} {g.oracle.calls} {dt:.1f}")
        law = top.dag_law() if top.is_dag() else "-"
        print(f"{str(top)} | {n} {top.m} {int(top.is_dag())} {top.source_out()} {top.fwd()} {law} | "
              + " | ".join(cells), flush=True)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == "validate":
        return validate(quick="--quick" in argv)
    if cmd == "enumerate":
        n, mmax = int(argv[1]), int(argv[2])
        rest = [a for a in argv[3:] if not a.startswith("--")]
        budget = float(rest[0]) if rest else 600.0
        modes = ["path", "generic", "pathfull"]
        only_m = None
        pick = None
        for a in argv[3:]:
            if a.startswith("--modes="):
                modes = a[len("--modes="):].split(",")
            if a.startswith("--m="):
                only_m = int(a[len("--m="):])
            if a.startswith("--pick="):
                pick = [int(x) for x in a[len("--pick="):].split(",")]
        enumerate_cmd(n, mmax, budget, modes, only_m, pick)
        return 0
    if cmd == "strategy":
        top = Topology.parse(argv[1])
        mode = "full"
        for a in argv[2:]:
            if a.startswith("--") and a[2:] in MODES:
                mode = a[2:]
        v, g, dt = solve(top, budget_seconds=3600, **MODES[mode])
        print(f"# {top}  value {v}  nodes {g.nodes}  z3 {g.oracle.calls}  {dt:.1f}s")
        M, tau = g.initial()
        g.strategy(M, tau, v)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
