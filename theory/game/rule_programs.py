#!/usr/bin/env python3
"""Exact worst-case charged cost of uniform rule programs on small rooted topologies.

A rule program is a deterministic comparison-addition program defined by the
topology alone. On a fixed topology it is a decision tree over the weight
cone, and this module walks that tree symbolically: weights are `z3` reals
greater than zero, every comparison branches on each strict outcome the
current cone leaves feasible (the strict adversary of the `generic` and `path`
games of `crates/unsorted-game`), and additions and comparisons are charged
one each. An addition is charged only when it creates a form not already
held; the source-arc forms are held at the start. A comparison whose outcome
the cone already decides is not made (`--charge-decided` charges it). At every
leaf each label's form must equal the distance on the whole cone, checked
against every simple path, so an incorrect program is flagged `WRONG` rather
than scored. The cost of the program is the largest charge over all leaves.

Programs

  dijkstra     push-based Dijkstra with array selection: the least label of
               the frontier by |F| - 1 comparisons, then one addition per
               out-arc to a non-final vertex and one comparison per out-arc
               into an already-labelled vertex.
  local-final  dijkstra, except that a non-final labelled vertex all of whose
               in-neighbours are final is finalised with no selection.
  four-rule    local-final with two preprocessing rules on the topology:
               arcs (u, v) with v dominating u (every s-u path passes v) are
               deleted, and a selection ranges only over the frontier of the
               earliest unfinished strongly connected component in one fixed
               topological order of the components (ties between incomparable
               components broken by least vertex). Every label it holds is a
               simple-path form.
  pooled       four-rule, except that a selection ranges over the frontier of
               every unfinished component of least rank, the rank being the
               longest chain of components above it, so that incomparable
               components share one selection.
  five-rule    four-rule with out-final exclusion: a labelled non-final
               vertex all of whose out-arcs kept by the pruning rule enter
               final vertices takes no part in a selection, and is finalised
               later by the local-final rule. Which vertices are excluded is a
               function of the topology and the final set, so the program is
               still defined by the topology alone. Correctness: let v be the
               selected vertex, P a shortest s-v path, z the last final vertex
               of P and x the vertex after z. As in four-rule, x is labelled
               and lies in the current component. If x != v, the vertex after
               x on P is not final, since z is the last final one, so x is not
               excluded; x was then a candidate, and label(x) <= len(P[s..x])
               < len(P) <= label(v) contradicts the choice of v. The least
               label over the non-excluded frontier is therefore a distance.
               If every labelled non-final vertex of the current component
               were excluded, every non-final vertex of it would be labelled
               (on a path into an unlabelled one, the vertex after the last
               final vertex is labelled, non-final, in the component, and has
               an out-arc into a non-final vertex) and every in-neighbour of
               an excluded vertex would be final (a non-final one would be an
               excluded vertex with an out-arc into a non-final vertex), so
               the local-final rule would already have fired. The fallback of
               selecting over the whole frontier is therefore never reached;
               a row where it is reached is flagged five-rule-FALLBACK.
  six-rule     five-rule with the hub rule, tried after the local-final
               rule and before a selection: call a vertex x h-exact when x is
               labelled and every kept in-neighbour of x other than h is final.
               If a non-final vertex h of the current component has no final
               in-neighbour and every kept in-neighbour of h is h-exact, each
               of them is relaxed into h and h is finalised with no selection.
               Every push skips the out-neighbours the pushed vertex's form
               already runs through; before a hub is finalised no form runs
               through a non-final vertex, so the skip changes nothing there. Among several such h, the one with the most
               in-neighbours is taken, then the least. Correctness: the label of
               an h-exact x is the least length of an s-x path avoiding h, since
               the vertex before x on such a path is a final in-neighbour; a
               shortest s-h path does not pass h before its last arc, so the
               vertex before h on it has its distance as label, and every other candidate is the
               length of a walk into h. A skipped arc u>x closes a walk through
               x, never shorter than the label x already holds. The condition
               that h have no final in-neighbour is not needed for correctness;
               it keeps the rule from replacing a cheaper selection.
  seven-rule   six-rule with a guarded relay (R7), prefix finalisation after
               a selection, and a hub rule that admits a labelled hub.
               Candidates of a selection are the labelled non-final vertices
               of the current component that five-rule does not exclude and
               that are not marked as relayed with their current label.
               Useful arcs of u: kept out-arcs u>v with v not final and not
               on u's form. After the local-final rule and the hub rule, and
               at most once between two finalisations, the program may relay:
               a candidate u is eligible when it has exactly one useful arc
               u>v, v is unlabelled and in the current component, and every
               path in the kept arcs among non-final vertices from another
               labelled non-final vertex into u passes v (a search over the
               non-final vertices of the current component).
               Eligible u are tried by fewest non-final kept in-neighbours,
               then least vertex, and u is relayed into v, relaxing u>v and
               marking u, when for every other candidate x one of two clauses
               holds:
                 H(x)  once x is final and its out-neighbours labelled, no
                       labelled non-final vertex has all its in-neighbours
                       final and the hub rule chooses v as its hub. Reads:
                       the in-neighbours of v and their in-neighbours, the
                       out-neighbours of x, the in-neighbours of every
                       labelled vertex for the local-final test, and the
                       scan of the current component the hub rule makes.
                 D(x)  2 (in_x - in_u) >= a_x + [x has a useful arc into v],
                       in_y the number of kept in-arcs of y from non-final
                       vertices and a_x the number of useful arcs of x.
                       Radius one: the in-arcs of u and x, the out-arcs of x.
               A relay is tagged H when H holds for every other candidate, D
               when D does, HD when both, and mixed otherwise. After a
               selection of v, every non-final vertex on v's form whose label
               is the prefix of that form up to it is finalised with v. The
               hub rule admits a labelled h; since h has no final
               in-neighbour, its label can only come from a relay.
               Correctness: let v be the selected vertex, P a shortest s-v
               path, z the last final vertex of P and x the vertex after z.
               As in five-rule, x is labelled, lies in the current component,
               and if x != v it is not excluded; its label is len(P[s..x])
               since z is final and pushed. If x is marked as relayed, its
               successor y on P is not final and not on x's form (a vertex
               on that form is closer than x, and y is farther), and x>y is
               kept, being on a shortest path; so x>y is x's one useful arc,
               the relay relaxed it with the label x still holds, and the
               label of y is len(P[s..y]): a relayed vertex passes exactness
               to its unique useful successor, which lies in the current
               component with x and v. Repeating along P, the first vertex
               after z that is not marked is a candidate, or is v, with label
               the length of its prefix of P, so the least candidate label is
               a distance. A vertex finalised as a prefix of v's form has a
               prefix of a shortest path as its label. With h labelled, the
               hub rule's argument is unchanged: the label h held is the
               length of a walk through one of its in-neighbours. A selection
               with no candidate left falls back to the whole frontier and is
               flagged seven-rule-FALLBACK.
               What is argued about cost: clause H carries an exchange
               argument for the opening. With |F| candidates, where u's path
               to v is the least label, the relay settles u and v together by
               one selection where six-rule makes at most two; where another
               candidate x is the least label, both select x, and H makes
               six-rule's next step the hub rule on v, which forms
               w(u) + w(u>v) itself, so the relay's addition is one six-rule
               makes as well; where u is the least label and x is below u's
               path to v, six-rule selects u first and its push of u makes
               the same addition. The argument covers the opening and
               does not bound what follows. Clause D carries no argument: it
               compares local terms of the two programs' costs (the
               relaxations into x that six-rule pays while x is unsettled,
               against the useful arcs of x the relay pays for when x wins)
               and is a stated heuristic. Every label is a simple-path form.
  eight-rule   seven-rule with a piece pass, tried after the hub rule and
               before the relay. The non-final vertices of the current
               component, with the kept arcs among them, fall into pieces,
               the components of the underlying undirected graph. A piece is
               eligible when it is a tree on at least two vertices and the
               form of each labelled vertex of it runs through final vertices
               only, besides the vertex itself. Its abstract piece numbers its
               vertices 1.. in increasing order, with a source arc into each
               labelled one and the kept arcs among them. On the abstract
               piece the program compares two costs, each the worst case with
               every comparison charged whatever the evaluation charges:
               seven-rule, and two passes from a root r, which relax every
               arc from child to parent bottom-up and then every arc from
               parent to child top-down over the tree searched breadth-first
               from r, least vertex first, skipping an arc u>v when u's form
               runs through v. Over the eligible pieces in order of least
               vertex, the first on which the passes from the cheapest root
               (least vertex first) cost strictly less than seven-rule is
               settled by those passes on the real state, and all its
               vertices are finalised and pushed; when there is none,
               seven-rule continues. Which piece is passed, from which root,
               is a function of the topology, the final set and the forms
               held, and the abstract costs are finite computations on the
               piece.
               Correctness: a kept in-neighbour of a piece vertex outside the
               piece is final and pushed, since a non-final one of the current
               component would lie in the piece and a later component has no
               arc into the current one. Let P be a shortest s-v path for v in
               the piece, y the last vertex of P outside the piece and x the
               vertex after it. The arcs of P are kept, y is a final, pushed
               in-neighbour of x, so label(x) <= len(P[s..x]), and the rest of
               P lies in the piece, so it is the tree path from x to v. The
               bottom-up pass carries the least label into the root's side of
               each tree edge and the top-down pass carries it out, so after
               both passes the label of v is at most len(P); every label is
               the length of a path, so it is the distance. A skipped arc
               closes a walk. Every label is a simple-path form.
  nine-rule    seven-rule (without the piece pass) with three changes to
               a selection. The out-final exclusion reads only the kept
               out-arcs into the current component: a labelled non-final
               vertex of it all of whose kept out-arcs into the component
               enter final vertices is not a candidate. A candidate y is left
               out when another candidate x holds a label known to be less:
               decided by the cone when decided comparisons are not charged,
               implied by the comparisons made on the branch under the memo
               measure, and never when every comparison is charged. The
               candidates are scanned in increasing order of next(x), the
               number of labelled non-final vertices of the current component
               that the exclusion would keep right after x is finalised and
               its out-neighbours labelled, least vertex first among
               equals, so that the candidate whose win leaves the largest
               next selection is scanned last and the least of the others is known when it
               wins. The relay's candidates are these candidates.
               Correctness: any scan order returns the least candidate, and a
               vertex left out as a known loser is not the least. For the
               exclusion, let v be the selected vertex, P a shortest s-v path,
               z its last final vertex and x the vertex after z; x is labelled
               with len(P[s..x]) and lies in the current component C. If
               x != v, its successor y on P is not final, and y is reached
               from x and reaches v, so y lies in C: x has a kept out-arc into
               a non-final vertex of C and is not excluded. The rest of
               seven-rule's argument is unchanged.
  two-sweep    on the fed bidirected path P_k only: arcs 0>i for every i and
               i>i+1, i+1>i; relax i-1>i for i = 2..k, then i+1>i for
               i = k-1..1, skipping the second when the label of i+1 already
               runs through i. Every label it holds is a simple-path form.
  passes       local-final after a fixed number of alternating fixed-order
               relaxation passes, best over all vertex orders and p <= 2
               (opt-in with --passes; exponential in n).

Inputs

  (default)    every rooted topology on at most four vertices with a recorded
               value: the table compiled into crates/unsorted-game/src/main.rs
               and experiments/2026-10-10-opt-dist-n4-m7.
  --n5         every five-vertex topology with a recorded value:
               experiments/2026-10-10-opt-dist-n5-m6 (both games) and
               experiments/2026-10-10-opt-dist-n5-m7-path (path game).
  --specials   the larger decided topologies listed in SPECIALS, each with the
               claim that holds its value; a row where the four-rule program
               costs more than the value is flagged four-rule-ABOVE-VALUE,
               and likewise for five-rule and six-rule.
  --bipath K,K,...
               the fed bidirected path P_k for each k given: two-sweep and
               four-rule, with the time four-rule took. Every vertex of P_k is
               fed and P_k is one component, so the two preprocessing rules
               delete nothing there and four-rule is local-final.
  --random N,M_LO,M_HI,COUNT,SEED
               COUNT draws of a rooted topology on N vertices (every vertex
               reachable from 0, no arc into 0) with a number of arcs uniform
               in M_LO..M_HI and the arcs uniform among those sets, from
               random.Random(SEED); repeated draws are evaluated once. Values
               are not known for these rows, so four-rule, five-rule and
               six-rule are compared with one another: every row where
               five-rule differs from four-rule or six-rule from five-rule,
               or a program is wrong or falls back, is printed, then the
               counts. All three hold simple-path forms only, so on a row
               where one costs less than another, the dearer one is above the
               path-register value.
  --random-seven N,M_LO,M_HI,COUNT,SEED
               the draws of --random with the same arguments, every distinct
               arc set printed in draw order with the six-rule and seven-rule
               costs, the number of relays over the decision tree and each
               relay as depth:u>v:clause, the depth being the number of final
               vertices; flags seven-rule-BELOW-SIX-RULE, -ABOVE-SIX-RULE,
               -WRONG, -FALLBACK, then the counts. --jobs J evaluates rows in
               J processes; the output is the same.
  --stars J,J,...
               the bidirected star with J fed leaves and an unfed hub, arcs
               0>i, i>h and h>i for i = 1..J with h = J+1: four-rule,
               five-rule, six-rule and seven-rule beside 4J - 3.
  --table FILE,FILE,...
               every row of the stdout of `unsorted-game table` runs, with the
               values they decide; an undecided row is evaluated and left out
               of the comparison with the value. Where its path-register cell
               reads `>=k`, the value is at least k (the solver's lower bound,
               from search and its static bound), and since the four-rule,
               five-rule, six-rule and seven-rule programs hold simple-path
               forms only, it is at most the least cost among those of them
               that hold the distances at every leaf: the row is flagged
               PATH-VALUE-k when that least cost is k, and PATH-VALUE-IN-k..c
               otherwise, PATH-BOUND-CONFLICT when that cost is below k, and
               PATH-BOUND-NONE when all four are wrong. A row that relabels
               a topology of SPECIALS is flagged SPECIAL[name].
  --eight-rule with the default input, --n5, --specials, --table FILE,...,
               --bipath K,..., --random N,M_LO,M_HI,COUNT,SEED, or
               --fed-trees T (every fed bidirected tree on T non-source
               vertices with a nonempty fed set, up to relabelling, by number
               fed then arc list) or --tree-plus T,COUNT,SEED (COUNT distinct
               draws of a uniform labelled tree on T vertices with both arcs
               of each edge, source arcs into 2..T-1 uniform vertices, and one
               more uniform arc, from random.Random(SEED)): seven-rule and
               eight-rule on every row, each piece pass of eight-rule's tree
               as depth:piece@root:pass<seven-rule, flags
               eight-rule-BELOW-SEVEN-RULE, -ABOVE-SEVEN-RULE, -FALLBACK,
               <prog>-WRONG, <prog>-ABOVE-VALUE and -BELOW-VALUE, then the
               counts. A table row the solver leaves at `>=k` is flagged
               PATH-VALUE-k when the lesser correct cost of the two is k, and
               PATH-VALUE-IN-k..c, PATH-BOUND-CONFLICT or PATH-BOUND-NONE
               otherwise, as for --table.
               --jobs J evaluates rows in J processes; the output is the same.
  --nine-rule  with the inputs of --eight-rule: seven-rule and nine-rule on
               every row, with the same flags and counts.
  --charge-memo
               the memo measure, for any input: a comparison is free only
               when the comparisons made earlier on its branch imply its
               outcome by transitivity, a form being less than every form
               that contains it; every other comparison is charged, those
               the cone decides included. It is the cost of a program that
               keeps a record of its comparisons and reads it, with no
               reasoning about the cone.
  --pieces T,B every abstract piece on 2..T vertices, each tree edge one way
               or both below B vertices and both from B, with a nonempty fed
               set and every vertex reachable from the source, up to
               relabelling: seven-rule's cost and the cheapest two
               passes with their root, every comparison charged, and per size
               the pieces where the passes are cheaper.
  --relabel FILE,FILE,...
               reads the stdout of --random runs and counts the rows where
               five-rule costs less up to relabelling of the non-source
               vertices, and checks each against the decided topologies of
               the two tables and SPECIALS.
  --numberings with the default input, --n5, --specials, --table FILE,...,
               --solves FILE,... (the stdout of `unsorted-game solve` runs in
               the path game, one row per `# arcs mode path value V` header)
               or --arcs ARCS;ARCS (no value): every row with a known value,
               or every given row, evaluated on every distinct relabelling of
               its non-source vertices when n <= 6, and on the distinct ones
               among COUNT random relabellings drawn by
               random.Random(SEED).shuffle when n >= 7 (--sample COUNT,SEED,
               default 300,9401, drawn afresh for every row); the arc set as
               recorded is always the first, and each relabelling is written
               with its arcs sorted. With --table, a row the solver leaves at
               `>=k` takes the value k when the least cost of the four
               programs on the arc set as recorded is k, as above, and the
               least cost over every numbering is printed beside it. Per row
               and program: the cost as recorded and the number of numberings
               at each cost; flags <prog>-NUMBERING-DEPENDENT,
               <prog>-ABOVE-VALUE-ON-NUMBERING[arcs] and -BELOW-, -WRONG- and
               -FALLBACK-ON-NUMBERING[arcs], each naming the first numbering
               it holds on; then per program the rows where the cost depends
               on the numbering, and the rows where it equals the value as
               recorded and on every numbering. The programs are four-rule,
               five-rule, six-rule and seven-rule, or those given by
               --programs P,P,...; --jobs J evaluates rows in J processes, and
               the output is the same.

On the tables and SPECIALS a row where five-rule leaves a vertex out of some
selection is flagged five-rule-EXCLUDES, and the number of such rows is
printed after the summary; a row where the hub rule of six-rule applies is
flagged six-rule-HUB, and rows where six-rule costs less or more than
five-rule are flagged six-rule-BELOW-FIVE-RULE and six-rule-ABOVE-FIVE-RULE.
A row where seven-rule relays is flagged seven-rule-RELAY[...] with each relay
of its decision tree as depth:u>v:clause, the number of such rows by clause is
printed after the summary, and rows where seven-rule costs less or more than
six-rule are flagged seven-rule-BELOW-SIX-RULE and seven-rule-ABOVE-SIX-RULE.

The value a program is measured against is the path-register value, the game
whose forms the label-setting programs hold; the unrestricted value is
printed beside it and agrees on every row read here.

The programs read the vertex numbers. Array selection scans the frontier in
increasing order, keeping the lesser label at each comparison; the local-final
rule takes the least eligible vertex; components the topological order leaves
incomparable are taken least vertex first; the hub rule and the relay break
ties by least vertex; a push relaxes out-arcs in the order the arcs are
listed. A comparison whose outcome the cone already decides is not charged,
and which comparisons the cone decides depends on the order in which labels
meet, so the cost of a program is its cost on the arc set as numbered, and two
numberings of one topology can cost different amounts; the value is the same
on every numbering. --numberings measures the cost of each program over the
numberings of a topology, and with --charge-decided the cost with every
comparison charged.

    python theory/game/rule_programs.py [--n5 | --specials | --bipath 3,4,5
                                         | --random 5,8,11,400,1 | --stars 2,3,4
                                         | --random-seven 5,9,11,1500,9101 [--jobs 7]
                                         | --table OUT.txt,OUT.txt
                                         | --relabel OUT.txt,OUT.txt]
                                        [--passes] [--charge-decided]
    python theory/game/rule_programs.py --eight-rule
                                        [--n5 | --specials | --table OUT.txt,OUT.txt
                                         | --bipath 3,4,5 | --random 6,9,11,1500,9301
                                         | --fed-trees 6 | --tree-plus 6,200,9312]
                                        [--charge-decided] [--jobs 3]
    python theory/game/rule_programs.py --nine-rule [inputs of --eight-rule]
                                        [--charge-decided | --charge-memo] [--jobs 3]
    python theory/game/rule_programs.py --pieces 5,5 [--jobs 3]
    python theory/game/rule_programs.py --numberings
                                        [--n5 | --specials | --table OUT.txt,OUT.txt
                                         | --solves OUT.txt,OUT.txt | --arcs 'ARCS;ARCS']
                                        [--programs P,P] [--sample 300,9401] [--jobs 3]
                                        [--charge-decided]
"""
from __future__ import annotations

import itertools
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import z3

ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ("dijkstra", "local-final", "four-rule", "pooled", "five-rule", "six-rule",
            "seven-rule")


# ----------------------------------------------------------------- topology
def parse_arcs(s: str) -> list[tuple[int, int]]:
    arcs = []
    for a in s.split(","):
        u, v = a.split(">")
        arcs.append((int(u), int(v)))
    return arcs


def vertex_count(arcs: list[tuple[int, int]]) -> int:
    return max(max(u, v) for u, v in arcs) + 1


def simple_paths(arcs, n):
    """Arc-count vectors of every simple path from 0, per target vertex."""
    out = {v: [] for v in range(1, n)}
    adj: dict[int, list[tuple[int, int]]] = {}
    for i, (u, v) in enumerate(arcs):
        adj.setdefault(u, []).append((v, i))

    def rec(u, seen, form):
        for v, i in adj.get(u, []):
            if v in seen:
                continue
            f = list(form)
            f[i] += 1
            out[v].append(tuple(f))
            rec(v, seen | {v}, f)

    rec(0, {0}, [0] * len(arcs))
    return out


def reach(arcs, src, banned=()):
    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for a, b in arcs:
            if a == u and b not in seen and b not in banned:
                seen.add(b)
                stack.append(b)
    return seen


def active_arcs(arcs):
    """Indices of arcs (u, v) for which v does not dominate u. An arc into a
    dominator of its tail closes a cycle through that dominator and is never
    on a shortest path."""
    keep = []
    for i, (u, v) in enumerate(arcs):
        if u == 0 or u in reach(arcs, 0, banned=(v,)):
            keep.append(i)
    return keep


def component_rank(arcs, n):
    """Rank of each vertex's strongly connected component in topological
    order: the length of the longest chain of components above it."""
    r = {v: reach(arcs, v) for v in range(n)}
    comp: dict[int, int] = {}
    cid = 0
    for v in range(n):
        if v in comp:
            continue
        for u in range(n):
            if u in r[v] and v in r[u]:
                comp[u] = cid
        cid += 1
    rank: dict[int, int] = {}

    def rk(c):
        if c in rank:
            return rank[c]
        best = 0
        for u in range(n):
            if comp[u] == c:
                for a, b in arcs:
                    if b == u and comp[a] != c:
                        best = max(best, rk(comp[a]) + 1)
        rank[c] = best
        return best

    return {v: rk(comp[v]) for v in range(n)}


def component_order(arcs, n):
    """Position of each vertex's strongly connected component in one
    topological order of the components: Kahn's algorithm, the component with
    the least vertex first among those available."""
    r = {v: reach(arcs, v) for v in range(n)}
    comp = {v: min(u for u in range(n) if u in r[v] and v in r[u]) for v in range(n)}
    comps = sorted(set(comp.values()))
    preds = {c: {comp[a] for a, b in arcs if comp[b] == c and comp[a] != c} for c in comps}
    pos: dict[int, int] = {}
    while len(pos) < len(comps):
        c = min(c for c in comps if c not in pos and preds[c] <= pos.keys())
        pos[c] = len(pos)
    return {v: pos[comp[v]] for v in range(n)}


def fed_cycle(k: int, feeds) -> str:
    """The cycle 1 -> 2 -> ... -> k -> 1 with source arcs into `feeds`."""
    arcs = [f"0>{v}" for v in sorted(feeds)]
    arcs += [f"{i}>{i % k + 1}" for i in range(1, k + 1)]
    return ",".join(arcs)


def fed_bipath(k: int) -> str:
    """The fed bidirected path P_k: 0>i for every i, and i>i+1, i+1>i."""
    arcs = [f"0>{i}" for i in range(1, k + 1)]
    for i in range(1, k):
        arcs += [f"{i}>{i + 1}", f"{i + 1}>{i}"]
    return ",".join(arcs)


# ----------------------------------------------------------------- symbolic run
class Run:
    def __init__(self, arcs, charge_decided=False):
        self.arcs = arcs
        self.n = vertex_count(arcs)
        self.m = len(arcs)
        self.w = [z3.Real(f"w{i}") for i in range(self.m)]
        self.s = z3.Solver()
        for x in self.w:
            self.s.add(x > 0)
        self.paths = simple_paths(arcs, self.n)
        self.charge_decided = bool(charge_decided)
        self.memo = charge_decided == "memo"
        self.facts = []
        self.worst = 0
        self.leaves = 0
        self.wrong = False
        self.exclusions = 0
        self.fallbacks = 0
        self.hubs = 0
        self.relays = 0
        self.relay_tags = []
        self.prefix = 0
        self.passes = 0
        self.pass_tags = []

    def lin(self, f):
        if not any(f):
            return z3.RealVal(0)
        return z3.Sum([c * self.w[i] for i, c in enumerate(f) if c])

    def feasible(self, c):
        self.s.push()
        self.s.add(c)
        r = self.s.check()
        self.s.pop()
        return r == z3.sat

    def implied(self, f, g):
        """Whether f < g follows from the comparisons made on this branch by
        transitivity, a form being less than every form containing it."""
        nodes = {f, g} | {a for a, _ in self.facts} | {b for _, b in self.facts}
        less = {a: set() for a in nodes}
        for a, b in self.facts:
            less[a].add(b)
        for a in nodes:
            for b in nodes:
                if a != b and all(x <= y for x, y in zip(a, b)):
                    less[a].add(b)
        seen, stack = {f}, [f]
        while stack:
            a = stack.pop()
            if a == g:
                return True
            for b in less[a]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        return False

    def cmp(self, f, g, cost, k_lt, k_gt):
        """Branch on f < g and f > g; each continuation takes the cost."""
        if self.memo:
            if self.implied(f, g):
                k_lt(cost)
                return
            if self.implied(g, f):
                k_gt(cost)
                return
            k_lt, k_gt = self._remember(f, g, k_lt), self._remember(g, f, k_gt)
        lf, lg = self.lin(f), self.lin(g)
        a = self.feasible(lf < lg)
        b = self.feasible(lf > lg)
        if a and b:
            self.s.push()
            self.s.add(lf < lg)
            k_lt(cost + 1)
            self.s.pop()
            self.s.push()
            self.s.add(lf > lg)
            k_gt(cost + 1)
            self.s.pop()
        elif a:
            k_lt(cost + (1 if self.charge_decided else 0))
        elif b:
            k_gt(cost + (1 if self.charge_decided else 0))
        # Only equality feasible: a lower-dimensional cone, which the strict
        # adversary never enters.

    def _remember(self, a, b, k):
        """k with a < b recorded for the branch it runs on."""
        def go(c):
            self.facts.append((a, b))
            k(c)
            self.facts.pop()
        return go

    def leaf(self, labels, cost):
        self.leaves += 1
        for v in range(1, self.n):
            f = labels.get(v)
            if f is None:
                self.wrong = True
                continue
            lf = self.lin(f)
            for p in self.paths[v]:
                if p != f and self.feasible(self.lin(p) < lf):
                    self.wrong = True
        self.worst = max(self.worst, cost)


def add_form(f, g):
    return tuple(a + b for a, b in zip(f, g))


def unit(m, i):
    f = [0] * m
    f[i] = 1
    return tuple(f)


# ----------------------------------------------------------------- programs
class Machine:
    """Labels, held forms and the relaxation step shared by the programs.

    Continuation-passing: every step takes the state and a continuation, so a
    comparison runs the rest of the program once per feasible outcome."""

    def __init__(self, run, prune=False, condense=None, exclude=False, hub=False):
        self.run = run
        self.exclude = exclude
        self.hub = hub
        arcs, n, m = run.arcs, run.n, run.m
        active = set(active_arcs(arcs)) if prune else set(range(m))
        self.inn = {v: [(u, i) for i, (u, vv) in enumerate(arcs) if vv == v and i in active]
                    for v in range(n)}
        self.outs = {u: [(v, i) for i, (uu, v) in enumerate(arcs) if uu == u and i in active]
                     for u in range(n)}
        kept = [arcs[i] for i in sorted(active)]
        if condense == "order":
            self.rank = component_order(kept, n)
        elif condense == "rank":
            self.rank = component_rank(kept, n)
        else:
            self.rank = {v: 0 for v in range(n)}

    def relax(self, labels, held, u, v, i, cost, k):
        run, m = self.run, self.run.m
        f = None if u == 0 else labels.get(u)
        if u != 0 and f is None:
            k(labels, held, cost)
            return
        g = unit(m, i) if u == 0 else add_form(f, unit(m, i))
        if ("done", i, g) in held:
            k(labels, held, cost)
            return
        held = held | {("done", i, g)}
        if g not in held:
            held = held | {g}
            cost += 1
        cur = labels.get(v)
        if cur is None:
            k({**labels, v: g}, held, cost)
        elif cur == g:
            k(labels, held, cost)
        else:
            run.cmp(g, cur, cost,
                    lambda c: k({**labels, v: g}, held, c),
                    lambda c: k(labels, held, c))

    def relax_seq(self, labels, held, items, cost, k):
        if not items:
            k(labels, held, cost)
            return
        (u, v, i), rest = items[0], items[1:]
        self.relax(labels, held, u, v, i, cost,
                   lambda L, H, c: self.relax_seq(L, H, rest, c, k))

    def push(self, labels, held, u, final, cost, k):
        f = labels.get(u) if self.hub else None
        items = [(u, v, i) for v, i in self.outs[u] if v not in final
                 and not (f is not None and self.through(f, v))]
        self.relax_seq(labels, held, items, cost, k)

    def through(self, f, v):
        """Whether the path form f enters v."""
        return any(f[i] for i, (_, b) in enumerate(self.run.arcs) if b == v)

    def hub_vertex(self, labels, final):
        """The hub rule of six-rule: (h, its in-arcs) or None."""
        n = self.run.n
        lo = min(self.rank[v] for v in range(n) if v not in final)
        best = None
        for h in range(1, n):
            if h in final or h in labels or self.rank[h] != lo:
                continue
            ins = self.inn[h]
            if not ins or any(u in final for u, _ in ins):
                continue
            if all(u in labels and all(x in final or x == h for x, _ in self.inn[u])
                   for u, _ in ins):
                if best is None or len(ins) > len(best[1]):
                    best = (h, ins)
        return best

    def select(self, labels, final, cost, k):
        lo = min(self.rank[v] for v in range(self.run.n) if v not in final)
        front = [v for v in sorted(labels) if v not in final and self.rank[v] == lo]
        if not front:
            # The earliest unfinished component has no labelled vertex, which
            # cannot happen when every vertex is reachable from the source.
            self.run.wrong = True
            self.run.leaf(labels, cost)
            return
        if self.exclude:
            kept = [v for v in front if not all(x in final for x, _ in self.outs[v])]
            if kept:
                self.run.exclusions += len(front) - len(kept)
                front = kept
            else:
                self.run.fallbacks += 1

        def go(best, rest, c):
            if not rest:
                k(best, c)
                return
            v, rest2 = rest[0], rest[1:]
            self.run.cmp(labels[v], labels[best], c,
                         lambda c2: go(v, rest2, c2),
                         lambda c2: go(best, rest2, c2))

        go(front[0], front[1:], cost)

    def start(self, k):
        held0 = {unit(self.run.m, i) for i in range(self.run.m)}
        self.push(dict(), held0, 0, {0}, 0, k)


def label_setting(run, local_final, prune=False, condense=None, order=None, passes=0,
                  exclude=False, hub=False):
    """dijkstra, local-final, four-rule, pooled, five-rule, and with `passes`
    the passes program. `condense` is None, "order" (four-rule, five-rule) or
    "rank" (pooled); `exclude` turns on out-final exclusion (five-rule), `hub`
    the hub rule (six-rule)."""
    mach = Machine(run, prune=prune, condense=condense, exclude=exclude, hub=hub)
    n = run.n

    def finalise(labels, held, v, final, cost, pass_state):
        final2 = final | {v}
        mach.push(labels, held, v, final2, cost,
                  lambda L, H, c: loop(L, H, final2, c, pass_state))

    def loop(labels, held, final, cost, pass_state):
        if len(final) == n:
            run.leaf(labels, cost)
            return
        if local_final:
            for v in range(1, n):
                if v not in final and v in labels and all(u in final for u, _ in mach.inn[v]):
                    finalise(labels, held, v, final, cost, pass_state)
                    return
        if hub:
            hv = mach.hub_vertex(labels, final)
            if hv is not None:
                h, ins = hv
                run.hubs += 1
                mach.relax_seq(labels, held, [(u, h, i) for u, i in ins], cost,
                               lambda L, H, c: finalise(L, H, h, final, c, pass_state))
                return
        if pass_state is not None and pass_state[0] > 0:
            p, fwd = pass_state
            seq = [v for v in (order if fwd else order[::-1]) if v not in final]
            items = [(u, v, i) for v in seq for u, i in mach.inn[v]]
            mach.relax_seq(labels, held, items, cost,
                           lambda L, H, c: loop(L, H, final, c, (p - 1, not fwd)))
            return
        if not any(v not in final for v in labels):
            run.wrong = True
            run.leaf(labels, cost)
            return
        mach.select(labels, final, cost,
                    lambda v, c: finalise(labels, held, v, final, c, pass_state))

    mach.start(lambda L, H, c: loop(L, H, {0}, c, (passes, True) if passes else None))


def two_sweep(run):
    """Two sweeps on the fed bidirected path; see the module docstring."""
    k = run.n - 1
    idx = {a: i for i, a in enumerate(run.arcs)}
    mach = Machine(run)
    if sorted(run.arcs) != sorted(parse_arcs(fed_bipath(k))):
        raise ValueError("two-sweep is defined on the fed bidirected path only")

    def backward(labels, held, i, cost):
        if i < 1:
            run.leaf(labels, cost)
            return
        through = labels[i + 1][idx[(i, i + 1)]] > 0
        if through:
            backward(labels, held, i - 1, cost)
            return
        mach.relax(labels, held, i + 1, i, idx[(i + 1, i)], cost,
                   lambda L, H, c: backward(L, H, i - 1, c))

    forward = [(i - 1, i, idx[(i - 1, i)]) for i in range(2, k + 1)]
    mach.start(lambda L, H, c: mach.relax_seq(L, H, forward, c,
                                              lambda L2, H2, c2: backward(L2, H2, k - 1, c2)))


class RelayMachine(Machine):
    """The machine of seven-rule: candidates, the relay's eligibility, the
    hub rule admitting a labelled hub, selection over the candidates, and the
    order of a path form."""

    def useful(self, labels, u, final):
        """Kept out-arcs of u into non-final vertices not on u's form."""
        f = labels.get(u)
        return [(v, i) for v, i in self.outs[u]
                if v not in final and not (f is not None and self.through(f, v))]

    def current(self, final):
        return min(self.rank[v] for v in range(self.run.n) if v not in final)

    def candidates(self, labels, final, held):
        """(frontier of the current component, candidates of a selection)."""
        lo = self.current(final)
        front = [v for v in sorted(labels) if v not in final and self.rank[v] == lo]
        kept = [v for v in front
                if not all(x in final for x, _ in self.outs[v])
                and ("relay", v, labels[v]) not in held]
        return front, kept

    def dominated(self, labels, final, u, v):
        """Every path in the kept arcs among non-final vertices from a labelled
        non-final vertex other than u and v into u passes v."""
        roots = [x for x in labels if x not in final and x != u and x != v]
        seen = set(roots)
        stack = list(roots)
        while stack:
            a = stack.pop()
            if a == u:
                return False
            for b, _ in self.outs[a]:
                if b in final or b == v or b in seen:
                    continue
                seen.add(b)
                stack.append(b)
        return u not in seen

    def hub_vertex(self, labels, final):
        """The hub rule of six-rule, admitting a labelled h."""
        n = self.run.n
        lo = min(self.rank[v] for v in range(n) if v not in final)
        best = None
        for h in range(1, n):
            if h in final or self.rank[h] != lo:
                continue
            ins = self.inn[h]
            if not ins or any(u in final for u, _ in ins):
                continue
            if all(u in labels and all(x in final or x == h for x, _ in self.inn[u])
                   for u, _ in ins):
                if best is None or len(ins) > len(best[1]):
                    best = (h, ins)
        return best

    def relay_options(self, labels, final, held):
        """Every eligible (u, v, arc) in the order they are tried."""
        lo = self.current(final)
        _, kept = self.candidates(labels, final, held)
        opts = []
        for u in kept:
            us = self.useful(labels, u, final)
            if len(us) != 1:
                continue
            v, i = us[0]
            if self.rank[v] != lo or v in labels:
                continue
            if not self.dominated(labels, final, u, v):
                continue
            key = (sum(1 for x, _ in self.inn[u] if x not in final), u)
            opts.append((key, u, v, i))
        opts.sort()
        return [(u, v, i) for _, u, v, i in opts]

    def select_candidates(self, labels, final, held, cost, k):
        front, kept = self.candidates(labels, final, held)
        if not front:
            self.run.wrong = True
            self.run.leaf(labels, cost)
            return
        if kept:
            front = kept
        else:
            self.run.fallbacks += 1

        def go(best, rest, c):
            if not rest:
                k(best, c)
                return
            v, rest2 = rest[0], rest[1:]
            self.run.cmp(labels[v], labels[best], c,
                         lambda c2: go(v, rest2, c2),
                         lambda c2: go(best, rest2, c2))

        go(front[0], front[1:], cost)

    def path_order(self, f):
        """(vertex, arc) along the simple path form f from the source."""
        arcs = self.run.arcs
        seq, cur = [], 0
        used = set()
        while True:
            nxt = [(i, b) for i, (a, b) in enumerate(arcs) if a == cur and f[i] and i not in used]
            if not nxt:
                return seq
            i, b = nxt[0]
            used.add(i)
            seq.append((b, i))
            cur = b


class OrderMachine(RelayMachine):
    """The machine of nine-rule: the out-final exclusion over arcs into the
    current component, known losers left out of a selection, and a scan
    order by the size of the selection each candidate's win leaves."""

    def candidates(self, labels, final, held):
        lo = self.current(final)
        front = [v for v in sorted(labels) if v not in final and self.rank[v] == lo]
        kept = [v for v in front
                if not all(x in final or self.rank[x] != lo for x, _ in self.outs[v])
                and ("relay", v, labels[v]) not in held]
        kept = [y for y in kept
                if not any(x != y and self.known_less(labels[x], labels[y]) for x in kept)]
        return front, kept

    def known_less(self, f, g):
        """Whether f < g is free under the measure: decided by the cone when
        decided comparisons are not charged, implied by the branch's own
        comparisons under the memo measure, never when all are charged."""
        run = self.run
        if run.memo:
            return run.implied(f, g)
        if run.charge_decided:
            return False
        return not run.feasible(run.lin(f) > run.lin(g))

    def next_size(self, labels, final, x):
        """Labelled non-final vertices of the current component that the
        exclusion keeps right after x is finalised and its out-neighbours
        labelled."""
        lo = self.current(final)
        f2 = final | {x}
        lab = {v for v in labels if v not in f2} | {y for y, _ in self.outs[x] if y not in f2}
        return sum(1 for v in lab if self.rank[v] == lo
                   and not all(y in f2 or self.rank[y] != lo for y, _ in self.outs[v]))

    def select_candidates(self, labels, final, held, cost, k):
        front, kept = self.candidates(labels, final, held)
        if not front:
            self.run.wrong = True
            self.run.leaf(labels, cost)
            return
        if kept:
            front = kept
        else:
            self.run.fallbacks += 1
        front = sorted(front, key=lambda x: (self.next_size(labels, final, x), x))

        def go(best, rest, c):
            if not rest:
                k(best, c)
                return
            v, rest2 = rest[0], rest[1:]
            self.run.cmp(labels[v], labels[best], c,
                         lambda c2: go(v, rest2, c2),
                         lambda c2: go(best, rest2, c2))

        go(front[0], front[1:], cost)


def nonfinal_in(mach, final, y):
    """Kept in-neighbours of y that are not final."""
    return [a for a, _ in mach.inn[y] if a not in final]


def clause_h(mach, labels, final, v, x):
    """H(x): once x is final and its out-neighbours labelled, no labelled
    non-final vertex has all its in-neighbours final, and the hub rule
    chooses v."""
    final2 = final | {x}
    labelled = set(labels)
    labelled |= {b for b, _ in mach.outs[x] if b not in final2}
    for y in labelled:
        if y not in final2 and all(a in final2 for a, _ in mach.inn[y]):
            return False
    hv = mach.hub_vertex({y: None for y in labelled}, final2)
    return hv is not None and hv[0] == v


def clause_d(mach, labels, final, u, v, x):
    """D(x): 2 (in_x - in_u) >= a_x + [x has a useful arc into v]."""
    in_x = len(nonfinal_in(mach, final, x))
    in_u = len(nonfinal_in(mach, final, u))
    us_x = mach.useful(labels, x, final)
    xv = int(any(b == v for b, _ in us_x))
    return 2 * (in_x - in_u) >= len(us_x) + xv


def choose_relay(mach, labels, final, options, kept):
    """The first eligible option for which every other candidate satisfies H
    or D, with its clause tag, or None."""
    for u, v, i in options:
        others = [x for x in kept if x != u]
        hs = [clause_h(mach, labels, final, v, x) for x in others]
        ds = [clause_d(mach, labels, final, u, v, x) for x in others]
        if all(h or d for h, d in zip(hs, ds)):
            tag = ("H" if all(hs) else "") + ("D" if all(ds) else "")
            return (u, v, i), tag or "mixed"
    return None


def seven_rule(run, piece_pass=False, machine=None):
    """six-rule with the guarded relay, prefix finalisation and the hub rule
    admitting a labelled hub; with `piece_pass`, eight-rule, and with
    OrderMachine as `machine`, nine-rule. See the module docstring."""
    mach = (machine or RelayMachine)(run, prune=True, condense="order", exclude=True, hub=True)
    n = run.n
    arc_of = {(u, v): i for v in range(n) for u, i in mach.inn[v]}

    def push_all(labels, held, vs, final, cost, k):
        if not vs:
            k(labels, held, cost)
            return
        mach.push(labels, held, vs[0], final, cost,
                  lambda L, H, c: push_all(L, H, vs[1:], final, c, k))

    def finalise(labels, held, v, final, cost, with_prefix=False):
        vs = [v]
        if with_prefix:
            pre = [0] * run.m
            extra = []
            for y, i in mach.path_order(labels[v]):
                pre[i] = 1
                if y == v:
                    break
                if y not in final and labels.get(y) == tuple(pre):
                    extra.append(y)
            run.prefix += len(extra)
            vs = extra + [v]
        final2 = final | set(vs)
        push_all(labels, held, vs, final2, cost, lambda L, H, c: loop(L, H, final2, c, False))

    def loop(labels, held, final, cost, relayed):
        if len(final) == n:
            run.leaf(labels, cost)
            return
        for v in range(1, n):
            if v not in final and v in labels and all(u in final for u, _ in mach.inn[v]):
                finalise(labels, held, v, final, cost)
                return
        hv = mach.hub_vertex(labels, final)
        if hv is not None:
            h, ins = hv
            run.hubs += 1
            mach.relax_seq(labels, held, [(u, h, i) for u, i in ins], cost,
                           lambda L, H, c: finalise(L, H, h, final, c))
            return
        d = piece_choice(mach, labels, final) if piece_pass else None
        if d is not None:
            piece, adj, root, pc, sc = d
            run.passes += 1
            run.pass_tags.append((len(final), tuple(piece), root, pc, sc))
            items = tree_items(adj, set(piece), root, arc_of)
            final2 = final | set(piece)
            relax_skip(mach, labels, held, items, cost,
                       lambda L, H, c: push_all(L, H, list(piece), final2, c,
                                                lambda L2, H2, c2: loop(L2, H2, final2, c2, False)))
            return
        if not relayed:
            options = mach.relay_options(labels, final, held)
            if options:
                _, kept = mach.candidates(labels, final, held)
                choice = choose_relay(mach, labels, final, options, kept)
                if choice is not None:
                    (u, v, i), tag = choice
                    run.relay_tags.append((len(final), u, v, tag))
                    run.relays += 1
                    mark = ("relay", u, labels[u])
                    mach.relax(labels, held | {mark}, u, v, i, cost,
                               lambda L, H, c: loop(L, H, final, c, True))
                    return
        if not any(v not in final for v in labels):
            run.wrong = True
            run.leaf(labels, cost)
            return
        mach.select_candidates(labels, final, held, cost,
                               lambda v, c: finalise(labels, held, v, final, c, with_prefix=True))

    mach.start(lambda L, H, c: loop(L, H, {0}, c, False))


def relay_string(tags):
    """Each relay as depth:u>v:clause, the depth being the number of final
    vertices."""
    return ",".join("%d:%d>%d:%s" % t for t in tags) if tags else "-"


# Worst-case cost with every comparison charged, of seven-rule and of two passes
# from each root, on each abstract piece met so far (per process).
_PIECE_SEVEN: dict[str, tuple[int, bool]] = {}
_PIECE_PASS: dict[tuple[str, int], tuple[int, bool]] = {}


def tree_items(adj, verts, root, arc_of):
    """The relaxations (u, v, arc) of two passes over a tree rooted at root:
    child into parent bottom-up, then parent into child top-down, each where
    the arc exists. The tree is searched breadth-first, least vertex first."""
    parent = {root: None}
    pre = [root]
    q = [root]
    while q:
        x = q.pop(0)
        for y in sorted(adj[x]):
            if y in verts and y not in parent:
                parent[y] = x
                pre.append(y)
                q.append(y)
    up, down = [], []
    for v in reversed(pre):
        p = parent[v]
        if p is not None and (v, p) in arc_of:
            up.append((v, p, arc_of[(v, p)]))
    for v in pre:
        p = parent[v]
        if p is not None and (p, v) in arc_of:
            down.append((p, v, arc_of[(p, v)]))
    return up + down


def relax_skip(mach, labels, held, items, cost, k):
    """Relax each (u, v, arc) in turn, skipping it when u's form runs through v."""
    if not items:
        k(labels, held, cost)
        return
    (u, v, i), rest = items[0], items[1:]
    f = labels.get(u)
    if f is not None and mach.through(f, v):
        relax_skip(mach, labels, held, rest, cost, k)
        return
    mach.relax(labels, held, u, v, i, cost,
               lambda L, H, c: relax_skip(mach, L, H, rest, c, k))


def piece_pass_cost(abs_arcs, root):
    """(cost, wrong) of two passes from root on an abstract piece, every
    comparison charged."""
    key = (abs_arcs, root)
    if key not in _PIECE_PASS:
        run = Run(parse_arcs(abs_arcs), True)
        mach = Machine(run)
        verts = set(range(1, run.n))
        adj = {v: set() for v in verts}
        arc_of = {}
        for i, (u, v) in enumerate(run.arcs):
            if u != 0:
                adj[u].add(v)
                adj[v].add(u)
                arc_of[(u, v)] = i
        items = tree_items(adj, verts, root, arc_of)
        mach.start(lambda L, H, c: relax_skip(mach, L, H, items, c,
                                              lambda L2, H2, c2: run.leaf(L2, c2)))
        _PIECE_PASS[key] = (run.worst, run.wrong)
    return _PIECE_PASS[key]


def piece_seven_cost(abs_arcs):
    """(cost, wrong) of seven-rule on an abstract piece, every comparison
    charged."""
    if abs_arcs not in _PIECE_SEVEN:
        c, w, _ = evaluate(abs_arcs, "seven-rule", True)
        _PIECE_SEVEN[abs_arcs] = (c, w)
    return _PIECE_SEVEN[abs_arcs]


def abstract_piece(mach, labels, piece):
    """The abstract piece of a sorted piece: its vertices renumbered 1.. in
    order, a source arc into each labelled one, and the kept arcs among them."""
    idx = {v: j + 1 for j, v in enumerate(piece)}
    arcs = [(0, idx[v]) for v in piece if v in labels]
    arcs += [(idx[u], idx[v]) for v in piece for u, _ in mach.inn[v] if u in idx]
    return ",".join(f"{a}>{b}" for a, b in sorted(arcs)), idx


def piece_choice(mach, labels, final):
    """(piece, adjacency, root, pass cost, seven-rule cost) for the first
    eligible piece of the current component, in order of least vertex, whose
    abstract piece two passes from some root settle strictly cheaper than
    seven-rule, both with every comparison charged; the root is the cheapest,
    least vertex first. None if there is no such piece."""
    lo = mach.current(final)
    vs = {v for v in range(mach.run.n) if v not in final and mach.rank[v] == lo}
    adj = {v: set() for v in vs}
    edges = set()
    for v in vs:
        for u, _ in mach.inn[v]:
            if u in vs and u != v:
                adj[v].add(u)
                adj[u].add(v)
                edges.add(frozenset((u, v)))
    seen = set()
    for r in sorted(vs):
        if r in seen:
            continue
        piece, stack = [], [r]
        seen.add(r)
        while stack:
            x = stack.pop()
            piece.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        piece.sort()
        if len(piece) < 2 or sum(1 for e in edges if e <= set(piece)) != len(piece) - 1:
            continue
        if not all(label_through_final(mach, labels, final, v) for v in piece):
            continue
        abs_arcs, idx = abstract_piece(mach, labels, piece)
        s, _ = piece_seven_cost(abs_arcs)
        best = None
        for v in piece:
            c, w = piece_pass_cost(abs_arcs, idx[v])
            if not w and (best is None or c < best[0]):
                best = (c, v)
        if best is not None and best[0] < s:
            return piece, adj, best[1], best[0], s
    return None


def label_through_final(mach, labels, final, v):
    """Whether v is unlabelled or its form runs through final vertices only,
    besides v itself."""
    f = labels.get(v)
    if f is None:
        return True
    return all(not c or mach.run.arcs[i][1] in final or mach.run.arcs[i][1] == v
               for i, c in enumerate(f))


def pass_string(tags):
    """Each piece pass as depth:piece@root:pass<seven-rule, the depth being the
    number of final vertices."""
    if not tags:
        return "-"
    return ";".join(f"{d}:{'.'.join(map(str, p))}@{r}:{pc}<{sc}" for d, p, r, pc, sc in tags)


def evaluate(arcs_s, program, charge_decided=False, stats=None, **kw):
    """(worst-case cost, flagged wrong, number of leaves) of one program.
    With `stats` a dict, the number of vertices excluded from selections and of
    selections that fell back to the whole frontier, of hub-rule steps and of
    relays, summed over the tree, are stored in it under "exclusions",
    "fallbacks", "hubs" and "relays", and the relays as (depth, u, v, clause)
    under "relay_tags"."""
    run = Run(parse_arcs(arcs_s), charge_decided)
    if program == "dijkstra":
        label_setting(run, local_final=False)
    elif program == "local-final":
        label_setting(run, local_final=True)
    elif program == "four-rule":
        label_setting(run, local_final=True, prune=True, condense="order")
    elif program == "pooled":
        label_setting(run, local_final=True, prune=True, condense="rank")
    elif program == "five-rule":
        label_setting(run, local_final=True, prune=True, condense="order", exclude=True)
    elif program == "six-rule":
        label_setting(run, local_final=True, prune=True, condense="order", exclude=True, hub=True)
    elif program == "seven-rule":
        seven_rule(run)
    elif program == "eight-rule":
        seven_rule(run, piece_pass=True)
    elif program == "nine-rule":
        seven_rule(run, machine=OrderMachine)
    elif program == "two-sweep":
        two_sweep(run)
    elif program == "passes":
        label_setting(run, local_final=True, **kw)
    else:
        raise ValueError(program)
    if stats is not None:
        stats["exclusions"] = run.exclusions
        stats["fallbacks"] = run.fallbacks
        stats["hubs"] = run.hubs
        stats["relays"] = run.relays
        stats["relay_tags"] = list(run.relay_tags)
        stats["passes"] = run.passes
        stats["pass_tags"] = sorted(set(run.pass_tags))
    return run.worst, run.wrong, run.leaves


def best_passes(arcs_s, charge_decided=False):
    n = vertex_count(parse_arcs(arcs_s))
    best = None
    for order in itertools.permutations(range(1, n)):
        for p in (1, 2):
            w, wrong, _ = evaluate(arcs_s, "passes", charge_decided, order=list(order), passes=p)
            if not wrong and (best is None or w < best[0]):
                best = (w, order, p)
    return best


# ----------------------------------------------------------------- inputs
def _value(cell: str):
    tok = cell.split()[0] if cell.split() else ""
    return int(tok) if tok.isdigit() else None


def solver_table(path: Path, n: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        gv = _value(parts[3]) if len(parts) > 3 else None
        rows.append((parts[0], n, _value(parts[2]), gv))
    return rows


def path_bounds(path: Path):
    """{arcs: k} for each row whose path-register search stopped at its
    budget, the cell reading `>=k`."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        tok = parts[2].split()[0] if len(parts) > 2 and parts[2].split() else ""
        if tok.startswith(">=") and tok[2:].isdigit():
            out[parts[0]] = int(tok[2:])
    return out


def recorded_n4():
    src = (ROOT / "crates/unsorted-game/src/main.rs").read_text(encoding="utf-8")
    body = src.split("const RECORDED")[1].split("];")[0]
    rows = []
    for m_ in re.finditer(r'\("([^"]+)",\s*(Some\((\d+)\)|None),\s*(Some\((\d+)\)|None)\)', body):
        arcs = m_.group(1)
        pv = int(m_.group(3)) if m_.group(3) else None
        gv = int(m_.group(5)) if m_.group(5) else None
        rows.append((arcs, vertex_count(parse_arcs(arcs)), pv, gv))
    rows += solver_table(ROOT / "experiments/2026-10-10-opt-dist-n4-m7/stdout.txt", 4)
    return rows


def recorded_n5():
    rows = []
    for name in ("experiments/2026-10-10-opt-dist-n5-m6/stdout.txt",
                 "experiments/2026-10-10-opt-dist-n5-m7-path/stdout.txt"):
        rows += solver_table(ROOT / name, 5)
    return rows


# Decided topologies outside the two tables: (name, arcs, path-register value,
# claim holding the value). Where the unrestricted value is decided it is equal.
SPECIALS = [
    ("C_4", fed_cycle(4, (1, 2, 3, 4)), 9, "C-070"),
    ("H_2, 1>2 subdivided", "0>1,0>2,0>3,1>4,4>2,2>1,2>3,3>2", 8, "C-179"),
    ("bidirected triangle fed at 1, 2", "0>1,0>2,1>2,1>3,2>1,2>3,3>1,3>2", 7, "C-069"),
    ("fed bidirected triangle less a chord", "0>1,0>2,0>3,1>2,1>3,2>1,2>3,3>1", 9, "C-074"),
    ("complete, four vertices", "0>1,0>2,0>3,1>2,1>3,2>1,2>3,3>1,3>2", 9, "C-073"),
]
SPECIALS += [(f"cycle k={k} fed at {','.join(map(str, F))}", fed_cycle(k, F), v, "C-180")
             for k, F, v in [(2, (1,), 1), (2, (1, 2), 3), (3, (1, 2, 3), 6), (3, (1, 2), 4),
                             (4, (1, 3), 5), (4, (1, 2, 3), 7), (5, (1,), 4), (5, (1, 4), 6),
                             (5, (1, 2, 3), 8), (5, (1, 2, 4), 8), (6, (1, 2), 7), (6, (1, 4), 7),
                             (6, (1, 2, 3), 9), (7, (1, 5), 8)]]
SPECIALS += [
    ("cycle k=6 fed at 1,3,5", fed_cycle(6, (1, 3, 5)), 9, "C-181"),
    ("cycle k=8 fed at 1,5", fed_cycle(8, (1, 5)), 9, "C-182"),
    ("cycle k=7 fed at 1,3,5", fed_cycle(7, (1, 3, 5)), 10, "C-183"),
    ("C_2 and C_2 sharing the source", "0>1,0>2,1>2,2>1,0>3,0>4,3>4,4>3", 6, "C-185"),
    ("C_2 and fed P_2 sharing the source", "0>1,0>2,1>2,2>1,0>3,0>4,3>4", 5, "C-185"),
    ("C_2 and half-fed C_2 sharing the source", "0>1,0>2,1>2,2>1,0>3,3>4,4>3", 4, "C-185"),
    ("C_3 and half-fed C_2 sharing the source", "0>1,0>2,0>3,1>2,2>3,3>1,0>4,4>5,5>4", 7, "C-185"),
    ("C_2 and C_3 sharing the source", "0>1,0>2,1>2,2>1,0>3,0>4,0>5,3>4,4>5,5>3", 9, "C-186"),
    ("C_3 and fed P_2 sharing the source", "0>1,0>2,0>3,1>2,2>3,3>1,0>4,0>5,4>5", 8, "C-188"),
]
SPECIALS += [(f"glued pieces {i + 1}", arcs, v, "C-189") for i, (arcs, v) in enumerate([
    ("0>1,0>2,1>2,2>1,2>3,1>4,3>4,4>3", 8),
    ("0>1,0>2,1>2,2>1,2>3,2>4,3>4,4>3", 8),
    ("0>1,0>2,1>2,2>1,0>3,2>3,3>2", 7),
    ("0>1,0>2,0>3,1>2,2>3,3>1,3>4", 7),
    ("0>1,1>2,0>3,2>3,3>2", 4),
    ("0>1,0>2,1>2,2>1,2>3", 4),
    ("0>1,0>2,0>3,1>2,2>1,2>3", 5),
    ("0>1,0>2,0>3,1>2,2>1,3>1", 5),
    ("0>1,1>2,2>1,0>3,2>3,3>1", 4),
    ("0>1,1>2,2>1,0>3,3>4,4>3,2>3,4>1", 5),
])]
SPECIALS += [
    ("star, hub 4 not fed, leaves 1, 2, 3 fed", "0>1,0>2,0>3,1>4,2>4,3>4,4>1,4>2,4>3", 9, "C-219"),
    ("C_2 fed at 2 and C_2 fed at 4, glued both ways", "4>1,0>2,1>2,2>1,2>3,0>4,3>4,4>3", 8, "C-218"),
    ("cycle k=5 fed at 1,2,3,4", fed_cycle(5, (1, 2, 3, 4)), 10, "C-211"),
    ("cycle k=6 fed at 1,2,4,5", fed_cycle(6, (1, 2, 4, 5)), 11, "C-214"),
]


# ----------------------------------------------------------------- reports
def five_against_four(costs):
    four, five = costs[PROGRAMS.index("four-rule")], costs[PROGRAMS.index("five-rule")]
    six = costs[PROGRAMS.index("six-rule")]
    flags = []
    if five < four:
        flags.append("five-rule-BELOW-FOUR-RULE")
    if five > four:
        flags.append("five-rule-ABOVE-FOUR-RULE")
    if six < five:
        flags.append("six-rule-BELOW-FIVE-RULE")
    if six > five:
        flags.append("six-rule-ABOVE-FIVE-RULE")
    return flags


def seven_against_six(costs):
    six, seven = costs[PROGRAMS.index("six-rule")], costs[PROGRAMS.index("seven-rule")]
    flags = []
    if seven < six:
        flags.append("seven-rule-BELOW-SIX-RULE")
    if seven > six:
        flags.append("seven-rule-ABOVE-SIX-RULE")
    return flags


def relay_summary(relaying, rows):
    by = Counter(relaying)
    print(f"# seven-rule relays on {len(relaying)} of {rows} rows"
          + (" (" + ", ".join(f"clause {k} on {v}" for k, v in sorted(by.items())) + ")"
             if by else ""))


def gap_summary(gaps):
    for k, g in gaps.items():
        if g:
            exact = sum(1 for x in g if x == 0)
            print(f"# {k}: rows {len(g)}, exact on {exact}, cost - V histogram {sorted(Counter(g).items())}")


def run_table(rows, charge, with_passes, bounds=None):
    print("arcs | n m | V_path V_gen | " + " ".join(PROGRAMS)
          + (" passes(order,p)" if with_passes else "") + " | flags")
    gaps = {p: [] for p in PROGRAMS}
    excluding = 0
    relaying = []
    undecided = []
    if with_passes:
        gaps["passes"] = []
    for arcs, n, pv, gv in rows:
        V = pv if pv is not None else gv
        m = len(parse_arcs(arcs))
        costs, flags = [], []
        for p in PROGRAMS:
            st = {}
            c, wrong, _ = evaluate(arcs, p, charge, stats=st)
            costs.append(c)
            if wrong:
                flags.append(f"{p}-WRONG")
            if st["fallbacks"]:
                flags.append(f"{p}-FALLBACK")
            if st["exclusions"] and p == "five-rule":
                flags.append(f"{p}-EXCLUDES")
                excluding += 1
            if st["hubs"]:
                flags.append(f"{p}-HUB")
            if st["relays"]:
                flags.append(f"{p}-RELAY[{relay_string(st['relay_tags'])}]")
                relaying.append("+".join(sorted({t[3] for t in st["relay_tags"]})))
            if V is not None:
                gaps[p].append(c - V)
                if c < V:
                    flags.append(f"{p}-BELOW-VALUE")
        if pv is not None and gv is not None and pv != gv:
            flags.append("GAMES-DIFFER")
        if bounds is not None:
            name = special_names().get((n, m, canonical(arcs)))
            if name:
                flags.append(f"SPECIAL[{name}]")
        if bounds and arcs in bounds and V is None:
            lo = bounds[arcs]
            ups = [costs[PROGRAMS.index(p)]
                   for p in ("four-rule", "five-rule", "six-rule", "seven-rule")
                   if f"{p}-WRONG" not in flags]
            hi = min(ups) if ups else None
            if hi is None:
                flags.append("PATH-BOUND-NONE")
            elif hi < lo:
                flags.append(f"PATH-BOUND-CONFLICT-{lo}>{hi}")
            else:
                undecided.append((lo, hi))
                flags.append(f"PATH-VALUE-{lo}" if hi == lo else f"PATH-VALUE-IN-{lo}..{hi}")
        flags += five_against_four(costs)
        flags += seven_against_six(costs)
        ps = ""
        if with_passes:
            y = best_passes(arcs, charge)
            ps = f" {y[0]}({''.join(map(str, y[1]))},{y[2]})" if y else " -"
            if y and V is not None:
                gaps["passes"].append(y[0] - V)
        print(f"{arcs} | {n} {m} | {pv if pv is not None else '-'} {gv if gv is not None else '-'} | "
              f"{' '.join(map(str, costs))}{ps} | {' '.join(flags)}")
    gap_summary(gaps)
    print(f"# five-rule leaves a vertex out of a selection on {excluding} of {len(gaps['five-rule'])} rows")
    relay_summary(relaying, len(rows))
    if undecided:
        closed = sum(1 for lo, hi in undecided if lo == hi)
        print(f"# undecided rows {len(undecided)}: the least of the correct four-rule, five-rule, six-rule and "
              f"seven-rule costs equals "
              f"the solver's lower bound on {closed}, deciding the path-register value; open intervals "
              f"{sorted(Counter((lo, hi) for lo, hi in undecided if lo != hi).items())}")


def run_specials(charge):
    print("name | arcs | n m | V claim | " + " ".join(PROGRAMS) + " | flags")
    gaps = {p: [] for p in PROGRAMS}
    excluding = 0
    relaying = []
    for name, arcs, V, claim in SPECIALS:
        a = parse_arcs(arcs)
        costs, flags = [], []
        for p in PROGRAMS:
            st = {}
            c, wrong, _ = evaluate(arcs, p, charge, stats=st)
            costs.append(c)
            gaps[p].append(c - V)
            if wrong:
                flags.append(f"{p}-WRONG")
            if st["fallbacks"]:
                flags.append(f"{p}-FALLBACK")
            if st["exclusions"] and p == "five-rule":
                flags.append(f"{p}-EXCLUDES")
                excluding += 1
            if st["hubs"]:
                flags.append(f"{p}-HUB")
            if st["relays"]:
                flags.append(f"{p}-RELAY[{relay_string(st['relay_tags'])}]")
                relaying.append("+".join(sorted({t[3] for t in st["relay_tags"]})))
            if c < V:
                flags.append(f"{p}-BELOW-VALUE")
            if p in ("four-rule", "five-rule", "six-rule", "seven-rule") and c > V:
                flags.append(f"{p}-ABOVE-VALUE")
        flags += five_against_four(costs)
        flags += seven_against_six(costs)
        print(f"{name} | {arcs} | {vertex_count(a)} {len(a)} | {V} {claim} | "
              f"{' '.join(map(str, costs))} | {' '.join(flags)}")
    gap_summary(gaps)
    print(f"# five-rule leaves a vertex out of a selection on {excluding} of {len(gaps['five-rule'])} rows")
    relay_summary(relaying, len(SPECIALS))


def star(j: int) -> str:
    """The bidirected star with j fed leaves 1..j and the unfed hub j+1."""
    h = j + 1
    arcs = [f"0>{i}" for i in range(1, j + 1)]
    arcs += [f"{i}>{h}" for i in range(1, j + 1)] + [f"{h}>{i}" for i in range(1, j + 1)]
    return ",".join(arcs)


def run_stars(js, charge):
    print("j | n m | 4j-3 | four-rule five-rule six-rule seven-rule | flags")
    for j in js:
        arcs = star(j)
        costs, flags = [], []
        for p in ("four-rule", "five-rule", "six-rule", "seven-rule"):
            st = {}
            c, wrong, _ = evaluate(arcs, p, charge, stats=st)
            costs.append(c)
            if wrong:
                flags.append(f"{p}-WRONG")
            if st["fallbacks"]:
                flags.append(f"{p}-FALLBACK")
            if st["relays"]:
                flags.append(f"{p}-RELAY[{relay_string(st['relay_tags'])}]")
        print(f"{j} | {j + 2} {3 * j} | {4 * j - 3} | {' '.join(map(str, costs))} | "
              f"{' '.join(flags)}", flush=True)


def run_bipath(ks, charge):
    print("k | n m | 4k-4 | two-sweep | four-rule | four-rule leaves seconds | flags")
    for k in ks:
        arcs = fed_bipath(k)
        a = parse_arcs(arcs)
        flags = []
        sweep, wrong, _ = evaluate(arcs, "two-sweep", charge)
        if wrong:
            flags.append("two-sweep-WRONG")
        t = time.perf_counter()
        four, wrong, leaves = evaluate(arcs, "four-rule", charge)
        t = time.perf_counter() - t
        if wrong:
            flags.append("four-rule-WRONG")
        if four > sweep:
            flags.append("four-rule-ABOVE-TWO-SWEEP")
        print(f"{k} | {k + 1} {len(a)} | {4 * k - 4} | {sweep} | {four} | {leaves} {t:.1f} | "
              f"{' '.join(flags)}", flush=True)


def random_rooted(n, m, rng):
    arcs_all = [(u, v) for u in range(n) for v in range(1, n) if u != v]
    while True:
        arcs = rng.sample(arcs_all, m)
        if len(reach(arcs, 0)) == n:
            return ",".join(f"{u}>{v}" for u, v in sorted(arcs))


def run_random(n, m_lo, m_hi, count, seed, charge):
    rng = random.Random(seed)
    print(f"# n {n}, m {m_lo}..{m_hi}, {count} draws, seed {seed}")
    print("arcs | n m | four-rule five-rule six-rule | flags")
    seen = set()
    stats = Counter()
    for _ in range(count):
        arcs = random_rooted(n, rng.randint(m_lo, m_hi), rng)
        if arcs in seen:
            continue
        seen.add(arcs)
        c4, w4, _ = evaluate(arcs, "four-rule", charge)
        st = {}
        c5, w5, _ = evaluate(arcs, "five-rule", charge, stats=st)
        st6 = {}
        c6, w6, _ = evaluate(arcs, "six-rule", charge, stats=st6)
        stats["rows"] += 1
        stats["five-rule excludes"] += st["exclusions"] > 0
        stats["five-rule below"] += c5 < c4
        stats["five-rule above"] += c5 > c4
        stats["six-rule hub"] += st6["hubs"] > 0
        stats["six-rule below five"] += c6 < c5
        stats["six-rule above five"] += c6 > c5
        stats["wrong"] += w4 or w5 or w6
        stats["fallback"] += st["fallbacks"] > 0 or st6["fallbacks"] > 0
        flags = five_against_four([{"four-rule": c4, "five-rule": c5, "six-rule": c6}.get(p, 0)
                                   for p in PROGRAMS])
        for name, w in (("four-rule", w4), ("five-rule", w5), ("six-rule", w6)):
            if w:
                flags.append(f"{name}-WRONG")
        if st["fallbacks"]:
            flags.append("five-rule-FALLBACK")
        if st6["fallbacks"]:
            flags.append("six-rule-FALLBACK")
        if st6["hubs"]:
            flags.append("six-rule-HUB")
        if c5 != c4 or c6 != c5 or w4 or w5 or w6 or st["fallbacks"] or st6["fallbacks"]:
            print(f"{arcs} | {n} {len(parse_arcs(arcs))} | {c4} {c5} {c6} | {' '.join(flags)}",
                  flush=True)
    print("# " + ", ".join(f"{k} {stats[k]}" for k in
                           ("rows", "five-rule excludes", "five-rule below", "five-rule above",
                            "six-rule hub", "six-rule below five", "six-rule above five",
                            "wrong", "fallback")))


def random_draws(n, m_lo, m_hi, count, seed):
    """The distinct arc sets of --random's draws, in draw order."""
    rng = random.Random(seed)
    seen, rows = set(), []
    for _ in range(count):
        arcs = random_rooted(n, rng.randint(m_lo, m_hi), rng)
        if arcs not in seen:
            seen.add(arcs)
            rows.append(arcs)
    return rows


def _six_seven(job):
    arcs, charge = job
    c6, w6, _ = evaluate(arcs, "six-rule", charge)
    st = {}
    c7, w7, _ = evaluate(arcs, "seven-rule", charge, stats=st)
    return arcs, c6, w6, c7, w7, st


def run_random_seven(n, m_lo, m_hi, count, seed, charge, jobs=1):
    rows = random_draws(n, m_lo, m_hi, count, seed)
    print(f"# n {n}, m {m_lo}..{m_hi}, {count} draws, seed {seed}, {len(rows)} distinct arc sets")
    print("arcs | n m | six-rule seven-rule | relays | relay steps | flags")
    work = [(a, charge) for a in rows]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.Pool(jobs) as pool:
            results = list(pool.imap(_six_seven, work, chunksize=8))
    else:
        results = map(_six_seven, work)
    stats = Counter()
    clauses = Counter()
    for arcs, c6, w6, c7, w7, st in results:
        flags = []
        if w6:
            flags.append("six-rule-WRONG")
        if w7:
            flags.append("seven-rule-WRONG")
        if st["fallbacks"]:
            flags.append("seven-rule-FALLBACK")
        if c7 < c6:
            flags.append("seven-rule-BELOW-SIX-RULE")
        if c7 > c6:
            flags.append("seven-rule-ABOVE-SIX-RULE")
        stats["rows"] += 1
        stats["relaying"] += st["relays"] > 0
        stats["below"] += c7 < c6
        stats["above"] += c7 > c6
        stats["wrong"] += bool(w6 or w7)
        stats["fallback"] += st["fallbacks"] > 0
        if st["relays"]:
            clauses["+".join(sorted({t[3] for t in st["relay_tags"]}))] += 1
        if c7 < c6:
            clauses["below " + "+".join(sorted({t[3] for t in st["relay_tags"]}))] += 1
        print(f"{arcs} | {n} {len(parse_arcs(arcs))} | {c6} {c7} | {st['relays']} | "
              f"{relay_string(st['relay_tags'])} | {' '.join(flags)}", flush=True)
    print(f"# rows {stats['rows']}, seven-rule relays on {stats['relaying']}, "
          f"below six-rule {stats['below']}, above six-rule {stats['above']}, "
          f"wrong {stats['wrong']}, fallback {stats['fallback']}")
    print("# rows by clause: " + (", ".join(f"{k} {v}" for k, v in sorted(clauses.items())) or "none"))


def special_names():
    """{(n, m, canonical arc list): name} over SPECIALS."""
    global _SPECIAL_NAMES
    if _SPECIAL_NAMES is None:
        _SPECIAL_NAMES = {}
        for name, arcs, _, _ in SPECIALS:
            a = parse_arcs(arcs)
            _SPECIAL_NAMES[(vertex_count(a), len(a), canonical(arcs))] = name
    return _SPECIAL_NAMES


_SPECIAL_NAMES = None


def canonical(arcs_s):
    """The least sorted arc list over every relabelling of the non-source
    vertices: equal for two arc sets exactly when one relabels the other."""
    arcs = parse_arcs(arcs_s)
    n = vertex_count(arcs)
    best = None
    for perm in itertools.permutations(range(1, n)):
        lab = (0,) + perm
        key = tuple(sorted((lab[u], lab[v]) for u, v in arcs))
        if best is None or key < best:
            best = key
    return best


def run_relabel(paths):
    """Reads the rows a --random run flags five-rule-BELOW-FOUR-RULE and counts
    them up to relabelling of the non-source vertices, per number of vertices,
    and checks each against every decided topology of the tables and SPECIALS
    with the same numbers of vertices and arcs."""
    below = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "five-rule-BELOW-FOUR-RULE" in line and not line.startswith("#"):
                below.append(line.split("|")[0].strip())
    sizes = {(vertex_count(parse_arcs(a)), len(parse_arcs(a))) for a in below}
    decided = [r[0] for r in recorded_n4() + recorded_n5() if r[2] is not None or r[3] is not None]
    decided += [arcs for _, arcs, _, _ in SPECIALS]
    decided_canon = {canonical(a) for a in decided
                     if (vertex_count(parse_arcs(a)), len(parse_arcs(a))) in sizes}
    print("arcs | n m | relabelling class | decided")
    classes: dict[tuple, int] = {}
    per_n: dict[int, set] = {}
    for a in below:
        c = canonical(a)
        classes.setdefault(c, len(classes))
        n = vertex_count(parse_arcs(a))
        per_n.setdefault(n, set()).add(c)
        print(f"{a} | {n} {len(parse_arcs(a))} | {classes[c]} | {'yes' if c in decided_canon else 'no'}")
    hits = sum(1 for c in classes if c in decided_canon)
    print(f"# arc sets {len(below)}, topologies up to relabelling {len(classes)} ("
          + ", ".join(f"{len(v)} on {n} vertices" for n, v in sorted(per_n.items()))
          + f"), relabellings of a decided topology {hits}")


# ----------------------------------------------------------------- eight-rule
def prufer_trees(t):
    """The edge lists of the labelled trees on 1..t, one per Pruefer sequence."""
    if t == 1:
        yield []
        return
    if t == 2:
        yield [(1, 2)]
        return
    for seq in itertools.product(range(1, t + 1), repeat=t - 2):
        deg = {v: 1 for v in range(1, t + 1)}
        for x in seq:
            deg[x] += 1
        edges = []
        for x in seq:
            leaf = min(v for v in deg if deg[v] == 1)
            edges.append((leaf, x))
            deg[leaf] -= 1
            deg[x] -= 1
        u, v = [w for w in deg if deg[w] == 1]
        edges.append((u, v))
        yield edges


def tree_pieces(t, bidirected_only):
    """(arcs, number fed) of every rooted topology whose non-source part is a
    tree on t vertices with each edge one way or both (both only, with
    `bidirected_only`), whose fed set is nonempty and whose every vertex is
    reachable from the source, up to relabelling; each
    as its canonical arc list, in order of first appearance."""
    seen = set()
    for edges in prufer_trees(t):
        orients = [(2,)] * len(edges) if bidirected_only else [(0, 1, 2)] * len(edges)
        for o in itertools.product(*orients):
            inner = []
            for (a, b), d in zip(edges, o):
                if d in (0, 2):
                    inner.append((a, b))
                if d in (1, 2):
                    inner.append((b, a))
            for r in range(1, t + 1):
                for lab in itertools.combinations(range(1, t + 1), r):
                    arcs = sorted([(0, x) for x in lab] + inner)
                    if len(reach(arcs, 0)) != t + 1:
                        continue
                    c = canonical(",".join(f"{a}>{b}" for a, b in arcs))
                    if c in seen:
                        continue
                    seen.add(c)
                    yield ",".join(f"{a}>{b}" for a, b in c), len(lab)


def tree_plus_draws(t, count, seed):
    """COUNT distinct rooted topologies: a uniform labelled tree on 1..t (a
    Pruefer sequence from random.Random(SEED)) with both arcs of every edge,
    source arcs into a uniform set of 2..t-1 vertices, and one further arc
    uniform among the absent ones between non-source vertices."""
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < count:
        seq = [rng.randint(1, t) for _ in range(t - 2)]
        deg = {v: 1 for v in range(1, t + 1)}
        for x in seq:
            deg[x] += 1
        edges = []
        for x in seq:
            leaf = min(v for v in deg if deg[v] == 1)
            edges.append((leaf, x))
            deg[leaf] -= 1
            deg[x] -= 1
        u, v = [w for w in deg if deg[w] == 1]
        edges.append((u, v))
        arcs = set()
        for a, b in edges:
            arcs |= {(a, b), (b, a)}
        for x in rng.sample(range(1, t + 1), rng.randint(2, t - 1)):
            arcs.add((0, x))
        extra = [(a, b) for a in range(1, t + 1) for b in range(1, t + 1)
                 if a != b and (a, b) not in arcs]
        arcs.add(rng.choice(extra))
        arcs_s = ",".join(f"{a}>{b}" for a, b in sorted(arcs))
        if len(reach(sorted(arcs), 0)) == t + 1 and arcs_s not in seen:
            seen.add(arcs_s)
            rows.append(arcs_s)
    return rows


def _seven_eight(job):
    name, arcs, value, bound, charge, prog = job
    c7, w7, _ = evaluate(arcs, "seven-rule", charge)
    st = {}
    c8, w8, _ = evaluate(arcs, prog, charge, stats=st)
    return name, arcs, value, bound, c7, w7, c8, w8, st


def eight_rows(argv):
    """[(name, arcs, value, solver lower bound)] of the input --eight-rule reads."""
    if "--specials" in argv:
        return [(name, arcs, v, None) for name, arcs, v, _ in SPECIALS]
    if "--bipath" in argv:
        return [(f"P_{k}", fed_bipath(k), None, None)
                for k in (int(x) for x in argv[argv.index("--bipath") + 1].split(","))]
    if "--random" in argv:
        n, m_lo, m_hi, count, seed = (int(x) for x in argv[argv.index("--random") + 1].split(","))
        return [("-", a, None, None) for a in random_draws(n, m_lo, m_hi, count, seed)]
    if "--fed-trees" in argv:
        t = int(argv[argv.index("--fed-trees") + 1])
        rows = sorted(tree_pieces(t, True), key=lambda r: (r[1], r[0]))
        return [("-", a, None, None) for a, _ in rows]
    if "--tree-plus" in argv:
        t, count, seed = (int(x) for x in argv[argv.index("--tree-plus") + 1].split(","))
        return [("-", a, None, None) for a in tree_plus_draws(t, count, seed)]
    if "--table" in argv:
        rows = []
        for path in argv[argv.index("--table") + 1].split(","):
            bounds = path_bounds(Path(path))
            for arcs, _, pv, _ in solver_table(Path(path), 0):
                rows.append(("-", arcs, pv, bounds.get(arcs)))
        return rows
    rows = recorded_n5() if "--n5" in argv else recorded_n4()
    return [("-", arcs, pv, None) for arcs, _, pv, _ in rows]


def charge_string(charge):
    if charge == "memo":
        return "charged unless the branch's own comparisons imply them"
    return "charged" if charge else "not charged"


def run_eight(rows, charge, jobs=1, prog="eight-rule"):
    print(f"# {prog} against seven-rule, comparisons the cone decides "
          + charge_string(charge)
          + ("; piece costs always charged" if prog == "eight-rule" else ""))
    print(f"name | arcs | V | seven-rule {prog} | piece passes | flags")
    work = [(name, arcs, v, b, charge, prog) for name, arcs, v, b in rows]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.Pool(jobs) as pool:
            results = pool.imap(_seven_eight, work, chunksize=1)
            _eight_report(results, prog)
    else:
        _eight_report(map(_seven_eight, work), prog)


def _eight_report(results, p8="eight-rule"):
    cnt = Counter()
    for name, arcs, value, bound, c7, w7, c8, w8, st in results:
        flags = []
        cnt["rows"] += 1
        for prog, w in (("seven-rule", w7), (p8, w8)):
            if w:
                flags.append(f"{prog}-WRONG")
                cnt[f"{prog} wrong"] += 1
        if st["fallbacks"]:
            flags.append(f"{p8}-FALLBACK")
            cnt[f"{p8} fallback"] += 1
        if value is None and bound is not None:
            ups = [c for c, w in ((c7, w7), (c8, w8)) if not w]
            if not ups:
                flags.append("PATH-BOUND-NONE")
            elif min(ups) == bound:
                flags.append(f"PATH-VALUE-{bound}")
                value = bound
                cnt["decided by bound"] += 1
            elif min(ups) < bound:
                flags.append("PATH-BOUND-CONFLICT")
            else:
                flags.append(f"PATH-VALUE-IN-{bound}..{min(ups)}")
        if value is not None:
            cnt["with value"] += 1
            for prog, c in (("seven-rule", c7), (p8, c8)):
                if c > value:
                    flags.append(f"{prog}-ABOVE-VALUE")
                    cnt[f"{prog} above value"] += 1
                if c < value:
                    flags.append(f"{prog}-BELOW-VALUE")
                    cnt[f"{prog} below value"] += 1
        if c8 < c7:
            flags.append(f"{p8}-BELOW-SEVEN-RULE")
            cnt[f"{p8} below seven-rule"] += 1
        if c8 > c7:
            flags.append(f"{p8}-ABOVE-SEVEN-RULE")
            cnt[f"{p8} above seven-rule"] += 1
        if st["passes"]:
            cnt["pass fires"] += 1
        vcell = f">={bound}" if bound is not None else ("-" if value is None else str(value))
        print(f"{name} | {arcs} | {vcell} | {c7} {c8} | {pass_string(st['pass_tags'])} | "
              f"{' '.join(flags)}", flush=True)
    keys = ("rows", "with value", "decided by bound", "seven-rule above value",
            "seven-rule below value", f"{p8} above value", f"{p8} below value",
            f"{p8} below seven-rule", f"{p8} above seven-rule", "pass fires",
            "seven-rule wrong", f"{p8} wrong", f"{p8} fallback")
    print("# " + ", ".join(f"{k} {cnt[k]}" for k in keys))


def _piece_job(item):
    arcs, nl = item
    s, sw = piece_seven_cost(arcs)
    n = vertex_count(parse_arcs(arcs))
    best = None
    for r in range(1, n):
        c, w = piece_pass_cost(arcs, r)
        if not w and (best is None or c < best[0]):
            best = (c, r)
    return arcs, n - 1, nl, s, sw, best


def run_pieces(t_max, bidirected_from, jobs=1):
    items = []
    for t in range(2, t_max + 1):
        items += list(tree_pieces(t, t >= bidirected_from))
    print(f"# {len(items)} pieces: trees on 2..{t_max} vertices, each edge one way or both"
          f" below {bidirected_from} vertices and both from {bidirected_from}, a nonempty"
          " fed set; every comparison charged")
    print("arcs | t fed | seven-rule pass@root | pass - seven-rule")
    if jobs > 1:
        import multiprocessing
        pool = multiprocessing.Pool(jobs)
        results = pool.imap(_piece_job, items, chunksize=4)
    else:
        pool, results = None, map(_piece_job, items)
    cnt = Counter()
    for arcs, t, nl, s, sw, best in results:
        pc = "-" if best is None else f"{best[0]}@{best[1]}"
        d = "-" if best is None else best[0] - s
        cnt[(t, "pieces")] += 1
        if best is not None and best[0] < s:
            cnt[(t, "pass cheaper")] += 1
        print(f"{arcs} | {t} {nl} | {s}{' WRONG' if sw else ''} {pc} | {d}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    for t in range(2, t_max + 1):
        print(f"# t {t}: pieces {cnt[(t, 'pieces')]}, two passes cheaper than seven-rule on "
              f"{cnt[(t, 'pass cheaper')]}")


# ----------------------------------------------------------------- numberings
NUMBERED = ("four-rule", "five-rule", "six-rule", "seven-rule")
EXHAUSTIVE_N = 6


def relabelled(arcs, lab) -> str:
    return ",".join(f"{a}>{b}" for a, b in sorted((lab[u], lab[v]) for u, v in arcs))


def numberings(arcs_s, count, seed):
    """The arc set as recorded, then the other distinct relabellings of its
    non-source vertices, each with its arcs sorted: all of them when
    n <= EXHAUSTIVE_N, else those among `count` shuffles by random.Random(seed)."""
    arcs = parse_arcs(arcs_s)
    n = vertex_count(arcs)
    out = [arcs_s]
    seen = {relabelled(arcs, tuple(range(n)))}
    if n <= EXHAUSTIVE_N:
        perms = (tuple(p) for p in itertools.permutations(range(1, n)))
    else:
        rng = random.Random(seed)

        def draws():
            p = list(range(1, n))
            for _ in range(count):
                rng.shuffle(p)
                yield tuple(p)

        perms = draws()
    for p in perms:
        b = relabelled(arcs, (0,) + p)
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _numbered(job):
    """Per program, (cost, wrong, fallbacks) on each numbering of one row."""
    arcs_s, programs, charge, count, seed = job
    nums = numberings(arcs_s, count, seed)
    res = {}
    for p in programs:
        res[p] = []
        for b in nums:
            st = {}
            c, wrong, _ = evaluate(b, p, charge, stats=st)
            res[p].append((c, wrong, st["fallbacks"]))
    return nums, res


def solve_rows(paths):
    """(arcs, value) from the `# arcs  mode path  value V` header lines of
    `unsorted-game solve` runs."""
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            m_ = re.match(r"# (\S+)\s+mode path\s+value (\d+)\b", line)
            if m_:
                rows.append((m_.group(1), int(m_.group(2))))
    return rows


def run_numberings(rows, programs, charge, count, seed, jobs=1, bounds=None):
    """rows: (arcs, value or None, note). A row with a `>=k` bound in `bounds`
    and no value takes k when the least correct cost of the four programs on
    the arc set as recorded is k."""
    big = sum(1 for a, _, _ in rows if vertex_count(parse_arcs(a)) > EXHAUSTIVE_N)
    print(f"# {len(rows)} rows; every distinct relabelling of the non-source vertices when "
          f"n <= {EXHAUSTIVE_N} ({len(rows) - big} rows), the distinct ones among {count} "
          f"random relabellings by random.Random({seed}).shuffle when n > {EXHAUSTIVE_N} "
          f"({big} rows); the arc set as recorded first; comparisons decided by the cone "
          + charge_string(charge))
    print("arcs | n m | V | numberings | " + " ; ".join(
        f"{p} as-recorded [cost x count]" for p in programs) + " | flags")
    work = [(a, programs, charge, count, seed) for a, _, _ in rows]
    if jobs > 1:
        import multiprocessing
        pool = multiprocessing.Pool(jobs)
        results = pool.imap(_numbered, work, chunksize=1)
    else:
        pool = None
        results = map(_numbered, work)
    stats = {p: Counter() for p in programs}
    known = 0
    undecided = []
    for (arcs, V, note), (nums, res) in zip(rows, results):
        a = parse_arcs(arcs)
        flags = []
        if V is None and bounds and arcs in bounds:
            lo = bounds[arcs]
            ups = [res[p][0][0] for p in NUMBERED if p in res and not res[p][0][1]]
            anyup = [c for p in NUMBERED if p in res for c, w, _ in res[p] if not w]
            hi, hi_any = (min(ups) if ups else None), (min(anyup) if anyup else None)
            if hi is None:
                flags.append("PATH-BOUND-NONE")
            elif hi < lo or (hi_any is not None and hi_any < lo):
                flags.append(f"PATH-BOUND-CONFLICT-{lo}>{min(hi, hi_any)}")
            else:
                flags.append(f"PATH-VALUE-{lo}" if hi == lo else f"PATH-VALUE-IN-{lo}..{hi}")
                if hi_any < hi:
                    flags.append(f"PATH-VALUE-{lo}-ON-NUMBERING" if hi_any == lo
                                 else f"PATH-VALUE-IN-{lo}..{hi_any}-ON-NUMBERING")
                if hi == lo:
                    V, note = lo, "by cost"
                else:
                    undecided.append((lo, hi, hi_any))
        known += V is not None
        cells = []
        for p in programs:
            cs = [c for c, _, _ in res[p]]
            hist = sorted(Counter(cs).items())
            cells.append(f"{p} {cs[0]} [" + " ".join(f"{c}x{k}" for c, k in hist) + "]")
            st = stats[p]
            st["rows"] += 1
            if len(hist) > 1:
                flags.append(f"{p}-NUMBERING-DEPENDENT")
                st["dependent"] += 1
            for tag, hit in (("ABOVE-VALUE", lambda c, w, f: V is not None and c > V),
                             ("BELOW-VALUE", lambda c, w, f: V is not None and c < V),
                             ("WRONG", lambda c, w, f: w),
                             ("FALLBACK", lambda c, w, f: f > 0)):
                ex = next((b for b, r in zip(nums, res[p]) if hit(*r)), None)
                if ex is not None:
                    flags.append(f"{p}-{tag}-ON-NUMBERING[{ex}]")
                    st[tag] += 1
            if V is not None:
                st["exact as recorded"] += cs[0] == V
                st["exact on every numbering"] += all(c == V for c in cs)
        v_cell = "-" if V is None else f"{V}" + (f" {note}" if note else "")
        print(f"{arcs} | {vertex_count(a)} {len(a)} | {v_cell} | {len(nums)} | "
              f"{' ; '.join(cells)} | {' '.join(flags)}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    print(f"# rows {len(rows)}, with a known value {known}")
    for p in programs:
        st = stats[p]
        print(f"# {p}: depends on the numbering on {st['dependent']} rows; equals the value as "
              f"recorded on {st['exact as recorded']} of {known} and on every numbering on "
              f"{st['exact on every numbering']}; above the value on some numbering on "
              f"{st['ABOVE-VALUE']}, below on {st['BELOW-VALUE']}; wrong on {st['WRONG']}, "
              f"fallback on {st['FALLBACK']}")
    if undecided:
        print(f"# undecided rows {len(undecided)}: (lower bound, least cost as recorded, least cost "
              f"on any numbering) {sorted(Counter(undecided).items())}")


def numbering_rows(argv):
    """(rows, bounds) for --numberings: rows (arcs, value or None, note)."""
    bounds = None
    if "--table" in argv:
        bounds = {}
        rows = []
        for path in argv[argv.index("--table") + 1].split(","):
            bs = path_bounds(Path(path))
            for arcs, _, pv, gv in solver_table(Path(path), 0):
                V = pv if pv is not None else gv
                if V is not None or arcs in bs:
                    rows.append((arcs, V, ""))
            bounds.update(bs)
    elif "--solves" in argv:
        rows = [(a, v, "solve") for a, v in solve_rows(argv[argv.index("--solves") + 1].split(","))]
    elif "--arcs" in argv:
        rows = [(a, None, "") for a in argv[argv.index("--arcs") + 1].split(";")]
    elif "--specials" in argv:
        rows = [(arcs, V, claim) for _, arcs, V, claim in SPECIALS]
    else:
        recorded = recorded_n5() if "--n5" in argv else recorded_n4()
        rows = [(arcs, pv if pv is not None else gv, "") for arcs, _, pv, gv in recorded
                if pv is not None or gv is not None]
    return rows, bounds


def main(argv):
    charge = "memo" if "--charge-memo" in argv else "--charge-decided" in argv
    for prog in ("eight-rule", "nine-rule"):
        if f"--{prog}" in argv:
            rows = eight_rows(argv)
            if "--limit" in argv:
                rows = rows[: int(argv[argv.index("--limit") + 1])]
            jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
            run_eight(rows, charge, jobs, prog)
            return 0
    if "--pieces" in argv:
        t_max, bfrom = (int(x) for x in argv[argv.index("--pieces") + 1].split(","))
        jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
        run_pieces(t_max, bfrom, jobs)
        return 0
    if "--numberings" in argv:
        rows, bounds = numbering_rows(argv)
        if "--limit" in argv:
            rows = rows[: int(argv[argv.index("--limit") + 1])]
        programs = (argv[argv.index("--programs") + 1].split(",") if "--programs" in argv
                    else list(NUMBERED))
        count, seed = ((int(x) for x in argv[argv.index("--sample") + 1].split(","))
                       if "--sample" in argv else (300, 9401))
        jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
        run_numberings(rows, programs, charge, count, seed, jobs, bounds)
        return 0
    if "--bipath" in argv:
        ks = [int(x) for x in argv[argv.index("--bipath") + 1].split(",")]
        run_bipath(ks, charge)
        return 0
    if "--stars" in argv:
        run_stars([int(x) for x in argv[argv.index("--stars") + 1].split(",")], charge)
        return 0
    if "--relabel" in argv:
        run_relabel(argv[argv.index("--relabel") + 1].split(","))
        return 0
    if "--random-seven" in argv:
        n, m_lo, m_hi, count, seed = (int(x) for x in
                                      argv[argv.index("--random-seven") + 1].split(","))
        jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
        run_random_seven(n, m_lo, m_hi, count, seed, charge, jobs)
        return 0
    if "--random" in argv:
        n, m_lo, m_hi, count, seed = (int(x) for x in argv[argv.index("--random") + 1].split(","))
        run_random(n, m_lo, m_hi, count, seed, charge)
        return 0
    if "--specials" in argv:
        run_specials(charge)
        return 0
    bounds = None
    if "--table" in argv:
        bounds = {}
        rows = []
        for path in argv[argv.index("--table") + 1].split(","):
            for arcs, _, pv, gv in solver_table(Path(path), 0):
                rows.append((arcs, vertex_count(parse_arcs(arcs)), pv, gv))
            bounds.update(path_bounds(Path(path)))
    else:
        rows = recorded_n5() if "--n5" in argv else recorded_n4()
    if "--limit" in argv:
        rows = rows[: int(argv[argv.index("--limit") + 1])]
    run_table(rows, charge, "--passes" in argv, bounds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
