//! Exact distance-only complexity of a small topology, by the transcript-cone
//! game of Cai, arXiv:2609.04825, Theorem J.3, under the strict adversary.
//!
//! Two games are played. `generic` lets the program hold any form and has the
//! adversary answer every comparison strictly; its value is a lower bound on
//! `OPT_DIST`, and is the game the paper's own certificate for `H_2` is in.
//! `path` restricts the program to forms below some path coefficientwise, the
//! path-register programs the paper names as the weaker model, under the same
//! adversary; it is an upper bound on `generic`, and where the two agree the
//! strict-adversary value is exact. The Python reference in
//! `theory/game/opt_dist.py` plays the same two games with `z3` deciding the
//! cones, and also the two games against the full adversary, which this
//! program does not.
//!
//! Two more modes charge comparisons only. `path-free` holds every sub-path
//! form from the start, so its value is the least worst-case number of
//! comparisons between two sums of sub-paths that locates the cell of the
//! weight vector; `generic-free` holds every 0/1 form, so its queries are
//! every linear form with coefficients in {-1, 0, 1}. Both are the depth of a
//! linear decision tree for the distance-only problem with additions free, and
//! each is at least `log2` of the number of cells, the number of shortest-path
//! trees the topology admits.
//!
//!     unsorted-game validate [--quick]
//!     unsorted-game enumerate N MMAX [SECONDS] [--modes=M,M] [--m=K] [--pick=I,I]
//!     unsorted-game table ARCS[;ARCS...] [SECONDS] [--modes=M,M]
//!     unsorted-game solve ARCS [--mode=MODE] [--seconds=S] [--min-entry | --feed-regions=SPEC [OPTIONS] | --cell-regions=I,J | --given=X:Y;... --hold=F;...] [--from=B] [--strategy]
//!     unsorted-game frontier ARCS AMAX [--mode=MODE] [--seconds=S] [--all-sub] [--strategy]
//!     unsorted-game certify ARCS R --out=FILE
//!
//! `certify` shows that the unrestricted program cannot win within `R` under
//! the strict adversary and writes a refutation certificate to `FILE`, in the
//! format `game/certify.rs` describes, for `scripts/check_refutation.py` to
//! check without this program.
//!
//! `frontier` budgets comparisons and additions separately and prints, for
//! every addition budget from `fwd` to `AMAX`, the least comparison budget a
//! program needs on every branch; with `--all-sub` the path-register program
//! starts holding every sub-path form and only comparisons are counted.
//!
//! With `--min-entry`, `solve` restricts the adversary: of the two answers to
//! a comparison it must give one that keeps the most cells with the fewest
//! direct source arcs in their tree, and chooses only between answers that
//! tie. The program still has to locate the cell, so the value is a lower
//! bound on the strict-adversary value.
//!
//! With `--feed-regions=V[:KIND[:K]]`, `solve` confines the adversary to
//! regions around the source arc `s>V`, of weight `b`. `above` is the one
//! region where `b` exceeds the sum of all other weights, so that no shortest
//! path uses `s>V` when `V` has another path from `s` (the regions need one;
//! without it `solve` stops). `below` is one region for each simple path `P` from `s`
//! to `V` avoiding `s>V`: `P` is the strict shortest such path and
//! `P - W/K < b < P`, `W` the sum of all weights, so `b` undercuts the
//! distance to `V` through its other in-arcs by less than `W/K` (no lower
//! limit when `K = 0`; `K` is at most 100). `low` is the one region where `b`
//! is below every other source arc, the whole orthant when there is none. `near` is `below` mirrored, one region per `P` with
//! `P < b < P + W/K`. `both` takes the union of `above` and `below`, `wedge`
//! that of `near` and `below`, and `all` every region but the thin ones below. `KIND` defaults to `both` and
//! `K` to 16. A confined adversary can only lower the value, so the value is
//! a lower bound on the strict-adversary value.
//!
//! The thin kinds separate scales against every other weight instead of against
//! `W`. `tbelow` and `tnear` are `below` and `near` with `|b - P| < w/K` for
//! every other arc weight `w`, so `b` deviates from the distance to `V`
//! through its other in-arcs by less than any weight; `thin` is their union.
//! `tlow` is the one region where `b < w/K` for every other arc weight `w`,
//! and `thuge` the one where `b` exceeds `K` times the sum of all other
//! weights. `tq` is one region per in-neighbour `u` of `V` with a source arc
//! and side: `|b - (b_u + c(u>V))| < w/K` for every other weight `w`, the path
//! `s>u>V` not required to be shortest. Kinds are joined with `+`, as in
//! `thin+tlow`.
//!
//! The thin kinds need `K >= 1`; `all` does not include them. An unknown
//! component of `KIND` stops `solve`.
//!
//! Options next to `--feed-regions` (each is echoed in the result line; all
//! but `--hold` and `--given` are ignored without it):
//! - `--keep=KIND`: the adversary must keep some region of `KIND` meeting the
//!   cone after every answer; an answer may then be forced (a strategy prints
//!   `(forced)`), and the solver prunes counting the cells of kept regions
//!   only.
//! - `--prefer=KIND`: of the answers allowed, the adversary gives only those
//!   that keep a region of `KIND` meeting the cone whenever one does; the
//!   solver then prunes with the least chain of a counted cell.
//! - `--small=T>H[:K]`: in every region, the weight of arc `T>H` is below
//!   every other weight but `b`, divided by `K` (16 by default, 1 to 100).
//! - `--hold=F;F` holds the listed forms before the first move, and
//!   `--given=X:Y;X:Y` asserts `X > Y` in every region, so the game starts
//!   from a prefix; a form is written as arcs joined with `+`, as `0>1+1>2`.
//!
//! Each confined adversary only restricts the strict adversary, so the value
//! is again a lower bound on the strict-adversary value (from the prefix, for
//! `--hold` and `--given`).
//!
//! Without `--feed-regions` or `--cell-regions`, `--hold` and `--given` start
//! the unconfined game from that prefix: its value is the strict-adversary
//! value of the game whose root cone carries the given answers and whose
//! program already holds the listed forms. The result line echoes them as
//! `prefix`. This path tags cells with one region, so it takes at most
//! seven labels, as `--feed-regions` does; it combines with neither
//! `--min-entry` nor `--cell-regions`. In path mode a held form should be a
//! sub-path form, or the value is not a path-register value.
//!
//! `--from=B` makes `solve` try budgets from `B` up, taking each lower one as
//! lost without searching it; the value printed is then the least winning
//! budget at or above `B`, the strict-adversary value only when `B` is at most
//! that value. The result line echoes `from B`, and a run stopped by its time
//! limit reports only the budgets it refuted.
//!
//! With `--cell-regions=I,J,...`, `solve` confines the adversary to the
//! listed cells of the root cone, numbered as `--cell-regions=list` prints
//! them; the value is again a lower bound on the strict-adversary value.
//!
//! Every command takes `--cache-gb=G`, the memory the caches of one game may
//! hold (4 by default). The caches only memoise, so the bound changes the
//! time a search takes and never a value or a least budget. Node counts, the
//! `states` and `cones` counts, and the forced loss under `--min-entry` are
//! counted over the states the search expanded, which do depend on the bound;
//! `solve` prints how many cache generations were dropped, and where that is
//! zero they are those of an unbounded run.
//!
//! An arc list is written `0>1,0>2,1>2`; the source is vertex 0. A mode is
//! one of `path`, `generic`, `path-free`, `generic-free`.

#![allow(clippy::needless_range_loop)]

mod game;
mod lp;
mod topo;

use std::time::{Duration, Instant};

use game::{set_cache_bytes, Certifier, Game, Mode};
use lp::{Cvec, MAXM};
use topo::{enumerate_topologies, lane, unit, Form, Topology};

/// The atomic region kinds of `--feed-regions`.
const ATOMS: [&str; 9] = [
    "above", "below", "near", "low", "tbelow", "tnear", "tlow", "thuge", "tq",
];

/// The atomic kinds a `KIND` names: its components joined with `+`, each an
/// atomic kind or one of the unions `both`, `wedge`, `thin`, `all`.
fn kind_atoms(kind: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    for c in kind.split('+') {
        let atoms: Vec<&'static str> = match c {
            "both" => vec!["above", "below"],
            "wedge" => vec!["near", "below"],
            "thin" => vec!["tnear", "tbelow"],
            "all" => vec!["above", "below", "near", "low"],
            _ => vec![*ATOMS
                .iter()
                .find(|&&a| a == c)
                .unwrap_or_else(|| panic!("unknown feed-regions kind {c}"))],
        };
        out.extend(atoms);
    }
    out
}

/// A form written as arcs joined with '+', e.g. `0>1+1>2`.
fn form_of(top: &Topology, t: &str) -> Form {
    t.split('+')
        .map(|a| {
            let (x, y) = a.split_once('>').expect("arc T>H");
            let arc = (x.parse::<u8>().unwrap(), y.parse::<u8>().unwrap());
            unit(
                top.arcs
                    .iter()
                    .position(|&z| z == arc)
                    .expect("no such arc"),
            )
        })
        .sum()
}

/// The rows of `--given=X:Y;...`, each answer `X > Y` for forms `X`, `Y`.
fn given_rows(top: &Topology, gv: &str) -> Vec<Cvec> {
    gv.split(';')
        .map(|pair| {
            let (x, y) = pair.split_once(':').expect("--given=X:Y");
            let (fx, fy) = (form_of(top, x), form_of(top, y));
            let mut c = [0i16; MAXM];
            for i in 0..top.m() {
                c[i] = lane(fx, i) as i16 - lane(fy, i) as i16;
            }
            c
        })
        .collect()
}

/// The regions of `--feed-regions=V[:KIND[:K]]` (see the module comment),
/// each with the atomic kind it belongs to.
fn feed_region_list(top: &Topology, spec: &str) -> Vec<(&'static str, Vec<Cvec>)> {
    let mut parts = spec.split(':');
    let v: u8 = parts.next().unwrap().parse().expect("feed-regions vertex");
    let kind = parts.next().unwrap_or("both");
    let k: i16 = parts.next().map_or(16, |x| {
        let k: u16 = x.parse().expect("feed-regions K, an integer from 0 to 100");
        assert!(k <= 100, "feed-regions K is at most 100");
        k as i16
    });
    assert!(parts.next().is_none(), "feed-regions takes V[:KIND[:K]]");
    let m = top.m();
    let ib = top
        .arcs
        .iter()
        .position(|&a| a == (0, v))
        .expect("feed-regions: no source arc into V");
    let mut out = Vec::new();
    let atoms = kind_atoms(kind);
    let has = |x: &str| atoms.contains(&x);
    assert!(
        k > 0
            || !["tbelow", "tnear", "tlow", "thuge", "tq"]
                .iter()
                .any(|x| has(x)),
        "the thin kinds tbelow, tnear, tlow, thuge and tq need K >= 1"
    );
    if has("thuge") {
        // b above K times the sum of all other weights.
        let mut row = [0i16; MAXM];
        for i in 0..m {
            row[i] = if i == ib { 1 } else { -k };
        }
        out.push(("thuge", vec![row]));
    }
    if has("tlow") {
        // b below every other weight divided by K.
        let mut rows = Vec::new();
        for j in 0..m {
            if j != ib {
                let mut c = [0i16; MAXM];
                c[j] = 1;
                c[ib] = -k;
                rows.push(c);
            }
        }
        out.push(("tlow", rows));
    }
    if has("tq") {
        // For each in-neighbour u of V with a source arc: b within w/K of
        // b_u + c(u>V), for every other arc weight w, below and above; the
        // path s>u>V need not be shortest.
        for (iu, &(a, h)) in top.arcs.iter().enumerate() {
            if h != v || a == 0 {
                continue;
            }
            let Some(is) = top.arcs.iter().position(|&x| x == (0, a)) else {
                continue;
            };
            for side in [-1i16, 1] {
                let mut rows = Vec::new();
                let mut q = [0i16; MAXM];
                q[is] = 1;
                q[iu] = 1;
                let mut under = [0i16; MAXM];
                for i in 0..m {
                    under[i] = side * ((i == ib) as i16 - q[i]);
                }
                rows.push(under);
                for j in 0..m {
                    if j != ib {
                        let mut t = [0i16; MAXM];
                        for i in 0..m {
                            t[i] = k * side * (q[i] - (i == ib) as i16);
                        }
                        t[j] += 1;
                        rows.push(t);
                    }
                }
                out.push(("tq", rows));
            }
        }
    }
    if has("above") {
        let mut row = [0i16; MAXM];
        for i in 0..m {
            row[i] = if i == ib { 1 } else { -1 };
        }
        out.push(("above", vec![row]));
    }
    if has("low") {
        let mut rows = Vec::new();
        for (i, &(a, _)) in top.arcs.iter().enumerate() {
            if a == 0 && i != ib {
                let mut c = [0i16; MAXM];
                c[i] = 1;
                c[ib] = -1;
                rows.push(c);
            }
        }
        out.push(("low", rows));
    }
    if has("below") || has("near") || has("tbelow") || has("tnear") {
        let all = top.paths();
        let others: Vec<Form> = all[v as usize - 1]
            .iter()
            .copied()
            .filter(|&z| z != unit(ib))
            .collect();
        for &p in &others {
            for (side, thin) in [(-1i16, false), (1, false), (-1, true), (1, true)] {
                let name = match (side < 0, thin) {
                    (true, false) => "below",
                    (false, false) => "near",
                    (true, true) => "tbelow",
                    (false, true) => "tnear",
                };
                if !has(name) {
                    continue;
                }
                let mut rows = Vec::new();
                let mut under = [0i16; MAXM];
                let mut near = [0i16; MAXM];
                for i in 0..m {
                    let pi = lane(p, i) as i16;
                    let bi = (i == ib) as i16;
                    under[i] = side * (bi - pi);
                    near[i] = k * side * (pi - bi) + 1;
                }
                rows.push(under);
                if thin {
                    for j in 0..m {
                        if j != ib {
                            let mut t = [0i16; MAXM];
                            for i in 0..m {
                                t[i] = k * side * (lane(p, i) as i16 - (i == ib) as i16);
                            }
                            t[j] += 1;
                            rows.push(t);
                        }
                    }
                } else if k > 0 {
                    rows.push(near);
                }
                for &q in &others {
                    if q != p {
                        let mut c = [0i16; MAXM];
                        for i in 0..m {
                            c[i] = lane(q, i) as i16 - lane(p, i) as i16;
                        }
                        rows.push(c);
                    }
                }
                out.push((name, rows));
            }
        }
    }
    assert!(
        !out.is_empty(),
        "feed-regions {spec}: no region (does V have another path from s?)"
    );
    out
}

fn solve(top: &Topology, mode: Mode, budget: Duration, restrict: bool) -> (Option<u8>, Game, f64) {
    solve_in(Game::new(top.clone(), mode), budget, restrict)
}

fn solve_in(mut g: Game, budget: Duration, restrict: bool) -> (Option<u8>, Game, f64) {
    g.restrict = restrict;
    let t0 = Instant::now();
    let v = g.value(budget, None);
    (v, g, t0.elapsed().as_secs_f64())
}

/// Values recorded by the Python reference, `experiments/2026-09-09-opt-dist-*`,
/// and the two values the paper states for `H_2`. Each entry is
/// `(arcs, path value, generic value)`, with `None` where the reference did
/// not decide that game.
const RECORDED: &[(&str, Option<u8>, Option<u8>)] = &[
    ("0>1,0>2", Some(0), Some(0)),
    ("0>1,1>2", Some(1), Some(1)),
    ("0>1,0>2,1>2", Some(2), Some(2)),
    ("0>1,1>2,2>1", Some(1), Some(1)),
    ("0>1,0>2,1>2,2>1", Some(3), Some(3)),
    ("0>1,0>2,0>3", Some(0), Some(0)),
    ("0>1,0>2,1>3", Some(1), Some(1)),
    ("0>1,1>2,1>3", Some(2), Some(2)),
    ("0>1,1>2,2>3", Some(2), Some(2)),
    ("0>1,0>2,0>3,1>2", Some(2), Some(2)),
    ("0>1,0>2,1>2,1>3", Some(3), Some(3)),
    ("0>1,0>2,1>2,2>3", Some(3), Some(3)),
    ("0>1,0>2,1>3,2>3", Some(3), Some(3)),
    ("0>1,0>2,1>3,3>1", Some(1), Some(1)),
    ("0>1,0>2,1>3,3>2", Some(3), Some(3)),
    ("0>1,1>2,1>3,2>1", Some(2), Some(2)),
    ("0>1,1>2,1>3,2>3", Some(4), Some(4)),
    ("0>1,1>2,2>1,2>3", Some(2), Some(2)),
    ("0>1,1>2,2>3,3>1", Some(2), Some(2)),
    ("0>1,1>2,2>3,3>2", Some(2), Some(2)),
    ("0>1,0>2,0>3,1>2,1>3", Some(4), Some(4)),
    ("0>1,0>2,0>3,1>2,2>1", Some(3), Some(3)),
    ("0>1,0>2,0>3,1>2,2>3", Some(4), Some(4)),
    ("0>1,0>2,0>3,1>2,3>2", Some(4), Some(4)),
    ("0>1,0>2,1>2,1>3,2>1", Some(4), Some(4)),
    ("0>1,0>2,1>2,1>3,2>3", Some(5), Some(5)),
    ("0>1,0>2,1>2,1>3,3>1", Some(3), Some(3)),
    ("0>1,0>2,1>2,1>3,3>2", Some(5), Some(5)),
    ("0>1,0>2,1>2,2>3,3>1", Some(4), Some(4)),
    ("0>1,0>2,1>2,2>3,3>2", Some(3), Some(3)),
    ("0>1,0>2,1>3,2>3,3>1", Some(4), Some(4)),
    ("0>1,0>2,1>3,3>1,3>2", Some(3), Some(3)),
    ("0>1,1>2,1>3,2>1,2>3", Some(4), Some(4)),
    ("0>1,1>2,1>3,2>1,3>1", Some(2), Some(2)),
    ("0>1,1>2,1>3,2>1,3>2", Some(4), Some(4)),
    ("0>1,1>2,1>3,2>3,3>2", Some(5), Some(5)),
    ("0>1,1>2,2>1,2>3,3>1", Some(2), Some(2)),
    ("0>1,1>2,2>1,2>3,3>2", Some(2), Some(2)),
    ("0>1,1>2,2>3,3>1,3>2", Some(2), Some(2)),
    ("0>1,0>2,0>3,1>2,1>3,2>1", Some(5), None),
    ("0>1,0>2,0>3,1>2,1>3,2>3", Some(6), None),
    ("0>1,0>2,0>3,1>2,2>1,3>1", Some(5), None),
    ("0>1,0>2,0>3,1>2,2>3,3>1", Some(6), Some(6)),
    ("0>1,0>2,1>2,1>3,2>1,2>3", Some(6), None),
    ("0>1,0>2,1>2,1>3,2>1,3>1", Some(4), None),
    ("0>1,0>2,1>2,1>3,2>1,3>2", Some(6), None),
    ("0>1,0>2,1>2,1>3,2>3,3>1", Some(6), None),
    ("0>1,0>2,1>2,1>3,2>3,3>2", Some(6), None),
    ("0>1,0>2,1>2,1>3,3>1,3>2", Some(5), None),
    ("0>1,0>2,1>2,2>3,3>1,3>2", Some(4), None),
    ("0>1,0>2,1>3,2>3,3>1,3>2", Some(5), None),
    ("0>1,1>2,1>3,2>1,2>3,3>1", Some(4), None),
    ("0>1,1>2,1>3,2>1,2>3,3>2", Some(5), None),
    ("0>1,1>2,2>1,2>3,3>1,3>2", Some(2), None),
];

/// `H_2` of the paper. Theorem 5.1 gives a program with two additions and five
/// comparisons, so `OPT_DIST <= 7`; the unrestricted program's strict-adversary
/// value is a lower bound on `OPT_DIST`, and both games here return 7.
const H2: &str = "0>1,0>2,0>3,1>2,1>3,2>1,3>1";

fn validate(quick: bool) -> i32 {
    let mut failures = 0;
    println!("recorded values, experiments/2026-09-09-opt-dist-*");
    let budget = Duration::from_secs(if quick { 60 } else { 3600 });
    for &(arcs, pv, gv) in RECORDED {
        let top = Topology::parse(arcs);
        if quick && top.m() > 5 {
            continue;
        }
        for (mode, expect) in [(Mode::Path, pv), (Mode::Generic, gv)] {
            let Some(expect) = expect else { continue };
            let (v, g, dt) = solve(&top, mode, budget, false);
            let ok = v == Some(expect);
            failures += !ok as i32;
            let law = if top.is_dag() {
                top.dag_law().to_string()
            } else {
                "-".into()
            };
            println!(
                "  {:28} m={} law={} {:7} game={:?} expect={} nodes={} lp={} {:.1}s {}",
                arcs,
                top.m(),
                law,
                mode.name(),
                v,
                expect,
                g.nodes,
                g.lps,
                dt,
                if ok { "ok" } else { "MISMATCH" }
            );
        }
    }
    if quick {
        return failures;
    }
    println!("H_2, Theorem 5.1: C = 4, P = 2, C_2 = 5, so OPT_DIST <= 7; expected game value 7");
    let h2 = Topology::parse(H2);
    for mode in [Mode::Path, Mode::Generic] {
        let (v, g, dt) = solve(&h2, mode, budget, false);
        let verdict = match v {
            Some(7) => "ok",
            Some(_) => "MISMATCH",
            None => "timed out",
        };
        failures += matches!(v, Some(x) if x != 7) as i32;
        println!(
            "  {:7} value {:?} (bound reached {}) nodes={} lp={} {:.1}s {}",
            mode.name(),
            v,
            g.lower_bound_reached(),
            g.nodes,
            g.lps,
            dt,
            verdict
        );
    }
    failures
}

fn enumerate_cmd(
    n: usize,
    mmax: usize,
    budget: Duration,
    modes: &[Mode],
    only_m: Option<usize>,
    pick: Option<Vec<usize>>,
) {
    let mut tops = enumerate_topologies(n, mmax);
    let mut header = format!(
        "# {} rooted topologies on {n} vertices with up to {mmax} arcs",
        tops.len()
    );
    if let Some(k) = only_m {
        tops.retain(|t| t.m() == k);
        header.push_str(&format!(", m = {k} only"));
    }
    if let Some(p) = &pick {
        tops = tops
            .into_iter()
            .enumerate()
            .filter(|(i, _)| p.contains(i))
            .map(|(_, t)| t)
            .collect();
        let ps: Vec<String> = p.iter().map(|i| i.to_string()).collect();
        header.push_str(&format!(", positions {}", ps.join(",")));
    }
    print_table(&header, &tops, budget, modes);
}

/// The enumeration table for the topologies named on the command line, in
/// the same format, so that a family played one member at a time is drawn by
/// the same script as the enumeration.
fn table_cmd(list: &str, budget: Duration, modes: &[Mode]) {
    let tops: Vec<Topology> = list.split(';').map(Topology::parse).collect();
    let header = format!("# {} named rooted topologies", tops.len());
    print_table(&header, &tops, budget, modes);
}

fn print_table(header: &str, tops: &[Topology], budget: Duration, modes: &[Mode]) {
    println!("{header}");
    let cols: Vec<String> = modes
        .iter()
        .map(|m| format!("{} nodes lp seconds", m.name()))
        .collect();
    println!("# arcs | n m dag d_s fwd law | {}", cols.join(" | "));
    for top in tops {
        let n = top.n;
        let mut cells = Vec::new();
        for &mode in modes {
            let (v, g, dt) = solve(top, mode, budget, false);
            let cell = match v {
                Some(v) => v.to_string(),
                None => format!(">={}", g.lower_bound_reached()),
            };
            cells.push(format!("{cell} {} {} {:.1}", g.nodes, g.lps, dt));
        }
        let law = if top.is_dag() {
            top.dag_law().to_string()
        } else {
            "-".into()
        };
        println!(
            "{top} | {n} {} {} {} {} {law} | {}",
            top.m(),
            top.is_dag() as u8,
            top.source_out(),
            top.fwd(),
            cells.join(" | ")
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || {
        eprintln!("usage:\n  unsorted-game validate [--quick]\n  unsorted-game enumerate N MMAX [SECONDS] [--modes=M,M] [--m=K] [--pick=I,I]\n  unsorted-game solve ARCS [--mode=MODE] [--seconds=S] [--min-entry | --feed-regions=V[:KIND[:K]] [--keep=V:KIND:K] | --cell-regions=I,J|list] [--given=X:Y;...] [--hold=F;...] [--from=B] [--strategy]\n  unsorted-game certify ARCS R --out=FILE\n  unsorted-game frontier ARCS AMAX [--mode=MODE] [--seconds=S] [--all-sub] [--strategy]\nevery command takes [--cache-gb=G]\nmodes: path, generic, path-free, generic-free");
        std::process::exit(2);
    };
    if args.is_empty() {
        usage();
    }
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .find_map(|a| a.strip_prefix(name).map(|s| s.to_string()))
    };
    if let Some(gb) = flag("--cache-gb=") {
        let gb: f64 = gb.parse().expect("GB");
        set_cache_bytes((gb * (1u64 << 30) as f64) as usize);
    }
    match args[0].as_str() {
        "validate" => {
            let failures = validate(args.iter().any(|a| a == "--quick"));
            std::process::exit(failures.min(1));
        }
        "enumerate" => {
            if args.len() < 3 {
                usage();
            }
            let n: usize = args[1].parse().expect("N");
            let mmax: usize = args[2].parse().expect("MMAX");
            let secs: f64 = args
                .get(3)
                .filter(|a| !a.starts_with("--"))
                .map_or(600.0, |a| a.parse().expect("SECONDS"));
            let modes: Vec<Mode> = flag("--modes=")
                .map(|s| {
                    s.split(',')
                        .map(|x| Mode::parse(x).expect("mode"))
                        .collect()
                })
                .unwrap_or_else(|| vec![Mode::Path, Mode::Generic]);
            let only_m = flag("--m=").map(|s| s.parse().expect("K"));
            let pick =
                flag("--pick=").map(|s| s.split(',').map(|x| x.parse().expect("I")).collect());
            enumerate_cmd(n, mmax, Duration::from_secs_f64(secs), &modes, only_m, pick);
        }
        "table" => {
            if args.len() < 2 {
                usage();
            }
            let secs: f64 = args
                .get(2)
                .filter(|a| !a.starts_with("--"))
                .map_or(600.0, |a| a.parse().expect("SECONDS"));
            let modes: Vec<Mode> = flag("--modes=")
                .map(|s| {
                    s.split(',')
                        .map(|x| Mode::parse(x).expect("mode"))
                        .collect()
                })
                .unwrap_or_else(|| vec![Mode::Path, Mode::Generic]);
            table_cmd(&args[1], Duration::from_secs_f64(secs), &modes);
        }
        "solve" => {
            if args.len() < 2 {
                usage();
            }
            let top = Topology::parse(&args[1]);
            let mode = flag("--mode=").map_or(Mode::Generic, |s| Mode::parse(&s).expect("mode"));
            let secs: f64 = flag("--seconds=").map_or(3600.0, |s| s.parse().expect("S"));
            let restrict = args.iter().any(|a| a == "--min-entry");
            let spec = flag("--feed-regions=");
            let cells_spec = flag("--cell-regions=");
            let prefix = flag("--given=").is_some() || flag("--hold=").is_some();
            assert!(
                !(restrict && (spec.is_some() || cells_spec.is_some() || prefix)),
                "--min-entry combines with none of --feed-regions, --cell-regions, --given, --hold"
            );
            assert!(
                !(cells_spec.is_some() && prefix),
                "--cell-regions combines with neither --given nor --hold"
            );
            assert!(
                !(spec.is_some() && cells_spec.is_some()),
                "--feed-regions and --cell-regions do not combine"
            );
            let game = match (&spec, &cells_spec) {
                (Some(sp), _) => {
                    let named = feed_region_list(&top, sp);
                    let names: Vec<String> = named.iter().map(|(n, _)| n.to_string()).collect();
                    let mut regions: Vec<Vec<Cvec>> = named.into_iter().map(|(_, r)| r).collect();
                    // `--keep=KIND` and `--prefer=KIND` name atomic kinds of
                    // the regions of `--feed-regions`.
                    let pick = |flag_name: &str, kk: &str| -> Vec<bool> {
                        let atoms = kind_atoms(kk);
                        let out: Vec<bool> =
                            names.iter().map(|n| atoms.contains(&n.as_str())).collect();
                        assert!(
                            out.iter().any(|&x| x),
                            "{flag_name}={kk} names no region of --feed-regions"
                        );
                        out
                    };
                    let keep = flag("--keep=").map(|kk| pick("--keep", &kk));
                    let prefer = flag("--prefer=").map(|kk| pick("--prefer", &kk));
                    let v: u8 = sp
                        .split(':')
                        .next()
                        .unwrap()
                        .parse()
                        .expect("feed-regions vertex");
                    let ib = top.arcs.iter().position(|&a| a == (0, v)).unwrap();
                    if let Some(sm) = flag("--small=") {
                        // Arc T>H below every weight but b and its own, divided
                        // by K, in every region.
                        let mut it = sm.split(':');
                        let (t, h) = it.next().unwrap().split_once('>').expect("--small=T>H[:K]");
                        let arc = (t.parse::<u8>().expect("T"), h.parse::<u8>().expect("H"));
                        let ks: i16 = it.next().map_or(16, |x| {
                            let k: u16 = x.parse().expect("--small K, an integer from 1 to 100");
                            assert!((1..=100).contains(&k), "--small K is from 1 to 100");
                            k as i16
                        });
                        assert!(it.next().is_none(), "--small takes T>H[:K]");
                        let ia = top
                            .arcs
                            .iter()
                            .position(|&a| a == arc)
                            .expect("--small: no such arc");
                        assert!(ia != ib, "--small: the arc must not be s>V");
                        for r in regions.iter_mut() {
                            for j in 0..top.m() {
                                if j != ia && j != ib {
                                    let mut c = [0i16; MAXM];
                                    c[j] = 1;
                                    c[ia] = -ks;
                                    r.push(c);
                                }
                            }
                        }
                    }
                    if let Some(gv) = flag("--given=") {
                        for c in given_rows(&top, &gv) {
                            for r in regions.iter_mut() {
                                r.push(c);
                            }
                        }
                    }
                    let mut g = Game::confined(top.clone(), mode, regions);
                    g.region_names = names;
                    assert!(
                        g.root_cells() > 0,
                        "the regions meet no cell of the root cone"
                    );
                    if let Some(hd) = flag("--hold=") {
                        for f in hd.split(';') {
                            g.add_held(form_of(&top, f));
                        }
                    }
                    if let Some(pr) = prefer {
                        g.prefer = pr;
                    }
                    if let Some(keep) = keep {
                        g.keep = keep;
                        assert!(g.root_meets_keep(), "--keep: the root meets no kept region");
                    }
                    g
                }
                (None, Some(cs)) => {
                    let all = Game::new(top.clone(), mode).cell_regions();
                    if cs == "list" {
                        for (k, (name, _)) in all.iter().enumerate() {
                            println!("cell {k}: {name}");
                        }
                        return;
                    }
                    let pick: Vec<Vec<Cvec>> = cs
                        .split(',')
                        .map(|x| all[x.parse::<usize>().expect("cell index")].1.clone())
                        .collect();
                    Game::confined(top.clone(), mode, pick)
                }
                (None, None) => match (flag("--given="), flag("--hold=")) {
                    (None, None) => Game::new(top.clone(), mode),
                    (gv, hd) => {
                        // One region: the given answers, which need not be
                        // invariant under the automorphisms.
                        let rows = gv.map_or(Vec::new(), |gv| given_rows(&top, &gv));
                        let mut g = Game::confined(top.clone(), mode, vec![rows]);
                        assert!(
                            g.root_cells() > 0,
                            "the given answers meet no cell of the root cone"
                        );
                        if let Some(hd) = hd {
                            for f in hd.split(';') {
                                g.add_held(form_of(&top, f));
                            }
                        }
                        g
                    }
                },
            };
            let mut game = game;
            game.from = flag("--from=").map(|s| s.parse().expect("--from=B"));
            let (v, mut g, dt) = solve_in(game, Duration::from_secs_f64(secs), restrict);
            println!(
                "# {top}  mode {}{}{}{}  value {}  corner {}  nodes {}  states {}  cones {}  lp {}  dropped {}  {dt:.1}s",
                mode.name(),
                flag("--from=").map_or(String::new(), |b| format!("  from {b}")),
                if restrict { format!("  adversary min-entry (forced loss {})", g.forced_loss) } else { String::new() },
                match (&spec, &cells_spec) {
                    (Some(sp), _) => format!(
                        "  adversary feed-regions {sp}{} ({} region-tagged cells{})",
                        ["keep", "prefer", "small", "hold", "given"]
                            .iter()
                            .filter_map(|f| flag(&format!("--{f}=")).map(|x| format!(" --{f}={x}")))
                            .collect::<String>(),
                        g.root_cells(),
                        g.root_regions()
                    ),
                    (None, Some(cs)) => format!("  adversary cell-regions {cs} ({} region-tagged cells)", g.root_cells()),
                    (None, None) => ["hold", "given"]
                        .iter()
                        .filter_map(|f| flag(&format!("--{f}=")).map(|x| format!(" --{f}={x}")))
                        .reduce(|a, b| a + &b)
                        .map_or(String::new(), |x| format!("  prefix{x}")),
                },
                v.map_or(format!(">={}", g.lower_bound_reached()), |v| v.to_string()),
                top.corner(),
                g.nodes,
                g.states(),
                g.cones_built(),
                g.lps,
                g.generations_dropped()
            );
            if let (Some(v), true) = (v, args.iter().any(|a| a == "--strategy")) {
                let held = g.initial();
                g.strategy(&held, v);
            }
        }
        "frontier" => {
            if args.len() < 3 {
                usage();
            }
            let top = Topology::parse(&args[1]);
            let amax: u8 = args[2].parse().expect("AMAX");
            let mode = flag("--mode=").map_or(Mode::Path, |s| Mode::parse(&s).expect("mode"));
            let secs: f64 = flag("--seconds=").map_or(3600.0, |s| s.parse().expect("S"));
            let all_sub = args.iter().any(|a| a == "--all-sub");
            let show = args.iter().any(|a| a == "--strategy");
            frontier_cmd(
                &top,
                mode,
                amax,
                Duration::from_secs_f64(secs),
                all_sub,
                show,
            );
        }
        "certify" => {
            if args.len() < 3 {
                usage();
            }
            let top = Topology::parse(&args[1]);
            let r: u8 = args[2].parse().expect("R");
            match flag("--out=") {
                Some(path) => certify_cmd(&top, r, &path),
                None => usage(),
            }
        }
        _ => usage(),
    }
}

/// Refutes budget `r` from the start of the unrestricted game and writes the
/// certificate to `path`; exits non-zero, writing no file, if the program
/// wins within `r`.
fn certify_cmd(top: &Topology, r: u8, path: &str) {
    let mut g = Game::new(top.clone(), Mode::Generic);
    let file = std::fs::File::create(path).expect("certificate file");
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    let t0 = Instant::now();
    let mut c = Certifier::new(&mut g, &mut out);
    let result = c.run(r);
    let s = std::mem::take(&mut c.stats);
    drop(c);
    let dt = t0.elapsed().as_secs_f64();
    match result {
        Ok(()) => {
            println!(
                "# {top}  refuted budget {r}  cells {}  states {}  leaves {}  inner {}  answered {}  implied {} (by lp {})  points {}  bytes {}  nodes {}  cones {}  dropped {}  {dt:.1}s",
                g.root_cells(),
                s.states,
                s.leaves,
                s.inner,
                s.answered,
                s.implied,
                s.implied_lp,
                s.points,
                s.bytes,
                g.nodes,
                g.cones_built(),
                g.generations_dropped()
            );
        }
        Err(e) => {
            println!("# {top}  budget {r}: no certificate: {e}  {dt:.1}s");
            // No partial certificate is left behind.
            drop(out);
            let _ = std::fs::remove_file(path);
            std::process::exit(1);
        }
    }
}

/// The least comparison budget at each addition budget, under the strict
/// adversary; with `show`, one strategy wherever that least budget drops.
fn frontier_cmd(top: &Topology, mode: Mode, amax: u8, budget: Duration, all_sub: bool, show: bool) {
    let mut g = Game::new(top.clone(), mode);
    let held = if all_sub {
        g.all_sub_forms()
    } else {
        g.initial()
    };
    let a0 = if all_sub { 0 } else { top.fwd() as u8 };
    println!(
        "# {top}  mode {}  cells {}  fwd {}  corner {}  held {}{}",
        mode.name(),
        g.root_cells(),
        top.fwd(),
        top.corner(),
        held.len(),
        if all_sub {
            "  (every sub-path form held)"
        } else {
            ""
        }
    );
    let t0 = Instant::now();
    let mut last: Option<u8> = None;
    for ra in a0..=amax {
        match g.least_comparisons(&held, ra, 250, budget) {
            Some(rc) => {
                println!(
                    "additions <= {ra}: comparisons {rc}  total {}  nodes {}  {:.1}s",
                    rc as u32 + ra as u32,
                    g.nodes,
                    t0.elapsed().as_secs_f64()
                );
                eprintln!(
                    "cache (cones, table, split table, chains, generations dropped) {:?}",
                    g.cache_sizes()
                );
                if show && last.is_none_or(|l| rc < l) {
                    g.strategy_split(&held, rc, ra);
                }
                last = Some(rc);
            }
            None => {
                println!(
                    "additions <= {ra}: undecided within the budget  nodes {}  {:.1}s",
                    g.nodes,
                    t0.elapsed().as_secs_f64()
                );
                return;
            }
        }
    }
}
