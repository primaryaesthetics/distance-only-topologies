//! The fed cycle `C_k` in the strict path-register game, against restricted
//! adversaries.
//!
//! Arcs are numbered `0..2k`: arc `i < k` is the source arc `b_{i+1}` into
//! vertex `i + 1`, arc `k + i` is the cycle arc `c_{i+1}` from vertex `i + 1`
//! to vertex `i + 2` (mod `k`). A form is a 0/1 mask over the arcs; the
//! program holds the zero form and the arc weights, may add two held forms
//! whose sum lies below some path from the source (at most one source arc, and
//! not every cycle arc), and may compare two held forms. A comparison answered
//! `x > y` adds the row `x − y > 0` to the transcript; the cone of a
//! transcript is the set of positive weightings satisfying all of its rows.
//!
//! A cell of `C_k` is the set of weightings with one shortest-path tree; it is
//! named by its entry set `S`, the vertices whose shortest path is their own
//! source arc. The single-entry cell `E_s` (`S = {s}`) has the `k − 1` facets
//! `f_j = b_j − b_s − c_s − … − c_{j−1} > 0`, `j ≠ s`. A leaf is a transcript
//! whose cone lies inside one cell, decided by exact linear programs, at
//! which the distance form of every vertex in that cell is held.
//!
//! Every adversary here gives only strict answers that leave a nonempty open
//! subset of the cone, so each is a restriction of the strict adversary and
//! the value of the game against it is a lower bound on the strict
//! path-register value; against `Full` it is that value.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

use crate::lp::{dot, interior_point, Cvec, MAXM};

/// A 0/1 form over the arcs.
pub type Form = u16;

/// The outcome `x > y`: the positive part of `x − y` in the low 16 bits and
/// its negative part in the high 16 bits.
pub type Row = u32;

/// The cycle arcs of `Game::set_worlds`; the first four are the draw of
/// `fed_cycle_worlds.py` at seed 1.
const CYCLE: [i64; 8] = [23700, 68200, 96700, 92100, 51300, 77900, 38900, 64100];

/// A world shape: the positions from `s` whose facet values are far from the
/// apex, then those near it, each listed from the smallest value up.
pub type Shape = (Vec<usize>, Vec<usize>);

/// Every order of `v`.
pub fn permutations(v: &[usize]) -> Vec<Vec<usize>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = vec![];
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// The shapes named by `spec` for `C_k`: `all` (every split into far and
/// near, every order within each), `twin` (all near or all far), `near`,
/// `far`, or patterns `near:2,1,3` and `far:2,3,1` joined by `+`.
pub fn shapes(spec: &str, k: usize) -> Result<Vec<Shape>, String> {
    let pos: Vec<usize> = (1..k).collect();
    let mut out = vec![];
    match spec {
        "all" | "twin" | "near" | "far" => {
            for mask in 0u32..(1 << (k - 1)) {
                let far: Vec<usize> = pos
                    .iter()
                    .copied()
                    .filter(|&u| (mask >> (u - 1)) & 1 == 1)
                    .collect();
                let near: Vec<usize> = pos
                    .iter()
                    .copied()
                    .filter(|&u| (mask >> (u - 1)) & 1 == 0)
                    .collect();
                let keep = match spec {
                    "all" => true,
                    "twin" => far.is_empty() || near.is_empty(),
                    "near" => far.is_empty(),
                    _ => near.is_empty(),
                };
                if !keep {
                    continue;
                }
                for f in permutations(&far) {
                    for n in permutations(&near) {
                        out.push((f.clone(), n));
                    }
                }
            }
        }
        _ => {
            for part in spec.split('+') {
                let bad = || format!("bad pattern {part}");
                let (regime, order) = part.split_once(':').ok_or_else(bad)?;
                let order: Vec<usize> = order
                    .split(',')
                    .map(|t| t.parse().map_err(|_| bad()))
                    .collect::<Result<_, _>>()?;
                let mut sorted = order.clone();
                sorted.sort_unstable();
                if sorted != pos {
                    return Err(format!("pattern {part} is not an order of 1..{}", k - 1));
                }
                match regime {
                    "near" => out.push((vec![], order)),
                    "far" => out.push((order, vec![])),
                    _ => return Err(bad()),
                }
            }
        }
    }
    Ok(out)
}

/// Generic cycle-arc weights by which the lexicographic family and the fixed
/// tie rules decide two forms of equal mass.
const GAMMA: [i64; 8] = [3, 5, 2, 7, 11, 13, 17, 19];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Adversary {
    /// Every strict answer; the cells kept are all cells of `C_k`.
    Full,
    /// Every strict answer after which some single-entry cell still meets the
    /// cone.
    Single,
    /// The strict answers that keep the most single-entry cells meeting the
    /// cone.
    MinEntry,
    /// Answers after which some point of a fixed finite set of weightings,
    /// each inside a single-entry cell, still satisfies every row.
    Points,
    /// The lexicographic family `b = T·β + …`, `c = 1 + ε·γ`, `T ≫ k ≫ ε`:
    /// a comparison is decided by an adaptively built order on the `β`, then
    /// by the number of cycle arcs on each side, then by a fixed generic `γ`.
    Lex,
    /// Worlds `s`: a comparison is answered by its sign at the apex `a_s` of
    /// `E_s`, the point with `b_s = β`, `c = γ` where every facet vanishes;
    /// a row vanishing there is answered either way that keeps the facet
    /// values `ε_j = f_j` of the vanishing rows an open cone.
    Apex,
    /// Worlds `s` whose cone stays thick at the apex face `A_s` of `E_s`
    /// (`b_j = walk_s(j)` for all `j`): the rows not vanishing on `A_s` are
    /// positive together somewhere on it and the vanishing rows leave the
    /// facet values an open cone.
    Thick,
    /// The adversary of Theorem F against search-raw programs: while two or
    /// more single-entry cells meet the cone, only rows `b_p − b_q` and rows
    /// free of source arcs may be compared; the former are answered by the
    /// interval rule, the latter at `c = γ`; after isolation every row is
    /// answered at the survivor's apex.
    TheoremF,
    /// The extended interval rule: Theorem F's interval rule carried to every
    /// row with two source arcs through Lemma H's usefulness test. While two
    /// or more single-entry cells meet the cone, a row with two source arcs
    /// gets, of its answers that keep an alive cell, the one with the
    /// lexicographically largest score (alive cells in the survivor set `J`
    /// for which the row is useless, alive cells in `J`, alive cells); on
    /// equal scores, the answer that makes the later held form the larger.
    /// `J` folds the rows answered so far in sorted order. A row without two
    /// source arcs is answered at the apex of the lowest-numbered alive cell,
    /// or by the other answer when that one leaves no alive cell. After
    /// isolation the adversary is free (any answer that keeps the survivor's
    /// cell).
    Eir,
}

/// How an adversary chooses between two answers that keep equally much.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ties {
    /// Either answer; the program must win both.
    Free,
    /// The side with more cycle arcs is larger; then a side with a source arc
    /// is larger than one without; then the higher source index is larger;
    /// then `γ` decides.
    Fixed,
    /// Rows of equal mass carrying two different source arcs by the interval
    /// rule on the alive single-entry cells, all others as `Fixed`.
    Interval,
}

/// The lexicographic family's rule for two candidates for the least source
/// arc compared against each other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LexKill {
    /// The side with more cycle arcs is larger; equal mass is the adversary's
    /// choice.
    Heavy,
    /// The adversary chooses which candidate loses.
    Free,
}

/// The lexicographic family's rule for a candidate compared with a
/// non-candidate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LexMixed {
    /// The non-candidate is larger.
    NonCandidate,
    /// As two non-candidates: the side with more cycle arcs is larger.
    Heavy,
}

/// Defects injected on purpose, to check that the search notices them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mutation {
    None,
    /// A leaf needs its outputs only, not a cone inside the cell.
    NoCertificate,
    /// `TheoremF` without the interval rule: before isolation a source-arc
    /// row may remove any alive cell, after isolation any answer that keeps
    /// the survivor's cell is allowed.
    NoInterval,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub k: usize,
    pub adversary: Adversary,
    pub ties: Ties,
    pub lex_kill: LexKill,
    pub lex_mixed: LexMixed,
    /// Points per single-entry cell and regime mask for `Points`.
    pub points: usize,
    pub regimes: u8,
    pub seed: u64,
    /// Key the transposition table up to the rotations of the cycle; sound
    /// only for a rotation-invariant adversary.
    pub symmetry: bool,
    /// Prune with the out-edge bound: a leaf in `E_s` needs, for every
    /// `j ≠ s`, a row with `+b_j`.
    pub out_edge_bound: bool,
    /// Search only the root moves `i mod n == split.0`, one per rotation
    /// orbit under `symmetry`; the value is the least over all `n` parts.
    pub split: (usize, usize),
    /// Entries each memo may hold before it is dropped; memos only memoise.
    pub cache_entries: usize,
    /// The largest budget tried.
    pub cap: usize,
    pub print: bool,
    /// Close the point set of `Points` under the rotations of the cycle.
    pub rotate_points: bool,
    /// `Points` only: keep, where possible, the waste invariant — some cell
    /// `t` with a near point alive has `waste_t + alive ≥ k` — and count the
    /// comparisons at which no allowed answer keeps it.
    pub waste_check: bool,
    pub mutation: Mutation,
}

impl Config {
    pub fn new(k: usize, adversary: Adversary) -> Config {
        Config {
            k,
            adversary,
            ties: Ties::Free,
            lex_kill: LexKill::Heavy,
            lex_mixed: LexMixed::NonCandidate,
            points: 3,
            regimes: 7,
            seed: 1,
            symmetry: false,
            out_edge_bound: false,
            split: (0, 1),
            cache_entries: 1_000_000,
            cap: 30,
            print: false,
            rotate_points: true,
            waste_check: false,
            mutation: Mutation::None,
        }
    }
}

/// The row `x − y`, read as `x > y`.
pub fn row_of(x: Form, y: Form) -> Row {
    let pos = x & !y;
    let neg = y & !x;
    (pos as u32) | ((neg as u32) << 16)
}

fn cvec_of(r: Row, m: usize) -> Cvec {
    let mut c = [0i16; MAXM];
    for (i, v) in c.iter_mut().enumerate().take(m) {
        *v = ((r >> i) & 1) as i16 - ((r >> (16 + i)) & 1) as i16;
    }
    c
}

fn negated(c: &Cvec) -> Cvec {
    let mut d = *c;
    for v in d.iter_mut() {
        *v = -*v;
    }
    d
}

/// The transcript `tau` (kept sorted) with the row `r` added.
fn with_row(tau: &[Row], r: Row) -> Vec<Row> {
    let mut t = tau.to_vec();
    let pos = t.partition_point(|&v| v < r);
    t.insert(pos, r);
    t
}

/// The cyclic interval `(p, q]` of vertices, as a mask.
fn interval(p: usize, q: usize, k: usize) -> u64 {
    let mut mask = 0u64;
    let mut v = (p + 1) % k;
    loop {
        mask |= 1 << v;
        if v == q {
            return mask;
        }
        v = (v + 1) % k;
    }
}

/// `(p, q)` when the row is `b_p − b_q` and nothing else.
fn raw_pair(c: &Cvec, k: usize) -> Option<(usize, usize)> {
    if c[k..2 * k].iter().any(|&v| v != 0) {
        return None;
    }
    let pos: Vec<usize> = (0..k).filter(|&i| c[i] == 1).collect();
    let neg: Vec<usize> = (0..k).filter(|&i| c[i] == -1).collect();
    (pos.len() == 1 && neg.len() == 1).then(|| (pos[0], neg[0]))
}

fn ceil_log2(n: usize) -> usize {
    let mut d = 0;
    while (1usize << d) < n {
        d += 1;
    }
    d
}

type Witnesses = Rc<Vec<Vec<i128>>>;

pub struct Game {
    pub cfg: Config,
    k: usize,
    m: usize,
    sub_path: Vec<bool>,
    /// Per cell, the distance forms of the vertices it does not enter.
    outputs: Vec<Vec<Form>>,
    /// Per cell, the rows whose positivity defines it.
    facets: Vec<Vec<Cvec>>,
    entry_sets: Vec<u16>,
    /// `(cell, weighting)` for the `Points` adversary.
    pub points: Vec<(usize, Vec<i64>)>,
    tt: HashMap<Vec<u8>, (i16, i16)>,
    implied_memo: HashMap<(Vec<Row>, Row), bool>,
    cert_memo: HashMap<(Vec<Row>, usize), bool>,
    alive_memo: HashMap<Vec<Row>, (u64, Witnesses)>,
    root_seen: Vec<(u8, (Form, Form))>,
    root_mine: Vec<(u8, (Form, Form))>,
    /// The most kept cells any comparison the search evaluated removes
    /// whichever way it is answered.
    pub forced_loss: usize,
    /// Per point of `Points`: whether every facet value is below every
    /// cycle-arc weight there (near the apex).
    pub point_near: Vec<bool>,
    /// Comparisons at which no allowed answer keeps the waste invariant.
    pub violations: u64,
    pub nodes: u64,
    pub lps: u64,
    pub tt_clears: u64,
    started: Instant,
    depth: usize,
}

impl Game {
    pub fn new(cfg: Config) -> Game {
        let k = cfg.k;
        assert!((2..=8).contains(&k), "k must lie in 2..=8");
        let m = 2 * k;
        let all_b = (1usize << k) - 1;
        let sub_path = (0..(1usize << m))
            .map(|z| {
                let b = z & all_b;
                let c = z >> k;
                if b == 0 {
                    c != all_b
                } else if b.count_ones() == 1 {
                    let into = (b.trailing_zeros() as usize + k - 1) % k;
                    (c >> into) & 1 == 0
                } else {
                    false
                }
            })
            .collect();
        let mut order: Vec<usize> = (1..=all_b).collect();
        order.sort_by_key(|&s| (s.count_ones(), s.trailing_zeros()));
        let (mut outputs, mut facets, mut entry_sets) = (vec![], vec![], vec![]);
        for smask in order {
            if cfg.adversary != Adversary::Full && smask.count_ones() != 1 {
                continue;
            }
            // The distance form of each vertex in the cell.
            let mut d: Vec<Form> = vec![0; k];
            let first = smask.trailing_zeros() as usize;
            let mut cur: Form = 0;
            for t in 0..k {
                let v = (first + t) % k;
                if (smask >> v) & 1 == 1 {
                    cur = 1 << v;
                } else {
                    cur |= 1 << (k + (v + k - 1) % k);
                }
                d[v] = cur;
            }
            let (mut outs, mut fs) = (vec![], vec![]);
            for v in 0..k {
                let pv = (v + k - 1) % k;
                let via = d[pv] | (1 << (k + pv));
                if (smask >> v) & 1 == 1 {
                    if via & (all_b as Form) == 1 << v {
                        continue; // around the whole cycle: always longer
                    }
                    fs.push(cvec_of(row_of(via, 1 << v), m));
                } else {
                    outs.push(d[v]);
                    fs.push(cvec_of(row_of(1 << v, d[v]), m));
                }
            }
            outputs.push(outs);
            facets.push(fs);
            entry_sets.push(smask as u16);
        }
        Game {
            k,
            m,
            sub_path,
            outputs,
            facets,
            entry_sets,
            points: vec![],
            tt: HashMap::new(),
            implied_memo: HashMap::new(),
            cert_memo: HashMap::new(),
            alive_memo: HashMap::new(),
            root_seen: vec![],
            root_mine: vec![],
            forced_loss: 0,
            point_near: vec![],
            violations: 0,
            nodes: 0,
            lps: 0,
            tt_clears: 0,
            started: Instant::now(),
            depth: 0,
            cfg,
        }
    }

    pub fn cells(&self) -> usize {
        self.entry_sets.len()
    }

    fn mass(&self, x: Form) -> u32 {
        (x >> self.k).count_ones()
    }

    fn gamma_sign(&self, x: Form, y: Form) -> i64 {
        let k = self.k;
        let d: i64 = (0..k)
            .map(|i| (((x >> (k + i)) & 1) as i64 - ((y >> (k + i)) & 1) as i64) * GAMMA[i])
            .sum();
        assert!(d != 0, "gamma must separate distinct cycle-arc sets");
        d
    }

    // ------------------------------------------------------- lexicographic

    /// `gt[p]`: the mask of `q` with `β_p > β_q`, transitively closed, from
    /// the rows with a source arc on both sides.
    fn closure(&self, tau: &[Row]) -> [u8; 8] {
        let k = self.k;
        let bmask = (1u32 << k) - 1;
        let mut gt = [0u8; 8];
        for &r in tau {
            let (pb, nb) = (r & bmask, (r >> 16) & bmask);
            if pb != 0 && nb != 0 {
                gt[pb.trailing_zeros() as usize] |= 1 << nb.trailing_zeros();
            }
        }
        for _ in 0..k {
            for p in 0..k {
                let mut acc = gt[p];
                for q in 0..k {
                    if (gt[p] >> q) & 1 == 1 {
                        acc |= gt[q];
                    }
                }
                gt[p] = acc;
            }
        }
        gt
    }

    /// Source arcs not known to exceed another: the candidates for the least.
    fn candidates(&self, gt: &[u8; 8]) -> u8 {
        (0..self.k)
            .filter(|&p| gt[p] == 0)
            .fold(0u8, |c, p| c | (1 << p))
    }

    fn lex_answers(&self, x: Form, y: Form, gt: &[u8; 8], cand: u8) -> Vec<Row> {
        let k = self.k;
        let bm: Form = (1 << k) - 1;
        let (bx, by) = (x & bm, y & bm);
        let (mx, my) = (self.mass(x), self.mass(y));
        let (xl, yl) = (row_of(x, y), row_of(y, x));
        if bx == by {
            if mx != my {
                return vec![if mx > my { xl } else { yl }];
            }
            return vec![if self.gamma_sign(x, y) > 0 { xl } else { yl }];
        }
        if bx == 0 {
            return vec![yl];
        }
        if by == 0 {
            return vec![xl];
        }
        let (p, q) = (bx.trailing_zeros() as usize, by.trailing_zeros() as usize);
        if (gt[p] >> q) & 1 == 1 {
            return vec![xl];
        }
        if (gt[q] >> p) & 1 == 1 {
            return vec![yl];
        }
        let (pc, qc) = ((cand >> p) & 1 == 1, (cand >> q) & 1 == 1);
        if pc != qc && self.cfg.lex_mixed == LexMixed::NonCandidate {
            return vec![if pc { yl } else { xl }];
        }
        if pc && qc && self.cfg.lex_kill == LexKill::Free {
            return vec![xl, yl];
        }
        match mx.cmp(&my) {
            std::cmp::Ordering::Greater => vec![xl],
            std::cmp::Ordering::Less => vec![yl],
            std::cmp::Ordering::Equal => vec![xl, yl],
        }
    }

    // ------------------------------------------------------- tie rules

    fn fixed_larger(&self, x: Form, y: Form) -> Row {
        let bm: Form = (1 << self.k) - 1;
        let (mx, my) = (self.mass(x), self.mass(y));
        if mx != my {
            return if mx > my { row_of(x, y) } else { row_of(y, x) };
        }
        let (bx, by) = (x & bm, y & bm);
        if bx != by {
            if by == 0 {
                return row_of(x, y);
            }
            if bx == 0 {
                return row_of(y, x);
            }
            return if bx > by { row_of(x, y) } else { row_of(y, x) };
        }
        if self.gamma_sign(x, y) > 0 {
            row_of(x, y)
        } else {
            row_of(y, x)
        }
    }

    /// The interval set `J` of a transcript: the intersection, over its rows
    /// `b_p − b_q`, of the cyclic intervals `(p, q]`.
    fn interval_set(&self, tau: &[Row]) -> u64 {
        let k = self.k;
        tau.iter()
            .filter_map(|&t| raw_pair(&cvec_of(t, self.m), k))
            .fold((1u64 << k) - 1, |j, (p, q)| j & interval(p, q, k))
    }

    fn interval_larger(&mut self, x: Form, y: Form, tau: &[Row]) -> Row {
        let k = self.k;
        let bm: Form = (1 << k) - 1;
        let (bx, by) = (x & bm, y & bm);
        let (mx, my) = (self.mass(x), self.mass(y));
        let raw = bx != 0 && by != 0 && bx != by && (x >> k) == (y >> k);
        if !raw && bx != by && bx != 0 && by != 0 && mx != my {
            return if mx > my { row_of(x, y) } else { row_of(y, x) };
        }
        if bx == by || bx == 0 || by == 0 {
            return self.fixed_larger(x, y);
        }
        let (p, q) = (bx.trailing_zeros() as usize, by.trailing_zeros() as usize);
        let alive = self.alive_info(tau, None).0;
        let j = self.interval_set(tau);
        let a_side = (alive & !(1u64 << p)) & j & interval(p, q, k);
        let b_side = (alive & !(1u64 << q)) & j & interval(q, p, k);
        if a_side.count_ones() >= b_side.count_ones() {
            row_of(x, y)
        } else {
            row_of(y, x)
        }
    }

    // ------------------------------------------------------- cones

    /// Whether the cone of `tau` lies in `r > 0` (so `r` tells nothing).
    fn implied(&mut self, tau: &[Row], r: Row) -> bool {
        if matches!(
            self.cfg.adversary,
            Adversary::Full
                | Adversary::Single
                | Adversary::MinEntry
                | Adversary::TheoremF
                | Adversary::Eir
        ) {
            let witnesses = self.alive_info(tau, None).1;
            let c = cvec_of(r, self.m);
            if witnesses
                .iter()
                .any(|w| !w.is_empty() && dot(&c, w, self.m) < 0)
            {
                return false;
            }
        }
        let key = (tau.to_vec(), r);
        if let Some(&v) = self.implied_memo.get(&key) {
            return v;
        }
        let mut rows: Vec<Cvec> = tau.iter().map(|&t| cvec_of(t, self.m)).collect();
        rows.push(negated(&cvec_of(r, self.m)));
        self.lps += 1;
        let v = interior_point(self.m, &rows).is_none();
        self.implied_memo.insert(key, v);
        v
    }

    /// Whether the cone of `tau` lies inside cell `s`.
    fn certified(&mut self, tau: &[Row], s: usize) -> bool {
        if self.cfg.mutation == Mutation::NoCertificate {
            return true;
        }
        let key = (tau.to_vec(), s);
        if let Some(&v) = self.cert_memo.get(&key) {
            return v;
        }
        let base: Vec<Cvec> = tau.iter().map(|&t| cvec_of(t, self.m)).collect();
        let mut inside = true;
        for f in self.facets[s].clone() {
            let mut rows = base.clone();
            rows.push(negated(&f));
            self.lps += 1;
            if interior_point(self.m, &rows).is_some() {
                inside = false;
                break;
            }
        }
        self.cert_memo.insert(key, inside);
        inside
    }

    /// The kept cells meeting the cone of `tau`, with an interior point of
    /// each (empty for the others). `parent` is the same for `tau` less one
    /// row: a parent point satisfying every row is reused without a program.
    fn alive_info(&mut self, tau: &[Row], parent: Option<Witnesses>) -> (u64, Witnesses) {
        if let Some(v) = self.alive_memo.get(tau) {
            return (v.0, v.1.clone());
        }
        let base: Vec<Cvec> = tau.iter().map(|&t| cvec_of(t, self.m)).collect();
        let mut out = 0u64;
        let mut wit: Vec<Vec<i128>> = vec![vec![]; self.cells()];
        for s in 0..self.cells() {
            if let Some(pw) = &parent {
                if pw[s].is_empty() {
                    continue;
                }
                if base.iter().all(|c| dot(c, &pw[s], self.m) > 0) {
                    out |= 1 << s;
                    wit[s] = pw[s].clone();
                    continue;
                }
            }
            let mut rows = base.clone();
            rows.extend(self.facets[s].iter().copied());
            self.lps += 1;
            if let Some(w) = interior_point(self.m, &rows) {
                out |= 1 << s;
                wit[s] = w;
            }
        }
        let rc = Rc::new(wit);
        self.alive_memo.insert(tau.to_vec(), (out, rc.clone()));
        (out, rc)
    }

    fn alive_after(&mut self, tau: &[Row], r: Row) -> u64 {
        let parent = self.alive_info(tau, None).1;
        self.alive_info(&with_row(tau, r), Some(parent)).0
    }

    // ------------------------------------------------------- apex worlds

    /// Arc weights at the apex of `E_s`: `c_i = 7^i`, `b_s = 7^k`, and
    /// `b_j = b_s + c_s + … + c_{j−1}`. For rows of path forms, whose
    /// coefficients on these values stay below 7 in absolute value, a row
    /// vanishes here only when it vanishes on the whole apex face.
    fn apex_values(&self, s: usize) -> Vec<i64> {
        let k = self.k;
        let gamma: Vec<i64> = (0..k as u32).map(|i| 7i64.pow(i)).collect();
        let mut val = vec![0i64; 2 * k];
        val[k..].copy_from_slice(&gamma);
        for t in 0..k {
            let j = (s + t) % k;
            val[j] = 7i64.pow(k as u32) + (0..t).map(|u| gamma[(s + u) % k]).sum::<i64>();
        }
        val
    }

    fn value_at(&self, c: &Cvec, val: &[i64]) -> i64 {
        val.iter().enumerate().map(|(i, &v)| c[i] as i64 * v).sum()
    }

    /// The facet coefficients of a row in the coordinates of `E_s`: its
    /// source-arc coefficients other than `b_s`.
    fn facet_part(&self, c: &Cvec, s: usize) -> Cvec {
        let mut e = [0i16; MAXM];
        for j in (0..self.k).filter(|&j| j != s) {
            e[j] = c[j];
        }
        e
    }

    /// Apex worlds consistent with `tau`.
    fn apex_worlds(&mut self, tau: &[Row]) -> u64 {
        if let Some(v) = self.alive_memo.get(tau) {
            return v.0;
        }
        let mut out = 0u64;
        for s in 0..self.k {
            let val = self.apex_values(s);
            let mut ok = true;
            let mut eps_rows: Vec<Cvec> = vec![];
            for &r in tau {
                let c = cvec_of(r, self.m);
                let v = self.value_at(&c, &val);
                if v < 0 {
                    ok = false;
                    break;
                }
                if v == 0 {
                    eps_rows.push(self.facet_part(&c, s));
                }
            }
            if ok && !eps_rows.is_empty() {
                self.lps += 1;
                ok = interior_point(self.k, &eps_rows).is_some();
            }
            if ok {
                out |= 1 << s;
            }
        }
        self.alive_memo.insert(tau.to_vec(), (out, Rc::new(vec![])));
        out
    }

    /// A row restricted to the apex face of `E_s`, in the coordinates `b_s`
    /// (at index `s`) and `c` (at `k..2k`).
    fn on_apex_face(&self, c: &Cvec, s: usize) -> Cvec {
        let k = self.k;
        let mut a = [0i16; MAXM];
        a[k..2 * k].copy_from_slice(&c[k..2 * k]);
        let mut bsum = 0i16;
        for t in 0..k {
            let j = (s + t) % k;
            bsum += c[j];
            for u in 0..t {
                a[k + (s + u) % k] += c[j];
            }
        }
        a[s] = bsum;
        a
    }

    fn thick_worlds(&mut self, tau: &[Row]) -> u64 {
        if let Some(v) = self.alive_memo.get(tau) {
            return v.0;
        }
        let mut out = 0u64;
        for s in 0..self.k {
            let mut on_face: Vec<Cvec> = vec![];
            let mut eps_rows: Vec<Cvec> = vec![];
            for &r in tau {
                let c = cvec_of(r, self.m);
                let a = self.on_apex_face(&c, s);
                if a.iter().all(|&v| v == 0) {
                    eps_rows.push(self.facet_part(&c, s));
                } else {
                    on_face.push(a);
                }
            }
            // The columns of b_j, j != s, are zero and do not matter.
            self.lps += 1;
            let mut ok = interior_point(self.m, &on_face).is_some();
            if ok && !eps_rows.is_empty() {
                self.lps += 1;
                ok = interior_point(self.k, &eps_rows).is_some();
            }
            if ok {
                out |= 1 << s;
            }
        }
        self.alive_memo.insert(tau.to_vec(), (out, Rc::new(vec![])));
        out
    }

    /// Rows of `tau` that do not vanish on the apex face of `E_s`, which no
    /// certificate of `E_s` thick at its apex can use.
    fn useless_rows(&self, tau: &[Row], s: usize) -> usize {
        tau.iter()
            .filter(|&&r| {
                self.on_apex_face(&cvec_of(r, self.m), s)
                    .iter()
                    .any(|&v| v != 0)
            })
            .count()
    }

    /// The vertices `s` for which the row `r` (read as `r > 0`, a row of two
    /// path forms with sources `p` (positive) and `q` (negative)) is useless:
    /// by Lemma H a single row is an edge `p -> q` of the certificate of `E_s`
    /// with surplus `C(s..q) - C(s..p) - (cycle-arc part of r)` on the cycle
    /// arcs, usable only when that vector is nonnegative.
    fn useless_set(&self, c: &Cvec, p: usize, q: usize) -> u64 {
        let k = self.k;
        let walk = |s: usize, t: usize, i: usize| -> i16 {
            // arc i (on i -> i+1) lies on the walk from s to t
            i16::from((i + k - s) % k < (t + k - s) % k)
        };
        let mut u = 0u64;
        for s in 0..k {
            if (0..k).any(|i| walk(s, q, i) - walk(s, p, i) - c[k + i] < 0) {
                u |= 1 << s;
            }
        }
        u
    }

    /// The survivor set `J` of the extended interval rule: the rows of `tau`
    /// with two source arcs in sorted order, each intersecting `J` with its
    /// useless set unless that would empty it.
    fn eir_set(&self, tau: &[Row]) -> u64 {
        let k = self.k;
        let mut j = (1u64 << k) - 1;
        for &t in tau {
            let c = cvec_of(t, self.m);
            let pos: Vec<usize> = (0..k).filter(|&i| c[i] == 1).collect();
            let neg: Vec<usize> = (0..k).filter(|&i| c[i] == -1).collect();
            if pos.len() == 1 && neg.len() == 1 {
                let u = self.useless_set(&c, pos[0], neg[0]);
                if j & u != 0 {
                    j &= u;
                }
            }
        }
        j
    }

    /// The answers of the extended interval rule (`Adversary::Eir`).
    fn eir_answers(&mut self, x: Form, y: Form, tau: &[Row]) -> Vec<Row> {
        let k = self.k;
        let alive = self.alive_info(tau, None).0;
        let (r, rev) = (row_of(x, y), row_of(y, x));
        let mut out: Vec<Row> = vec![];
        if alive.count_ones() <= 1 {
            for o in [r, rev] {
                if self.alive_after(tau, o) != 0 {
                    out.push(o);
                }
            }
            return out;
        }
        let c = cvec_of(r, self.m);
        let pos: Vec<usize> = (0..k).filter(|&i| c[i] == 1).collect();
        let neg: Vec<usize> = (0..k).filter(|&i| c[i] == -1).collect();
        if pos.len() == 1 && neg.len() == 1 {
            let j = self.eir_set(tau) & alive;
            let nc = negated(&c);
            let mut scored = vec![];
            for (o, oc, p, q) in [(r, c, pos[0], neg[0]), (rev, nc, neg[0], pos[0])] {
                let kept = self.alive_after(tau, o);
                let u = self.useless_set(&oc, p, q);
                scored.push((
                    o,
                    kept,
                    (kept & j & u).count_ones(),
                    (kept & j).count_ones(),
                ));
            }
            // `max_by_key` keeps the last of equal maxima: on a tie, `rev`,
            // the answer that makes `y` the larger.
            let pick = scored
                .iter()
                .filter(|s| s.1 != 0)
                .max_by_key(|s| (s.2, s.3, s.1.count_ones()))
                .map(|s| s.0);
            out.extend(pick);
        } else {
            // No source arc, or one: answer at the apex of the lowest alive
            // cell. That apex need not lie near the cone; when its answer
            // leaves no alive cell, the cone implies the other one.
            let s = alive.trailing_zeros() as usize;
            let v = self.value_at(&c, &self.apex_values(s));
            assert!(v != 0, "EIR: a row vanishes at the apex");
            let (a, b) = if v > 0 { (r, rev) } else { (rev, r) };
            out.push(if self.alive_after(tau, a) != 0 { a } else { b });
        }
        for &o in &out {
            assert!(
                self.alive_after(tau, o) != 0,
                "EIR: an answer leaves no single-entry cell"
            );
        }
        out
    }

    /// The answers of Theorem F's adversary; empty when the comparison lies
    /// outside the search-raw class. Asserts F1 (a single-entry cell meets
    /// the cone exactly when its vertex never lost a source-arc row) and F2
    /// (every answer keeps a single-entry cell, and the survivor's apex
    /// satisfies every row) on every comparison it answers.
    fn theorem_f_answers(&mut self, x: Form, y: Form, tau: &[Row]) -> Vec<Row> {
        let k = self.k;
        let all = (1u64 << k) - 1;
        let alive = self.alive_info(tau, None).0;
        let (r, rev) = (row_of(x, y), row_of(y, x));
        let c = cvec_of(r, self.m);
        let no_interval = self.cfg.mutation == Mutation::NoInterval;
        let mut out: Vec<Row> = vec![];
        if alive.count_ones() >= 2 {
            let lost = tau
                .iter()
                .filter_map(|&t| raw_pair(&cvec_of(t, self.m), k))
                .fold(0u64, |l, (p, _)| l | (1 << p));
            assert_eq!(
                alive,
                all & !lost,
                "F1: alive cells are not the never-lost vertices"
            );
            let j = self.interval_set(tau);
            if c[..k].iter().all(|&v| v == 0) {
                let v = self.value_at(&c, &self.apex_values(0));
                assert!(v != 0);
                out.push(if v > 0 { r } else { rev });
            } else if let Some((p, q)) = raw_pair(&c, k) {
                // r = b_p − b_q: "b_p larger" removes E_p.
                let p_loses = if no_interval {
                    alive & !(1u64 << p)
                } else {
                    (alive & !(1u64 << p)) & j & interval(p, q, k)
                };
                if p_loses != 0 {
                    out.push(r);
                } else {
                    assert!(
                        no_interval || (alive & !(1u64 << q)) & j & interval(q, p, k) != 0,
                        "F2: the interval rule has no side to keep"
                    );
                    out.push(rev);
                }
            } else {
                return vec![];
            }
        } else if no_interval {
            for o in [r, rev] {
                if self.alive_after(tau, o) != 0 {
                    out.push(o);
                }
            }
        } else {
            let s = alive.trailing_zeros() as usize;
            let val = self.apex_values(s);
            let v = self.value_at(&c, &val);
            if v != 0 {
                out.push(if v > 0 { r } else { rev });
            } else {
                for o in [r, rev] {
                    let mut eps_rows: Vec<Cvec> = vec![];
                    for &t in tau.iter().chain(std::iter::once(&o)) {
                        let tc = cvec_of(t, self.m);
                        let tv = self.value_at(&tc, &val);
                        assert!(tv >= 0, "F2: a row is negative at the survivor's apex");
                        if tv == 0 {
                            eps_rows.push(self.facet_part(&tc, s));
                        }
                    }
                    self.lps += 1;
                    if interior_point(k, &eps_rows).is_some() {
                        out.push(o);
                    }
                }
            }
        }
        for &o in &out {
            assert!(
                self.alive_after(tau, o) != 0,
                "F2: an answer leaves no single-entry cell"
            );
        }
        out
    }

    // ------------------------------------------------------- points

    fn point_cells(&mut self, tau: &[Row]) -> u64 {
        if let Some(v) = self.alive_memo.get(tau) {
            return v.0;
        }
        let rows: Vec<Cvec> = tau.iter().map(|&t| cvec_of(t, self.m)).collect();
        let mut out = 0u64;
        for (cell, w) in &self.points {
            if (out >> cell) & 1 == 0
                && rows.iter().all(|c| {
                    w.iter()
                        .enumerate()
                        .map(|(i, &x)| c[i] as i64 * x)
                        .sum::<i64>()
                        > 0
                })
            {
                out |= 1 << cell;
            }
        }
        self.alive_memo.insert(tau.to_vec(), (out, Rc::new(vec![])));
        out
    }

    /// Draws `points` weightings per single-entry cell from a seeded
    /// generator, cycling through the regimes in `regimes` (near the apex:
    /// cycle arcs in [1000, 9999], facet values in [1, 99]; far from it:
    /// cycle arcs in [1, 99], facet values in [1000, 99999]; mixed: [1, 9999]
    /// and [1, 99999]), and closes the set under the rotations of the cycle.
    /// A draw at which two distinct sub-path forms tie is drawn again, so no
    /// comparison the program can make vanishes at a point, and a point alive
    /// in the cone can be followed to a leaf. Asserts that every point lies
    /// strictly inside its cell.
    pub fn make_points(&mut self) {
        let k = self.k;
        assert_eq!(self.cells(), k);
        let mut st = self
            .cfg
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut rnd = |lo: i64, hi: i64| -> i64 {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((st >> 33) % ((hi - lo + 1) as u64)) as i64
        };
        let forms: Vec<usize> = (0..self.sub_path.len())
            .filter(|&z| self.sub_path[z])
            .collect();
        let generic = |w: &[i64]| -> bool {
            let mut sums: Vec<i64> = forms
                .iter()
                .map(|&z| {
                    (0..2 * k)
                        .filter(|&i| (z >> i) & 1 == 1)
                        .map(|i| w[i])
                        .sum()
                })
                .collect();
            sums.sort_unstable();
            sums.windows(2).all(|p| p[0] != p[1])
        };
        let regimes = [
            ((1000, 9999), (1, 99)),
            ((1, 99), (1000, 99999)),
            ((1, 9999), (1, 99999)),
        ];
        assert!(self.cfg.regimes & 7 != 0, "no regime selected");
        let mut base = vec![];
        for s in 0..k {
            for t in 0..self.cfg.points {
                let mut ri = t % 3;
                while (self.cfg.regimes >> ri) & 1 == 0 {
                    ri = (ri + 1) % 3;
                }
                let ((clo, chi), (elo, ehi)) = regimes[ri];
                let w = loop {
                    let mut w = vec![0i64; 2 * k];
                    for c in w[k..].iter_mut() {
                        *c = rnd(clo, chi);
                    }
                    let bs = rnd(1, 9999);
                    for u in 0..k {
                        let j = (s + u) % k;
                        let mut v = bs + (0..u).map(|q| w[k + (s + q) % k]).sum::<i64>();
                        if u > 0 {
                            v += rnd(elo, ehi);
                        }
                        w[j] = v;
                    }
                    if generic(&w) {
                        break w;
                    }
                };
                base.push((s, w));
            }
        }
        self.set_points(base);
    }

    /// Takes `(cell, weighting)` pairs as the point set of `Points`, closes it
    /// under the rotations of the cycle unless `rotate_points` is off, and
    /// asserts that every point lies
    /// strictly inside its cell and that no two distinct sub-path forms tie at
    /// it.
    pub fn set_points(&mut self, base: Vec<(usize, Vec<i64>)>) {
        let k = self.k;
        assert_eq!(self.cells(), k);
        let forms: Vec<usize> = (0..self.sub_path.len())
            .filter(|&z| self.sub_path[z])
            .collect();
        for (s, w) in &base {
            assert!(*s < k && w.len() == 2 * k, "a point has the wrong shape");
            let mut sums: Vec<i64> = forms
                .iter()
                .map(|&z| {
                    (0..2 * k)
                        .filter(|&i| (z >> i) & 1 == 1)
                        .map(|i| w[i])
                        .sum()
                })
                .collect();
            sums.sort_unstable();
            assert!(
                sums.windows(2).all(|p| p[0] != p[1]),
                "two sub-path forms tie at a point"
            );
        }
        let mut seen = HashSet::new();
        let turns = if self.cfg.rotate_points { k } else { 1 };
        for (s0, w) in base {
            for t in 0..turns {
                let mut w2 = vec![0i64; 2 * k];
                for i in 0..k {
                    w2[(i + t) % k] = w[i];
                    w2[k + (i + t) % k] = w[k + i];
                }
                if seen.insert(w2.clone()) {
                    self.points.push(((s0 + t) % k, w2));
                }
            }
        }
        self.point_near.clear();
        for (s, w) in &self.points {
            let mut fmax = 0;
            for f in &self.facets[*s] {
                let v: i64 = w.iter().enumerate().map(|(i, &x)| f[i] as i64 * x).sum();
                assert!(v > 0, "a point lies outside its cell");
                fmax = fmax.max(v);
            }
            let cmin = *w[k..].iter().min().unwrap();
            self.point_near.push(fmax < cmin);
        }
    }

    /// The lexicographic worlds of `theory/game/fed_cycle_worlds.py`, one per
    /// shape and cell: in `E_s`, `b_s = 10^9`, the cycle arcs `CYCLE`, and
    /// the facet value at position `u` from `s` equal to `10^7·rank` for the
    /// positions of `far` and to `rank` for those of `near`, ranks counted
    /// from 1 in the order listed. Taken as the point set, unrotated.
    pub fn set_worlds(&mut self, shapes: &[Shape]) {
        let k = self.k;
        let c = &CYCLE[..k];
        let mut base = vec![];
        for s in 0..k {
            for (far, near) in shapes {
                let mut eps = vec![0i64; k];
                for (r, &u) in far.iter().enumerate() {
                    eps[u] = 10_000_000 * (r as i64 + 1);
                }
                for (r, &u) in near.iter().enumerate() {
                    eps[u] = r as i64 + 1;
                }
                let mut w = vec![0i64; 2 * k];
                w[k..].copy_from_slice(c);
                for u in 0..k {
                    let walk: i64 = (0..u).map(|q| c[(s + q) % k]).sum();
                    w[(s + u) % k] = 1_000_000_000 + walk + eps[u];
                }
                base.push((s, w));
            }
        }
        self.cfg.rotate_points = false;
        self.set_points(base);
    }

    // ------------------------------------------------------- search

    fn alive_set(&mut self, tau: &[Row]) -> u64 {
        match self.cfg.adversary {
            Adversary::Points => self.point_cells(tau),
            Adversary::Thick => self.thick_worlds(tau),
            Adversary::Apex => self.apex_worlds(tau),
            Adversary::Lex => {
                let gt = self.closure(tau);
                self.candidates(&gt) as u64
            }
            Adversary::Full
            | Adversary::Single
            | Adversary::MinEntry
            | Adversary::TheoremF
            | Adversary::Eir => self.alive_info(tau, None).0,
        }
    }

    /// Whether the adversary can follow every kept cell to a leaf, which
    /// makes the Kraft bound over all of them sound.
    fn follows_every_cell(&self) -> bool {
        match self.cfg.adversary {
            Adversary::Full | Adversary::Single | Adversary::Points | Adversary::Apex => true,
            Adversary::Thick => self.cfg.ties == Ties::Free,
            Adversary::MinEntry | Adversary::Lex | Adversary::TheoremF | Adversary::Eir => false,
        }
    }

    fn missing_outputs(&self, held: &[Form], s: usize) -> usize {
        self.outputs[s]
            .iter()
            .filter(|w| held.binary_search(w).is_err())
            .count()
    }

    /// A lower bound on the operations left, and whether the state is a leaf.
    fn eval(&mut self, held: &[Form], tau: &[Row]) -> (usize, bool) {
        let alive = self.alive_set(tau);
        let nc = alive.count_ones() as usize;
        assert!(nc >= 1, "no kept cell meets the cone");
        let k = self.k;
        let covered = tau.iter().fold(0u32, |o, &t| o | (t & ((1u32 << k) - 1)));
        let mut chains = vec![];
        let mut needs = vec![];
        for s in (0..self.cells()).filter(|&s| (alive >> s) & 1 == 1) {
            let a = self.missing_outputs(held, s);
            chains.push(a);
            let entries = self.entry_sets[s] as u32;
            // A leaf in E_s has, for every j != s, a row with +b_j, or b_j
            // could be lowered inside the cone until f_j < 0 (the leaf certificate);
            // and a row of path forms has at most one positive source arc.
            let uncovered = if self.cfg.out_edge_bound
                && entries.count_ones() == 1
                && self.cfg.mutation != Mutation::NoCertificate
            {
                (((1u32 << k) - 1) & !entries & !covered).count_ones() as usize
            } else {
                0
            };
            needs.push(a + uncovered);
        }
        let least = *chains.iter().min().unwrap();
        if nc == 1 {
            let s = alive.trailing_zeros() as usize;
            let inside = self.certified(tau, s);
            if inside && least == 0 {
                return (0, true);
            }
            return ((least + usize::from(!inside)).max(needs[0]), false);
        }
        if self.follows_every_cell() {
            // Every kept cell holds a leaf: r >= a + log2 #{chain >= a}.
            chains.sort_unstable_by(|a, b| b.cmp(a));
            let mut lb = needs.iter().copied().max().unwrap_or(1).max(1);
            for (i, &a) in chains.iter().enumerate() {
                lb = lb.max(a + ceil_log2(i + 1));
            }
            return (lb, false);
        }
        let search = match self.cfg.adversary {
            // Each comparison removes at most one candidate.
            Adversary::Lex | Adversary::TheoremF => nc - 1,
            // A deterministic rule may remove several cells at once; two
            // alive cells still need one more operation.
            Adversary::Eir => 1,
            // The larger child keeps at least half of the kept cells.
            _ => ceil_log2(nc),
        };
        ((search + least).max(*needs.iter().min().unwrap()), false)
    }

    pub fn wins(&mut self, held: &[Form], tau: &[Row], r: usize) -> bool {
        self.nodes += 1;
        if self.nodes % (1 << 22) == 0 {
            eprintln!(
                "  nodes {} tt {} lps {} {:.0}s",
                self.nodes,
                self.tt.len(),
                self.lps,
                self.started.elapsed().as_secs_f64()
            );
        }
        if self.nodes & 0xffff == 0 {
            let lim = self.cfg.cache_entries;
            if self.implied_memo.len() > lim {
                self.implied_memo.clear();
            }
            if self.cert_memo.len() > lim / 2 {
                self.cert_memo.clear();
            }
            if self.alive_memo.len() > lim / 2 {
                self.alive_memo.clear();
            }
            if self.tt.len() > 2 * lim {
                self.tt.clear();
                self.tt_clears += 1;
            }
        }
        let (h, ready) = self.eval(held, tau);
        if ready {
            return true;
        }
        if h > r || r == 0 {
            return false;
        }
        let key = if self.cfg.symmetry {
            self.canonical_key(held, tau)
        } else {
            Self::key(held, tau)
        };
        if let Some(&(lose, win)) = self.tt.get(&key) {
            if win >= 0 && (win as usize) <= r {
                return true;
            }
            if lose >= r as i16 {
                return false;
            }
        }
        let res = self.expand(held, tau, r, false);
        let e = self.tt.entry(key).or_insert((-1, -1));
        if res {
            if e.1 < 0 || (r as i16) < e.1 {
                e.1 = r as i16;
            }
        } else if (r as i16) > e.0 {
            e.0 = r as i16;
        }
        res
    }

    fn rotate(&self, z: u32, t: usize) -> u32 {
        let k = self.k;
        let full = (1u32 << k) - 1;
        let rot = |x: u32| ((x << t) | (x >> (k - t))) & full;
        rot(z & full) | (rot((z >> k) & full) << k)
    }

    fn canonical_key(&self, held: &[Form], tau: &[Row]) -> Vec<u8> {
        let mut best = Self::key(held, tau);
        for t in 1..self.k {
            let mut h: Vec<Form> = held
                .iter()
                .map(|&z| self.rotate(z as u32, t) as Form)
                .collect();
            h.sort_unstable();
            let mut r: Vec<Row> = tau
                .iter()
                .map(|&x| self.rotate(x & 0xffff, t) | (self.rotate(x >> 16, t) << 16))
                .collect();
            r.sort_unstable();
            best = best.min(Self::key(&h, &r));
        }
        best
    }

    fn key(held: &[Form], tau: &[Row]) -> Vec<u8> {
        let mut v = Vec::with_capacity(1 + 2 * held.len() + 4 * tau.len());
        v.push(held.len() as u8);
        for &h in held {
            v.extend_from_slice(&h.to_le_bytes());
        }
        for &t in tau {
            v.extend_from_slice(&t.to_le_bytes());
        }
        v
    }

    /// At the root, one move per rotation orbit under `symmetry`, and only
    /// those this process is assigned by `split`.
    fn root_skip(&mut self, held: &[Form], tau: &[Row], kind: u8, x: Form, y: Form) -> bool {
        if !tau.is_empty() || held.len() != 2 * self.k + 1 {
            return false;
        }
        let turns = if self.cfg.symmetry { self.k } else { 1 };
        let canon = (0..turns)
            .map(|t| {
                let (a, b) = if t == 0 {
                    (x, y)
                } else {
                    (
                        self.rotate(x as u32, t) as Form,
                        self.rotate(y as u32, t) as Form,
                    )
                };
                (kind, (a.min(b), a.max(b)))
            })
            .min()
            .unwrap();
        if self.root_seen.contains(&canon) {
            return !self.root_mine.contains(&canon);
        }
        self.root_seen.push(canon);
        let mine = (self.root_seen.len() - 1) % self.cfg.split.1 == self.cfg.split.0;
        if mine {
            self.root_mine.push(canon);
        }
        !mine
    }

    /// The answers the adversary may give to comparing `x` with `y`, each the
    /// row `larger − smaller`; `None` when the comparison lies outside the
    /// program class or tells the program nothing.
    /// Operations of the state that a leaf in the single-entry cell `t`, thick
    /// at its apex, cannot use: built forms other than the distance forms of
    /// `t`, and rows not vanishing on the apex face of `E_t`.
    fn waste(&self, held: &[Form], tau: &[Row], t: usize) -> usize {
        let built = held
            .iter()
            .filter(|&&z| z.count_ones() >= 2 && !self.outputs[t].contains(&z))
            .count();
        built + self.useless_rows(tau, t)
    }

    /// The waste invariant: some cell with a near point alive has
    /// `waste_t + #cells alive ≥ k`.
    fn waste_invariant(&mut self, held: &[Form], tau: &[Row]) -> bool {
        let alive = self.point_cells(tau);
        let n = alive.count_ones() as usize;
        let rows: Vec<Cvec> = tau.iter().map(|&t| cvec_of(t, self.m)).collect();
        let mut near = 0u64;
        for (i, (cell, w)) in self.points.iter().enumerate() {
            if self.point_near[i]
                && rows.iter().all(|c| {
                    w.iter()
                        .enumerate()
                        .map(|(j, &x)| c[j] as i64 * x)
                        .sum::<i64>()
                        > 0
                })
            {
                near |= 1 << cell;
            }
        }
        (0..self.k)
            .filter(|&t| (near >> t) & 1 == 1)
            .any(|t| self.waste(held, tau, t) + n >= self.k)
    }

    fn allowed(
        &mut self,
        held: &[Form],
        x: Form,
        y: Form,
        tau: &[Row],
        gt: &[u8; 8],
        cand: u8,
    ) -> Option<Vec<Row>> {
        let opts = match self.cfg.adversary {
            Adversary::TheoremF => self.theorem_f_answers(x, y, tau),
            Adversary::Eir => self.eir_answers(x, y, tau),
            Adversary::Lex => self.lex_answers(x, y, gt, cand),
            adversary => {
                let here = self.alive_set(tau).count_ones();
                let mut kept = vec![];
                for o in [row_of(x, y), row_of(y, x)] {
                    let a = match adversary {
                        Adversary::Points => self.point_cells(&with_row(tau, o)),
                        Adversary::Thick => self.thick_worlds(&with_row(tau, o)),
                        Adversary::Apex => self.apex_worlds(&with_row(tau, o)),
                        _ => self.alive_after(tau, o),
                    };
                    kept.push((o, a.count_ones()));
                }
                let best = kept[0].1.max(kept[1].1);
                let loss = (here - best) as usize;
                if loss > self.forced_loss {
                    self.forced_loss = loss;
                    if loss >= 2 {
                        let ts: Vec<String> = tau.iter().map(|&t| self.fmt_row(t)).collect();
                        println!(
                            "# forced loss {}: compare {} ? {} after {}",
                            loss,
                            self.fmt(x),
                            self.fmt(y),
                            if ts.is_empty() {
                                "nothing".into()
                            } else {
                                ts.join(" ; ")
                            }
                        );
                    }
                }
                let tie = kept[0].1 == kept[1].1;
                let keep_best = || {
                    kept.iter()
                        .filter(|p| p.1 == best)
                        .map(|p| p.0)
                        .collect::<Vec<Row>>()
                };
                match (adversary, self.cfg.ties) {
                    (Adversary::Thick, Ties::Fixed) if tie => {
                        // Keep the worlds that hold the most useless rows.
                        let mut score = vec![];
                        for &(o, _) in &kept {
                            let t2 = with_row(tau, o);
                            let w = self.thick_worlds(&t2);
                            score.push(
                                (0..self.k)
                                    .filter(|&s| (w >> s) & 1 == 1)
                                    .map(|s| self.useless_rows(&t2, s))
                                    .sum::<usize>(),
                            );
                        }
                        if score[0] != score[1] {
                            vec![if score[0] > score[1] {
                                kept[0].0
                            } else {
                                kept[1].0
                            }]
                        } else {
                            vec![self.fixed_larger(x, y)]
                        }
                    }
                    (Adversary::Thick, Ties::Fixed) => keep_best(),
                    (Adversary::MinEntry, Ties::Fixed) if tie => vec![self.fixed_larger(x, y)],
                    (Adversary::MinEntry, Ties::Interval) if tie => {
                        vec![self.interval_larger(x, y, tau)]
                    }
                    (Adversary::MinEntry, _) => keep_best(),
                    (Adversary::Points, _) if self.cfg.waste_check => {
                        let all: Vec<Row> = kept.iter().filter(|p| p.1 >= 1).map(|p| p.0).collect();
                        let good: Vec<Row> = all
                            .iter()
                            .copied()
                            .filter(|&o| self.waste_invariant(held, &with_row(tau, o)))
                            .collect();
                        if good.is_empty() && self.waste_invariant(held, tau) {
                            self.violations += 1;
                            if self.violations <= 20 {
                                let ts: Vec<String> =
                                    tau.iter().map(|&t| self.fmt_row(t)).collect();
                                let hs: Vec<String> = held
                                    .iter()
                                    .filter(|z| z.count_ones() >= 2)
                                    .map(|&z| self.fmt(z))
                                    .collect();
                                println!(
                                    "# violation: compare {} ? {} after [{}] holding [{}]",
                                    self.fmt(x),
                                    self.fmt(y),
                                    ts.join(" ; "),
                                    hs.join(", ")
                                );
                            }
                            all
                        } else if good.is_empty() {
                            all
                        } else {
                            good
                        }
                    }
                    _ => kept.iter().filter(|p| p.1 >= 1).map(|p| p.0).collect(),
                }
            }
        };
        if opts.is_empty() || (opts.len() == 1 && self.implied(tau, opts[0])) {
            return None;
        }
        Some(opts)
    }

    fn expand(&mut self, held: &[Form], tau: &[Row], r: usize, print: bool) -> bool {
        let gt = self.closure(tau);
        let cand = self.candidates(&gt);
        for i in 0..held.len() {
            let x = held[i];
            if x == 0 {
                continue;
            }
            for &y in &held[i + 1..] {
                if self.root_skip(held, tau, 0, x, y) {
                    continue;
                }
                let Some(opts) = self.allowed(held, x, y, tau, &gt, cand) else {
                    continue;
                };
                if opts
                    .iter()
                    .all(|&o| self.wins(held, &with_row(tau, o), r - 1))
                {
                    if print {
                        self.print_comparison(x, y, &opts, held, tau, r);
                    }
                    return true;
                }
            }
        }
        for i in 0..held.len() {
            for j in (i + 1)..held.len() {
                let (x, y) = (held[i], held[j]);
                if x & y != 0 || x == 0 {
                    continue;
                }
                let z = x | y;
                if !self.sub_path[z as usize] || held.binary_search(&z).is_ok() {
                    continue;
                }
                if self.root_skip(held, tau, 1, x, y) {
                    continue;
                }
                let mut h2 = held.to_vec();
                let pos = h2.partition_point(|&v| v < z);
                h2.insert(pos, z);
                if self.wins(&h2, tau, r - 1) {
                    if print {
                        println!(
                            "{}add {} + {} = {}",
                            self.pad(),
                            self.fmt(x),
                            self.fmt(y),
                            self.fmt(z)
                        );
                        self.depth += 1;
                        self.strategy(&h2, tau, r - 1);
                        self.depth -= 1;
                    }
                    return true;
                }
            }
        }
        false
    }

    // ------------------------------------------------------- output

    fn pad(&self) -> String {
        "  ".repeat(self.depth)
    }

    fn print_comparison(
        &mut self,
        x: Form,
        y: Form,
        opts: &[Row],
        held: &[Form],
        tau: &[Row],
        r: usize,
    ) {
        println!("{}compare {} ? {}", self.pad(), self.fmt(x), self.fmt(y));
        for &o in opts {
            let (larger, smaller) = if o == row_of(x, y) { (x, y) } else { (y, x) };
            println!(
                "{}-> {} > {}",
                self.pad(),
                self.fmt(larger),
                self.fmt(smaller)
            );
            self.depth += 1;
            self.strategy(held, &with_row(tau, o), r - 1);
            self.depth -= 1;
        }
    }

    /// Prints a winning strategy from this state at the least budget that
    /// wins, which is at most `r`.
    fn strategy(&mut self, held: &[Form], tau: &[Row], r: usize) {
        if self.eval(held, tau).1 {
            let s = self.alive_set(tau).trailing_zeros() as usize;
            let set = self.entry_sets[s];
            let entries: Vec<String> = (0..self.k)
                .filter(|v| (set >> v) & 1 == 1)
                .map(|v| (v + 1).to_string())
                .collect();
            println!(
                "{}ready: entry {} (budget left {})",
                self.pad(),
                entries.join(","),
                r
            );
            return;
        }
        let least = (0..=r)
            .find(|&rr| self.wins(held, tau, rr))
            .expect("the state wins within r");
        assert!(self.expand(held, tau, least, true));
    }

    fn fmt_row(&self, r: Row) -> String {
        format!(
            "{} > {}",
            self.fmt((r & 0xffff) as Form),
            self.fmt((r >> 16) as Form)
        )
    }

    pub fn fmt(&self, z: Form) -> String {
        if z == 0 {
            return "0".into();
        }
        let k = self.k;
        let mut parts: Vec<String> = (0..k)
            .filter(|i| (z >> i) & 1 == 1)
            .map(|i| format!("b{}", i + 1))
            .collect();
        parts.extend(
            (0..k)
                .filter(|i| (z >> (k + i)) & 1 == 1)
                .map(|i| format!("c{}", i + 1)),
        );
        parts.join("+")
    }

    /// The least budget at most `cap` the program wins with, deepening one
    /// budget at a time from the root's lower bound; prints each budget's
    /// verdict and, with `print`, the winning strategy.
    pub fn value(&mut self) -> Option<usize> {
        let k = self.k;
        let mut held: Vec<Form> = (0..2 * k).map(|i| 1 << i).collect();
        held.push(0);
        held.sort_unstable();
        let (h, _) = self.eval(&held, &[]);
        for r in h..=self.cfg.cap {
            let t0 = Instant::now();
            let w = self.wins(&held, &[], r);
            println!(
                "# k={} budget {} {}  nodes {} tt {} lps {} ({:.1}s)",
                k,
                r,
                if w { "wins" } else { "loses" },
                self.nodes,
                self.tt.len(),
                self.lps,
                t0.elapsed().as_secs_f64()
            );
            if w {
                if self.cfg.print {
                    self.strategy(&held, &[], r);
                }
                return Some(r);
            }
        }
        None
    }
}
