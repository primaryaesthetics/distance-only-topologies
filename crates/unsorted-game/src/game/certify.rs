//! A refutation certificate: a record, checkable without this program, that
//! the unrestricted program cannot win the strict game within a budget.
//!
//! The certificate is a list of states, each with a budget it loses at. A
//! state is written as its held forms and its transcript, and comes with
//! integer points strictly inside its cone. A state is either a leaf, refuted
//! by the cell bound, or an inner state, whose every legal move is answered:
//!
//! * an addition producing a form not yet held leads to the state with that
//!   form added, which must itself be listed at one less budget or more, up
//!   to an automorphism of the topology;
//! * a comparison of two distinct held forms is either answered by one sign,
//!   whose child state must be listed at one less budget or more, or shown
//!   to reveal nothing: its answer is a non-negative combination of the
//!   transcript rows and the coordinates, given by its multipliers.
//!
//! The points of a state are taken from cells of its cone. A leaf lists, for
//! one addition count `a`, enough cells whose distance forms need at least
//! `a` additions that `a + ceil(log2 #cells) > r`; an inner state lists two
//! cells, or one cell some of whose distance forms are not held, which shows
//! it is not ready.
//!
//! Lines, after a header of `unsorted-refutation 1`, `arcs ARCS`,
//! `budget R`, are one state each, fields separated by `|`:
//!
//!     r | held forms | transcript rows | points | a | entries
//!
//! A form, a row and a point are comma-separated integers, one per arc, and
//! several of them are separated by spaces; each point is followed by `:c`,
//! the addition count of its cell's distance forms from the held forms. An
//! entry `i,j,s` answers the
//! comparison of held forms `i` and `j` (positions on the line, from 0) with
//! `sign(form_i - form_j) = s`; an entry `i,j,s,D,l_1,...,l_k` claims that
//! `D * s * (form_i - form_j) - sum l_t * row_t` is non-negative in every
//! coordinate, with `D > 0`, `l_t >= 0` and one `l_t` per transcript row. A
//! leaf has no entries. The last line is `end STATES`.

use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

use super::{ceil_log2, feasible, neg, normalise, Cone, Game};
use crate::lp::Cvec;
use crate::topo::{add, lane, Form};

/// Counts gathered while a certificate is written.
#[derive(Default, Debug)]
pub struct CertStats {
    pub states: u64,
    pub leaves: u64,
    pub inner: u64,
    pub answered: u64,
    pub implied: u64,
    pub implied_lp: u64,
    pub points: u64,
    pub bytes: u64,
}

/// An exact rational, kept in lowest terms with a positive denominator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Q {
    n: i128,
    d: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Q {
    fn int(n: i128) -> Q {
        Q { n, d: 1 }
    }
    fn new(n: i128, d: i128) -> Q {
        let g = gcd(n, d).max(1);
        let s = if d < 0 { -1 } else { 1 };
        Q {
            n: s * n / g,
            d: s * d / g,
        }
    }
    fn sub(self, o: Q) -> Q {
        Q::new(self.n * o.d - o.n * self.d, self.d * o.d)
    }
    fn mul(self, o: Q) -> Q {
        Q::new(self.n * o.n, self.d * o.d)
    }
    fn div(self, o: Q) -> Q {
        Q::new(self.n * o.d, self.d * o.n)
    }
    fn neg(self) -> bool {
        self.n < 0
    }
    fn pos(self) -> bool {
        self.n > 0
    }
    fn lt(self, o: Q) -> bool {
        self.n * o.d < o.n * self.d
    }
}

/// Non-negative multipliers `l` with `sum_t l_t * gens_t <= v` in every
/// coordinate, as `(D, D * l)` with integer entries, or `None` if there are
/// none. A phase-one simplex over exact rationals with Bland's rule.
fn nonneg_combination(m: usize, gens: &[Cvec], v: &[i64]) -> Option<(i128, Vec<i128>)> {
    let k = gens.len();
    if v[..m].iter().all(|&x| x >= 0) {
        return Some((1, vec![0; k]));
    }
    // Rows: sum_t gens_t[i] l_t + s_i = v_i, with rows of negative v_i
    // negated and given an artificial. Columns: l (k), s (m), art (m), rhs.
    let cols = k + 2 * m + 1;
    let rhs = cols - 1;
    let mut t: Vec<Vec<Q>> = Vec::with_capacity(m + 1);
    let mut basic = Vec::with_capacity(m);
    let mut is_art = vec![false; m];
    for i in 0..m {
        let sign: i128 = if v[i] < 0 { -1 } else { 1 };
        let mut row = vec![Q::int(0); cols];
        for j in 0..k {
            row[j] = Q::int(sign * gens[j][i] as i128);
        }
        row[k + i] = Q::int(sign);
        row[rhs] = Q::int(sign * v[i] as i128);
        if sign < 0 {
            row[k + m + i] = Q::int(1);
            basic.push(k + m + i);
            is_art[i] = true;
        } else {
            basic.push(k + i);
        }
        t.push(row);
    }
    let mut obj = vec![Q::int(0); cols];
    for i in 0..m {
        if is_art[i] {
            for j in 0..cols {
                if j >= k + m && j < k + 2 * m {
                    continue;
                }
                obj[j] = obj[j].sub(t[i][j]);
            }
        }
    }
    t.push(obj);
    loop {
        let q = (0..k + m).find(|&j| t[m][j].neg());
        let Some(q) = q else { break };
        let mut p: Option<usize> = None;
        for i in 0..m {
            if !t[i][q].pos() {
                continue;
            }
            let ratio = t[i][rhs].div(t[i][q]);
            match p {
                None => p = Some(i),
                Some(r) => {
                    let best = t[r][rhs].div(t[r][q]);
                    if ratio.lt(best) || (ratio == best && basic[i] < basic[r]) {
                        p = Some(i);
                    }
                }
            }
        }
        let p = p?;
        let piv = t[p][q];
        for j in 0..cols {
            t[p][j] = t[p][j].div(piv);
        }
        for i in 0..=m {
            if i == p || t[i][q].n == 0 {
                continue;
            }
            let f = t[i][q];
            for j in 0..cols {
                let x = t[i][j].sub(f.mul(t[p][j]));
                t[i][j] = x;
            }
        }
        basic[p] = q;
    }
    // Phase one ends with the artificials at zero exactly when feasible; the
    // objective row's right-hand side is minus their sum.
    if t[m][rhs].n != 0 {
        return None;
    }
    let mut lam = vec![Q::int(0); k];
    for i in 0..m {
        if basic[i] < k {
            lam[basic[i]] = t[i][rhs];
        }
    }
    let mut den: i128 = 1;
    for x in &lam {
        den = den / gcd(den, x.d) * x.d;
    }
    let out: Vec<i128> = lam.iter().map(|x| x.n * (den / x.d)).collect();
    for i in 0..m {
        let s: i128 = (0..k).map(|j| out[j] * gens[j][i] as i128).sum();
        assert!(s <= den * v[i] as i128, "combination check");
    }
    assert!(out.iter().all(|&x| x >= 0));
    Some((den, out))
}

fn fmt_form(z: Form, m: usize) -> String {
    (0..m)
        .map(|i| lane(z, i).to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn fmt_ints<T: ToString>(v: &[T]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub struct Certifier<'a> {
    game: &'a mut Game,
    out: &'a mut dyn Write,
    /// The largest budget each written state is refuted at, by state key up
    /// to automorphism.
    written: HashMap<u128, u8>,
    pub stats: CertStats,
}

impl<'a> Certifier<'a> {
    pub fn new(game: &'a mut Game, out: &'a mut dyn Write) -> Self {
        Certifier {
            game,
            out,
            written: HashMap::new(),
            stats: CertStats::default(),
        }
    }

    fn line(&mut self, s: &str) -> Result<(), String> {
        self.stats.bytes += s.len() as u64 + 1;
        writeln!(self.out, "{s}").map_err(|e| e.to_string())
    }

    /// Writes a certificate that the program loses from the start within
    /// `r`; an error if the program wins.
    pub fn run(&mut self, r: u8) -> Result<(), String> {
        let held = self.game.initial();
        let root = self.game.root.clone();
        if self.game.wins(&held, &root, r) {
            return Err(format!("the program wins within {r}; nothing to refute"));
        }
        let arcs = self.game.top.to_string();
        self.line("unsorted-refutation 1")?;
        self.line(&format!("arcs {arcs}"))?;
        self.line(&format!("budget {r}"))?;
        self.refute(&held, &root, r)?;
        let n = self.stats.states;
        self.line(&format!("end {n}"))?;
        self.out.flush().map_err(|e| e.to_string())
    }

    /// The addition count of each cell's distance forms from `held`.
    fn cell_chains(&mut self, held: &[Form], cone: &Cone) -> Vec<u8> {
        let mut out = Vec::with_capacity(cone.cells.len());
        for k in 0..cone.cells.len() {
            let cell = cone.cells[k];
            let targets: Vec<Form> = (0..self.game.labels)
                .map(|i| self.game.paths[i][cell[i] as usize])
                .collect();
            out.push(self.game.chain(held, &targets));
        }
        out
    }

    /// Writes the states refuting `(held, cone)` at budget `r`, children
    /// first; the state is known to lose at `r`.
    fn refute(&mut self, held: &[Form], cone: &Rc<Cone>, r: u8) -> Result<(), String> {
        let key = self.game.state_key(held, cone);
        if self.written.get(&key).is_some_and(|&b| b >= r) {
            return Ok(());
        }
        let m = self.game.m;
        if self.game.unready(held, cone).is_empty() {
            return Err("a ready state was to be refuted".into());
        }
        let chains = self.cell_chains(held, cone);
        // The cell bound at each addition count a: a + ceil log2 N_a.
        let mut best: Option<(u8, usize)> = None;
        let top = chains.iter().copied().max().unwrap_or(0);
        for a in 0..=top {
            let na = chains.iter().filter(|&&c| c >= a).count();
            if na == 0 {
                continue;
            }
            if a as u32 + ceil_log2(na) as u32 > r as u32 {
                let need = if a > r { 1 } else { (1usize << (r - a)) + 1 };
                if best.is_none_or(|(_, n)| need < n) {
                    best = Some((a, need));
                }
            }
        }
        let tau: Vec<String> = cone.tau.iter().map(|c| fmt_ints(&c[..m])).collect();
        let forms: Vec<String> = held.iter().map(|&z| fmt_form(z, m)).collect();
        if let Some((a, need)) = best {
            let pts: Vec<String> = (0..cone.cells.len())
                .filter(|&k| chains[k] >= a)
                .take(need)
                .map(|k| format!("{}:{}", fmt_ints(&cone.witnesses[k][..m]), chains[k]))
                .collect();
            self.stats.points += pts.len() as u64;
            let s = format!(
                "{r}|{}|{}|{}|{a}|",
                forms.join(" "),
                tau.join(" "),
                pts.join(" ")
            );
            self.line(&s)?;
            self.stats.leaves += 1;
            self.stats.states += 1;
            self.written.insert(key, r);
            return Ok(());
        }
        if r == 0 {
            return Err("an unready state at budget 0 escaped the cell bound".into());
        }
        // Not ready: two cells, or one cell whose forms are not all held.
        let (pts, a) = if cone.cells.len() >= 2 {
            (vec![0usize, 1], 0u8)
        } else {
            (vec![0usize], 1u8)
        };
        if a == 1 && chains[0] == 0 {
            return Err("an unready state with one cell and every form held".into());
        }
        let mut entries: Vec<String> = Vec::new();
        for i in 0..held.len() {
            for j in i..held.len() {
                let Some(z) = add(held[i], held[j], m) else {
                    return Err("an addition beyond the lane cap at an inner state".into());
                };
                if held.binary_search(&z).is_ok() {
                    continue;
                }
                let mut h2 = held.to_vec();
                let pos = h2.partition_point(|&v| v < z);
                h2.insert(pos, z);
                self.refute(&h2, cone, r - 1)?;
            }
        }
        for i in 0..held.len() {
            for j in i + 1..held.len() {
                let (x, y) = (held[i], held[j]);
                let e = self.comparison(held, cone, r, i, j, x, y)?;
                entries.push(e);
            }
        }
        let pts: Vec<String> = pts
            .iter()
            .map(|&k| format!("{}:{}", fmt_ints(&cone.witnesses[k][..m]), chains[k]))
            .collect();
        self.stats.points += pts.len() as u64;
        let s = format!(
            "{r}|{}|{}|{}|{a}|{}",
            forms.join(" "),
            tau.join(" "),
            pts.join(" "),
            entries.join(" ")
        );
        self.line(&s)?;
        self.stats.inner += 1;
        self.stats.states += 1;
        self.written.insert(key, r);
        Ok(())
    }

    /// The entry answering the comparison of `x = held[i]` with `y = held[j]`
    /// at a state losing at `r`, after writing the child it points to.
    #[allow(clippy::too_many_arguments)]
    fn comparison(
        &mut self,
        held: &[Form],
        cone: &Rc<Cone>,
        r: u8,
        i: usize,
        j: usize,
        x: Form,
        y: Form,
    ) -> Result<String, String> {
        let m = self.game.m;
        // `c = flip * (x - y) / g`, primitive with its first nonzero entry
        // positive, so `c > 0` is the answer `sign(x - y) = flip`.
        let (c, flip) = normalise(x, y, m);
        let diff: Vec<i64> = (0..m)
            .map(|t| lane(x, t) as i64 - lane(y, t) as i64)
            .collect();
        let implied = |s: i64, game_tau: &[Cvec]| -> Option<String> {
            let v: Vec<i64> = diff.iter().map(|&d| s * d).collect();
            let (den, lam) = nonneg_combination(m, game_tau, &v)?;
            Some(format!(
                "{i},{j},{s},{den}{}",
                lam.iter().map(|l| format!(",{l}")).collect::<String>()
            ))
        };
        let tau = cone.tau.clone();
        let nc = neg(&c);
        let known = if diff.iter().all(|&d| d >= 0) {
            Some(1)
        } else if diff.iter().all(|&d| d <= 0) {
            Some(-1)
        } else if tau.binary_search(&c).is_ok() {
            Some(flip as i64)
        } else if tau.binary_search(&nc).is_ok() {
            Some(-(flip as i64))
        } else {
            None
        };
        if let Some(s) = known {
            self.stats.implied += 1;
            return implied(s, &tau)
                .ok_or_else(|| "a known answer without multipliers".to_string());
        }
        let a = self.game.child_cone(cone, &c);
        let b = self.game.child_cone(cone, &nc);
        match (feasible(&a), feasible(&b)) {
            (true, true) => {
                self.stats.answered += 1;
                let (child, s) = if !self.game.wins(held, &a, r - 1) {
                    (a, flip as i64)
                } else if !self.game.wins(held, &b, r - 1) {
                    (b, -(flip as i64))
                } else {
                    return Err("both answers win at a losing state".into());
                };
                self.refute(held, &child, r - 1)?;
                Ok(format!("{i},{j},{s}"))
            }
            (fa, fb) => {
                let s = if fa {
                    flip as i64
                } else if fb {
                    -(flip as i64)
                } else {
                    return Err("a comparison with no feasible answer".into());
                };
                self.stats.implied += 1;
                self.stats.implied_lp += 1;
                implied(s, &tau).ok_or_else(|| "a forced answer without multipliers".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::MAXM;

    fn row(v: &[i16]) -> Cvec {
        let mut c = [0i16; MAXM];
        c[..v.len()].copy_from_slice(v);
        c
    }

    #[test]
    fn a_chain_of_two_rows_implies_their_sum() {
        // w0 > w1 and w1 > w2 give w0 > w2.
        let gens = [row(&[1, -1, 0]), row(&[0, 1, -1])];
        let (d, l) = nonneg_combination(3, &gens, &[1, 0, -1]).unwrap();
        assert!(d > 0);
        assert_eq!(l, vec![d, d]);
    }

    #[test]
    fn an_open_answer_has_no_multipliers() {
        let gens = [row(&[1, -1, 0])];
        assert!(nonneg_combination(3, &gens, &[0, 1, -1]).is_none());
    }

    #[test]
    fn the_orthant_alone_implies_a_non_negative_difference() {
        assert_eq!(nonneg_combination(2, &[], &[2, 0]), Some((1, vec![])));
    }
}
