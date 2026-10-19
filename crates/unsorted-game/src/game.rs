//! The transcript-cone game under the strict adversary.
//!
//! A state is a set `M` of held forms and a transcript `tau` of strict
//! comparison outcomes, each a linear form asserted positive on the weights,
//! which together cut the open orthant to an open polyhedral cone. The program
//! moves by adding two held forms or by comparing two held forms, and must win
//! on every outcome the cone leaves feasible. It has won when every label has a
//! held form equal to its distance on the whole cone.
//!
//! On a full-dimensional cone the cone's cells — the assignments of one
//! strict-minimum path to each label that hold on some open subset — decide
//! readiness and possibility without any further geometry, and a child cone's
//! cells are a subset of its parent's, so each child is built from its parent
//! by one feasibility test per parent cell, with the parent cell's witness
//! point reused whenever it already satisfies the new outcome.
//!
//! The value is found by iterative deepening with a transposition table keyed
//! on the state up to the topology's automorphisms, and a lower bound from the
//! cells' addition chains: under strict answers every leaf of a winning tree
//! lies in one cell, a leaf in cell `c` has spent at least `chain(c)` additions,
//! and the comparison nodes form a binary tree, so by Kraft's inequality a
//! budget `r` must satisfy `sum over cells c of 2^(chain(c) - r) <= 1`, which
//! implies `r >= a + log2 |{cells with chain >= a}|` for all `a`.
//!
//! In the free-addition games the program holds every form it could ever
//! build from the start and pays only for comparisons, so the value is the
//! depth of a linear decision tree that locates the weight vector's cell —
//! the assignment of one shortest path to each label — with queries drawn
//! from the differences of held forms. The path-register version holds every
//! sub-path form, so its queries compare two sums of sub-paths; the
//! unrestricted version holds every 0/1 form over the arcs, so its queries
//! are every linear form with coefficients in {-1, 0, 1}. With nothing to
//! add, the chain bound reduces to `r >= log2 |cells|`.
//!
//! The same game is also played with the two resources budgeted separately, a
//! comparison budget and an addition budget that every branch must respect,
//! which traces the resource region of Cai's §2.3 under the strict adversary.
//! There the bound is `rc >= log2 |cells|` and `ra >= max chain(c)`, since every
//! cell holds a leaf.

use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::lp::{dot, gcd, interior_point, Cvec, MAXM};
use crate::topo::{add, below, lane, unit, Form, Topology};

mod certify;
pub use certify::Certifier;

const MAXL: usize = 8;

/// One strict-minimum path per label, by index into `paths[label]`.
type Cell = [u8; MAXL];

/// The cone of a transcript and the cells meeting it, each with a point of
/// the cell inside the cone. A child keeps the parent's point wherever it
/// still lies inside, so points are shared, not copied.
struct Cone {
    tau: Vec<Cvec>,
    cells: Vec<Cell>,
    witnesses: Vec<Rc<[i128]>>,
}

/// Bytes the allocator spends on one allocation besides its contents.
const ALLOC_OVERHEAD: usize = 16;

impl Cone {
    /// An upper estimate of the heap bytes this cone keeps alive. Every point
    /// it refers to is counted in full, also when shared with another cone,
    /// since a shared point outlives the cone that found it.
    fn bytes(&self, m: usize) -> usize {
        let point = 16 * m + 2 * std::mem::size_of::<usize>() + ALLOC_OVERHEAD;
        std::mem::size_of::<Cone>()
            + ALLOC_OVERHEAD
            + self.tau.capacity() * std::mem::size_of::<Cvec>()
            + ALLOC_OVERHEAD
            + self.cells.capacity() * std::mem::size_of::<Cell>()
            + ALLOC_OVERHEAD
            + self.witnesses.capacity() * std::mem::size_of::<Rc<[i128]>>()
            + ALLOC_OVERHEAD
            + self.witnesses.len() * point
    }
}

/// A memo table of bounded size in two generations. Entries go into the
/// young one; when its estimated size reaches half the limit, the old one is
/// dropped and the young one takes its place, and an entry found in the old
/// one is moved back into the young one. Every table here caches the value
/// of a pure function of its key, so a dropped entry costs only its
/// recomputation and never changes an answer; it changes which states the
/// search expands again, so node counts and statistics gathered over the
/// expanded states (such as `forced_loss`) may differ.
struct Memo<K, V> {
    young: HashMap<K, (V, usize)>,
    old: HashMap<K, (V, usize)>,
    bytes: usize,
    limit: usize,
    per_entry: usize,
    rotations: u64,
}

impl<K: Hash + Eq + Copy, V: Clone> Memo<K, V> {
    fn new(limit: usize) -> Self {
        Memo {
            young: HashMap::new(),
            old: HashMap::new(),
            bytes: 0,
            limit,
            // Key and value in a hash table at its usual load, with the
            // spare capacity a doubling table carries.
            per_entry: 2 * (std::mem::size_of::<(K, (V, usize))>() + 1) * 8 / 7,
            rotations: 0,
        }
    }

    fn get(&mut self, k: &K) -> Option<V> {
        if let Some((v, _)) = self.young.get(k) {
            return Some(v.clone());
        }
        let (v, extra) = self.old.remove(k)?;
        self.insert_sized(*k, v.clone(), extra);
        Some(v)
    }

    fn insert(&mut self, k: K, v: V) {
        self.insert_sized(k, v, 0);
    }

    /// Inserts with `extra` bytes owned by the value beyond its own size.
    fn insert_sized(&mut self, k: K, v: V, extra: usize) {
        if self.bytes >= self.limit / 2 {
            self.old = std::mem::take(&mut self.young);
            self.bytes = 0;
            self.rotations += 1;
        }
        if self.young.insert(k, (v, extra)).is_none() {
            self.bytes += self.per_entry + extra;
        }
    }

    /// The entry for `k`, found in either generation or inserted as `default`.
    fn entry(&mut self, k: K, default: V) -> &mut V {
        if !self.young.contains_key(&k) {
            let (v, extra) = self.old.remove(&k).unwrap_or((default, 0));
            self.insert_sized(k, v, extra);
        }
        &mut self.young.get_mut(&k).unwrap().0
    }

    fn len(&self) -> usize {
        self.young.len() + self.old.len()
    }
}

#[derive(Clone, Copy)]
struct Entry {
    win: u8,
    lose: i16,
    lb: u8,
}

const NO_WIN: u8 = u8::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The program is unrestricted; the adversary answers strictly.
    Generic,
    /// The program holds only sub-path forms; the adversary answers strictly.
    Path,
    /// Every 0/1 form is held from the start and additions cost nothing.
    GenericFree,
    /// Every sub-path form is held from the start and additions cost nothing.
    PathFree,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "generic" => Some(Mode::Generic),
            "path" => Some(Mode::Path),
            "generic-free" => Some(Mode::GenericFree),
            "path-free" => Some(Mode::PathFree),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Generic => "generic",
            Mode::Path => "path",
            Mode::GenericFree => "generic-free",
            Mode::PathFree => "path-free",
        }
    }

    /// Whether the program may hold only sub-path forms.
    pub fn path_only(self) -> bool {
        matches!(self, Mode::Path | Mode::PathFree)
    }

    /// Whether additions are free and every holdable form is held at the start.
    pub fn free_add(self) -> bool {
        matches!(self, Mode::GenericFree | Mode::PathFree)
    }
}

pub struct Game {
    pub top: Topology,
    m: usize,
    labels: usize,
    path_only: bool,
    free_add: bool,
    held0: Vec<Form>,
    paths: Vec<Vec<Form>>,
    path_forms: Vec<Form>,
    sub_forms: Vec<Form>,
    autos: Vec<Vec<usize>>,
    root: Rc<Cone>,
    cones: Memo<u128, Rc<Cone>>,
    cones_built: u64,
    tt: Memo<u128, Entry>,
    chain_memo: Memo<u128, u8>,
    split_tt: Memo<(u128, u8, u8), bool>,
    /// The largest budget shown losing from the start, by the search or by
    /// the static lower bound at the root.
    root_lose: i16,
    /// When set, the adversary must answer each comparison with an outcome
    /// that keeps the most cells of fewest entries (see `min_entry_cells`);
    /// it chooses freely only between outcomes that tie. A restricted
    /// adversary can only lower the value, so the value stays a lower bound.
    pub restrict: bool,
    /// When set, `value` tries budgets from this one up, taking every lower
    /// budget as lost without searching it: the value it returns is then the
    /// least winning budget at or above `from`.
    pub from: Option<u8>,
    /// Regions the adversary is confined to, each a list of rows asserted
    /// positive; empty for the unconfined game. A cell of a cone is then an
    /// assignment of paths together with one region, stored in the last slot
    /// of the cell, and the cone meets it when the transcript, the paths'
    /// strict-minimum rows and the region's rows have a common interior
    /// point. A confined adversary can only lower the value.
    regions: Vec<Vec<Cvec>>,
    /// Regions the adversary must keep meeting the cone (indexed as
    /// `regions`); empty for no such duty. An answer that would leave none
    /// of them is not given, so an answer may be forced, and the solver
    /// then prunes counting the cells of kept regions only.
    pub keep: Vec<bool>,
    /// Regions the adversary keeps meeting the cone whenever an allowed
    /// answer does (indexed as `regions`); empty for none. The solver then
    /// prunes with the least chain of a cell only.
    pub prefer: Vec<bool>,
    /// A name for each region, printed at the leaves of a strategy.
    pub region_names: Vec<String>,
    entry: Vec<Vec<bool>>,
    min_entries: usize,
    /// Under `restrict`, the most min-entry cells any comparison met so far
    /// removes whichever way it is answered.
    pub forced_loss: usize,
    pub nodes: u64,
    pub lps: u64,
    deadline: Instant,
    pub timed_out: bool,
    started: Instant,
    next_report: Instant,
}

#[derive(Clone)]
enum Move {
    Add {
        x: Form,
        y: Form,
        z: Form,
    },
    Cmp {
        x: Form,
        y: Form,
        c: Cvec,
        flip: i16,
    },
}

/// `x - y` as a primitive integer vector with its first nonzero entry
/// positive, and the sign that was divided out.
fn normalise(x: Form, y: Form, m: usize) -> (Cvec, i16) {
    let mut c = [0i16; MAXM];
    let mut g: i128 = 0;
    for i in 0..m {
        c[i] = lane(x, i) as i16 - lane(y, i) as i16;
        g = gcd(g, c[i] as i128);
    }
    if g == 0 {
        return (c, 1);
    }
    let mut flip = 1i16;
    for i in 0..m {
        c[i] /= g as i16;
    }
    for i in 0..m {
        if c[i] != 0 {
            if c[i] < 0 {
                flip = -1;
            }
            break;
        }
    }
    if flip < 0 {
        for v in c.iter_mut() {
            *v = -*v;
        }
    }
    (c, flip)
}

fn neg(c: &Cvec) -> Cvec {
    let mut d = *c;
    for v in d.iter_mut() {
        *v = -*v;
    }
    d
}

#[inline]
fn mix(h: u64, x: u64) -> u64 {
    let mut h = h ^ x;
    h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

struct Fp(u64, u64);

impl Fp {
    fn new(tag: u64) -> Fp {
        Fp(
            mix(0x1234_5678_9ABC_DEF0, tag),
            mix(0x0FED_CBA9_8765_4321, tag),
        )
    }
    fn u64(&mut self, x: u64) {
        self.0 = mix(self.0, x);
        self.1 = mix(self.1, x.rotate_left(17) ^ 0xA5A5_A5A5_A5A5_A5A5);
    }
    fn form(&mut self, z: Form) {
        self.u64(z as u64);
        self.u64((z >> 64) as u64);
    }
    fn cvec(&mut self, c: &Cvec, m: usize) {
        for i in 0..m {
            self.u64(c[i] as i64 as u64);
        }
    }
    fn done(self) -> u128 {
        ((self.0 as u128) << 64) | self.1 as u128
    }
}

fn fp_tau(tau: &[Cvec], m: usize) -> u128 {
    let mut f = Fp::new(1);
    f.u64(tau.len() as u64);
    for c in tau {
        f.cvec(c, m);
    }
    f.done()
}

/// How often a running search writes its progress to stderr.
const REPORT_EVERY: Duration = Duration::from_secs(600);

/// Whether some cell meets the cone, that is, the transcript can occur.
fn feasible(cone: &Cone) -> bool {
    !cone.cells.is_empty()
}

fn ceil_log2(k: usize) -> u8 {
    let mut d = 0u8;
    while (1usize << d) < k {
        d += 1;
    }
    d
}

/// The bound on the memory the caches of one game may hold, 4 GiB unless set.
static CACHE_BYTES: AtomicUsize = AtomicUsize::new(4 << 30);

/// Bounds the memory the caches of every game made from now on may hold.
pub fn set_cache_bytes(bytes: usize) {
    CACHE_BYTES.store(bytes, Ordering::Relaxed);
}

impl Game {
    /// A game whose caches hold about the set bound: half for cones, a
    /// quarter for each transposition table (a command uses one of the two),
    /// a sixteenth for chains.
    pub fn new(top: Topology, mode: Mode) -> Game {
        Game::confined(top, mode, Vec::new())
    }

    /// A game whose adversary answers only with outcomes that leave some
    /// region of `regions` meeting the cone (see the `regions` field). The
    /// regions need not be invariant under the topology's automorphisms, so
    /// states are keyed without them.
    pub fn confined(top: Topology, mode: Mode, regions: Vec<Vec<Cvec>>) -> Game {
        let cache_bytes = CACHE_BYTES.load(Ordering::Relaxed);
        let m = top.m();
        let labels = top.n - 1;
        assert!(labels <= MAXL, "at most {MAXL} labels");
        let paths = top.paths();
        let mut path_forms: Vec<Form> = paths.iter().flatten().copied().collect();
        path_forms.sort_unstable();
        path_forms.dedup();
        let mut sub_forms: Vec<Form> = Vec::new();
        for &p in &path_forms {
            let idx: Vec<usize> = (0..m).filter(|&i| lane(p, i) != 0).collect();
            for mask in 0u32..(1u32 << idx.len()) {
                let mut z: Form = 0;
                for (b, &i) in idx.iter().enumerate() {
                    if mask & (1 << b) != 0 {
                        z |= unit(i);
                    }
                }
                sub_forms.push(z);
            }
        }
        sub_forms.sort_unstable();
        sub_forms.dedup();
        assert!(
            regions.is_empty() || labels < MAXL,
            "at most {} labels with regions",
            MAXL - 1
        );
        assert!(regions.len() <= 256, "at most 256 regions");
        let autos = if regions.is_empty() {
            top.automorphisms()
        } else {
            vec![(0..m).collect()]
        };
        let mut held0: Vec<Form> = if !mode.free_add() {
            (0..m).map(unit).chain(std::iter::once(0)).collect()
        } else if mode.path_only() {
            sub_forms.clone()
        } else {
            (0..(1u32 << m))
                .map(|mask| {
                    (0..m)
                        .filter(|&i| mask >> i & 1 == 1)
                        .fold(0, |z, i| z | unit(i))
                })
                .collect()
        };
        held0.sort_unstable();
        let mut game = Game {
            top,
            m,
            labels,
            path_only: mode.path_only(),
            free_add: mode.free_add(),
            held0,
            paths,
            path_forms,
            sub_forms,
            autos,
            root: Rc::new(Cone {
                tau: Vec::new(),
                cells: Vec::new(),
                witnesses: Vec::new(),
            }),
            cones: Memo::new(cache_bytes / 2),
            cones_built: 0,
            tt: Memo::new(cache_bytes / 4),
            chain_memo: Memo::new(cache_bytes / 16),
            split_tt: Memo::new(cache_bytes / 4),
            root_lose: -1,
            restrict: false,
            from: None,
            regions,
            keep: Vec::new(),
            prefer: Vec::new(),
            region_names: Vec::new(),
            entry: Vec::new(),
            min_entries: 0,
            forced_loss: 0,
            nodes: 0,
            lps: 0,
            deadline: Instant::now() + Duration::from_secs(3600 * 24 * 365),
            timed_out: false,
            started: Instant::now(),
            next_report: Instant::now() + REPORT_EVERY,
        };
        game.entry = game
            .paths
            .iter()
            .map(|ps| {
                ps.iter()
                    .map(|&z| (0..game.m).filter(|&i| lane(z, i) != 0).count() == 1)
                    .collect()
            })
            .collect();
        game.root_cone();
        game.min_entries = (0..game.root.cells.len())
            .map(|k| game.entries(&game.root.cells[k]))
            .min()
            .unwrap_or(0);
        game
    }

    /// The labels a cell serves by a single arc from the source.
    fn entries(&self, cell: &Cell) -> usize {
        (0..self.labels)
            .filter(|&i| self.entry[i][cell[i] as usize])
            .count()
    }

    /// The cells of a cone with as few entries as any cell of the topology.
    fn min_entry_cells(&self, cone: &Cone) -> usize {
        cone.cells
            .iter()
            .filter(|c| self.entries(c) == self.min_entries)
            .count()
    }

    /// The outcomes of comparison `c` the adversary may give: both feasible
    /// children, or under `restrict` those keeping the most min-entry cells.
    /// Empty when the comparison cannot split the cone.
    fn answers(&mut self, cone: &Cone, c: &Cvec) -> Vec<(Rc<Cone>, i16)> {
        let a = self.child_cone(cone, c);
        let b = self.child_cone(cone, &neg(c));
        let mut out: Vec<(Rc<Cone>, i16)> = [(a, 1i16), (b, -1i16)]
            .into_iter()
            .filter(|(k, _)| feasible(k))
            .collect();
        if out.len() < 2 {
            return Vec::new();
        }
        if !self.keep.is_empty() || !self.prefer.is_empty() {
            if !self.keep.is_empty() {
                let keep = &self.keep;
                out.retain(|(k, _)| k.cells.iter().any(|c| keep[c[MAXL - 1] as usize]));
                assert!(!out.is_empty(), "a kept region left both answers");
            }
            if !self.prefer.is_empty() {
                let pr = &self.prefer;
                let meets = |k: &Cone| k.cells.iter().any(|c| pr[c[MAXL - 1] as usize]);
                if out.iter().any(|(k, _)| meets(k)) {
                    out.retain(|(k, _)| meets(k));
                }
            }
            return out;
        }
        if self.restrict {
            let keep: Vec<usize> = out.iter().map(|(k, _)| self.min_entry_cells(k)).collect();
            let here = self.min_entry_cells(cone);
            let loss = here - keep[0].max(keep[1]);
            self.forced_loss = self.forced_loss.max(loss);
            if keep[0] != keep[1] {
                out.remove(if keep[0] > keep[1] { 1 } else { 0 });
            }
        }
        out
    }

    /// Adds a form to those held before the first move.
    pub fn add_held(&mut self, z: Form) {
        if let Err(pos) = self.held0.binary_search(&z) {
            self.held0.insert(pos, z);
        }
    }

    /// The forms held before the first move: the zero form and the arc
    /// weights, or in a free-addition game every form the program may hold.
    pub fn initial(&self) -> Vec<Form> {
        self.held0.clone()
    }

    // ---------------------------------------------------------------- cones

    fn strict_rows(&self, label: usize, p: usize, rows: &mut Vec<Cvec>) {
        let pf = self.paths[label][p];
        for (q, &qf) in self.paths[label].iter().enumerate() {
            if q == p {
                continue;
            }
            let mut c = [0i16; MAXM];
            for i in 0..self.m {
                c[i] = lane(qf, i) as i16 - lane(pf, i) as i16;
            }
            rows.push(c);
        }
    }

    fn root_cone(&mut self) {
        let mut cells = Vec::new();
        let mut witnesses = Vec::new();
        let mut chosen: Cell = [0; MAXL];
        let mut rows: Vec<Cvec> = Vec::new();
        if self.regions.is_empty() {
            self.extend_cells(0, &mut chosen, &mut rows, &mut cells, &mut witnesses);
        } else {
            for r in 0..self.regions.len() {
                rows.clear();
                rows.extend_from_slice(&self.regions[r]);
                chosen[MAXL - 1] = r as u8;
                self.extend_cells(0, &mut chosen, &mut rows, &mut cells, &mut witnesses);
            }
        }
        self.root = Rc::new(Cone {
            tau: Vec::new(),
            cells,
            witnesses,
        });
        self.cones_built += 1;
    }

    fn extend_cells(
        &mut self,
        label: usize,
        chosen: &mut Cell,
        rows: &mut Vec<Cvec>,
        cells: &mut Vec<Cell>,
        witnesses: &mut Vec<Rc<[i128]>>,
    ) {
        if label == self.labels {
            return;
        }
        for p in 0..self.paths[label].len() {
            let mark = rows.len();
            self.strict_rows(label, p, rows);
            self.lps += 1;
            if let Some(w) = interior_point(self.m, rows) {
                chosen[label] = p as u8;
                if label + 1 == self.labels {
                    cells.push(*chosen);
                    witnesses.push(w.into());
                } else {
                    self.extend_cells(label + 1, chosen, rows, cells, witnesses);
                }
            }
            rows.truncate(mark);
        }
    }

    /// The cone of `parent`'s transcript with `c > 0` added; built from the
    /// parent's cells if not already known. Feasible iff it has a cell.
    fn child_cone(&mut self, parent: &Cone, c: &Cvec) -> Rc<Cone> {
        let mut tau = parent.tau.clone();
        let pos = tau.partition_point(|t| t < c);
        tau.insert(pos, *c);
        let key = fp_tau(&tau, self.m);
        if let Some(cone) = self.cones.get(&key) {
            return cone;
        }
        let mut cells = Vec::new();
        let mut witnesses = Vec::new();
        let mut rows: Vec<Cvec> = Vec::new();
        for (k, &cell) in parent.cells.iter().enumerate() {
            if dot(c, &parent.witnesses[k], self.m) > 0 {
                cells.push(cell);
                witnesses.push(parent.witnesses[k].clone());
                continue;
            }
            rows.clear();
            rows.extend_from_slice(&tau);
            for label in 0..self.labels {
                self.strict_rows(label, cell[label] as usize, &mut rows);
            }
            if !self.regions.is_empty() {
                rows.extend_from_slice(&self.regions[cell[MAXL - 1] as usize]);
            }
            self.lps += 1;
            if let Some(w) = interior_point(self.m, &rows) {
                cells.push(cell);
                witnesses.push(w.into());
            }
        }
        tau.shrink_to_fit();
        cells.shrink_to_fit();
        witnesses.shrink_to_fit();
        let cone = Rc::new(Cone {
            tau,
            cells,
            witnesses,
        });
        let bytes = cone.bytes(self.m);
        self.cones.insert_sized(key, cone.clone(), bytes);
        self.cones_built += 1;
        cone
    }

    // ------------------------------------------------------------ readiness

    /// The held form equal to label `i`'s distance on the whole cone, if any.
    fn ready_form(&self, held: &[Form], cone: &Cone, i: usize) -> Option<Form> {
        let cells = &cone.cells;
        let p0 = cells[0][i];
        if cells.iter().any(|c| c[i] != p0) {
            return None;
        }
        let z = self.paths[i][p0 as usize];
        held.binary_search(&z).ok().map(|_| z)
    }

    fn unready(&self, held: &[Form], cone: &Cone) -> Vec<usize> {
        (0..self.labels)
            .filter(|&i| self.ready_form(held, cone, i).is_none())
            .collect()
    }

    /// Labels among `open` that no held form can serve anywhere on the cone.
    fn missing(&self, held: &[Form], cone: &Cone, open: &[usize]) -> usize {
        let cells = &cone.cells;
        open.iter()
            .filter(|&&i| {
                !cells
                    .iter()
                    .any(|c| held.binary_search(&self.paths[i][c[i] as usize]).is_ok())
            })
            .count()
    }

    // ---------------------------------------------------------------- chain

    /// The least number of additions producing every target from the held
    /// forms, searching only forms below some target coefficientwise.
    fn chain(&mut self, held: &[Form], targets: &[Form]) -> u8 {
        let mut need: Vec<Form> = targets
            .iter()
            .copied()
            .filter(|t| held.binary_search(t).is_err())
            .collect();
        need.sort_unstable();
        need.dedup();
        if need.is_empty() {
            return 0;
        }
        let m = self.m;
        let useful: Vec<Form> = held
            .iter()
            .copied()
            .filter(|&z| need.iter().any(|&t| below(z, t, m)))
            .collect();
        let mut f = Fp::new(2);
        f.u64(useful.len() as u64);
        for &z in &useful {
            f.form(z);
        }
        f.u64(need.len() as u64);
        for &z in &need {
            f.form(z);
        }
        let key = f.done();
        if let Some(d) = self.chain_memo.get(&key) {
            return d;
        }
        let mut depth = need.len() as u8;
        while !Self::reach(m, &need, useful.clone(), need.clone(), depth) {
            depth += 1;
        }
        self.chain_memo.insert(key, depth);
        depth
    }

    fn reach(m: usize, need: &[Form], held: Vec<Form>, want: Vec<Form>, depth: u8) -> bool {
        if want.is_empty() {
            return true;
        }
        if depth == 0 || want.len() > depth as usize {
            return false;
        }
        for i in 0..held.len() {
            for j in i..held.len() {
                let Some(z) = add(held[i], held[j], m) else {
                    continue;
                };
                if held.binary_search(&z).is_ok() || !need.iter().any(|&t| below(z, t, m)) {
                    continue;
                }
                let mut h2 = held.clone();
                let pos = h2.partition_point(|&v| v < z);
                h2.insert(pos, z);
                let w2: Vec<Form> = want.iter().copied().filter(|&v| v != z).collect();
                if Self::reach(m, need, h2, w2, depth - 1) {
                    return true;
                }
            }
        }
        false
    }

    fn lower_bound(&mut self, held: &[Form], cone: &Cone) -> u8 {
        // Under regions a leaf may meet several cells with one assignment of
        // paths; the bound counts assignments, each needing its own leaf.
        let mut distinct: Vec<Cell> = Vec::new();
        if !self.regions.is_empty() {
            // Under `keep` only cells of kept regions are counted: the
            // adversary can follow any one of them, since an answer that
            // keeps it keeps a kept region, so each still needs a leaf.
            let keep = &self.keep;
            distinct = cone
                .cells
                .iter()
                .filter(|c| keep.is_empty() || keep[c[MAXL - 1] as usize])
                .map(|c| {
                    let mut d = *c;
                    d[MAXL - 1] = 0;
                    d
                })
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
        }
        let cells: &[Cell] = if self.regions.is_empty() {
            &cone.cells
        } else {
            &distinct
        };
        let n_cells = cells.len();
        let mut chains: Vec<u8> = Vec::with_capacity(n_cells);
        let mut targets: Vec<Form> = Vec::with_capacity(self.labels);
        for k in 0..n_cells {
            let cell = cells[k];
            targets.clear();
            for i in 0..self.labels {
                targets.push(self.paths[i][cell[i] as usize]);
            }
            chains.push(self.chain(held, &targets));
        }
        if self.restrict {
            // The restricted adversary keeps at least half of the min-entry
            // cells at each comparison, since each lies in some child, so a
            // leaf is reached only after `log2` of their number comparisons,
            // in one of them; nothing is claimed about the other cells.
            let low: Vec<usize> = (0..n_cells)
                .filter(|&k| self.entries(&cone.cells[k]) == self.min_entries)
                .collect();
            let pool: Vec<u8> = if low.is_empty() {
                chains.clone()
            } else {
                low.iter().map(|&k| chains[k]).collect()
            };
            let least = *pool.iter().min().unwrap_or(&0);
            return (least + ceil_log2(low.len().max(1))).max(1);
        }
        if !self.prefer.is_empty() {
            return (*chains.iter().min().unwrap_or(&0)).max(1);
        }
        chains.sort_unstable_by(|a, b| b.cmp(a));
        let mut best = 1u8;
        for (i, &a) in chains.iter().enumerate() {
            best = best.max(a + ceil_log2(i + 1));
        }
        // Kraft over the counted cells: every counted cell (path assignment)
        // holds its own leaf, a leaf in cell c at depth <= r passes at most
        // r - chain(c) comparison nodes, and the comparison nodes form a
        // binary tree, so sum_c 2^(chain(c) - r) <= 1.
        let sum: u128 = chains
            .iter()
            .map(|&a| 1u128 << a.min(127))
            .fold(0, u128::saturating_add);
        let r = (u128::BITS - sum.saturating_sub(1).leading_zeros()) as u8;
        best.max(r)
    }

    // ----------------------------------------------------------------- keys

    fn state_key(&self, held: &[Form], cone: &Cone) -> u128 {
        let tau = &cone.tau;
        let m = self.m;
        let mut best: Option<u128> = None;
        let mut forms: Vec<Form> = Vec::with_capacity(held.len());
        let mut cons: Vec<Cvec> = Vec::with_capacity(tau.len());
        for sigma in &self.autos {
            forms.clear();
            cons.clear();
            if sigma.iter().enumerate().all(|(i, &s)| i == s) {
                forms.extend_from_slice(held);
                cons.extend_from_slice(tau);
            } else {
                for &z in held {
                    let mut img: Form = 0;
                    for j in 0..m {
                        img |= lane(z, sigma[j]) << (8 * j);
                    }
                    forms.push(img);
                }
                forms.sort_unstable();
                for c in tau {
                    let mut img = [0i16; MAXM];
                    for j in 0..m {
                        img[j] = c[sigma[j]];
                    }
                    cons.push(img);
                }
                cons.sort_unstable();
            }
            let mut f = Fp::new(3);
            f.u64(forms.len() as u64);
            for &z in &forms {
                f.form(z);
            }
            f.u64(cons.len() as u64);
            for c in &cons {
                f.cvec(c, m);
            }
            let k = f.done();
            if best.is_none_or(|b| k < b) {
                best = Some(k);
            }
        }
        best.unwrap()
    }

    // ---------------------------------------------------------------- moves

    fn tier_form(&self, z: Form) -> u8 {
        if self.path_forms.binary_search(&z).is_ok() {
            0
        } else if self.sub_forms.binary_search(&z).is_ok() {
            1
        } else {
            2
        }
    }

    fn moves(&self, held: &[Form], cone: &Cone) -> Vec<Move> {
        let m = self.m;
        let tau = &cone.tau;
        let mut cmps: Vec<(u8, Form, Form, Move)> = Vec::new();
        let mut adds: Vec<(u8, Form, Form, Move)> = Vec::new();
        for i in 0..held.len() {
            let x = held[i];
            for j in i..held.len() {
                let y = held[j];
                if let Some(z) = add(x, y, m).filter(|_| !self.free_add) {
                    if held.binary_search(&z).is_err()
                        && (!self.path_only || self.sub_forms.binary_search(&z).is_ok())
                    {
                        adds.push((self.tier_form(z), x, y, Move::Add { x, y, z }));
                    }
                }
            }
            if x == 0 {
                continue;
            }
            for j in i + 1..held.len() {
                let y = held[j];
                let (c, flip) = normalise(x, y, m);
                if c.iter().all(|&v| v == 0) {
                    continue;
                }
                let nc = neg(&c);
                if tau.binary_search(&c).is_ok() || tau.binary_search(&nc).is_ok() {
                    continue;
                }
                let tx = self.tier_form(x);
                let ty = self.tier_form(y);
                let tier = if tx == 0 && ty == 0 {
                    0
                } else if tx <= 1 && ty <= 1 {
                    1
                } else {
                    2
                };
                cmps.push((tier, x, y, Move::Cmp { x, y, c, flip }));
            }
        }
        cmps.sort_by_key(|(t, x, y, _)| (*t, *x, *y));
        adds.sort_by_key(|(t, x, y, _)| (*t, *x, *y));
        if self.free_add {
            // Many pairs of held forms have the same difference; one query each.
            let mut seen: std::collections::HashSet<Cvec> = Default::default();
            cmps.retain(|(_, _, _, mv)| match mv {
                Move::Cmp { c, .. } => seen.insert(*c),
                Move::Add { .. } => true,
            });
        }
        cmps.into_iter()
            .chain(adds)
            .map(|(_, _, _, mv)| mv)
            .collect()
    }

    // --------------------------------------------------------------- search

    fn wins(&mut self, held: &[Form], cone: &Cone, r: u8) -> bool {
        let key = self.state_key(held, cone);
        if let Some(e) = self.tt.get(&key) {
            if e.win <= r {
                return true;
            }
            if e.lose >= r as i16 {
                return false;
            }
        }
        if self.tick() {
            return false;
        }
        let open = self.unready(held, cone);
        if open.is_empty() {
            self.record_win(key, 0);
            return true;
        }
        let lb = match self.tt.get(&key).map(|e| e.lb) {
            Some(lb) if lb > 0 => lb,
            _ => {
                let lb = self.lower_bound(held, cone);
                self.tt
                    .entry(
                        key,
                        Entry {
                            win: NO_WIN,
                            lose: -1,
                            lb: 0,
                        },
                    )
                    .lb = lb;
                lb
            }
        };
        if r == 0 || r < lb {
            self.record_lose(key, r);
            return false;
        }
        for mv in self.moves(held, cone) {
            match mv {
                Move::Add { z, .. } => {
                    let mut h2 = held.to_vec();
                    let pos = h2.partition_point(|&v| v < z);
                    h2.insert(pos, z);
                    let w = self.wins(&h2, cone, r - 1);
                    if self.timed_out {
                        return false;
                    }
                    if w {
                        self.record_win(key, r);
                        return true;
                    }
                }
                Move::Cmp { c, .. } => {
                    let mut outcomes: Vec<Rc<Cone>> =
                        self.answers(cone, &c).into_iter().map(|(k, _)| k).collect();
                    if outcomes.is_empty() {
                        continue;
                    }
                    // The adversary answers first with the outcome that
                    // leaves the most labels unserved.
                    let hurt: Vec<usize> = outcomes
                        .iter()
                        .map(|k| self.missing(held, k, &open))
                        .collect();
                    if hurt.len() == 2 && hurt[1] > hurt[0] {
                        outcomes.swap(0, 1);
                    }
                    let mut all = true;
                    for k in outcomes {
                        let w = self.wins(held, &k, r - 1);
                        if self.timed_out {
                            return false;
                        }
                        if !w {
                            all = false;
                            break;
                        }
                    }
                    if all {
                        self.record_win(key, r);
                        return true;
                    }
                }
            }
        }
        self.record_lose(key, r);
        false
    }

    /// The strategy printers may have to search again where the caches
    /// dropped a state, and a known win must not be lost to the clock.
    fn lift_deadline(&mut self) {
        self.deadline = Instant::now() + Duration::from_secs(3600 * 24 * 365);
        self.timed_out = false;
    }

    /// Counts a node; true once the deadline has passed. Writes the size of
    /// the caches to stderr every `REPORT_EVERY`.
    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes & 1023 != 0 {
            return false;
        }
        let now = Instant::now();
        if now >= self.next_report {
            self.next_report = now + REPORT_EVERY;
            eprintln!(
                "progress {:.0}s  nodes {}  lp {}  cones built {}  cache {:?}",
                (now - self.started).as_secs_f64(),
                self.nodes,
                self.lps,
                self.cones_built,
                self.cache_sizes()
            );
        }
        if now > self.deadline {
            self.timed_out = true;
            return true;
        }
        false
    }

    fn record_win(&mut self, key: u128, r: u8) {
        let e = self.tt.entry(
            key,
            Entry {
                win: NO_WIN,
                lose: -1,
                lb: 0,
            },
        );
        e.win = e.win.min(r);
    }

    fn record_lose(&mut self, key: u128, r: u8) {
        let e = self.tt.entry(
            key,
            Entry {
                win: NO_WIN,
                lose: -1,
                lb: 0,
            },
        );
        e.lose = e.lose.max(r as i16);
    }

    /// The least winning budget by iterative deepening; `None` on timeout.
    pub fn value(&mut self, budget: Duration, cap: Option<u8>) -> Option<u8> {
        assert!(
            !(self.restrict && !self.regions.is_empty()),
            "the min-entry adversary is not defined on region-tagged cells"
        );
        self.deadline = Instant::now() + budget;
        let held = self.initial();
        let root = self.root.clone();
        let cap = cap.unwrap_or((2 * self.m * (self.top.n - 1)).min(250) as u8);
        let start = if self.unready(&held, &root).is_empty() {
            0
        } else {
            self.lower_bound(&held, &root)
        };
        self.root_lose = self.root_lose.max(start as i16 - 1);
        let start = self.from.map_or(start, |f| f.max(start));
        for r in start..=cap {
            if self.wins(&held, &root, r) {
                return Some(r);
            }
            if self.timed_out {
                return None;
            }
            self.root_lose = self.root_lose.max(r as i16);
        }
        None
    }

    /// One more than the largest budget shown losing from the start.
    pub fn lower_bound_reached(&self) -> u8 {
        (self.root_lose + 1) as u8
    }

    // ------------------------------------------------------------- strategy

    /// Whether the root cone meets a kept region.
    pub fn root_meets_keep(&self) -> bool {
        self.root
            .cells
            .iter()
            .any(|c| self.keep[c[MAXL - 1] as usize])
    }

    /// The names of the regions the root cone meets, empty without regions.
    pub fn root_regions(&self) -> String {
        let root = self.root.clone();
        self.leaf_regions(&root)
    }

    /// The names of the regions a leaf cone meets, empty without regions.
    fn leaf_regions(&self, cone: &Cone) -> String {
        if self.region_names.is_empty() {
            return String::new();
        }
        let mut names: Vec<&str> = cone
            .cells
            .iter()
            .map(|c| self.region_names[c[MAXL - 1] as usize].as_str())
            .collect();
        names.sort_unstable();
        names.dedup();
        format!("   [{}]", names.join(" "))
    }

    /// Prints one winning strategy from a state known to win within `r`.
    pub fn strategy(&mut self, held: &[Form], r: u8) {
        self.lift_deadline();
        let root = self.root.clone();
        self.strategy_at(held, &root, r, 0);
    }

    fn strategy_at(&mut self, held: &[Form], cone: &Cone, r: u8, depth: usize) {
        let pad = "  ".repeat(depth);
        let open = self.unready(held, cone);
        if open.is_empty() {
            let parts: Vec<String> = (0..self.labels)
                .map(|i| {
                    let z = self.ready_form(held, cone, i).unwrap();
                    format!("d({}) = {}", i + 1, self.top.fmt_form(z))
                })
                .collect();
            println!(
                "{pad}ready: {}{}",
                parts.join("; "),
                self.leaf_regions(cone)
            );
            return;
        }
        for mv in self.moves(held, cone) {
            match mv {
                Move::Add { x, y, z } => {
                    let mut h2 = held.to_vec();
                    let pos = h2.partition_point(|&v| v < z);
                    h2.insert(pos, z);
                    if self.wins(&h2, cone, r - 1) {
                        println!(
                            "{pad}add {} + {}",
                            self.top.fmt_form(x),
                            self.top.fmt_form(y)
                        );
                        self.strategy_at(&h2, cone, r - 1, depth);
                        return;
                    }
                }
                Move::Cmp { x, y, c, flip } => {
                    let outcomes = self.answers(cone, &c);
                    if outcomes.is_empty() {
                        continue;
                    }
                    if outcomes.iter().all(|(k, _)| self.wins(held, k, r - 1)) {
                        println!(
                            "{pad}compare {} ? {}{}",
                            self.top.fmt_form(x),
                            self.top.fmt_form(y),
                            if outcomes.len() == 1 {
                                "   (forced)"
                            } else {
                                ""
                            }
                        );
                        for (k, s) in outcomes {
                            // `c > 0` is `x - y > 0` up to the flip.
                            let sym = if s * flip > 0 { ">" } else { "<" };
                            println!("{pad}  {sym}:");
                            self.strategy_at(held, &k, r - 1, depth + 2);
                        }
                        return;
                    }
                }
            }
        }
        panic!("no winning move at a winning state");
    }

    // ------------------------------------------------------- split budgets

    /// Every sub-path form together with the initial forms: the held set of a
    /// path-register program that has made every addition it could ever use.
    pub fn all_sub_forms(&self) -> Vec<Form> {
        let mut v = self.sub_forms.clone();
        v.extend(self.initial());
        v.sort_unstable();
        v.dedup();
        v
    }

    /// `(ceil log2 |cells|, max over cells of chain)`: a winning tree has a
    /// leaf in every cell of the cone, so it needs that comparison depth, and
    /// the branch ending in cell `c` makes at least `chain(c)` additions.
    fn split_lower_bound(&mut self, held: &[Form], cone: &Cone) -> (u8, u8) {
        let n_cells = cone.cells.len();
        let mut most = 0u8;
        let mut targets: Vec<Form> = Vec::with_capacity(self.labels);
        for k in 0..n_cells {
            let cell = cone.cells[k];
            targets.clear();
            for i in 0..self.labels {
                targets.push(self.paths[i][cell[i] as usize]);
            }
            most = most.max(self.chain(held, &targets));
        }
        (ceil_log2(n_cells), most)
    }

    /// Whether the program wins from `(held, cone)` with at most `rc`
    /// comparisons and at most `ra` additions on every branch.
    fn wins_split(&mut self, held: &[Form], cone: &Cone, rc: u8, ra: u8) -> bool {
        let key = (self.state_key(held, cone), rc, ra);
        if let Some(w) = self.split_tt.get(&key) {
            return w;
        }
        if self.tick() {
            return false;
        }
        if self.unready(held, cone).is_empty() {
            self.split_tt.insert(key, true);
            return true;
        }
        let (lc, la) = self.split_lower_bound(held, cone);
        if rc < lc || ra < la {
            self.split_tt.insert(key, false);
            return false;
        }
        for mv in self.moves(held, cone) {
            let w = match mv {
                Move::Add { z, .. } => {
                    if ra == 0 {
                        continue;
                    }
                    let mut h2 = held.to_vec();
                    let pos = h2.partition_point(|&v| v < z);
                    h2.insert(pos, z);
                    self.wins_split(&h2, cone, rc, ra - 1)
                }
                Move::Cmp { c, .. } => {
                    if rc == 0 {
                        continue;
                    }
                    let a = self.child_cone(cone, &c);
                    let b = self.child_cone(cone, &neg(&c));
                    if !feasible(&a) || !feasible(&b) {
                        continue;
                    }
                    self.wins_split(held, &a, rc - 1, ra) && self.wins_split(held, &b, rc - 1, ra)
                }
            };
            if self.timed_out {
                return false;
            }
            if w {
                self.split_tt.insert(key, true);
                return true;
            }
        }
        self.split_tt.insert(key, false);
        false
    }

    /// The least comparison budget that wins from `held` with at most `ra`
    /// additions; `None` on timeout or above `cap`.
    pub fn least_comparisons(
        &mut self,
        held: &[Form],
        ra: u8,
        cap: u8,
        budget: Duration,
    ) -> Option<u8> {
        self.deadline = Instant::now() + budget;
        self.timed_out = false;
        let root = self.root.clone();
        for rc in 0..=cap {
            if self.wins_split(held, &root, rc, ra) {
                return Some(rc);
            }
            if self.timed_out {
                return None;
            }
        }
        None
    }

    /// Prints one winning strategy within the split budget `(rc, ra)`.
    pub fn strategy_split(&mut self, held: &[Form], rc: u8, ra: u8) {
        self.lift_deadline();
        let root = self.root.clone();
        self.strategy_split_at(held, &root, rc, ra, 1);
    }

    fn strategy_split_at(&mut self, held: &[Form], cone: &Cone, rc: u8, ra: u8, depth: usize) {
        let pad = "  ".repeat(depth);
        if self.unready(held, cone).is_empty() {
            let parts: Vec<String> = (0..self.labels)
                .map(|i| {
                    let z = self.ready_form(held, cone, i).unwrap();
                    format!("d({}) = {}", i + 1, self.top.fmt_form(z))
                })
                .collect();
            println!("{pad}ready: {}", parts.join("; "));
            return;
        }
        for mv in self.moves(held, cone) {
            match mv {
                Move::Add { x, y, z } => {
                    if ra == 0 {
                        continue;
                    }
                    let mut h2 = held.to_vec();
                    let pos = h2.partition_point(|&v| v < z);
                    h2.insert(pos, z);
                    if self.wins_split(&h2, cone, rc, ra - 1) {
                        println!(
                            "{pad}add {} + {}",
                            self.top.fmt_form(x),
                            self.top.fmt_form(y)
                        );
                        self.strategy_split_at(&h2, cone, rc, ra - 1, depth);
                        return;
                    }
                }
                Move::Cmp { x, y, c, flip } => {
                    if rc == 0 {
                        continue;
                    }
                    let a = self.child_cone(cone, &c);
                    let b = self.child_cone(cone, &neg(&c));
                    if !feasible(&a) || !feasible(&b) {
                        continue;
                    }
                    if self.wins_split(held, &a, rc - 1, ra)
                        && self.wins_split(held, &b, rc - 1, ra)
                    {
                        println!(
                            "{pad}compare {} ? {}",
                            self.top.fmt_form(x),
                            self.top.fmt_form(y)
                        );
                        for (k, s) in [(a, 1i16), (b, -1i16)] {
                            let sym = if s * flip > 0 { ">" } else { "<" };
                            println!("{pad}  {sym}:");
                            self.strategy_split_at(held, &k, rc - 1, ra, depth + 2);
                        }
                        return;
                    }
                }
            }
        }
        panic!("no winning move at a winning state");
    }

    /// Each cell of the root cone as a region: its strict-minimum rows, with
    /// the shortest path of every label written out.
    pub fn cell_regions(&self) -> Vec<(String, Vec<Cvec>)> {
        self.root
            .cells
            .iter()
            .map(|cell| {
                let mut rows = Vec::new();
                for label in 0..self.labels {
                    self.strict_rows(label, cell[label] as usize, &mut rows);
                }
                let names: Vec<String> = (0..self.labels)
                    .map(|i| self.top.fmt_form(self.paths[i][cell[i] as usize]))
                    .collect();
                (names.join(" ; "), rows)
            })
            .collect()
    }

    /// The number of cells of the root cone.
    pub fn root_cells(&self) -> usize {
        self.root.cells.len()
    }

    pub fn cones_built(&self) -> u64 {
        self.cones_built
    }

    /// States the transposition table holds now; fewer than were visited once
    /// a generation has been dropped.
    pub fn states(&self) -> usize {
        self.tt.len()
    }

    /// Generations dropped by all caches so far; zero means nothing was
    /// recomputed and every count matches an unbounded run.
    pub fn generations_dropped(&self) -> u64 {
        self.cones.rotations
            + self.tt.rotations
            + self.split_tt.rotations
            + self.chain_memo.rotations
    }

    /// Entries held by the caches (cones, transposition table, split table,
    /// chain memo) and how many generations they dropped in all.
    pub fn cache_sizes(&self) -> (usize, usize, usize, usize, u64) {
        (
            self.cones.len(),
            self.tt.len(),
            self.split_tt.len(),
            self.chain_memo.len(),
            self.generations_dropped(),
        )
    }
}
