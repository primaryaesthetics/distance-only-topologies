//! Rooted topologies: parsing, enumeration up to isomorphism, simple paths,
//! automorphisms, and the two closed forms the tables are read against.

use crate::lp::MAXM;

/// A linear form with non-negative integer coefficients, packed eight bits per
/// arc. Arc `i` occupies bits `8i..8i+8`, and every coefficient stays below
/// 128 so that a sum of two forms never carries across a lane.
pub type Form = u128;

pub const LANE_CAP: u128 = 127;

#[inline]
pub fn lane(z: Form, i: usize) -> u128 {
    (z >> (8 * i)) & 0xff
}

#[inline]
pub fn unit(i: usize) -> Form {
    1u128 << (8 * i)
}

/// `x + y`, or `None` if some coefficient would exceed the lane cap.
#[inline]
pub fn add(x: Form, y: Form, m: usize) -> Option<Form> {
    let z = x + y;
    for i in 0..m {
        if lane(z, i) > LANE_CAP {
            return None;
        }
    }
    Some(z)
}

/// Whether `z <= t` coefficientwise.
#[inline]
pub fn below(z: Form, t: Form, m: usize) -> bool {
    for i in 0..m {
        if lane(z, i) > lane(t, i) {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub n: usize,
    pub arcs: Vec<(u8, u8)>,
}

impl Topology {
    pub fn m(&self) -> usize {
        self.arcs.len()
    }

    /// Parses `0>1,0>2,1>2`; the source is vertex 0.
    pub fn parse(text: &str) -> Topology {
        let mut arcs = Vec::new();
        let mut n = 0;
        for tok in text.split(',') {
            let (u, v) = tok.split_once('>').expect("arc as u>v");
            let u: u8 = u.trim().parse().expect("vertex");
            let v: u8 = v.trim().parse().expect("vertex");
            n = n.max(u as usize + 1).max(v as usize + 1);
            arcs.push((u, v));
        }
        assert!(arcs.len() <= MAXM, "at most {MAXM} arcs");
        Topology { n, arcs }
    }

    pub fn is_dag(&self) -> bool {
        let mut indeg = vec![0usize; self.n];
        for &(_, v) in &self.arcs {
            indeg[v as usize] += 1;
        }
        let mut stack: Vec<usize> = (0..self.n).filter(|&v| indeg[v] == 0).collect();
        let mut seen = 0;
        while let Some(u) = stack.pop() {
            seen += 1;
            for &(a, b) in &self.arcs {
                if a as usize == u {
                    indeg[b as usize] -= 1;
                    if indeg[b as usize] == 0 {
                        stack.push(b as usize);
                    }
                }
            }
        }
        seen == self.n
    }

    pub fn source_out(&self) -> usize {
        self.arcs.iter().filter(|&&(u, _)| u == 0).count()
    }

    pub fn reachable(&self) -> bool {
        let mut seen = vec![false; self.n];
        seen[0] = true;
        let mut stack = vec![0usize];
        let mut count = 1;
        while let Some(u) = stack.pop() {
            for &(a, b) in &self.arcs {
                if a as usize == u && !seen[b as usize] {
                    seen[b as usize] = true;
                    count += 1;
                    stack.push(b as usize);
                }
            }
        }
        count == self.n
    }

    /// Every simple source-to-`v` path as a 0/1 form, indexed by label
    /// `v - 1`. A shortest walk under non-negative weights has a simple
    /// representative, so these are the only forms a distance can take.
    pub fn paths(&self) -> Vec<Vec<Form>> {
        let mut out: Vec<Vec<Form>> = vec![Vec::new(); self.n - 1];
        let mut onpath = vec![false; self.n];
        onpath[0] = true;
        self.walk(0, 0, &mut onpath, &mut out);
        out
    }

    fn walk(&self, u: usize, used: Form, onpath: &mut Vec<bool>, out: &mut Vec<Vec<Form>>) {
        for (i, &(a, b)) in self.arcs.iter().enumerate() {
            if a as usize != u || onpath[b as usize] {
                continue;
            }
            let vec = used | unit(i);
            out[b as usize - 1].push(vec);
            onpath[b as usize] = true;
            self.walk(b as usize, vec, onpath, out);
            onpath[b as usize] = false;
        }
    }

    /// `fwd(G, s)`: the largest number of forward non-source arcs over rooted
    /// vertex orders (Cai, Theorem 3.1).
    pub fn fwd(&self) -> usize {
        let mut best = 0;
        let mut order: Vec<usize> = (1..self.n).collect();
        permute(&mut order, 0, &mut |ord| {
            let mut pos = vec![0usize; self.n];
            for (i, &v) in ord.iter().enumerate() {
                pos[v] = i + 1;
            }
            let rooted = ord.iter().all(|&v| {
                self.arcs
                    .iter()
                    .any(|&(u, w)| w as usize == v && pos[u as usize] < pos[v])
            });
            if !rooted {
                return;
            }
            let count = self
                .arcs
                .iter()
                .filter(|&&(u, v)| u != 0 && pos[u as usize] < pos[v as usize])
                .count();
            best = best.max(count);
        });
        best
    }

    /// Theorem 4.1's corner for a simple DAG, `(m - n + 1) + (m - d_s)`.
    pub fn dag_law(&self) -> usize {
        (self.m() + 1 - self.n) + (self.m() - self.source_out())
    }

    /// `fwd + (m - n + 1)`, the corner every topology is measured against.
    pub fn corner(&self) -> usize {
        self.fwd() + (self.m() + 1 - self.n)
    }

    /// Every relabelling of the non-source vertices that maps the arc set onto
    /// itself, as the permutation it induces on arc indices; identity first.
    pub fn automorphisms(&self) -> Vec<Vec<usize>> {
        let m = self.m();
        let mut out = Vec::new();
        let mut order: Vec<usize> = (1..self.n).collect();
        permute(&mut order, 0, &mut |perm| {
            let mut relabel = vec![0usize; self.n];
            for (i, &p) in perm.iter().enumerate() {
                relabel[i + 1] = p;
            }
            // Parallel arcs share endpoints, so each arc takes the first index
            // with the right endpoints that no earlier arc has taken; the
            // image is then a bijection, and the identity maps to itself.
            let mut image = Vec::with_capacity(m);
            let mut taken = vec![false; m];
            for &(u, v) in &self.arcs {
                let arc = (relabel[u as usize] as u8, relabel[v as usize] as u8);
                match (0..m).find(|&i| !taken[i] && self.arcs[i] == arc) {
                    Some(i) => {
                        taken[i] = true;
                        image.push(i);
                    }
                    None => return,
                }
            }
            out.push(image);
        });
        let id: Vec<usize> = (0..m).collect();
        out.sort_by_key(|s| *s != id);
        out
    }

    pub fn fmt_form(&self, z: Form) -> String {
        if z == 0 {
            return "0".to_string();
        }
        let mut parts = Vec::new();
        for (i, &(u, v)) in self.arcs.iter().enumerate() {
            let c = lane(z, i);
            if c != 0 {
                let coef = if c == 1 {
                    String::new()
                } else {
                    format!("{c}*")
                };
                parts.push(format!("{coef}w({u}>{v})"));
            }
        }
        parts.join(" + ")
    }
}

impl std::fmt::Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.arcs.iter().map(|(u, v)| format!("{u}>{v}")).collect();
        write!(f, "{}", parts.join(","))
    }
}

/// Visits every permutation of `items` in lexicographic order of positions.
fn permute(items: &mut Vec<usize>, k: usize, visit: &mut dyn FnMut(&[usize])) {
    if k == items.len() {
        visit(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// Every reachable simple rooted digraph on `n` vertices with at most
/// `max_arcs` arcs, one representative per isomorphism class, in the same
/// order and with the same canonical arc list as `theory/game/opt_dist.py`,
/// so that positions in the two tables agree. Arcs into the source and
/// self-loops are excluded.
pub fn enumerate_topologies(n: usize, max_arcs: usize) -> Vec<Topology> {
    let mut slots = Vec::new();
    for u in 0..n {
        for v in 1..n {
            if u != v {
                slots.push((u as u8, v as u8));
            }
        }
    }
    let mut perms: Vec<Vec<usize>> = Vec::new();
    let mut order: Vec<usize> = (1..n).collect();
    permute_lex(&mut order, &mut perms);
    let mut seen: std::collections::HashSet<Vec<(u8, u8)>> = Default::default();
    let mut out = Vec::new();
    for k in (n - 1)..=max_arcs {
        for chosen in combinations(&slots, k) {
            let top = Topology {
                n,
                arcs: chosen.clone(),
            };
            if !top.reachable() {
                continue;
            }
            let mut canon: Option<Vec<(u8, u8)>> = None;
            for perm in &perms {
                let mut relabel = vec![0usize; n];
                for (i, &p) in perm.iter().enumerate() {
                    relabel[i + 1] = p;
                }
                let mut cand: Vec<(u8, u8)> = chosen
                    .iter()
                    .map(|&(u, v)| (relabel[u as usize] as u8, relabel[v as usize] as u8))
                    .collect();
                cand.sort();
                if canon.as_ref().is_none_or(|c| cand < *c) {
                    canon = Some(cand);
                }
            }
            let canon = canon.unwrap();
            if seen.insert(canon.clone()) {
                out.push(Topology { n, arcs: canon });
            }
        }
    }
    out
}

/// Permutations in the order Python's `itertools.permutations` yields them.
fn permute_lex(items: &mut [usize], out: &mut Vec<Vec<usize>>) {
    fn rec(pool: &[usize], prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if pool.is_empty() {
            out.push(prefix.clone());
            return;
        }
        for i in 0..pool.len() {
            let mut rest = pool.to_vec();
            let x = rest.remove(i);
            prefix.push(x);
            rec(&rest, prefix, out);
            prefix.pop();
        }
    }
    rec(items, &mut Vec::new(), out);
}

/// Combinations in the order Python's `itertools.combinations` yields them.
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    if k > items.len() {
        return out;
    }
    loop {
        out.push(idx.iter().map(|&i| items[i].clone()).collect());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + items.len() - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_the_python_tables() {
        assert_eq!(enumerate_topologies(3, 4).len(), 5);
        assert_eq!(enumerate_topologies(4, 5).len(), 34);
        assert_eq!(enumerate_topologies(4, 6).len(), 49);
    }

    #[test]
    fn h2_fwd_and_corner() {
        let h2 = Topology::parse("0>1,0>2,0>3,1>2,1>3,2>1,3>1");
        assert_eq!(h2.fwd(), 2);
        assert_eq!(h2.corner(), 6);
        assert_eq!(h2.automorphisms().len(), 2);
    }

    #[test]
    fn parallel_arcs_give_bijective_automorphisms() {
        let t = Topology::parse("0>1,0>2,1>2,1>2");
        for sigma in t.automorphisms() {
            let mut seen = sigma.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..t.m()).collect::<Vec<_>>());
        }
        assert_eq!(t.automorphisms()[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn dag_law_on_a_diamond() {
        let t = Topology::parse("0>1,0>2,1>2");
        assert!(t.is_dag());
        assert_eq!(t.dag_law(), 2);
        assert_eq!(t.paths()[1].len(), 2);
    }
}
