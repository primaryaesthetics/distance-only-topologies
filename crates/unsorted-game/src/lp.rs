//! Exact feasibility of an open polyhedral cone in the positive orthant.
//!
//! The question is whether `{ w : w > 0, a·w > 0 for every row a }` is
//! nonempty, and if so, a rational point in it. Every inequality is strict and
//! homogeneous, so the set is nonempty exactly when `{ w >= 1, a·w >= 1 }` is,
//! and that is a linear program with no objective. It is solved by a phase-one
//! simplex in integer arithmetic: the tableau is kept as integers over one
//! common denominator, the determinant of the current basis, and each pivot is
//! an exact division. Nothing is rounded anywhere, so a verdict is a proof.
//!
//! The systems here have at most a dozen variables and a few dozen rows, with
//! entries that are differences of small coefficient vectors, so the
//! determinants stay far inside `i128`.

/// The largest number of arcs a topology may have. Coefficient vectors are
/// fixed-size arrays of this length.
pub const MAXM: usize = 16;

/// A linear form over the arc weights, as integer coefficients.
pub type Cvec = [i16; MAXM];

/// Decides `{ w > 0 : a·w > 0 for all a in rows }` and returns an integer
/// interior point if there is one. The point satisfies every row strictly,
/// which the caller may re-check at integer cost.
pub fn interior_point(m: usize, rows: &[Cvec]) -> Option<Vec<i128>> {
    let r = rows.len();
    if r == 0 {
        return Some(vec![1; m]);
    }
    // Substitute w = 1 + u with u >= 0: row a becomes a·u >= 1 - sum(a) = b.
    // Rows with b <= 0 are satisfied at u = 0 and get a basic slack; rows with
    // b > 0 get an artificial variable. Columns: u (m), slack (r), art (r), rhs.
    let cols = m + 2 * r + 1;
    let rhs = cols - 1;
    let mut t: Vec<Vec<i128>> = Vec::with_capacity(r + 1);
    let mut basic: Vec<usize> = Vec::with_capacity(r);
    let mut art_rows: Vec<usize> = Vec::new();
    for (i, a) in rows.iter().enumerate() {
        let b: i128 = 1 - a[..m].iter().map(|&x| x as i128).sum::<i128>();
        let mut row = vec![0i128; cols];
        if b > 0 {
            for j in 0..m {
                row[j] = a[j] as i128;
            }
            row[m + i] = -1;
            row[m + r + i] = 1;
            row[rhs] = b;
            basic.push(m + r + i);
            art_rows.push(i);
        } else {
            for j in 0..m {
                row[j] = -(a[j] as i128);
            }
            row[m + i] = 1;
            row[rhs] = -b;
            basic.push(m + i);
        }
        t.push(row);
    }
    if art_rows.is_empty() {
        return Some(vec![1; m]);
    }
    // Reduced costs for minimising the sum of the artificials.
    let mut obj = vec![0i128; cols];
    for &i in &art_rows {
        for j in 0..m {
            obj[j] -= t[i][j];
        }
        obj[m + i] += 1;
        obj[rhs] -= t[i][rhs];
    }
    t.push(obj);
    let objrow = r;
    let mut d: i128 = 1;
    let mut art_allowed = vec![true; r];
    loop {
        // Bland's rule: the lowest-indexed column with a negative reduced cost.
        let mut q = None;
        for j in 0..rhs {
            if j >= m + r && !art_allowed[j - m - r] {
                continue;
            }
            if t[objrow][j] < 0 {
                q = Some(j);
                break;
            }
        }
        let Some(q) = q else { break };
        // Ratio test on rows with a positive entry; ties to the lowest basic index.
        let mut p: Option<usize> = None;
        for i in 0..r {
            if t[i][q] <= 0 {
                continue;
            }
            match p {
                None => p = Some(i),
                Some(k) => {
                    let lhs = t[i][rhs].checked_mul(t[k][q]).expect("lp overflow");
                    let rhs_v = t[k][rhs].checked_mul(t[i][q]).expect("lp overflow");
                    if lhs < rhs_v || (lhs == rhs_v && basic[i] < basic[k]) {
                        p = Some(i);
                    }
                }
            }
        }
        let p = p.expect("phase one is bounded below");
        let piv = t[p][q];
        for i in 0..=r {
            if i == p {
                continue;
            }
            let f = t[i][q];
            for j in 0..cols {
                let v = t[i][j]
                    .checked_mul(piv)
                    .and_then(|x| x.checked_sub(f.checked_mul(t[p][j]).expect("lp overflow")))
                    .expect("lp overflow");
                debug_assert_eq!(v % d, 0);
                t[i][j] = v / d;
            }
        }
        if basic[p] >= m + r {
            art_allowed[basic[p] - m - r] = false;
        }
        basic[p] = q;
        d = piv;
        if d < 0 {
            d = -d;
            for row in t.iter_mut() {
                for v in row.iter_mut() {
                    *v = -*v;
                }
            }
        }
    }
    if t[objrow][rhs] != 0 {
        return None;
    }
    let mut w = vec![d; m];
    for i in 0..r {
        if basic[i] < m {
            w[basic[i]] += t[i][rhs];
        }
    }
    let g = w.iter().fold(0i128, |g, &x| gcd(g, x));
    if g > 1 {
        for x in w.iter_mut() {
            *x /= g;
        }
    }
    debug_assert!(rows.iter().all(|a| dot(a, &w, m) > 0));
    Some(w)
}

pub fn dot(a: &Cvec, w: &[i128], m: usize) -> i128 {
    let mut s = 0i128;
    for j in 0..m {
        s += a[j] as i128 * w[j];
    }
    s
}

pub fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(v: &[i16]) -> Cvec {
        let mut c = [0i16; MAXM];
        c[..v.len()].copy_from_slice(v);
        c
    }

    #[test]
    fn orthant_alone_is_feasible() {
        assert!(interior_point(3, &[]).is_some());
    }

    #[test]
    fn a_strict_cut_and_its_reverse() {
        let m = 2;
        let a = row(&[1, -1]);
        let b = row(&[-1, 1]);
        assert!(interior_point(m, &[a]).is_some());
        assert!(interior_point(m, &[b]).is_some());
        assert!(interior_point(m, &[a, b]).is_none());
    }

    #[test]
    fn a_cycle_of_strict_inequalities_is_infeasible() {
        let m = 3;
        let rows = [row(&[1, -1, 0]), row(&[0, 1, -1]), row(&[-1, 0, 1])];
        assert!(interior_point(m, &rows).is_none());
        assert!(interior_point(m, &rows[..2]).is_some());
    }

    #[test]
    fn negative_sum_rows_need_the_orthant() {
        // w0 - 2 w1 > 0 together with w1 - 2 w2 > 0 and w2 - 2 w0 > 0 is
        // impossible for positive weights.
        let m = 3;
        let rows = [row(&[1, -2, 0]), row(&[0, 1, -2]), row(&[-2, 0, 1])];
        assert!(interior_point(m, &rows).is_none());
        assert!(interior_point(m, &rows[..2]).is_some());
    }
}
