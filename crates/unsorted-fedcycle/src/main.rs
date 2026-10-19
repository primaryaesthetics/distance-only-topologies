//! The fed cycle `C_k` (source arcs `b_i` into every vertex `i`, cycle arcs
//! `c_i` from `i` to `i + 1` mod `k`) in the strict path-register game of Cai,
//! arXiv:2609.04825, played exactly against adversaries that may give only a
//! subset of the strict adversary's answers.
//!
//! Restricting the adversary can only lower the value, so the value against
//! each restricted adversary is a lower bound on the strict path-register
//! value of `C_k`; against `full` it is that value. Each adversary and the
//! leaf condition are described in `game.rs`.
//!
//!     unsorted-fedcycle --k=K[,K...] --adversary=A [options]
//!
//! Adversaries: `full`, `single` (keeps a single-entry cell), `min-entry`
//! (keeps the most single-entry cells), `points`, `lex`, `apex`, `thick`,
//! `theorem-f`, `eir` (Theorem F's interval rule carried to rows with two
//! source arcs: of the answers that keep an alive cell it gives the one with
//! the most alive survivors for which the row is useless, then the most alive
//! survivors, then the most alive cells, and on equal scores the one that
//! makes the later held form the larger; other rows at the apex of the lowest
//! alive cell; free after isolation).
//!
//! Options:
//!   --ties=free|fixed|interval   tie rule of `min-entry` (all three) and `thick` (`free`, `fixed`)
//!   --lex-kill=heavy|free        `lex`: two candidates compared against each other
//!   --lex-mixed=noncandidate|heavy  `lex`: a candidate compared with a non-candidate
//!   --points=N --regimes=MASK --seed=S --print-points   the point set of `points`
//!   --point-file=PATH            read the point set of `points` instead, one point
//!                                per line: the cell `s` (1-based), then `b_1..b_k`,
//!                                `c_1..c_k`; closed under rotation
//!   --no-rotation                with `--point-file`, keep the points as read
//!   --worlds=SPEC                `points`: the lexicographic worlds of
//!                                `theory/game/fed_cycle_worlds.py` (its draw of the
//!                                cycle arcs for k <= 4, a fixed one beyond), unrotated: `all`,
//!                                `twin`, `near`, `far`, or `near:2,1,3+far:2,3,1`;
//!                                `pairs` solves every one-near-one-far pair in turn
//!   --waste-check                `points`: keep the waste invariant where possible
//!                                and count the comparisons where it cannot be kept
//!   --symmetry                   key states up to rotation (rotation-invariant adversaries only)
//!   --out-edge-bound             prune with the out-edge bound (the leaf certificate)
//!   --split=I/N                  search only the root moves of part I of N
//!   --cache=N                    entries a memo holds before it is dropped
//!   --cap=R                      the largest budget tried
//!   --print                      print a winning strategy (one k only)
//!   --mutate=no-certificate|no-interval   inject a defect the search must notice
//!
//! Every line other than a printed strategy starts with `#`, so the output of
//! `--print` is a strategy file for `theory/game/fed_cycle_tree.py`.

mod game;
// The exact cone-feasibility test of the general game solver, shared.
#[allow(clippy::needless_range_loop)]
#[path = "../../unsorted-game/src/lp.rs"]
mod lp;

use std::time::Instant;

use game::{permutations, shapes, Adversary, Config, Game, LexKill, LexMixed, Mutation, Ties};

type Parsed = (Vec<usize>, Config, bool, Option<String>, Option<String>);

fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut ks = vec![];
    let mut cfg = Config::new(0, Adversary::Single);
    let mut adversary = None;
    let mut print_points = false;
    let mut point_file = None;
    let mut worlds = None;
    for a in args {
        let (flag, value) = a.split_once('=').unwrap_or((a.as_str(), ""));
        let bad = || format!("bad value in {a}");
        match flag {
            "--k" => {
                for v in value.split(',') {
                    ks.push(v.parse().map_err(|_| bad())?);
                }
            }
            "--adversary" => {
                adversary = Some(match value {
                    "full" => Adversary::Full,
                    "single" => Adversary::Single,
                    "min-entry" => Adversary::MinEntry,
                    "points" => Adversary::Points,
                    "lex" => Adversary::Lex,
                    "apex" => Adversary::Apex,
                    "thick" => Adversary::Thick,
                    "theorem-f" => Adversary::TheoremF,
                    "eir" => Adversary::Eir,
                    _ => return Err(bad()),
                })
            }
            "--ties" => {
                cfg.ties = match value {
                    "free" => Ties::Free,
                    "fixed" => Ties::Fixed,
                    "interval" => Ties::Interval,
                    _ => return Err(bad()),
                }
            }
            "--lex-kill" => {
                cfg.lex_kill = match value {
                    "heavy" => LexKill::Heavy,
                    "free" => LexKill::Free,
                    _ => return Err(bad()),
                }
            }
            "--lex-mixed" => {
                cfg.lex_mixed = match value {
                    "noncandidate" => LexMixed::NonCandidate,
                    "heavy" => LexMixed::Heavy,
                    _ => return Err(bad()),
                }
            }
            "--mutate" => {
                cfg.mutation = match value {
                    "no-certificate" => Mutation::NoCertificate,
                    "no-interval" => Mutation::NoInterval,
                    _ => return Err(bad()),
                }
            }
            "--points" => cfg.points = value.parse().map_err(|_| bad())?,
            "--regimes" => cfg.regimes = value.parse().map_err(|_| bad())?,
            "--seed" => cfg.seed = value.parse().map_err(|_| bad())?,
            "--cache" => cfg.cache_entries = value.parse().map_err(|_| bad())?,
            "--cap" => cfg.cap = value.parse().map_err(|_| bad())?,
            "--split" => {
                let (i, n) = value.split_once('/').ok_or_else(bad)?;
                cfg.split = (i.parse().map_err(|_| bad())?, n.parse().map_err(|_| bad())?);
            }
            "--symmetry" => cfg.symmetry = true,
            "--out-edge-bound" => cfg.out_edge_bound = true,
            "--print" => cfg.print = true,
            "--print-points" => print_points = true,
            "--point-file" => point_file = Some(value.to_string()),
            "--no-rotation" => cfg.rotate_points = false,
            "--worlds" => worlds = Some(value.to_string()),
            "--waste-check" => cfg.waste_check = true,
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    cfg.adversary = adversary.ok_or("--adversary is required")?;
    if ks.is_empty() {
        return Err("--k is required".into());
    }
    let rotation_invariant = match cfg.adversary {
        Adversary::Full | Adversary::Single | Adversary::Points => true,
        Adversary::MinEntry | Adversary::Thick => cfg.ties == Ties::Free,
        Adversary::Lex | Adversary::Apex | Adversary::TheoremF | Adversary::Eir => false,
    };
    if cfg.symmetry && !rotation_invariant {
        return Err("--symmetry needs a rotation-invariant adversary".into());
    }
    if cfg.ties != Ties::Free
        && !(cfg.adversary == Adversary::MinEntry
            || (cfg.adversary == Adversary::Thick && cfg.ties == Ties::Fixed))
    {
        return Err("this tie rule is not defined for this adversary".into());
    }
    if cfg.mutation != Mutation::None && cfg.adversary != Adversary::TheoremF {
        return Err("mutations apply to theorem-f".into());
    }
    if cfg.split.1 == 0 || cfg.split.0 >= cfg.split.1 {
        return Err("--split=I/N needs I < N".into());
    }
    if cfg.print && ks.len() > 1 {
        return Err("--print takes one k".into());
    }
    if cfg.adversary == Adversary::Full && ks.iter().any(|&k| k > 6) {
        return Err("full keeps every cell, at most 63, so k <= 6".into());
    }
    if point_file.is_some() && cfg.adversary != Adversary::Points {
        return Err("--point-file needs --adversary=points".into());
    }
    if !cfg.rotate_points && (point_file.is_none() || cfg.symmetry) {
        return Err("--no-rotation needs --point-file and no --symmetry".into());
    }
    if worlds.is_some()
        && (cfg.adversary != Adversary::Points || point_file.is_some() || cfg.symmetry)
    {
        return Err("--worlds needs --adversary=points, no --point-file and no --symmetry".into());
    }
    if let Some(spec) = &worlds {
        if spec != "pairs" {
            for &k in &ks {
                shapes(spec, k)?;
            }
        }
    }
    Ok((ks, cfg, print_points, point_file, worlds))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (ks, base, print_points, point_file, worlds) = match parse(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    println!("# {}", args.join(" "));
    if worlds.as_deref() == Some("pairs") {
        let show = |o: &[usize]| {
            o.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        for k in ks {
            let orders = permutations(&(1..k).collect::<Vec<usize>>());
            for n in &orders {
                for f in &orders {
                    let mut cfg = base.clone();
                    cfg.k = k;
                    let mut g = Game::new(cfg);
                    g.set_worlds(&[(vec![], n.clone()), (f.clone(), vec![])]);
                    let t0 = Instant::now();
                    let v = g.value();
                    println!(
                        "# k={} near:{} far:{} value {}  nodes {}  {:.1}s",
                        k,
                        show(n),
                        show(f),
                        v.map_or("none".to_string(), |v| v.to_string()),
                        g.nodes,
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
        }
        return;
    }
    for k in ks {
        let mut cfg = base.clone();
        cfg.k = k;
        let mut g = Game::new(cfg);
        if g.cfg.adversary == Adversary::Points {
            match (&point_file, &worlds) {
                (Some(path), _) => g.set_points(read_points(path, k)),
                (None, Some(spec)) => g.set_worlds(&shapes(spec, k).expect("checked")),
                (None, None) => g.make_points(),
            }
            println!("# k={k}: {} points", g.points.len());
            if print_points {
                for (s, w) in &g.points {
                    let ws: Vec<String> = w.iter().map(|x| x.to_string()).collect();
                    println!("# point E_{} {}", s + 1, ws.join(" "));
                }
            }
        }
        let t0 = Instant::now();
        let v = g.value();
        println!(
            "# k={} value {}  cap {}  forced loss {}  violations {}  nodes {}  lps {}  tt clears {}  {:.1}s",
            k,
            v.map_or("none".to_string(), |v| v.to_string()),
            g.cfg.cap,
            g.forced_loss,
            g.violations,
            g.nodes,
            g.lps,
            g.tt_clears,
            t0.elapsed().as_secs_f64()
        );
    }
}

/// The points of `path` for `C_k`; lines of another length and lines
/// starting with `#` are skipped.
fn read_points(path: &str, k: usize) -> Vec<(usize, Vec<i64>)> {
    let text = std::fs::read_to_string(path).expect("cannot read the point file");
    let mut out = vec![];
    for line in text.lines() {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let v: Vec<i64> = line
            .split_whitespace()
            .map(|t| t.parse().expect("a point file holds integers"))
            .collect();
        if v.len() != 2 * k + 1 {
            continue;
        }
        assert!(v[0] >= 1 && v[0] as usize <= k, "cell out of range");
        out.push((v[0] as usize - 1, v[1..].to_vec()));
    }
    assert!(!out.is_empty(), "no point for this k in the point file");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(k: usize, adversary: Adversary, edit: impl Fn(&mut Config)) -> Option<usize> {
        let mut cfg = Config::new(k, adversary);
        edit(&mut cfg);
        let mut g = Game::new(cfg);
        if adversary == Adversary::Points {
            g.make_points();
        }
        g.value()
    }

    #[test]
    fn two_cycle_costs_three_against_every_adversary() {
        for a in [
            Adversary::Full,
            Adversary::Single,
            Adversary::MinEntry,
            Adversary::Points,
            Adversary::Lex,
            Adversary::Apex,
            Adversary::Thick,
            Adversary::TheoremF,
            Adversary::Eir,
        ] {
            assert_eq!(value(2, a, |_| {}), Some(3), "{a:?}");
        }
    }

    #[test]
    fn three_cycle_costs_six() {
        for a in [
            Adversary::Single,
            Adversary::MinEntry,
            Adversary::TheoremF,
            Adversary::Apex,
            Adversary::Eir,
        ] {
            assert_eq!(value(3, a, |c| c.out_edge_bound = true), Some(6), "{a:?}");
        }
    }

    #[test]
    fn twin_worlds_carry_the_three_cycle() {
        let mut cfg = Config::new(3, Adversary::Points);
        cfg.out_edge_bound = true;
        let mut g = Game::new(cfg);
        g.set_worlds(&shapes("twin", 3).unwrap());
        assert_eq!(g.points.len(), 12);
        assert_eq!(g.value(), Some(6));
        let mut cfg = Config::new(3, Adversary::Points);
        cfg.out_edge_bound = true;
        let mut g = Game::new(cfg);
        g.set_worlds(&shapes("far:2,1", 3).unwrap());
        assert_eq!(g.value(), Some(5));
    }

    #[test]
    fn theorem_f_needs_its_certificate_and_its_interval_rule() {
        let no_cert = value(3, Adversary::TheoremF, |c| {
            c.mutation = Mutation::NoCertificate
        });
        let no_interval = value(3, Adversary::TheoremF, |c| {
            c.mutation = Mutation::NoInterval
        });
        assert!(no_cert < Some(6) && no_interval < Some(6));
    }
}
