# distance-only-topologies

Code, recorded runs and claim ledger for the paper

> **Exact Distance-Only Complexity of Small Topologies**
> Nikolai Khobotov ([ORCID 0009-0005-3612-830X](https://orcid.org/0009-0005-3612-830X))

## What the paper is about

Cai (arXiv:2609.04825) defines `OPT_DIST(G, s)`: the least worst-case number
of additions and comparisons that a deterministic program, specialised to one
rooted directed topology, needs in order to hold every shortest-path distance,
with no order of the vertices required. It is the value of a finite game, and
a uniform `O(m + n)` algorithm for directed shortest paths would make it
linear on every topology.

The paper plays that game exactly on small topologies: every rooted topology
with at most four vertices, every one with five vertices and at most six
arcs, and the 319 with five vertices and seven arcs. It determines the fed
cycle `C_k` for `k <= 5`, proves `OPT_DIST(C_k) >= 2k + 2` for all `k >= 5`,
gives the exact cost `k + 2f - 3` of the cycle fed at `f <= 5` of its
vertices, and shows that with additions free `C_k` needs exactly
`k - 1 + ceil(log2 k)` comparisons of sub-path sums, against `2k - 2` for
Dijkstra. It then measures how close a uniform program of local rules comes
to these values, and how that program behaves at scale.

## Layout

| Path | Contents |
| --- | --- |
| `crates/unsorted-game` | Rust player of the two strict games and the two free-addition games |
| `crates/unsorted-fedcycle` | exact player of the path-register game on `C_k` against restricted adversaries |
| `theory/game/` | Python reference players (`opt_dist.py`, `fed_cycle_check.py`, with `z3`) and the rule-program evaluator `rule_programs.py` |
| `ref/rule_programs/scale_probe.py` | the four-rule program on concrete weights, checked against a binary-heap Dijkstra |
| `scripts/` | the lemma and certificate checkers named in the paper, the plotting and census scripts of the recorded runs, and the two repository gates |
| `data/certificates/` | the `0/1` point sets of `C_5` and `C_6` |
| `experiments/` | 299 recorded runs, one directory each: `manifest.json`, `stdout.txt`, `stderr.txt`, and any figure the run wrote |
| `docs/ledger/CLAIMS.md` | 221 claims, each pointing to the run that records it |
| `docs/rules-supplement.pdf` | the supplement on the further local rules, with a map from the paper's rule and program names to those of `rule_programs.py` and the ledger |

## Building the Rust players

Rust 1.85 or later. The two crates have no dependencies.

```sh
cargo build --release
```

This produces `target/release/unsorted-game` and `target/release/unsorted-fedcycle`.

## Python

Python 3.12. `z3-solver` is needed by `theory/game/opt_dist.py`,
`theory/game/fed_cycle_check.py` and `theory/game/rule_programs.py`;
`matplotlib` and `numpy` only by the plotting scripts. The scripts under
`scripts/` that check lemmas and certificates use the standard library only.

```sh
python -m pip install -r requirements.txt
```

The scale probe on a road network reads `data/graphs/USA-road-d.NY.gr`, which
is fetched rather than stored here:

```sh
python scripts/fetch_corpus.py dimacs-ny
```

## Representative commands

From the paper's appendix, run from the repository root:

```sh
target/release/unsorted-game validate
target/release/unsorted-game enumerate 5 6 600 --modes=path,generic
target/release/unsorted-game table '0>1,0>2,0>3,0>4,1>2,2>3,3>4,4>1' 3600 \
       --modes=path,generic
target/release/unsorted-game frontier '0>1,0>2,0>3,0>4,0>5,1>2,2>3,3>4,4>5,5>1' 6 \
       --mode=path
target/release/unsorted-game solve '0>1,0>2,0>3,0>4,0>5,1>2,2>3,3>4,4>5,5>1' \
       --mode=generic --from=11 --seconds=72000 --cache-gb=8
target/release/unsorted-fedcycle --k=2,3,4 --adversary=theorem-f
python scripts/fed_cycle_zero_one.py cert \
       data/certificates/fed_cycle_c5_zero_one.txt 7
python scripts/check_walk_program.py 8 200 1
target/release/unsorted-game certify '0>1,0>2,0>3,0>4,1>2,2>3,3>4,4>1' 8 --out=c4.txt
python scripts/check_refutation.py c4.txt \
       --expect-arcs '0>1,0>2,0>3,0>4,1>2,2>3,3>4,4>1' --expect-budget 8
python theory/game/rule_programs.py --n5 --charge-decided
python theory/game/rule_programs.py --bipath 3,4,5,6,7 --charge-decided
python theory/game/rule_programs.py --eight-rule --charge-decided \
       --bipath 3,4,5,6
```

## Checking the record

Every number in the paper names, in a comment of its source, the claim ID in
`docs/ledger/CLAIMS.md` that it comes from, and every claim points to the
directory of a recorded run under `experiments/`. Two gates check this:

```sh
python scripts/check_claims.py         # every claim resolves; every cited ID is defined
python scripts/check_recorded_code.py  # each run's recorded code against this tree
```

A run's `manifest.json` holds the exact command, the host, the toolchain, and
the sha256 of every code file the run executed. Its `git_sha` field is the
commit of the development history the run started from; that history is not
published, and this repository starts a fresh one. The code a run executed
is identified by the recorded sha256 values: where the file here still
matches, `check_recorded_code.py` reports it as identical, and where it was
edited after the run, as changed.

New runs are recorded with `scripts/record_run.py`.

## Citing

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`.
