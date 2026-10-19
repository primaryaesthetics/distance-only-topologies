# Experiments

One directory per recorded run: `<date>-<slug>/`, containing `manifest.json`,
`stdout.txt` and `stderr.txt`. Produced by `scripts/record_run.py`, which fills
in the git sha, the host and CPU, the toolchain versions, the seed and the
checksum of any input file named on the command line.

A recorded run is never edited. To correct one, record a new run and mark the
old claim superseded in [../docs/ledger/CLAIMS.md](../docs/ledger/CLAIMS.md).

A number that did not come through here cannot be cited as evidence.
