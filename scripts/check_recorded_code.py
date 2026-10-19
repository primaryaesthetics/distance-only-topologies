#!/usr/bin/env python3
"""Recorded-code gate.

A manifest written by `scripts/record_run.py` names the commit its run started
from (`git_sha`) and the sha256 of every code file the run executed
(`code_sha256`). The commits are those of the development history in which
the runs were recorded; that history is not part of this repository, and the
`git_sha` field is kept as it was written. What this gate checks is what can
be checked from the files here: whether each recorded hash matches the file
of the same path in this tree.

Each code file of each run is put in one class:

  identical   the file here hashes to the recorded value
  crlf        as identical, but the run's checkout held the file with CRLF
              endings, so the bytes it executed differed by line endings only
  outside     a Rust run pins every source file of the workspace it was
              built in; this file belongs to a crate of that workspace that
              is not published here and that the published crates do not
              depend on (they have no dependencies)
  archived    the file here differs, and docs/ledger/code/<sha256> holds the
              exact bytes the run executed
  changed     the file here differs from the recorded value: the code was
              edited after the run, and the version that ran is the one at
              the manifest's commit of the development history
  missing    a file the run executed that is neither here nor archived

A run takes the worst class of its files, in the order above. Manifests
without `code_sha256` predate the field and are counted as `legacy`.

The gate fails when a run cited from the claim ledger has a `missing` file,
or names a program (a Python script, or the crate of a Rust binary) that is
not in this tree.

    python scripts/check_recorded_code.py            # summary
    python scripts/check_recorded_code.py --verbose  # every run and file not identical
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_run import is_workspace_source, normalised, runs_rust, sha256  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "ledger" / "CLAIMS.md"
ARCHIVE = ROOT / "docs" / "ledger" / "code"
CITED = re.compile(r"experiments/([\w.\-]+)")
CLASSES = ("identical", "crlf", "outside", "archived", "changed", "missing")


def classify(manifest: dict, path: str, recorded: str) -> str:
    here = ROOT / path
    if here.is_file() and sha256(normalised(here.read_bytes())) == recorded:
        return "crlf" if path in (manifest.get("code_crlf") or []) else "identical"
    kept = ARCHIVE / recorded
    if kept.is_file() and sha256(normalised(kept.read_bytes())) == recorded:
        return "archived"
    if here.is_file():
        return "changed"
    if runs_rust(manifest.get("command") or []) and is_workspace_source(path):
        return "outside"
    return "missing"


def programs_absent(command: list[str]) -> list[str]:
    """Programs a command names that this tree does not hold."""
    absent = []
    for argument in command:
        if argument.endswith(".py") and not (ROOT / argument).is_file():
            absent.append(argument)
    if runs_rust(command):
        program = command[0].replace("\\", "/").rsplit("/", 1)[-1]
        if program in ("cargo", "cargo.exe"):
            crate = next((command[i + 1] for i, a in enumerate(command[:-1]) if a in ("-p", "--package")), None)
        else:
            crate = program.removesuffix(".exe")
        if crate and not (ROOT / "crates" / crate / "Cargo.toml").is_file():
            absent.append(f"crates/{crate}")
    return absent


def survey() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]], dict[str, list[str]]]:
    runs: dict[str, str] = {}
    details: dict[str, list[tuple[str, str]]] = {}
    absent: dict[str, list[str]] = {}
    for manifest_path in sorted((ROOT / "experiments").glob("*/manifest.json")):
        run = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if gone := programs_absent(manifest.get("command") or []):
            absent[run] = gone
        hashes = manifest.get("code_sha256")
        if not isinstance(hashes, dict):
            runs[run] = "legacy"
            continue
        files = [(classify(manifest, path, value), path) for path, value in sorted(hashes.items())]
        worst = max((CLASSES.index(status) for status, _ in files), default=0)
        runs[run] = CLASSES[worst]
        details[run] = [(status, path) for status, path in files if status != "identical"]
    return runs, details, absent


def cited_runs() -> set[str]:
    if not LEDGER.exists():
        return set()
    return set(CITED.findall(LEDGER.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", action="store_true",
                        help="list every run and file that is not identical")
    args = parser.parse_args()

    runs, details, absent = survey()
    cited = cited_runs()
    failures = [f"{run}: a code file it executed is missing" for run, status in runs.items()
                if status == "missing" and run in cited]
    failures += [f"{run}: names {', '.join(gone)}, which is not in this tree"
                 for run, gone in absent.items() if run in cited]

    if args.verbose:
        for run, status in runs.items():
            if status in ("identical", "legacy"):
                continue
            print(f"{status:<10} {run}")
            counts = Counter(file_status for file_status, _ in details.get(run, []))
            for file_status, path in details.get(run, []):
                if file_status in ("changed", "archived", "missing"):
                    print(f"           {file_status:<10} {path}")
            for file_status in ("crlf", "outside"):
                if counts[file_status]:
                    print(f"           {file_status:<10} {counts[file_status]} file(s)")

    counts = Counter(runs.values())
    summary = ", ".join(f"{counts[name]} {name}" for name in (*CLASSES, "legacy"))
    print(f"recorded code: {len(runs)} runs: {summary}")
    if failures:
        for failure in failures:
            print(f"recorded code: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
