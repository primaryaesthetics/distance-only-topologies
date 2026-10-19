#!/usr/bin/env python3
"""Claim gate.

Two directions, both of which have to hold:

  * every claim in docs/ledger/CLAIMS.md points at evidence that exists —
    an experiment directory with a manifest, or a Lean theorem in a file that
    contains it;
  * every claim id cited anywhere in the repository is defined in the ledger.

Run from the repository root. Exits non-zero on any failure.
"""

from __future__ import annotations

import re
import os
import subprocess
import sys
from pathlib import Path

LEDGER = Path("docs/ledger/CLAIMS.md")
CLAIM_ID = re.compile(r"\bC-\d{3,}\b")
LEAN_POINTER = re.compile(r"^(?P<file>theory/[\w/]+\.lean)#(?P<theorem>[\w.']+)$")
LEAN_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.DOTALL)
LEAN_LINE_COMMENT = re.compile(r"--.*?$", re.MULTILINE)
EXPERIMENT_POINTER = re.compile(r"^experiments/[\w.\-]+/?$")

TEXT_SUFFIXES = {".md", ".rs", ".py", ".lean", ".toml", ".yml", ".yaml", ".cff"}


IGNORED_DIRS = {".git", "target", ".lake", "__pycache__", ".venv", ".pytest_cache"}
IGNORED_PREFIXES = ("data/graphs/",)


def walked_files() -> list[Path]:
    """Every file under the current directory except build output and fetched
    data, for a checkout that is not a git work tree (an unpacked archive)."""
    found = []
    for directory, subdirs, files in os.walk("."):
        subdirs[:] = [d for d in subdirs if d not in IGNORED_DIRS]
        for name in files:
            path = Path(directory, name).relative_to(".")
            if not path.as_posix().startswith(IGNORED_PREFIXES):
                found.append(path)
    return sorted(found)


def tracked_files() -> list[Path]:
    """Every file git would publish: tracked files, and untracked files that
    no exclude rule hides, so a file meets the gate before it is staged.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return walked_files()
    return [Path(name) for name in out.split("\0") if name]


def parse_ledger() -> tuple[dict[str, str], list[str]]:
    """Returns {claim id: evidence pointer} and a list of parse errors."""
    errors: list[str] = []
    claims: dict[str, str] = {}
    if not LEDGER.exists():
        return claims, [f"{LEDGER}: missing"]

    for number, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or not CLAIM_ID.fullmatch(cells[0]):
            continue
        if len(cells) < 4:
            errors.append(f"{LEDGER}:{number}: row has {len(cells)} columns, expected at least 4")
            continue
        claim_id, evidence = cells[0], cells[3]
        if claim_id in claims:
            errors.append(f"{LEDGER}:{number}: {claim_id} defined twice")
        claims[claim_id] = evidence
    return claims, errors


def strip_lean_comments(text: str) -> str:
    """Removes Lean comments, so that prose mentioning `sorry` is not mistaken
    for a `sorry` in a proof."""
    return LEAN_LINE_COMMENT.sub("", LEAN_BLOCK_COMMENT.sub("", text))


def check_evidence(claim_id: str, evidence: str) -> list[str]:
    evidence = evidence.strip("`")
    if match := LEAN_POINTER.match(evidence):
        path = Path(match["file"])
        if not path.exists():
            return [f"{claim_id}: {path} does not exist"]
        theorem = match["theorem"]
        code = strip_lean_comments(path.read_text(encoding="utf-8"))
        if not re.search(rf"\b(theorem|lemma|def|structure)\s+{re.escape(theorem)}\b", code):
            return [f"{claim_id}: {path} does not declare {theorem}"]
        if re.search(r"\bsorry\b", code):
            return [f"{claim_id}: {path} contains sorry"]
        return []
    if EXPERIMENT_POINTER.match(evidence):
        path = Path(evidence.rstrip("/"))
        if not path.is_dir():
            return [f"{claim_id}: {path} is not a directory"]
        if not (path / "manifest.json").exists():
            return [f"{claim_id}: {path}/manifest.json is missing"]
        return []
    return [f"{claim_id}: evidence {evidence!r} is neither an experiment directory nor a Lean pointer"]


def main() -> int:
    claims, errors = parse_ledger()

    for claim_id, evidence in sorted(claims.items()):
        errors.extend(check_evidence(claim_id, evidence))

    cited: dict[str, list[str]] = {}
    for path in tracked_files():
        if path.suffix not in TEXT_SUFFIXES or path == LEDGER:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for claim_id in CLAIM_ID.findall(line):
                cited.setdefault(claim_id, []).append(f"{path.as_posix()}:{number}")

    for claim_id, sites in sorted(cited.items()):
        if claim_id not in claims:
            for site in sites:
                errors.append(f"{site}: cites {claim_id}, which the ledger does not define")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"\nclaims: {len(errors)} failure(s)", file=sys.stderr)
        return 1

    print(f"claims: {len(claims)} defined, {len(cited)} cited, all resolving")
    return 0


if __name__ == "__main__":
    sys.exit(main())
