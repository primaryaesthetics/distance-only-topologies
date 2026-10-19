#!/usr/bin/env python3
"""Fetches the real-graph corpus described by data/manifests/*.json.

Graph data is never committed. This script reproduces the local copy and
verifies it against the recorded checksum, so a benchmark input is
reconstructible from the repository alone.

    python scripts/fetch_corpus.py            # everything not yet present
    python scripts/fetch_corpus.py dimacs-ny  # one entry
    python scripts/fetch_corpus.py --verify   # check what is already there
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "data" / "manifests"
TARGET = ROOT / "data" / "graphs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry: dict, verify_only: bool) -> bool:
    name = entry["name"]
    destination = TARGET / entry["file"]
    expected = entry.get("sha256")

    if destination.exists():
        actual = sha256(destination)
        if expected and actual != expected:
            print(f"{name}: checksum mismatch\n  expected {expected}\n  actual   {actual}",
                  file=sys.stderr)
            return False
        if not expected:
            print(f"{name}: present, no checksum recorded yet; actual {actual}")
            return True
        print(f"{name}: present, checksum ok")
        return True

    if verify_only:
        print(f"{name}: absent")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = entry["url"]
    print(f"{name}: fetching {url}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if entry.get("gzipped"):
        with gzip.open(temporary, "rb") as source, destination.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        temporary.unlink()
    else:
        temporary.replace(destination)

    actual = sha256(destination)
    if expected and actual != expected:
        print(f"{name}: checksum mismatch after download\n  expected {expected}\n  actual   {actual}",
              file=sys.stderr)
        return False
    if not expected:
        print(f"{name}: fetched; record sha256 {actual} in the manifest")
    else:
        print(f"{name}: fetched, checksum ok")
    return True


def main(argv: list[str]) -> int:
    verify_only = "--verify" in argv
    wanted = [a for a in argv if not a.startswith("--")]

    entries = []
    for path in sorted(MANIFESTS.glob("*.json")):
        entries.extend(json.loads(path.read_text(encoding="utf-8"))["graphs"])

    if wanted:
        entries = [e for e in entries if e["name"] in wanted]
        if not entries:
            print(f"no manifest entry matches {wanted}", file=sys.stderr)
            return 1

    ok = all(fetch(entry, verify_only) for entry in entries)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
