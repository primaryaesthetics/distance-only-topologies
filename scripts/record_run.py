#!/usr/bin/env python3
"""Records a benchmark run as evidence.

Creates experiments/<date>-<slug>/ containing manifest.json and the raw output
of the command. A timing number that did not come through here is not evidence
and cannot be cited from the claim ledger.

    python scripts/record_run.py dijkstra-ny -- \\
        cargo run -q -p unsorted-bench --release -- --graph data/USA-road-d.NY.gr

Besides the commit and the inputs, the manifest pins the code the command
executes, under `code_sha256` as a map from repository path to sha256:

  * a Python command: each script named on the command line and, transitively,
    each module of this repository it imports from its own directory;
  * a Rust command (`cargo`, or a binary under `target/`): every source file
    under `crates/*/src/`, every `Cargo.toml` and `Cargo.lock`, since a binary
    is built from the whole workspace and not from one file of it.

Text files are hashed with CRLF line endings normalised to LF, so that the
value verifies against the committed blob on any checkout. `code_crlf` lists
the files whose bytes on disk had CRLF endings and therefore differed from
what was hashed, and `code_digest` is one sha256 over the whole map.
`scripts/check_recorded_code.py` checks these against the record.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        elif system == "Windows":
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def working_tree_is_dirty() -> bool:
    """Whether the tree that produced this run differs from its commit.

    Runs recorded earlier the same day sit in experiments/ untracked, and they
    are output rather than input: they say nothing about whether this run can be
    reproduced. A tracked file under experiments/ that has been *modified* is a
    different matter, since a recorded run is never edited, and it counts.
    """
    for line in git("status", "--porcelain").splitlines():
        status, path = line[:2], line[3:].strip().strip('"')
        if status == "??" and path.startswith("experiments/"):
            continue
        return True
    return False


def corpus_hash(command: list[str]) -> str | None:
    """Hashes any argument that names an existing file, so the input is pinned."""
    digest = hashlib.sha256()
    seen = False
    for argument in command:
        path = Path(argument)
        try:
            is_file = path.is_file()
        except OSError:
            # An argument too long to be a file name, such as a list of arcs.
            is_file = False
        if is_file:
            digest.update(path.read_bytes())
            seen = True
    return digest.hexdigest() if seen else None


def normalised(data: bytes) -> bytes:
    """A text file's bytes with CRLF endings as LF; a binary file's unchanged.
    A file is binary when its first 8 KiB hold a NUL byte, as git decides."""
    return data if b"\0" in data[:8192] else data.replace(b"\r\n", b"\n")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_digest(hashes: dict[str, str]) -> str:
    """One sha256 over a code map, independent of the order it was built in."""
    return sha256("".join(f"{path}\0{value}\n" for path, value in sorted(hashes.items())).encode())


def is_workspace_source(path: str) -> bool:
    """Whether a repository path is part of what a Rust binary is built from."""
    name = path.rsplit("/", 1)[-1]
    if name == "Cargo.toml" or path == "Cargo.lock":
        return True
    parts = path.split("/")
    return parts[0] == "crates" and path.endswith(".rs") and "src" in parts[2:-1]


def runs_rust(command: list[str]) -> bool:
    program = command[0].replace("\\", "/") if command else ""
    name = program.rsplit("/", 1)[-1]
    return name in ("cargo", "cargo.exe") or "target/" in program


def python_entries(command: list[str], root: Path) -> list[Path]:
    """The scripts of this repository a command names."""
    found = []
    for argument in command:
        if not argument.endswith(".py"):
            continue
        path = (root / argument).resolve()
        if path.is_file() and path.is_relative_to(root.resolve()):
            found.append(path)
    return found


def python_code(command: list[str], root: Path) -> list[Path]:
    """Each named script and, transitively, each module it imports that lives
    in its own directory, which is where Python looks for it first."""
    seen: set[Path] = set()
    queue = python_entries(command, root)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_bytes(), filename=str(path))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.append(node.module)
        for name in names:
            stem = path.parent.joinpath(*name.split("."))
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                if candidate.is_file() and candidate not in seen:
                    queue.append(candidate)
    return sorted(seen)


def code_paths(command: list[str], root: Path = ROOT) -> list[str]:
    """Repository paths of the code a command executes."""
    paths = {path.relative_to(root.resolve()).as_posix() for path in python_code(command, root)}
    if runs_rust(command):
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=root, check=True,
        ).stdout
        paths.update(
            name for name in listed.split("\0")
            if name and is_workspace_source(name) and (root / name).is_file()
        )
    return sorted(paths)


def code_record(command: list[str], root: Path = ROOT) -> dict[str, object]:
    """The manifest fields that pin a command's code."""
    hashes: dict[str, str] = {}
    crlf: list[str] = []
    for path in code_paths(command, root):
        raw = (root / path).read_bytes()
        text = normalised(raw)
        hashes[path] = sha256(text)
        if text != raw:
            crlf.append(path)
    return {
        "code_sha256": hashes,
        "code_digest": code_digest(hashes),
        "code_crlf": crlf,
    }


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    split = argv.index("--")
    slug, command = argv[0], argv[split + 1:]
    if not slug or not command:
        print(__doc__, file=sys.stderr)
        return 2

    # Sampled before the run directory exists, since creating it would itself
    # make the tree dirty and every run would then disown its own evidence,
    # and before the command runs, since the commit and the code a run is
    # pinned to are the ones it started from.
    dirty = working_tree_is_dirty()
    sha = git("rev-parse", "HEAD")
    code = code_record(command)

    date = dt.date.today().isoformat()
    directory = ROOT / "experiments" / f"{date}-{slug}"
    if directory.exists():
        print(f"{directory} exists; a recorded run is never overwritten", file=sys.stderr)
        return 1
    directory.mkdir(parents=True)

    started = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    finished = dt.datetime.now(dt.timezone.utc)

    (directory / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    manifest = {
        "slug": slug,
        "command": command,
        "exit_code": completed.returncode,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "rustc": subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "corpus_sha256": corpus_hash(command),
        **code,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print(f"\nrecorded: {directory.relative_to(ROOT).as_posix()}")
    if manifest["git_dirty"]:
        print("warning: working tree was dirty; this run is not reproducible", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
