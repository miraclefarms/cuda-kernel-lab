#!/usr/bin/env python3
"""Confidentiality gate for this public repository.

The article repo (miraclefarms-content) runs its own check over its public paths.
This repo is outside that scan and is public, so it carries an equivalent gate.
The keyword list itself lives in tools/confidential-patterns.json, which is
gitignored — see tools/confidential-patterns.example.json for the shape.

  python3 tools/check_confidential.py            # scan tracked files
  python3 tools/check_confidential.py --staged   # commit gate
  python3 tools/check_confidential.py <path...>

Exit codes: 0 clean (warnings allowed), 1 at least one BLOCK.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

SCAN_SUFFIXES = {".md", ".markdown", ".txt", ".yaml", ".yml", ".json", ".csv",
                 ".py", ".sh", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp", ".cmake"}

# Pattern list is loaded from tools/confidential-patterns.json, which is NOT in
# version control. A published list of designators that must never be named is
# itself a disclosure of what confidential material exists, so the list stays
# local and only the example shape is committed.
PATTERNS_PATH = REPO / "tools" / "confidential-patterns.json"
EXAMPLE_PATH = REPO / "tools" / "confidential-patterns.example.json"

FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def load_patterns() -> tuple[list, list]:
    path = PATTERNS_PATH if PATTERNS_PATH.exists() else EXAMPLE_PATH
    if not PATTERNS_PATH.exists():
        print(f"[warn] {PATTERNS_PATH.name} missing — falling back to the example list, "
              f"which blocks nothing real. Copy the example and fill it in.")
    spec = json.loads(path.read_text(encoding="utf-8"))

    def compile_group(name: str) -> list:
        out = []
        for entry in spec.get(name, []):
            flags = 0
            for ch in entry.get("flags", ""):
                flags |= FLAGS.get(ch, 0)
            out.append((re.compile(entry["re"], flags), entry.get("label", name)))
        return out

    return compile_group("block"), compile_group("warn")


def tracked_files() -> list[pathlib.Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True).stdout.split()
    return [REPO / p for p in out]


def staged_files() -> list[pathlib.Path]:
    out = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--name-only",
                          "--diff-filter=ACM"], capture_output=True, text=True).stdout.split()
    return [REPO / p for p in out]


# This file necessarily contains every pattern it looks for, and policy text is
# allowed to name the rule it enforces. Both are marked rather than silently
# skipped, so a reader can see what was exempted and why.
SELF = pathlib.Path(__file__).resolve()
ALLOW_MARKER = "confidential-ok"


def scan(paths: list[pathlib.Path]) -> int:
    block_patterns, warn_patterns = load_patterns()
    blocks = warns = 0
    for path in paths:
        if path.suffix not in SCAN_SUFFIXES or not path.is_file():
            continue
        if path.resolve() == SELF:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        try:
            rel = path.relative_to(REPO)
        except ValueError:
            rel = path  # explicit path outside the repo: report it as given
        for n, line in enumerate(lines, 1):
            if ALLOW_MARKER in line:
                continue
            for pattern, label in block_patterns:
                for m in pattern.finditer(line):
                    print(f"BLOCK {rel}:{n}  {label}: {m.group(0)}")
                    blocks += 1
            for pattern, label in warn_patterns:
                for m in pattern.finditer(line):
                    print(f"WARN  {rel}:{n}  {label}: {m.group(0)}")
                    warns += 1
    print(f"\n{len(paths)} files scanned — {blocks} block, {warns} warn")
    if blocks:
        print("\nA hit is resolved by removing the content, not by rewording around the scanner.")
    return 1 if blocks else 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--staged"]:
        paths = staged_files()
    elif args:
        paths = [pathlib.Path(a).resolve() for a in args]
    else:
        paths = tracked_files()
    return scan(paths)


if __name__ == "__main__":
    raise SystemExit(main())
