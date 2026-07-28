#!/usr/bin/env python3
"""Audit that no active paper code references the superseded legacy contract.

The legacy P3/NMI research contract was archived under:
  - configs/archive/p3_legacy/
  - docs/archive/p3_legacy/
  - scripts/archive/p3_legacy/

New training code, paper mode, and result-generation code must not read the
legacy contract as a constraint source. This audit scans the active working
tree for textual references to legacy contract paths/identifiers.

Acceptance (R0-01):
    python scripts/contracts/audit_legacy_references.py --strict
    -> "active paper code references to legacy contract = 0" and exit code 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Legacy contract identifiers. A hit in an active file = one violation.
LEGACY_PATTERNS = [
    re.compile(r"configs/p3_"),
    re.compile(r"docs/p3_"),
    re.compile(r"p3_frozen_research_contract"),
    re.compile(r"p3_primary_task"),
    re.compile(r"nmi_execution_contract"),
    re.compile(r"configs/nmi_execution\.yaml"),
]

# Directories that are either the archive itself (allowed to reference the
# legacy contract), historical data/result records, or non-code bulk.
EXCLUDED_DIR_PREFIXES = (
    ".git",
    "configs/archive",
    "docs/archive",
    "scripts/archive",
    "artifacts",
    "results",
    "data",
    "logs",
    "snapshots",
    "backups",
    "checkpoints",
    "ckpts",
    "models",
    "external_tools",
    "benchmark_v21/external_data",
    "mrna_editflow.egg-info",
    "__pycache__",
    "node_modules",
)

# Files that legitimately contain the legacy patterns (the audit definition
# and its test fixtures). Everything else must be clean.
EXCLUDED_FILES = {
    "scripts/contracts/audit_legacy_references.py",
    "tests/test_audit_legacy_references.py",
}

TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".sh", ".txt", ".toml",
    ".cfg", ".ini", ".tsv", ".csv", ".rst",
}
TEXT_NAMES = {"Dockerfile", ".dockerignore", ".gitignore", ".gitattributes"}
MAX_FILE_BYTES = 5 * 1024 * 1024


def _is_excluded(rel: str) -> bool:
    if rel in EXCLUDED_FILES:
        return True
    for prefix in EXCLUDED_DIR_PREFIXES:
        if rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def iter_active_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_excluded(rel):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path, rel


def scan(root: Path):
    """Return list of (rel_path, line_no, pattern, line_text) violations."""
    violations = []
    for path, rel in iter_active_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in LEGACY_PATTERNS:
                if pat.search(line):
                    violations.append((rel, lineno, pat.pattern, line.strip()))
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any active reference is found")
    parser.add_argument("--json", action="store_true",
                        help="emit violations as JSON")
    args = parser.parse_args(argv)

    violations = scan(REPO_ROOT)
    n = len(violations)

    if args.json:
        print(json.dumps([
            {"file": f, "line": ln, "pattern": p, "text": t}
            for f, ln, p, t in violations
        ], indent=2, ensure_ascii=False))
    else:
        for f, ln, p, t in violations:
            print(f"VIOLATION {f}:{ln} pattern={p} :: {t[:120]}")

    print(f"active paper code references to legacy contract = {n}")
    if args.strict and n > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
