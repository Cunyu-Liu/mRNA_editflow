#!/usr/bin/env python3
"""Audit unsupported affirmative exact-event sampling claims.

Negative statements, prohibited-phrase lists and structural boolean fields
whose value is frozen to false are distinguished from affirmative claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
package_spec = importlib.util.spec_from_file_location(
    "mrna_editflow",
    REPO_ROOT / "__init__.py",
    submodule_search_locations=[str(REPO_ROOT)],
)
if package_spec is None or package_spec.loader is None:
    raise RuntimeError("cannot bind mrna_editflow to current worktree")
package_module = importlib.util.module_from_spec(package_spec)
sys.modules["mrna_editflow"] = package_module
package_spec.loader.exec_module(package_module)

from mrna_editflow.core.mk0.acceptance import canonical_json_bytes

CLAIM = re.compile(
    r"\b(?:exact[\s_-]+gillespie|exact[\s_-]+ctmc[\s_-]+sampling|"
    r"exact[\s_-]+continuous[\s_-]+time\s+markov\s+chain[\s_-]+sampling|"
    r"(?:true|genuine)\s+(?:ctmc|continuous[\s_-]+time\s+markov\s+chain)"
    r"(?:\s+sampler|\s+sampling)?)\b",
    re.I,
)

CLAUSE_BOUNDARY = re.compile(r"[.!?;。！？；]")
NEGATIVE_FIXTURE_MARKER = "mk0-claim-audit-negative-fixture"


def _claim_clause(line: str, match: re.Match[str]) -> str:
    """Return only the clause containing a claim match."""

    prefix = line[: match.start()]
    suffix = line[match.end() :]
    left_matches = list(CLAUSE_BOUNDARY.finditer(prefix))
    left = left_matches[-1].end() if left_matches else 0
    right_match = CLAUSE_BOUNDARY.search(suffix)
    right = match.end() + (right_match.start() if right_match else len(suffix))
    return line[left:right]


def _tracked_text_paths(root: Path) -> tuple[list[Path], str, list[str]]:
    """Resolve the complete tracked UTF-8 universe, with a test fallback."""

    discovery = "git_ls_files"
    skipped_binary: list[str] = []
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        relative_paths = [
            Path(item.decode("utf-8")) for item in output.split(b"\0") if item
        ]
        if not relative_paths:
            raise RuntimeError("empty tracked file universe")
    except (OSError, subprocess.CalledProcessError, RuntimeError, UnicodeDecodeError):
        # Unit tests use tiny temporary trees without Git metadata.
        discovery = "recursive_utf8_fallback_no_git_metadata"
        relative_paths = [
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts
            )
        ]
    text_paths: list[Path] = []
    for relative in sorted(set(relative_paths)):
        path = root / relative
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            skipped_binary.append(relative.as_posix())
            continue
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary.append(relative.as_posix())
            continue
        text_paths.append(path)
    return text_paths, discovery, skipped_binary


NEGATIVE = re.compile(
    r"\b(?:not|none|never|neither|false|unsupported|forbid(?:den)?|disallow(?:ed)?)\b|"
    r"不得|不是|禁止|不称为|不支持|严禁",
    re.I,
)


def _walk_false_bindings(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if (
                key == "exact_gillespie"
                and item is not False
                and not (isinstance(item, dict) and item.get("const") is False)
            ):
                violations.append(f"{child} must be false/const-false")
            violations.extend(_walk_false_bindings(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(_walk_false_bindings(item, f"{path}[{index}]"))
    return violations


def audit(root: Path) -> dict[str, Any]:
    paths, discovery, skipped_binary = _tracked_text_paths(root)
    affirmative: list[dict[str, Any]] = []
    audited_hits = 0
    structural_failures: list[str] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            matches = list(CLAIM.finditer(line))
            if not matches:
                continue
            stripped = line.strip()
            if NEGATIVE_FIXTURE_MARKER in line:
                # Explicitly marked source text used to test this auditor.  A
                # marker is narrow and auditable; surrounding negation is not
                # allowed to suppress real claims.
                audited_hits += len(matches)
                continue
            structural_identifier = "exact_gillespie" in stripped and (
                stripped.endswith(": bool")
                or '"exact_gillespie"' in stripped
                or "exact_gillespie=" in stripped
            )
            for match in matches:
                audited_hits += 1
                clause = _claim_clause(line, match)
                if not NEGATIVE.search(clause) and not structural_identifier:
                    affirmative.append(
                        {
                            "path": str(path.relative_to(root)),
                            "line": line_number,
                            "text": stripped,
                        }
                    )
        if path.suffix in {".json", ".yaml", ".yml"}:
            try:
                if path.suffix == ".json":
                    structured = json.loads(text)
                else:
                    import yaml

                    structured = yaml.safe_load(text)
                structural_failures.extend(
                    f"{path.relative_to(root)}:{message}"
                    for message in _walk_false_bindings(structured)
                )
            except Exception as error:
                structural_failures.append(
                    f"{path.relative_to(root)}:structured parse failed: "
                    f"{type(error).__name__}: {error}"
                )
    universe = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(paths)
    ]
    universe_sha256 = hashlib.sha256(
        json.dumps(universe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "mk0_exact_claim_audit_v1",
        "file_universe_discovery": discovery,
        "tracked_utf8_file_universe_sha256": universe_sha256,
        "files_audited": len(paths),
        "binary_files_skipped": skipped_binary,
        "claim_hits_reviewed": audited_hits,
        "unsupported_affirmative_claims": affirmative,
        "structural_false_binding_failures": structural_failures,
        "unsupported_affirmative_claim_count": len(affirmative),
        "pass": not affirmative and not structural_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
