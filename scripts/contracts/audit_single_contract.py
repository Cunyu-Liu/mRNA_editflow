#!/usr/bin/env python3
"""Fail-closed audit for the active UTR EditFlow V2 contract.

The audit combines structured invariant checks with a scan for legacy
constraint-source references outside versioned archive directories. Historical
results may mention old science; only active code/config/docs are forbidden
from treating those files as current authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_GOAL_SHA = "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"

ARCHIVE_PARTS = {"archive", "__pycache__", ".git"}
SKIP_PREFIXES = (
    "artifacts/",
    "results/",
    "data/",
    "logs/",
    "snapshots/",
    "backups/",
    "checkpoints/",
    "benchmark_v21/external_data/",
    "data_registry/search_artifacts/",
)
SKIP_FILES = {
    "scripts/contracts/audit_single_contract.py",
    "docs/contracts/v2_contract_conflict_matrix.md",
    "docs/contracts/mrna_editflow_contract.md",
    "docs/decision_log.md",
    "tests/test_audit_legacy_references.py",
}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".sh", ".toml", ".txt", ".csv"}

LEGACY_ACTIVE_PATTERNS = {
    "legacy_contract_id": re.compile(r"\bpublic_intervention_contract_v1\b"),
    "legacy_contract_path": re.compile(r"\bconfigs/public_intervention_contract\.yaml\b"),
    "legacy_question_path": re.compile(r"\bdocs/public_intervention_scientific_question\.md\b"),
    "legacy_claim_path": re.compile(r"\bdocs/public_intervention_claim_matrix\.md\b"),
    "sealed_assignment": re.compile(r"\brole\s*:\s*sealed_external_test\b"),
    "sealed_block": re.compile(r"\bsealed_external_dataset\s*:"),
    "flow_optional_assignment": re.compile(r"\bflow_optional\s*:\s*true\b", re.I),
    "cpu_fallback_assignment": re.compile(r"\bcpu_fallback_allowed\s*:\s*true\b", re.I),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_active_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in SKIP_FILES or any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if ARCHIVE_PARTS.intersection(path.relative_to(root).parts):
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        yield path, rel


def scan_legacy_references(root: Path) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for path, rel in iter_active_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, pattern in LEGACY_ACTIVE_PATTERNS.items():
                if pattern.search(line):
                    violations.append(
                        {
                            "kind": name,
                            "file": rel,
                            "line": line_no,
                            "text": line.strip()[:240],
                        }
                    )
    return violations


def _get(mapping: dict, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def audit(root: Path) -> dict[str, object]:
    contract_path = root / "configs/utr_editflow_execution_policy.yaml"
    goal_path = root / "docs/contracts/mrna_editflow_contract.md"
    readme_path = root / "README.md"
    errors: list[str] = []

    if not contract_path.is_file():
        return {
            "strict_pass": False,
            "errors": ["missing configs/utr_editflow_execution_policy.yaml"],
            "counters": {
                "active_predictor_only_fallback": 1,
                "active_flow_optional_clauses": 1,
                "active_cds_full_length_phase1_tasks": 1,
                "gse246381_sealed_wording": 1,
                "formal_neural_cpu_fallback_allowed": 1,
                "active_contract_ambiguity": 1,
            },
            "violations": [],
        }

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    goal_hash = sha256_path(goal_path) if goal_path.is_file() else None
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""

    predictor_fallback = int(
        _get(contract, "method", "predictor_role") != "support_only"
        or _get(contract, "method", "predictor_only_fallback_allowed") is not False
    )
    flow_optional = int(
        _get(contract, "method", "edit_flow_required") is not True
        or _get(contract, "method", "flow_optional") is not False
    )
    regions = _get(contract, "current_scope", "regions", default=[])
    cds_full_length = int(
        regions != ["five_utr", "three_utr"]
        or _get(contract, "current_scope", "cds") != "forbidden"
        or _get(contract, "current_scope", "full_length") != "forbidden"
    )
    gse_sealed = int(
        _get(contract, "gse246381", "historically_exposed") is not True
        or _get(contract, "gse246381", "role")
        != "historically_exposed_retrospective_external_stress_test"
        or _get(contract, "gse246381", "untouched_wording_allowed") is not False
    )
    cpu_fallback = int(
        _get(contract, "training", "formal_neural_device") != "cuda"
        or _get(contract, "training", "cpu_fallback_allowed") is not False
    )

    active_contract_files = [
        path
        for path in (root / "configs").glob("*contract*.yaml")
        if "archive" not in path.parts and path.name != "execution_contract.yaml"
    ]
    ambiguity = int(
        contract.get("contract_id") != "mrna_editflow_single_active_contract"
        or _get(contract, "goal_document", "sha256") != EXPECTED_GOAL_SHA
        or goal_hash != EXPECTED_GOAL_SHA
        or len(active_contract_files) != 1
        or "mrna_editflow_single_active_contract" not in readme
    )

    legacy_violations = scan_legacy_references(root)
    ambiguity += len(legacy_violations)

    counters = {
        "active_predictor_only_fallback": predictor_fallback,
        "active_flow_optional_clauses": flow_optional,
        "active_cds_full_length_phase1_tasks": cds_full_length,
        "gse246381_sealed_wording": gse_sealed,
        "formal_neural_cpu_fallback_allowed": cpu_fallback,
        "active_contract_ambiguity": ambiguity,
    }
    for name, value in counters.items():
        if value:
            errors.append(f"{name}={value}")

    return {
        "contract_id": contract.get("contract_id"),
        "goal_sha256_expected": EXPECTED_GOAL_SHA,
        "goal_sha256_actual": goal_hash,
        "active_contract_files": [p.relative_to(root).as_posix() for p in active_contract_files],
        "counters": counters,
        "violations": legacy_violations,
        "errors": errors,
        "strict_pass": not errors,
    }


def audit(root: Path) -> dict[str, object]:
    policy_path = root / "configs/utr_editflow_execution_policy.yaml"
    goal_path = root / "docs/contracts/mrna_editflow_contract.md"
    errors: list[str] = []
    if not policy_path.is_file() or not goal_path.is_file():
        return {
            "strict_pass": False,
            "errors": ["missing canonical contract or derived execution policy"],
            "counters": {"active_contract_count": 0, "parallel_active_contract_references": 1, "versioned_amendment_active_contract_files": 1, "derived_policy_hash_mismatch": 1, "active_predictor_only_fallback": 1, "active_flow_optional_clauses": 1, "active_cds_full_length_phase1_tasks": 1, "gse246381_sealed_wording": 1, "formal_neural_cpu_fallback_allowed": 1},
            "violations": [],
        }
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    goal_hash = sha256_path(goal_path)
    canonical_files = [path for path in (root / "docs/contracts").glob("*.md") if "archive" not in path.parts and path.name != "v2_contract_conflict_matrix.md"]
    active_count = len(canonical_files)
    versioned = [path for path in canonical_files if re.search(r"(?:_v\d|amendment)", path.name, re.I)]
    policy_mismatch = int(policy.get("execution_policy_id") != "utr_editflow_execution_policy" or policy.get("non_authoritative_derivative") is not True or policy.get("generated_from_contract_path") != goal_path.relative_to(root).as_posix() or policy.get("generated_from_contract_sha256") != goal_hash)
    predictor_fallback = int(_get(policy, "method", "predictor_role") != "support_only" or _get(policy, "method", "predictor_only_fallback_allowed") is not False)
    flow_optional = int(_get(policy, "method", "edit_flow_required") is not True or _get(policy, "method", "flow_optional") is not False)
    regions = _get(policy, "current_scope", "regions", default=[])
    cds_full_length = int(regions != ["five_utr", "three_utr"] or _get(policy, "current_scope", "cds") != "forbidden" or _get(policy, "current_scope", "full_length") != "forbidden")
    gse_sealed = int(_get(policy, "gse246381", "historically_exposed") is not True or _get(policy, "gse246381", "role") != "historically_exposed_retrospective_external_stress_test" or _get(policy, "gse246381", "untouched_wording_allowed") is not False)
    cpu_fallback = int(_get(policy, "training", "formal_neural_device") != "cuda" or _get(policy, "training", "cpu_fallback_allowed") is not False)
    legacy_violations = scan_legacy_references(root)
    parallel = int(active_count != 1 or any(path.resolve() != goal_path.resolve() for path in canonical_files) or bool(legacy_violations))
    counters = {
        "active_contract_count": active_count,
        "parallel_active_contract_references": parallel,
        "versioned_amendment_active_contract_files": len(versioned),
        "derived_policy_hash_mismatch": policy_mismatch,
        "active_predictor_only_fallback": predictor_fallback,
        "active_flow_optional_clauses": flow_optional,
        "active_cds_full_length_phase1_tasks": cds_full_length,
        "gse246381_sealed_wording": gse_sealed,
        "formal_neural_cpu_fallback_allowed": cpu_fallback,
    }
    for name, value in counters.items():
        if (name == "active_contract_count" and value != 1) or (name != "active_contract_count" and value):
            errors.append(f"{name}={value}")
    return {
        "canonical_contract_path": goal_path.relative_to(root).as_posix(),
        "contract_sha256": goal_hash,
        "derived_policy_path": policy_path.relative_to(root).as_posix(),
        "active_contract_files": [path.relative_to(root).as_posix() for path in canonical_files],
        "counters": counters,
        "violations": legacy_violations,
        "errors": errors,
        "strict_pass": not errors,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    report = audit(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for violation in report.get("violations", []):
            print(
                "VIOLATION "
                f"{violation['file']}:{violation['line']} "
                f"{violation['kind']} :: {violation['text']}"
            )
        for name, value in report["counters"].items():
            print(f"{name.replace('_', ' ')} = {value}")
        print(f"active contract strict pass = {str(report['strict_pass']).lower()}")

    return 1 if args.strict and not report["strict_pass"] else 0


if __name__ == "__main__":
    sys.exit(main())
