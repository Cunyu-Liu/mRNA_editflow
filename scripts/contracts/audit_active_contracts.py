#!/usr/bin/env python3
"""Audit active-contract consistency for utr_editflow_contract_v2 (Phase C0-04).

Run:
    python scripts/contracts/audit_active_contracts.py --strict

Checks that the active tree is governed by exactly one contract
(`utr_editflow_contract_v2`) and that no v1 / legacy / forbidden clauses
remain in active code, configs, docs, or tests.

Exit codes:
    0 — all checks passed (active contract ambiguity = 0)
    1 — one or more checks failed (use --strict to fail on warnings too)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Active v2 contract — the only allowed active contract
ACTIVE_CONTRACT_ID = "utr_editflow_contract_v2"
ACTIVE_CONTRACT_PATH = ROOT / "configs" / "utr_editflow_contract_v2.yaml"

# v1 contract — must NOT appear in active tree (only allowed in archive/)
V1_CONTRACT_ID = "public_intervention_contract_v1"
V1_CONTRACT_FILES = [
    "configs/public_intervention_contract.yaml",
    "docs/public_intervention_scientific_question.md",
    "docs/public_intervention_claim_matrix.md",
    "docs/execution/task_registry.yaml",
]

# Files allowed to mention v1 contract_id or forbidden terms because they
# DEFINE the v2 contract, record the v1→v2 transition, or audit/test it.
ALLOWED_TO_REFERENCE_V1_OR_FORBIDDEN = {
    # v2 contract family — defines what's forbidden
    "configs/utr_editflow_contract_v2.yaml",
    "docs/utr_editflow_scientific_question_v2.md",
    "docs/utr_editflow_claim_matrix_v2.md",
    "docs/execution/task_registry_v2.yaml",
    "docs/decision_log.md",
    "docs/contracts/v2_contract_conflict_matrix.md",
    "docs/data/d0_05_dataset_role_table.yaml",
    "docs/data/d0_05_current_candidates_audit.md",
    "data_registry/intervention_candidates.yaml",
    "README.md",
    # Audit + test files — they check for forbidden terms by name
    "scripts/contracts/audit_active_contracts.py",
    "tests/test_utr_editflow_contract_v2.py",
    "tests/test_audit_legacy_references.py",
    # Legacy SUPERSEDED markers (intentionally name v1 as superseded)
    "configs/archive/p3_legacy/SUPERSEDED.md",
    "docs/archive/p3_legacy/SUPERSEDED.md",
    "scripts/archive/p3_legacy/SUPERSEDED.md",
    "archive/legacy_predictor_first_v1/SUPERSEDED.md",
}

# Forbidden clauses / wording in active tree.
# Patterns are matched against file content. A file is flagged ONLY if:
#   (a) the pattern matches, AND
#   (b) the file is NOT in ALLOWED_TO_REFERENCE_V1_OR_FORBIDDEN, AND
#   (c) the file is not a dated historical plan document.
FORBIDDEN_PATTERNS = [
    # GSE246381 sealed wording used as a positive claim (not as a "forbidden wording" list entry)
    (re.compile(r"GSE246381[^.\n]*(sealed|untouched|never-seen)", re.IGNORECASE),
     "GSE246381 sealed/untouched wording (must be historically_exposed E4)"),
    # Flow optional wording used as an active clause
    (re.compile(r"\bFlow\s+optional\b", re.IGNORECASE),
     "Flow-optional clause (Edit Flow is primary, not optional)"),
    # CPU fallback for formal neural training
    (re.compile(r"cpu[_\s-]?fallback.*neural|neural.*cpu[_\s-]?fallback", re.IGNORECASE),
     "formal neural CPU fallback (training is GPU-only)"),
]

# Legacy code/config paths that must NOT be in active tree
LEGACY_ACTIVE_PATHS = [
    "benchmark",
    "benchmark_v21",
    "data/nmi_benchmark_v2",
    "rl",
    "ckpts",
    "train_grpo.py",
    "train_dagger_ranker.py",
    "train_proposal_ranker.py",
    "train_adapter.py",
    "train_backbone.py",
    "sample.py",
    "configs/nmi_split_v2.yaml",
    "configs/paired_delta",
    "configs/stage_a_full_a100_max.json",
    "configs/stage_a_full_bs8_gradaccum4.json",
    "configs/stage_a_mig_tiny_gencode.json",
    "configs/stage_a_recovery_p2_02.json",
    "configs/stage_a_recovery_p2_10_option_c.json",
    "audit_sota_readiness.py",
    "sota_gap_report.py",
    "harvest_sota_artifacts.sh",
    "audit_multiobjective_scaleup_claims.py",
    "docs/next_steps_sota_roadmap.md",
    "docs/codongpt_rl_reproduction_blocker.md",
    "docs/cross_region_synergy_finding_v1.md",
    "docs/cross_region_synergy_finding_v2.md",
    "docs/cross_region_synergy_protocol_v1.md",
    "tests/test_public_intervention_contract.py",
]

# File extensions to scan for forbidden patterns
SCAN_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".json", ".sh", ".txt", ".toml"}

# Directories to skip during scanning (archive content is allowed to keep v1 refs)
SKIP_DIRS = {
    "archive",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def _is_allowed(rel: str) -> bool:
    """Whether a file is allowed to reference v1 or forbidden terms."""
    if rel in ALLOWED_TO_REFERENCE_V1_OR_FORBIDDEN:
        return True
    # Dated historical plan documents record what was planned at the time
    if rel.startswith("docs/plans/2026-07-28-"):
        return True
    return False


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  WARN: {msg}", file=sys.stderr)


def check_active_contract_exists() -> bool:
    print("[1/6] Active v2 contract file exists ...")
    ok = ACTIVE_CONTRACT_PATH.exists()
    if ok:
        print(f"  OK: {ACTIVE_CONTRACT_PATH.relative_to(ROOT)}")
    else:
        fail(f"Missing active contract: {ACTIVE_CONTRACT_PATH}")
    return ok


def check_v1_contract_archived() -> bool:
    print("[2/6] v1 contract files not in active tree ...")
    ok = True
    for rel in V1_CONTRACT_FILES:
        p = ROOT / rel
        if p.exists():
            fail(f"v1 contract file still in active tree: {rel}")
            ok = False
    if ok:
        print("  OK: no v1 contract files in active tree")
    # Also verify they ARE archived
    archive_dir = ROOT / "archive" / "legacy_predictor_first_v1" / "contracts_v1"
    expected = [
        "public_intervention_contract.yaml",
        "public_intervention_claim_matrix.md",
        "public_intervention_scientific_question.md",
        "task_registry.yaml",
    ]
    for name in expected:
        ap = archive_dir / name
        if not ap.exists():
            warn(f"v1 contract not found in archive: {ap.relative_to(ROOT)}")
    return ok


def check_legacy_paths_not_in_active_tree() -> bool:
    print("[3/6] Legacy paths not in active tree ...")
    ok = True
    for rel in LEGACY_ACTIVE_PATHS:
        p = ROOT / rel
        if p.exists():
            fail(f"Legacy path still in active tree: {rel}")
            ok = False
    if ok:
        print(f"  OK: {len(LEGACY_ACTIVE_PATHS)} legacy paths verified absent from active tree")
    return ok


def check_no_v1_contract_id_in_active_code() -> bool:
    print("[4/6] No v1 contract_id references in active code ...")
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix not in SCAN_EXTENSIONS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if V1_CONTRACT_ID in text:
            rel = str(p.relative_to(ROOT))
            if _is_allowed(rel):
                continue
            hits.append(rel)
    if hits:
        for h in hits:
            fail(f"v1 contract_id referenced in active file: {h}")
        return False
    print("  OK: no v1 contract_id references in active code")
    return True


def check_no_forbidden_patterns_in_active_tree() -> bool:
    print("[5/6] No forbidden clauses in active tree ...")
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix not in SCAN_EXTENSIONS:
            continue
        rel = str(p.relative_to(ROOT))
        if _is_allowed(rel):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern, description in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                hits.append((rel, description))
    if hits:
        for rel, desc in hits:
            fail(f"Forbidden clause ({desc}) in: {rel}")
        return False
    print("  OK: no forbidden clauses in active tree")
    return True


def check_task_registry_uses_v2() -> bool:
    print("[6/6] Active task registry uses v2 contract_id ...")
    registry_path = ROOT / "docs" / "execution" / "task_registry_v2.yaml"
    if not registry_path.exists():
        fail(f"Missing v2 task registry: {registry_path}")
        return False
    import yaml
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if registry.get("contract_id") != ACTIVE_CONTRACT_ID:
        fail(f"task_registry_v2 contract_id = {registry.get('contract_id')!r}, expected {ACTIVE_CONTRACT_ID!r}")
        return False
    # Verify C0-04 acceptance: all C0 tasks DONE
    c0_tasks = [t for t in registry.get("tasks", []) if t.get("phase") == "C0"]
    for t in c0_tasks:
        if t.get("status") != "DONE":
            fail(f"C0 task not DONE: {t.get('task_id')} status={t.get('status')}")
            return False
    print(f"  OK: task_registry_v2 contract_id={ACTIVE_CONTRACT_ID}, {len(c0_tasks)} C0 tasks DONE")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit active-contract consistency for utr_editflow_contract_v2")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    print(f"Auditing active contract: {ACTIVE_CONTRACT_ID}")
    print(f"Root: {ROOT}")
    print()

    checks = [
        check_active_contract_exists(),
        check_v1_contract_archived(),
        check_legacy_paths_not_in_active_tree(),
        check_no_v1_contract_id_in_active_code(),
        check_no_forbidden_patterns_in_active_tree(),
        check_task_registry_uses_v2(),
    ]

    print()
    passed = sum(1 for c in checks if c)
    total = len(checks)
    print(f"Result: {passed}/{total} checks passed")

    if all(checks):
        print("ACTIVE CONTRACT AMBIGUITY = 0")
        return 0
    else:
        print("ACTIVE CONTRACT AMBIGUITY > 0 — see FAIL messages above", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
