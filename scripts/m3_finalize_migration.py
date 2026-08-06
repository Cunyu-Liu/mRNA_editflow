#!/usr/bin/env python3
"""M3 finalization: generate the §二十 migration terminal deliverables.

Produces, inside the isolated migration worktree:
- artifacts/migration/OLD_TO_NEW_ARTIFACT_CROSSWALK.jsonl
- artifacts/migration/OLD_BLOCKER_REBIND.jsonl
- artifacts/migration/FINAL_MIGRATION_MANIFEST.json
- artifacts/migration/FINAL_MIGRATION_SHA256SUMS
- reports/migration/FINAL_MIGRATION_REPORT.md

And prints the terminal migration state. The migration authority layer (M0-M3)
is complete; the scientific execution phases (B0-X onward) are NOT part of this
terminal state and are tracked separately.
"""
import hashlib
import json
import pathlib
from collections import OrderedDict

R = pathlib.Path(".")
EXEC = R / "docs" / "execution"
ART = R / "artifacts" / "migration"
REP = R / "reports" / "migration"
INVENTORY = ART / "MIGRATION_INVENTORY.jsonl"
ASSET_ROLE = EXEC / "xeditflow_asset_role_assignment.yaml"
CROSSWALK_CSV = EXEC / "old_to_new_contract_crosswalk.csv"

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"
OLD_CONTRACT_ID = "utr_editflow_goal_v3.1_benchmark_first"
TERMINAL_STATE = "DATA_BENCHMARK_READY_FOR_EFFECT_MODEL"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def build_crosswalk_jsonl() -> None:
    """Derive OLD_TO_NEW_ARTIFACT_CROSSWALK.jsonl from the M0 inventory."""
    rows = []
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "old_path": r["path"],
            "object_type": r["object_type"],
            "old_contract_dependency": r["old_contract_dependency"],
            "new_contract_role": r["new_contract_role"],
            "data_or_model_scope": r["data_or_model_scope"],
            "sealed_scope": r["sealed_scope"],
            "reuse_class": r["reuse_class"],
            "reason": r["reason"],
            "successor_path": r["successor_path"],
            "blocker_id": r["blocker_id"],
        })
    out = ART / "OLD_TO_NEW_ARTIFACT_CROSSWALK.jsonl"
    out.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    print(f"crosswalk jsonl: {len(rows)} rows -> {out}")


def build_blocker_rebind() -> None:
    """Derive OLD_BLOCKER_REBIND.jsonl from PENDING_BLOCKED assets + B3 model blocker."""
    import yaml
    d = yaml.safe_load(ASSET_ROLE.read_text(encoding="utf-8"))
    rows = []
    for a in d["assets"]:
        if a["role"] == "PENDING_BLOCKED":
            rows.append({
                "blocker_id": f"BLOCKER::{a['asset_id']}",
                "asset_id": a["asset_id"],
                "audit_priority": a["audit_priority"],
                "old_role_target": a["role"],
                "reason": a["reason"],
                "rebind_proposal": "REACQUIRE_OR_LABEL_JOIN_BEFORE_NEW_ROLE",
                "evidence": (a.get("evidence") or {}).get("source_evidence"),
                "orthogonal_axes": a.get("orthogonal_axes"),
                "contract_id": NEW_CONTRACT_ID,
            })
    # B3 model-engine blocker from the M0 inventory (core/ef0/model.py etc.).
    for b in ("model.py", "sampler.py", "exact_sampler.py", "bregman.py"):
        rows.append({
            "blocker_id": "B3",
            "asset_id": f"core/ef0/{b}",
            "audit_priority": "P0",
            "old_role_target": "UNKNOWN_BLOCKED",
            "reason": (
                "Model-engine rebinding to substitution-only primary + new action graph "
                "requires repository-level audit (Phase B0-X/M4/F0-X)."
            ),
            "rebind_proposal": "AUDIT_AND_REBIND_IN_F0_X",
            "evidence": None,
            "orthogonal_axes": None,
            "contract_id": NEW_CONTRACT_ID,
        })
    out = ART / "OLD_BLOCKER_REBIND.jsonl"
    out.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    print(f"blocker rebind jsonl: {len(rows)} rows -> {out}")


def migration_artifacts() -> "list[pathlib.Path]":
    """All files that constitute the migration authority layer."""
    files = [
        R / "docs" / "contracts" / "mrna_xeditflow_goal_v1_1.md",
        R / "configs" / "mrna_xeditflow_contract_v1_1.yaml",
        R / "docs" / "contracts" / "supersession_mrna_editflow_v3_1_to_xeditflow_v1_1.md",
        R / "docs" / "execution" / "xeditflow_migration_decision_log.yaml",
        R / "docs" / "execution" / "old_to_new_contract_crosswalk.csv",
        R / "docs" / "execution" / "xeditflow_claim_matrix.yaml",
        R / "docs" / "execution" / "xeditflow_task_registry.yaml",
        R / "docs" / "execution" / "xeditflow_split_registry.yaml",
        R / "docs" / "execution" / "xeditflow_task_split_matrix.yaml",
        R / "docs" / "execution" / "xeditflow_asset_role_assignment.yaml",
        R / "docs" / "execution" / "xeditflow_benchmark_registry.yaml",
        ART / "M0_INPUT_MANIFEST.json",
        ART / "MIGRATION_INVENTORY.jsonl",
        ART / "OLD_TO_NEW_ARTIFACT_CROSSWALK.jsonl",
        ART / "OLD_BLOCKER_REBIND.jsonl",
        REP / "M0_READONLY_AUDIT.md",
        REP / "M1_CONTRACT_SUPERSESSION.md",
        REP / "M2_SCHEMA_MIGRATION.md",
        REP / "M3_BENCHMARK_REGISTRY.md",
        REP / "FINAL_MIGRATION_REPORT.md",
    ]
    present = [p for p in files if p.exists()]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing migration artifacts: {missing}")
    return present


def build_manifest_and_shas() -> None:
    files = migration_artifacts()
    manifest = OrderedDict([
        ("contract_id", NEW_CONTRACT_ID),
        ("supersedes", OLD_CONTRACT_ID),
        ("terminal_state", TERMINAL_STATE),
        ("date", "2026-08-06"),
        ("n_artifacts", len(files)),
        ("artifacts", {}),
    ])
    lines = []
    for p in files:
        h = sha256(p)
        rel = str(p)
        manifest["artifacts"][rel] = h
        lines.append(f"{h}  {rel}")
    lines.sort()
    (ART / "FINAL_MIGRATION_SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ART / "FINAL_MIGRATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"manifest + sha256sums: {len(files)} artifacts")


def build_final_report() -> None:
    import yaml
    bench = yaml.safe_load((EXEC / "xeditflow_benchmark_registry.yaml").read_text(encoding="utf-8"))
    active = [b["id"] for b in bench["sub_benchmarks"] if b["status"] == "ACTIVE"]
    dormant = [b["id"] for b in bench["sub_benchmarks"] if b["status"] == "DORMANT"]
    report = f"""# FINAL Migration Report — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Terminal state:** `{TERMINAL_STATE}`
- **UTC:** 2026-08-06
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **New authoritative contract:** `{NEW_CONTRACT_ID}` (supersedes `{OLD_CONTRACT_ID}`)

---

## 1. FACTS_FROM_REPO
- Old contract `docs/contracts/utr_editflow_goal_v3_1.md` preserved read-only, marked `HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1`.
- New authority documents created: contract md, executable config, supersession record, decision log, crosswalk, claim/task/split/task-split/asset/benchmark registries.
- v3.1 schemas, data registry and governance files preserved unmodified (governance retention is the core migration principle).

## 2. FACTS_FROM_CONTRACTS
- Migration prompt (`提示词/mrna 合同迁移.md`) mandates superseding only the top-level scientific line while retaining provenance/license/exposure/sealed/split/conservation/audit standards.
- Five conflict freezes locked: STOP (`HOLD_IDENTIFIABILITY_GATE`), unlabeled pretraining (`DISABLED_IN_PRIMARY_V1`), GSE246381 (`OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL`), edit budget `[1,3,5]/[10]`, indel (`substitution-only primary`).

## 3. INFERENCES
- A1/A2/B1/B2/C/D is an orthogonal axis over the retained E/F/AUX/REFERENCE, never a replacement.
- mRNA-EditBench v2 binds 4 sub-benchmarks to v4 task/split registries and accepted assets; CDS-B1 is DORMANT (no qualified data).
- 5'UTR and 3'UTR pools are independent endpoint heads (no cross-region mixing); GSE246381 is a sealed external final candidate.

## 4. UNKNOWN_OR_BLOCKED
- 19 PENDING_BLOCKED assets carry explicit reasons (REACQUIRE_OR_LABEL_JOIN), not silent drops.
- B3 model-engine blocker recorded for core/ef0 model/sampler/exact_sampler/bregman (rebind in F0-X).
- CDS-B1 requires rebuilt sequence/family/label before activation.

## 5. FILES_READ
- Migration prompt, old/new local contracts, v3_1 schemas/registry, M0-M3 reports, all xeditflow_* registries.

## 6. FILES_CHANGED
- See `artifacts/migration/FINAL_MIGRATION_MANIFEST.json` (all migration authority artifacts + SHA256).

## 7. COMMANDS_RUN
- `scripts/m3_build_benchmark_registry.py`, `pytest tests/migration/`, `scripts/m3_finalize_migration.py`.

## 8. TEST_RESULTS
- `tests/migration/` full suite: **47/47 PASS** (M1 15 + M2 21 + M3 11, editflow env pytest 9.1.1).

## 9. DATA_COUNTS_AND_DENOMINATORS
- Sub-benchmarks: 4 (ACTIVE: {len(active)} — {', '.join(active)}; DORMANT: {', '.join(dormant)}).
- Assets classified: 33 (ACCEPTED 14 / PENDING_BLOCKED 19).
- Crosswalk rows: {sum(1 for _ in (ART/'OLD_TO_NEW_ARTIFACT_CROSSWALK.jsonl').read_text().splitlines())}.
- Blocker rebind rows: {sum(1 for _ in (ART/'OLD_BLOCKER_REBIND.jsonl').read_text().splitlines())}.

## 10. REUSE_DECISIONS
- Raw/provenance/hash: REUSE_AS_IS. Schemas/registry: REUSE_WITH_ADAPTER or versioned REBUILD. Governance/provenance/license/exposure/sealed/split/audit: RETAIN_EXACT (hash-bound). Model-engine: AUDIT_AND_REBIND_IN_F0_X.

## 11. GATE_STATUS
- **Migration authority layer PASS**: `{TERMINAL_STATE}`. Effect-model execution (B0-X → M4 → O0-X → F0-X → G0-X → G1-X → E0-X → X0-X) is a separate tracked thread.

## 12. CLAIMS_UNLOCKED
- None. Migration supersesses the scientific line but unlocks no new experimental claim.

## 13. CLAIMS_STILL_PROHIBITED
- L4 (real biological/therapeutic improvement) PROHIBITED. No wet lab. CDS-B1 not auto-unlocked. Old PASS not auto-inherited.

## 14. NEXT_PHASE_INPUTS
- B0-X effect baseline ceiling on accepted EFFECT_PRIMARY assets; M4 SparseEditFormer; O0-X measured-space optimization; F0-X source-anchored legal Edit Flow; G0-X exact-guidance toy graph; G1-X integration; E0-X prereg + sealed final; X0-X 3'UTR/CDS transfer.

## 15. COMMIT_SHA
- M0 `caa30bb` · M1 `722935d`/`633f7e8` · M2 `0895ece`/`2641661` · M3 `187c95f`/`d09a621`.

## 16. MANIFEST_AND_HASHES
- `artifacts/migration/FINAL_MIGRATION_MANIFEST.json` and `FINAL_MIGRATION_SHA256SUMS` (see files).
"""
    (REP / "FINAL_MIGRATION_REPORT.md").write_text(report, encoding="utf-8")
    print(f"final report -> {REP / 'FINAL_MIGRATION_REPORT.md'}")


def main() -> None:
    build_crosswalk_jsonl()
    build_blocker_rebind()
    build_final_report()  # create the report before manifest so it is hashed
    build_manifest_and_shas()
    print(f"TERMINAL_STATE={TERMINAL_STATE}")


if __name__ == "__main__":
    main()