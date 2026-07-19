"""Remediate near-neighbor violations and promote splits to paper_eligible=true.

Strategy: for each split, remove ALL val/test records that appear in ANY
near-neighbor violation from their respective role index files, and move
them to the excluded index.  Then re-run the exhaustive cross-role
near-neighbor audit.  If it passes (0 violations), promote the split to
paper_eligible=true.

This does NOT rebuild canonical records or derived views — only the
lightweight split index files, manifests, and audit artifacts are modified.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.reconstruction_v2_audit import (
    V2AuditError,
    exhaustive_cross_role_near_neighbor_audit,
    gene_symbol_alias_audit,
    promote_combined_bundle,
    promote_split_to_paper_eligible,
)
from mrna_editflow.data.split_contract import SPLIT_ROLES, sha256_file as contract_sha256_file


PROJECT_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow")
COMBINED_DIR = PROJECT_ROOT / "data/reconstructed/p0_data_reconstruction_v1/combined"
SPLIT_ROOT = PROJECT_ROOT / "benchmark/dev/p0_data_reconstruction_v1"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULT_PATH = DOCS_DIR / "p0_data_reconstruction_v2_remediation_result.json"

SPLIT_NAMES = [
    "combined_family",
    "gencode_family",
    "refseq_family",
    "gencode_to_refseq",
]


def _log(phase: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{phase}] {msg}", flush=True)


def _read_indices(path: Path) -> list[int]:
    with open(path, "r", encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()]


def _write_indices(path: Path, indices: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for idx in indices:
            fh.write(f"{idx}\n")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    _log("init", f"split_root={SPLIT_ROOT}")

    # ------------------------------------------------------------------
    # Load combined bundle for alias audit (reuse from v2 audit).
    # ------------------------------------------------------------------
    combined_records = load_records_jsonl(str(COMBINED_DIR / "combined_model_view.records.jsonl"))
    combined_meta = _load_jsonl(COMBINED_DIR / "combined_model_view.metadata.jsonl")
    assignments = json.loads((COMBINED_DIR / "family_assignments.json").read_text())
    family_evidence = _load_jsonl(COMBINED_DIR / "family_evidence.jsonl")
    alias_audit = gene_symbol_alias_audit(combined_records, combined_meta, assignments, family_evidence)
    _log("alias", f"alias_audit.passed={alias_audit['passed']}")

    split_results: dict[str, dict] = {}
    remediation_summary: dict[str, dict] = {}

    for name in SPLIT_NAMES:
        split_dir = SPLIT_ROOT / name
        _log(name, f"--- split {name} ---")
        t0 = time.time()

        # 1. Load existing violations.
        nn_audit_path = split_dir / "near_neighbor_audit.json"
        old_audit = json.loads(nn_audit_path.read_text())
        violations = old_audit.get("violations", [])
        _log(name, f"  existing violations: {len(violations)}")

        # 2. Collect val/test record indices to remove.
        val_to_remove: set[int] = set()
        test_to_remove: set[int] = set()
        for v in violations:
            for role, idx in [(v.get("role_a"), v.get("index_a")), (v.get("role_b"), v.get("index_b"))]:
                if role == "val" and idx is not None:
                    val_to_remove.add(idx)
                elif role == "test" and idx is not None:
                    test_to_remove.add(idx)
        _log(name, f"  val_to_remove: {len(val_to_remove)}")
        _log(name, f"  test_to_remove: {len(test_to_remove)}")

        # 3. Back up original index files.
        manifest = json.loads((split_dir / "split_manifest.json").read_text())
        val_idx_path = Path(manifest["split"]["roles"]["val"]["idx_path"])
        test_idx_path = Path(manifest["split"]["roles"]["test"]["idx_path"])
        for p in [val_idx_path, test_idx_path]:
            bak = p.with_suffix(p.suffix + ".v1.bak")
            if not bak.exists():
                shutil.copy2(p, bak)
                _log(name, f"  backed up {p.name} -> {bak.name}")

        # 4. Filter val.idx and test.idx.
        old_val = _read_indices(val_idx_path)
        old_test = _read_indices(test_idx_path)
        new_val = [i for i in old_val if i not in val_to_remove]
        new_test = [i for i in old_test if i not in test_to_remove]
        _write_indices(val_idx_path, new_val)
        _write_indices(test_idx_path, new_test)
        _log(name, f"  val: {len(old_val)} -> {len(new_val)} (removed {len(old_val) - len(new_val)})")
        _log(name, f"  test: {len(old_test)} -> {len(new_test)} (removed {len(old_test) - len(new_test)})")

        # 5. Update excluded.idx.
        excluded_path = split_dir / "excluded.idx"
        old_excluded: set[int] = set()
        if excluded_path.exists():
            old_excluded = set(_read_indices(excluded_path))
        new_excluded = sorted(old_excluded | val_to_remove | test_to_remove)
        _write_indices(excluded_path, new_excluded)
        old_excluded_count = manifest.get("split", {}).get("excluded", {}).get("count", 0)
        _log(name, f"  excluded: {old_excluded_count} -> {len(new_excluded)} (added {len(new_excluded) - old_excluded_count})")

        # 6. Update manifest's excluded section (so promote_split_to_paper_eligible picks it up).
        manifest["split"]["excluded"] = {
            "idx_path": str(excluded_path),
            "count": len(new_excluded),
            "reason": "near_neighbor_violation_or_prior_exclusion",
        }
        (split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        # 7. Re-run near-neighbor audit.
        records_path = manifest["records"]["path"]
        records = load_records_jsonl(records_path)
        roles = {
            "train": _read_indices(Path(manifest["split"]["roles"]["train"]["idx_path"])),
            "val": new_val,
            "test": new_test,
        }
        t_nn = time.time()
        _log(name, "  re-running near-neighbor audit ...")
        nn_audit = exhaustive_cross_role_near_neighbor_audit(records, roles)
        _log(name, f"  nn_audit.passed={nn_audit['passed']}")
        _log(name, f"  nn_audit.stats={nn_audit['stats']}")
        _log(name, f"  nn_audit phase elapsed: {time.time() - t_nn:.1f}s")

        if nn_audit["passed"]:
            t_promo = time.time()
            _log(name, "  promoting to paper_eligible=true ...")
            try:
                summary = promote_split_to_paper_eligible(
                    split_dir,
                    near_neighbor_audit=nn_audit,
                    alias_audit=alias_audit,
                )
                _log(name, f"  promoted: paper_eligible={summary['paper_eligible']}")
                _log(name, f"  manifest_sha256={summary['manifest_sha256']}")
                _log(name, f"  promotion elapsed: {time.time() - t_promo:.1f}s")
                split_results[name] = summary
            except V2AuditError as exc:
                _log(name, f"  PROMOTION FAILED: {exc}")
                return 1
        else:
            _log(name, f"  REMEDIATION INCOMPLETE: {nn_audit['stats']['n_violations']} violations remain")
            return 1

        remediation_summary[name] = {
            "val_removed": len(val_to_remove),
            "test_removed": len(test_to_remove),
            "excluded_before": old_excluded_count,
            "excluded_after": len(new_excluded),
            "val_before": len(old_val),
            "val_after": len(new_val),
            "test_before": len(old_test),
            "test_after": len(new_test),
            "violations_before": len(violations),
            "violations_after": nn_audit["stats"]["n_violations"],
        }
        _log(name, f"  split total elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Update combined manifest.
    # ------------------------------------------------------------------
    t0 = time.time()
    _log("combined", "updating combined_reconstruction_manifest.json ...")
    combined_update = promote_combined_bundle(COMBINED_DIR, split_results)
    _log("combined", f"  paper_eligible={combined_update['paper_eligible']}")
    _log("combined", f"  block_reasons={combined_update['block_reasons']}")
    _log("combined", f"  manifest_sha256={combined_update['manifest_sha256']}")
    _log("combined", f"  elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Persist remediation result.
    # ------------------------------------------------------------------
    result = {
        "artifact_kind": "p0_data_reconstruction_v2_remediation_result",
        "combined_dir": str(COMBINED_DIR),
        "split_root": str(SPLIT_ROOT),
        "alias_audit_passed": alias_audit["passed"],
        "remediation_summary": remediation_summary,
        "split_results": split_results,
        "combined_manifest": combined_update,
        "total_elapsed_seconds": round(time.time() - t_start, 2),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _log("done", f"result written to {RESULT_PATH}")
    _log("done", f"total elapsed: {time.time() - t_start:.1f}s")

    n_promoted = sum(1 for r in split_results.values() if r.get("paper_eligible"))
    _log("done", f"promoted {n_promoted}/{len(SPLIT_NAMES)} splits to paper_eligible=true")
    if n_promoted != len(SPLIT_NAMES):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
