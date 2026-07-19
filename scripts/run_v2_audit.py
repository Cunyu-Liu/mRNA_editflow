"""Runner for the v2 audit on the actual frozen v1 namespace.

Logs per-phase timing to stdout and writes the final audit result to
``docs/p0_data_reconstruction_v2_audit_result.json``.

Invocation (from /home/cunyuliu/mrna_editflow_goal):
    PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
        /home/cunyuliu/miniconda3/envs/editflow/bin/python \
        mrna_editflow/scripts/run_v2_audit.py
"""
from __future__ import annotations

import json
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
    record_split_audit_outcome,
)
from mrna_editflow.data.split_contract import SPLIT_ROLES, sha256_file as contract_sha256_file


PROJECT_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow")
COMBINED_DIR = PROJECT_ROOT / "data/reconstructed/p0_data_reconstruction_v1/combined"
SPLIT_ROOT = PROJECT_ROOT / "benchmark/dev/p0_data_reconstruction_v1"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULT_PATH = DOCS_DIR / "p0_data_reconstruction_v2_audit_result.json"

SPLIT_NAMES = [
    "combined_family",
    "gencode_family",
    "refseq_family",
    "gencode_to_refseq",
]


def _log(phase: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{phase}] {msg}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_indices(path: Path) -> list[int]:
    with open(path, "r", encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()]


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    _log("init", f"combined_dir={COMBINED_DIR}")
    _log("init", f"split_root={SPLIT_ROOT}")

    # ------------------------------------------------------------------
    # Phase 1: load combined bundle
    # ------------------------------------------------------------------
    t0 = time.time()
    _log("load", "loading combined_model_view.records.jsonl ...")
    combined_records = load_records_jsonl(str(COMBINED_DIR / "combined_model_view.records.jsonl"))
    _log("load", f"  records: {len(combined_records)}")

    combined_meta = _load_jsonl(COMBINED_DIR / "combined_model_view.metadata.jsonl")
    _log("load", f"  metadata rows: {len(combined_meta)}")

    assignments = json.loads((COMBINED_DIR / "family_assignments.json").read_text())
    _log("load", f"  assignments: {len(assignments)}")

    family_evidence = _load_jsonl(COMBINED_DIR / "family_evidence.jsonl")
    _log("load", f"  family_evidence rows: {len(family_evidence)}")
    _log("load", f"  phase elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 2: gene-symbol alias audit on combined bundle (run once)
    # ------------------------------------------------------------------
    t0 = time.time()
    _log("alias", "running gene_symbol_alias_audit on combined bundle ...")
    alias_audit = gene_symbol_alias_audit(
        combined_records, combined_meta, assignments, family_evidence
    )
    alias_artifact_path = COMBINED_DIR / "combined_alias_audit.json"
    alias_artifact_path.write_text(
        json.dumps(alias_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    _log("alias", f"  passed={alias_audit['passed']}")
    _log("alias", f"  stats={alias_audit['stats']}")
    _log("alias", f"  artifact: {alias_artifact_path}")
    _log("alias", f"  phase elapsed: {time.time() - t0:.1f}s")

    if not alias_audit["passed"]:
        _log("alias", "ALIAS AUDIT FAILED — aborting before any promotion.")
        return 2

    # ------------------------------------------------------------------
    # Phase 3: per-split near-neighbor audit + promotion
    # ------------------------------------------------------------------
    split_results: dict[str, dict] = {}
    near_neighbor_audits: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        split_dir = SPLIT_ROOT / name
        _log(name, f"--- split {name} ---")
        t0 = time.time()
        manifest = json.loads((split_dir / "split_manifest.json").read_text())
        records_path = manifest["records"]["path"]
        records = load_records_jsonl(records_path)
        _log(name, f"  loaded {len(records)} records from {records_path}")

        roles: dict[str, list[int]] = {}
        for role in SPLIT_ROLES:
            idx_path = Path(manifest["split"]["roles"][role]["idx_path"])
            roles[role] = _read_indices(idx_path)
            _log(name, f"  role {role}: {len(roles[role])} indices")

        t_nn = time.time()
        _log(name, "  running exhaustive_cross_role_near_neighbor_audit ...")
        nn_audit = exhaustive_cross_role_near_neighbor_audit(records, roles)
        _log(name, f"  nn_audit.passed={nn_audit['passed']}")
        _log(name, f"  nn_audit.stats={nn_audit['stats']}")
        _log(name, f"  nn_audit phase elapsed: {time.time() - t_nn:.1f}s")
        near_neighbor_audits[name] = nn_audit

        if nn_audit["passed"] and alias_audit["passed"]:
            t_promo = time.time()
            _log(name, "  promoting split to paper_eligible=true ...")
            try:
                summary = promote_split_to_paper_eligible(
                    split_dir,
                    near_neighbor_audit=nn_audit,
                    alias_audit=alias_audit,
                )
                _log(name, f"  promoted: paper_eligible={summary['paper_eligible']}")
                _log(name, f"  manifest_sha256={summary['manifest_sha256']}")
                _log(name, f"  promotion phase elapsed: {time.time() - t_promo:.1f}s")
                split_results[name] = summary
            except V2AuditError as exc:
                _log(name, f"  PROMOTION REFUSED: {exc}")
                t_rec = time.time()
                _log(name, "  recording audit outcome without promotion ...")
                split_results[name] = record_split_audit_outcome(
                    split_dir,
                    near_neighbor_audit=nn_audit,
                    alias_audit=alias_audit,
                )
                _log(name, f"  recorded: paper_eligible={split_results[name]['paper_eligible']}")
                _log(name, f"  block_reasons={split_results[name]['block_reasons']}")
                _log(name, f"  record phase elapsed: {time.time() - t_rec:.1f}s")
        else:
            t_rec = time.time()
            _log(name, "  audits failed — recording outcome without promotion ...")
            split_results[name] = record_split_audit_outcome(
                split_dir,
                near_neighbor_audit=nn_audit,
                alias_audit=alias_audit,
            )
            _log(name, f"  recorded: paper_eligible={split_results[name]['paper_eligible']}")
            _log(name, f"  block_reasons={split_results[name]['block_reasons']}")
            _log(name, f"  record phase elapsed: {time.time() - t_rec:.1f}s")
        _log(name, f"  split total elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 4: update combined manifest
    # ------------------------------------------------------------------
    t0 = time.time()
    _log("combined", "updating combined_reconstruction_manifest.json ...")
    combined_update = promote_combined_bundle(COMBINED_DIR, split_results)
    _log("combined", f"  paper_eligible={combined_update['paper_eligible']}")
    _log("combined", f"  block_reasons={combined_update['block_reasons']}")
    _log("combined", f"  manifest_sha256={combined_update['manifest_sha256']}")
    _log("combined", f"  phase elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 5: persist final audit result
    # ------------------------------------------------------------------
    result = {
        "artifact_kind": "p0_data_reconstruction_v2_audit_result",
        "combined_dir": str(COMBINED_DIR),
        "split_root": str(SPLIT_ROOT),
        "alias_audit": alias_audit,
        "alias_audit_path": str(alias_artifact_path),
        "alias_audit_sha256": contract_sha256_file(alias_artifact_path),
        "near_neighbor_audits": near_neighbor_audits,
        "split_results": split_results,
        "combined_manifest": combined_update,
        "total_elapsed_seconds": round(time.time() - t_start, 2),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _log("done", f"result written to {RESULT_PATH}")
    _log("done", f"total elapsed: {time.time() - t_start:.1f}s")

    n_promoted = sum(1 for r in split_results.values() if r.get("paper_eligible"))
    n_failed = len(SPLIT_NAMES) - n_promoted
    _log("done", f"promoted {n_promoted}/{len(SPLIT_NAMES)} splits; {n_failed} recorded with violations")
    if n_failed > 0:
        _log("done", "near-neighbor violations found — see audit artifacts for details")
    return 0 if n_promoted == len(SPLIT_NAMES) else 0  # 0 = audits completed (regardless of outcome)


if __name__ == "__main__":
    sys.exit(main())
