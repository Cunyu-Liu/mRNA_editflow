#!/usr/bin/env python3
"""Record a fail-closed GP0 preflight without starting training.

This utility is deliberately a preflight recorder, not a GP0 trainer.  It
reads metadata and aggregate counts only, binds the current D1/B0/foundation
inputs, records GPU/process state, and writes an auditable WAITING marker when
the upstream data binding or formal GP0 infrastructure is incomplete.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


CONTRACT_ID = "utr_editflow_goal_v2"
CONTRACT_SHA256 = "3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791"
SCIENTIFIC_QUESTION_ID = "RQ-UTR-EDITFLOW-V2"
FORBIDDEN_NEW_LABEL_ACCESSIONS = {"GSE246381"}
REQUIRED_GP0_ENTRYPOINTS = (
    "scripts/gp0/train_gp0.py",
    "scripts/gp0/evaluate_gp0.py",
)


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except OSError as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(error),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    stat = path.stat()
    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": known_sha256 or _sha256(path),
            "sha256_source": "supplied_prior_verified_binding"
            if known_sha256
            else "computed_during_preflight",
        }
    )
    return record


def _scan_jsonl_records(path: Path) -> dict[str, Any]:
    total = 0
    paired = 0
    malformed = 0
    by_accession: collections.Counter[str] = collections.Counter()
    paired_by_accession: collections.Counter[str] = collections.Counter()
    by_region: collections.Counter[str] = collections.Counter()
    paired_by_region: collections.Counter[str] = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            accession = str(row.get("accession", "UNKNOWN"))
            region = str(row.get("region", "UNKNOWN"))
            by_accession[accession] += 1
            by_region[region] += 1
            if row.get("source_sequence") is not None and row.get(
                "candidate_sequence"
            ) is not None:
                paired += 1
                paired_by_accession[accession] += 1
                paired_by_region[region] += 1
    return {
        "total_records": total,
        "paired_records": paired,
        "non_paired_records": total - paired,
        "malformed_records": malformed,
        "accessions": len(by_accession),
        "by_accession": dict(sorted(by_accession.items())),
        "paired_by_accession": dict(sorted(paired_by_accession.items())),
        "by_region": dict(sorted(by_region.items())),
        "paired_by_region": dict(sorted(paired_by_region.items())),
    }


def _scan_exposure_ledger(path: Path) -> dict[str, Any]:
    policy: dict[str, set[tuple[Any, ...]]] = collections.defaultdict(set)
    selected = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            accession = str(row.get("accession", "UNKNOWN"))
            key = (
                row.get("data_role"),
                row.get("evidence_grade"),
                row.get("exposure_status"),
                bool(row.get("historically_exposed")),
                bool(row.get("labels_allowed_for_new_training")),
                bool(row.get("labels_allowed_for_new_hyperparameter_selection")),
            )
            policy[accession].add(key)
            if accession in FORBIDDEN_NEW_LABEL_ACCESSIONS:
                selected.add((accession, key))
    return {
        "accession_policy": {
            accession: [list(item) for item in sorted(values, key=str)]
            for accession, values in sorted(policy.items())
        },
        "forbidden_contract_accession_rows": [
            [accession, list(key)] for accession, key in sorted(selected, key=str)
        ],
    }


def _scan_split_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    study = payload.get("study_disjoint", {})
    by_accession = study.get("by_accession_split", {})
    covered = {
        accession: sum(int(value) for value in splits.values())
        for accession, splits in by_accession.items()
    }
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256(path),
        "study_disjoint_n_total": study.get("n_total"),
        "study_disjoint_by_accession": covered,
        "all_split_summaries": payload,
    }


def _gpu_snapshot() -> dict[str, Any]:
    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    torch_audit: dict[str, Any]
    try:
        import torch

        torch_audit = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None,
        }
    except Exception as error:  # pragma: no cover - environment dependent
        torch_audit = {"import_or_cuda_error": repr(error)}
    return {
        "nvidia_smi_gpu": gpu_query,
        "nvidia_smi_compute_apps": process_query,
        "torch": torch_audit,
        "formal_training_started": False,
        "safe_exclusive_gpu_confirmed": False,
    }


def _disk_snapshot(paths: Iterable[Path]) -> dict[str, Any]:
    result = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            result[str(path)] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        except OSError as error:
            result[str(path)] = {"error": repr(error)}
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_checksums(root: Path) -> str:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "artifact_checksums.sha256":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(root)}")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument("--exposure-ledger-sha256", required=True)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--b0-card", type=Path, required=True)
    parser.add_argument("--foundation-manifest", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-sha256", required=True)
    args = parser.parse_args()

    started = time.time()
    root = args.run_root
    root.mkdir(parents=True, exist_ok=False)

    git_head = _run(["git", "rev-parse", "HEAD"], cwd=args.repo)
    git_status = _run(["git", "status", "--short", "--branch"], cwd=args.repo)
    canonical = _scan_jsonl_records(args.data)
    exposure = _scan_exposure_ledger(args.exposure_ledger)
    split_summary = _scan_split_summary(args.split_summary)
    b0_card = json.loads(args.b0_card.read_text(encoding="utf-8"))
    foundation = json.loads(args.foundation_manifest.read_text(encoding="utf-8"))

    b0_unique = {
        track: card.get("counts", {}).get("unique_records")
        for track, card in b0_card.get("track_cards", {}).items()
    }
    paired_by_accession = canonical["paired_by_accession"]
    covered_by_accession = split_summary.get("study_disjoint_by_accession", {})
    missing_by_accession = {
        accession: count - int(covered_by_accession.get(accession, 0))
        for accession, count in paired_by_accession.items()
        if count != int(covered_by_accession.get(accession, 0))
    }
    forbidden_rows = exposure["forbidden_contract_accession_rows"]
    entrypoints = {
        relative: {
            "path": str(args.repo / relative),
            "exists": (args.repo / relative).exists(),
        }
        for relative in REQUIRED_GP0_ENTRYPOINTS
    }

    blockers = []
    if canonical["paired_records"] != 134059 or canonical["accessions"] != 11:
        blockers.append(
            {
                "id": "D1_CURRENT_COUNT_MISMATCH",
                "expected": {"paired_records": 134059, "accessions": 11},
                "observed": {
                    "paired_records": canonical["paired_records"],
                    "accessions": canonical["accessions"],
                },
            }
        )
    if any(value != 134059 for value in b0_unique.values()):
        blockers.append(
            {
                "id": "B0_SPLIT_CARD_NOT_BOUND_TO_CURRENT_D1",
                "current_d1_paired": canonical["paired_records"],
                "b0_card_unique_records": b0_unique,
                "missing_by_accession_against_study_split": missing_by_accession,
            }
        )
    if forbidden_rows:
        blockers.append(
            {
                "id": "GSE246381_HIGH_LEVEL_POLICY_CONFLICT",
                "contract_rule": "labels_allowed_for_new_training=false and labels_allowed_for_new_hyperparameter_selection=false",
                "ledger_rows": forbidden_rows,
            }
        )
    if not all(item["exists"] for item in entrypoints.values()):
        blockers.append(
            {
                "id": "GP0_FORMAL_ENTRYPOINT_MISSING",
                "entrypoints": entrypoints,
            }
        )
    blockers.append(
        {
            "id": "GPU_SAFE_SLOT_NOT_CONFIRMED",
            "reason": "existing compute processes are present; no formal GP0 launch is permitted until a non-interfering resource allocation is recorded",
        }
    )

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_manifest = {
        "schema_version": "gp0_data_binding_v1",
        "d1_canonical_records": _file_binding(args.data, known_sha256=args.data_sha256),
        "exposure_ledger": _file_binding(
            args.exposure_ledger, known_sha256=args.exposure_ledger_sha256
        ),
        "d1_aggregate_scan": canonical,
        "final_labels_accessed": False,
        "gse246381_new_label_policy": "FORBIDDEN_BY_HIGH_LEVEL_CONTRACT",
    }
    split_manifest = {
        "schema_version": "gp0_split_binding_v1",
        "b0_card": _file_binding(args.b0_card),
        "split_summary": split_summary,
        "b0_card_unique_records": b0_unique,
        "current_d1_paired_records_not_covered_by_study_split": missing_by_accession,
        "final_labels_accessed": False,
    }
    foundation_manifest = {
        "schema_version": "gp0_foundation_binding_v1",
        "fm0_manifest_path": str(args.foundation_manifest),
        "fm0_manifest_sha256": _sha256(args.foundation_manifest),
        "model_id": foundation.get("model_id"),
        "revision": foundation.get("revision"),
        "checkpoint_sha256": args.foundation_checkpoint_sha256,
        "checkpoint_sha256_in_fm0": foundation.get("checkpoint_sha256"),
        "license": foundation.get("license"),
        "exact_sequence_overlap": "NOT_AVAILABLE_NOT_ASSERTED",
    }
    code_manifest = {
        "repo": str(args.repo),
        "git_head_stdout": git_head["stdout"].strip(),
        "git_status": git_status,
        "entrypoints": entrypoints,
    }
    preflight = {
        "schema_version": "gp0_preflight_v1",
        "run_id": root.name,
        "generated_at_utc": now,
        "goal_contract": {
            "id": CONTRACT_ID,
            "sha256": CONTRACT_SHA256,
        },
        "scientific_question_id": SCIENTIFIC_QUESTION_ID,
        "phase_id": "GP0",
        "task_id": "GP0-01",
        "parent_run_id": "EF0_true_utr_ctmc_gpu_20260803T032203Z_4f4d124_s20260803",
        "data": data_manifest,
        "split": split_manifest,
        "foundation": foundation_manifest,
        "code": code_manifest,
        "gpu": _gpu_snapshot(),
        "disk": _disk_snapshot((Path("/home"), Path("/mnt"))),
        "blockers": blockers,
        "formal_training_started": False,
        "final_labels_accessed": False,
        "paper_eligibility": False,
        "state": "WAITING_FOR_UPSTREAM_DATA_SYNC_AND_GP0_INFRASTRUCTURE",
        "evidence_level": "E0_PRETRAINING_NOT_STARTED",
    }

    _write_json(root / "provenance/data_manifest.json", data_manifest)
    _write_json(root / "provenance/split_manifest.json", split_manifest)
    _write_json(root / "provenance/foundation_manifest.json", foundation_manifest)
    _write_json(root / "provenance/code_manifest.json", code_manifest)
    _write_json(root / "preflight/gp0_preflight.json", preflight)
    _write_json(root / "failure/gp0_preflight_stop.json", {
        "status": "NOT_STARTED",
        "state": preflight["state"],
        "stop_rule": "do not launch formal GP0 until data/split/policy/infrastructure blockers are resolved",
        "blockers": blockers,
    })

    run_manifest = {
        "schema_version": "gp0-run-manifest/v1",
        "run_id": root.name,
        "phase_id": "GP0",
        "task_id": "GP0-01",
        "parent_run_id": preflight["parent_run_id"],
        "goal_contract": preflight["goal_contract"],
        "scientific_question_id": SCIENTIFIC_QUESTION_ID,
        "git_commit": code_manifest["git_head_stdout"],
        "data_manifest_sha256": _sha256(root / "provenance/data_manifest.json"),
        "split_manifest_sha256": _sha256(root / "provenance/split_manifest.json"),
        "foundation_checkpoint": f"{foundation.get('model_id')}@{foundation.get('revision')}",
        "foundation_checkpoint_sha256": args.foundation_checkpoint_sha256,
        "exposure_ledger_version": args.exposure_ledger_sha256,
        "seed": None,
        "training_manifest_complete": False,
        "training_started": False,
        "gpu_formal_verified": False,
        "state": preflight["state"],
        "evidence_level": preflight["evidence_level"],
        "paper_eligibility": False,
        "final_labels_accessed": False,
        "known_deviations": blockers,
        "created_at_utc": now,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(root / "run_manifest.json", run_manifest)
    _write_json(root / "summary/summary.json", {
        "run_id": root.name,
        "status": "NOT_STARTED",
        "state": preflight["state"],
        "current_d1_paired_records": canonical["paired_records"],
        "current_d1_accessions": canonical["accessions"],
        "b0_card_unique_records": b0_unique,
        "blocker_count": len(blockers),
        "formal_claims": [],
        "scientific_claim_boundary": "No GP0 training or scientific conclusion was established.",
    })
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs/preflight.log").write_text(
        json.dumps(
            {
                "started_at_epoch": started,
                "finished_at_epoch": time.time(),
                "run_id": root.name,
                "state": preflight["state"],
                "blocker_ids": [item["id"] for item in blockers],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "WAITING_FOR_UPSTREAM_DATA_SYNC_AND_GP0_INFRASTRUCTURE").write_text(
        "Formal GP0 was not started. See run_manifest.json and failure/gp0_preflight_stop.json.\n",
        encoding="utf-8",
    )
    (root / "artifact_checksums.sha256").write_text(
        _artifact_checksums(root), encoding="utf-8"
    )
    print(json.dumps({
        "run_root": str(root),
        "state": preflight["state"],
        "paired_records": canonical["paired_records"],
        "accessions": canonical["accessions"],
        "b0_card_unique_records": b0_unique,
        "blocker_ids": [item["id"] for item in blockers],
        "formal_training_started": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
