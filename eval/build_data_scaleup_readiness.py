"""Build an honest data scale-up readiness report.

This report consolidates P3 data-scale evidence that otherwise lives across
manifests, queue status files, and roadmap notes. It is deliberately read-only:
queued jobs and parser/tooling readiness are reported as such, not promoted into
completed RefSeq, MPRA, leakage, or true data-scale claims.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "Data scale-up readiness is infrastructure and corpus-governance evidence. "
    "Do not claim RefSeq-scale training, real MPRA/TE/stability prediction, "
    "family-disjoint leakage-safe splits, or a data-scale law until the raw "
    "files, cleaned records, manifests, split/leakage audits, and downstream "
    "T1-T7/scale-law evaluations are complete."
)

GENCODE_MANIFEST = "data/processed/gencode_human_transcripts.data_manifest.json"
REFSEQ_MANIFEST = "data/processed/refseq_human_rna.data_manifest.json"
REFSEQ_RAW = "data/raw/human.1.rna.gbff.gz"
REFSEQ_RECORDS = "data/processed/refseq_human_rna.records.jsonl"
MPRA_CANDIDATES: tuple[str, ...] = (
    "data/processed/mpra_5utr.jsonl",
    "data/processed/mpra_5utr.data_manifest.json",
    "data/processed/mpra_mrl.jsonl",
    "data/processed/mpra_mrl.data_manifest.json",
    "data/processed/*mpra*.jsonl",
    "data/processed/*mpra*.json",
)
MPRA_TABLE_ENV_VARS: tuple[str, ...] = (
    "MRNA_EDITFLOW_MPRA_TE_TABLE",
    "MPRA_TE_TABLE",
    "MPRA_TABLE",
)
MPRA_TABLE_PATTERNS: tuple[str, ...] = (
    "data/raw/*mpra*.csv",
    "data/raw/*mpra*.tsv",
    "data/raw/*mrl*.csv",
    "data/raw/*mrl*.tsv",
    "data/processed/*mpra*.csv",
    "data/processed/*mpra*.tsv",
    "data/processed/*mrl*.csv",
    "data/processed/*mrl*.tsv",
)
STABILITY_TABLE_ENV_VARS: tuple[str, ...] = (
    "MRNA_EDITFLOW_STABILITY_TABLE",
    "STABILITY_TABLE",
    "HALF_LIFE_TABLE",
)
STABILITY_TARGET_ENV_VARS: tuple[str, ...] = (
    "MRNA_EDITFLOW_STABILITY_TARGET_COL",
    "STABILITY_TARGET_COL",
    "HALF_LIFE_TARGET_COL",
)
STABILITY_TABLE_PATTERNS: tuple[str, ...] = (
    "data/raw/*stability*.csv",
    "data/raw/*stability*.tsv",
    "data/raw/*half_life*.csv",
    "data/raw/*half_life*.tsv",
    "data/raw/*degradation*.csv",
    "data/raw/*degradation*.tsv",
    "data/processed/*stability*.csv",
    "data/processed/*stability*.tsv",
    "data/processed/*half_life*.csv",
    "data/processed/*half_life*.tsv",
    "data/processed/*degradation*.csv",
    "data/processed/*degradation*.tsv",
)
MPRA_PREDICTOR_REPORTS: tuple[str, ...] = (
    "benchmark/mpra_te_predictor*/report.json",
    "benchmark/mpra_te_predictor*.json",
)
STABILITY_PREDICTOR_REPORTS: tuple[str, ...] = (
    "benchmark/stability_predictor*/report.json",
    "benchmark/stability_predictor*.json",
)
FAMILY_LEAKAGE_PROTOCOL_REPORTS: tuple[str, ...] = (
    "benchmark/family_leakage_protocol*/report.json",
    "benchmark/family_leakage_protocol*.json",
    "benchmark/gencode_family_leakage_protocol/report.json",
    "benchmark/refseq_family_leakage_protocol/report.json",
)
DATASET_MANIFEST_AUDIT = "docs/dataset_manifest_audit.json"
STAGE_A_DOWNSTREAM_EVAL_AUDIT = "docs/stage_a_downstream_eval_readiness.json"


def _path(project_root: str, rel_or_abs: str) -> str:
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(project_root, rel_or_abs)


def _rel(project_root: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, project_root)
    except ValueError:
        return path


def _load_json_if_exists(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _read_jsonl_tail(path: str, n: int = 3) -> list[Mapping[str, object]]:
    if not os.path.exists(path):
        return []
    rows: list[Mapping[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, Mapping):
                rows.append(row)
    return rows[-max(0, int(n)) :]


def _progress_tail_from_declared_path(
    project_root: str,
    declared_path: object,
    *,
    n: int = 8,
) -> list[Mapping[str, object]]:
    path = _local_candidate_for_declared_path(project_root, declared_path)
    if not path:
        return []
    return _read_jsonl_tail(path, n=n)


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_candidate_for_declared_path(project_root: str, declared_path: object) -> Optional[str]:
    if not isinstance(declared_path, str) or not declared_path:
        return None
    if os.path.isabs(declared_path):
        if os.path.exists(declared_path):
            return declared_path
        marker = "/mrna_editflow/"
        if marker in declared_path:
            rel = declared_path.split(marker, 1)[1]
            return os.path.join(project_root, rel)
        return declared_path
    return os.path.join(project_root, declared_path)


def _file_audit(project_root: str, declared_path: object) -> dict[str, object]:
    local_path = _local_candidate_for_declared_path(project_root, declared_path)
    exists = bool(local_path and os.path.exists(local_path))
    return {
        "declared_path": declared_path if isinstance(declared_path, str) else None,
        "local_path": _rel(project_root, local_path) if local_path else None,
        "exists": exists,
        "size_bytes": os.path.getsize(local_path) if exists and local_path else None,
        "sha256": _sha256(local_path) if exists and local_path else None,
    }


def _table_candidate_paths(
    project_root: str,
    *,
    env_vars: Sequence[str],
    patterns: Sequence[str],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(path: str, source: str, label: str) -> None:
        candidate_path = _path(project_root, path) if not os.path.isabs(path) else path
        key = os.path.abspath(candidate_path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "path": _rel(project_root, candidate_path),
                "source": source,
                "source_label": label,
                "exists": os.path.exists(candidate_path),
                "size_bytes": os.path.getsize(candidate_path)
                if os.path.exists(candidate_path)
                else None,
                "sha256": _sha256(candidate_path),
            }
        )

    for env_name in env_vars:
        value = os.environ.get(env_name)
        if value:
            add(value, "env", env_name)
    for pattern in patterns:
        for path in sorted(glob.glob(_path(project_root, pattern))):
            add(path, "glob", pattern)
    return candidates


def _split_counts_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "other": 0, "missing": 0}
    for row in rows:
        split = str(row.get("split", "") or "").strip().lower()
        if not split:
            counts["missing"] += 1
        elif split in {"train", "val", "test"}:
            counts[split] += 1
        else:
            counts["other"] += 1
    return counts


def _empty_table_audit(
    dataset: str,
    *,
    env_vars: Sequence[str],
    patterns: Sequence[str],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "status": "no_candidate_table_configured",
        "ready_for_predictor_input": False,
        "schema_valid": False,
        "official_split_ready": False,
        "selected_table": None,
        "candidate_tables": [],
        "env_vars": list(env_vars),
        "candidate_patterns": list(patterns),
    }


def _audit_mpra_table(project_root: str) -> dict[str, object]:
    candidates = _table_candidate_paths(
        project_root,
        env_vars=MPRA_TABLE_ENV_VARS,
        patterns=MPRA_TABLE_PATTERNS,
    )
    if not candidates:
        return _empty_table_audit(
            "mpra_te",
            env_vars=MPRA_TABLE_ENV_VARS,
            patterns=MPRA_TABLE_PATTERNS,
        )

    audited: list[dict[str, object]] = []
    for candidate in candidates:
        row = dict(candidate)
        selected_path = _local_candidate_for_declared_path(project_root, row.get("path"))
        if not selected_path or not os.path.exists(selected_path):
            row.update(
                {
                    "schema_valid": False,
                    "official_split_ready": False,
                    "ready_for_predictor_input": False,
                    "parse_error": "candidate table does not exist locally",
                }
            )
            audited.append(row)
            continue
        try:
            from mrna_editflow.data.prepare_mpra import read_mpra_table

            rows = read_mpra_table(selected_path)
            split_counts = _split_counts_from_rows(rows)
            official_split_ready = bool(
                rows
                and split_counts["train"] > 0
                and split_counts["val"] > 0
                and split_counts["test"] > 0
                and split_counts["other"] == 0
                and split_counts["missing"] == 0
            )
            row.update(
                {
                    "schema_valid": bool(rows),
                    "n_rows": len(rows),
                    "split_counts": split_counts,
                    "sequence_length": {
                        "min": min(len(str(r.get("sequence", ""))) for r in rows)
                        if rows
                        else None,
                        "max": max(len(str(r.get("sequence", ""))) for r in rows)
                        if rows
                        else None,
                    },
                    "target_column": "mrl",
                    "official_split_ready": official_split_ready,
                    "ready_for_predictor_input": official_split_ready,
                    "parse_error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - message is surfaced in report.
            row.update(
                {
                    "schema_valid": False,
                    "official_split_ready": False,
                    "ready_for_predictor_input": False,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            )
        audited.append(row)

    selected = next(
        (row for row in audited if row.get("ready_for_predictor_input")),
        next((row for row in audited if row.get("schema_valid")), audited[0]),
    )
    if selected.get("ready_for_predictor_input"):
        status = "schema_and_official_split_ready"
    elif selected.get("schema_valid"):
        status = "schema_valid_missing_official_split"
    else:
        status = "candidate_table_invalid"
    return {
        "dataset": "mpra_te",
        "status": status,
        "ready_for_predictor_input": bool(selected.get("ready_for_predictor_input")),
        "schema_valid": bool(selected.get("schema_valid")),
        "official_split_ready": bool(selected.get("official_split_ready")),
        "selected_table": selected.get("path"),
        "candidate_tables": audited,
        "env_vars": list(MPRA_TABLE_ENV_VARS),
        "candidate_patterns": list(MPRA_TABLE_PATTERNS),
    }


def _stability_target_col_from_env() -> Optional[str]:
    for env_name in STABILITY_TARGET_ENV_VARS:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _audit_stability_table(project_root: str) -> dict[str, object]:
    candidates = _table_candidate_paths(
        project_root,
        env_vars=STABILITY_TABLE_ENV_VARS,
        patterns=STABILITY_TABLE_PATTERNS,
    )
    if not candidates:
        return _empty_table_audit(
            "stability_half_life",
            env_vars=STABILITY_TABLE_ENV_VARS,
            patterns=STABILITY_TABLE_PATTERNS,
        )

    target_col = _stability_target_col_from_env()
    audited: list[dict[str, object]] = []
    for candidate in candidates:
        row = dict(candidate)
        selected_path = _local_candidate_for_declared_path(project_root, row.get("path"))
        if not selected_path or not os.path.exists(selected_path):
            row.update(
                {
                    "schema_valid": False,
                    "official_split_ready": False,
                    "ready_for_predictor_input": False,
                    "parse_error": "candidate table does not exist locally",
                }
            )
            audited.append(row)
            continue
        try:
            from mrna_editflow.eval.stability_predictor import read_stability_table

            rows, selected_target = read_stability_table(
                selected_path,
                target_col=target_col,
            )
            split_counts = _split_counts_from_rows(rows)
            official_split_ready = bool(
                rows
                and split_counts["train"] > 0
                and split_counts["val"] > 0
                and split_counts["test"] > 0
                and split_counts["other"] == 0
                and split_counts["missing"] == 0
            )
            row.update(
                {
                    "schema_valid": bool(rows),
                    "n_rows": len(rows),
                    "split_counts": split_counts,
                    "sequence_length": {
                        "min": min(len(str(r.get("sequence", ""))) for r in rows)
                        if rows
                        else None,
                        "max": max(len(str(r.get("sequence", ""))) for r in rows)
                        if rows
                        else None,
                    },
                    "target_column": selected_target,
                    "official_split_ready": official_split_ready,
                    "ready_for_predictor_input": official_split_ready,
                    "parse_error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - message is surfaced in report.
            row.update(
                {
                    "schema_valid": False,
                    "official_split_ready": False,
                    "ready_for_predictor_input": False,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            )
        audited.append(row)

    selected = next(
        (row for row in audited if row.get("ready_for_predictor_input")),
        next((row for row in audited if row.get("schema_valid")), audited[0]),
    )
    if selected.get("ready_for_predictor_input"):
        status = "schema_and_official_split_ready"
    elif selected.get("schema_valid"):
        status = "schema_valid_missing_official_split"
    else:
        status = "candidate_table_invalid"
    return {
        "dataset": "stability_half_life",
        "status": status,
        "ready_for_predictor_input": bool(selected.get("ready_for_predictor_input")),
        "schema_valid": bool(selected.get("schema_valid")),
        "official_split_ready": bool(selected.get("official_split_ready")),
        "selected_table": selected.get("path"),
        "candidate_tables": audited,
        "env_vars": list(STABILITY_TABLE_ENV_VARS),
        "target_env_vars": list(STABILITY_TARGET_ENV_VARS),
        "target_col_override": target_col,
        "candidate_patterns": list(STABILITY_TABLE_PATTERNS),
    }


def _latest_status(project_root: str, pattern: str) -> tuple[Optional[str], Optional[Mapping[str, object]]]:
    candidates = glob.glob(_path(project_root, pattern))
    if not candidates:
        return None, None
    latest = max(candidates, key=os.path.getmtime)
    return latest, _load_json_if_exists(latest)


def _manifest_audit(project_root: str, rel_path: str, expected_dataset: str) -> dict[str, object]:
    path = _path(project_root, rel_path)
    payload = _load_json_if_exists(path)
    if payload is None:
        return {
            "dataset": expected_dataset,
            "manifest_path": rel_path,
            "manifest_exists": False,
            "manifest_ready": False,
            "status": "missing_manifest",
            "records": _file_audit(project_root, None),
        }

    dataset_payload = payload.get("dataset", {})
    dataset_name = (
        dataset_payload.get("name") if isinstance(dataset_payload, Mapping) else None
    )
    clean = payload.get("clean_summary", {})
    raw = payload.get("raw_summary", {})
    drops = payload.get("cleaning_drop_counts", {})
    records_sha = payload.get("records_sha256")
    records = _file_audit(project_root, payload.get("records_path"))
    sha_matches = (
        bool(records.get("exists"))
        and isinstance(records_sha, str)
        and records.get("sha256") == records_sha
    )
    clean_n = clean.get("n_records") if isinstance(clean, Mapping) else None
    manifest_ready = (
        dataset_name == expected_dataset
        and isinstance(records_sha, str)
        and len(records_sha) == 64
        and isinstance(clean_n, int)
        and clean_n > 0
    )
    if not manifest_ready:
        status = "manifest_incomplete"
    elif sha_matches:
        status = "manifest_and_local_records_verified"
    elif records.get("exists"):
        status = "manifest_ready_local_records_sha_mismatch"
    else:
        status = "manifest_ready_records_not_local"
    return {
        "dataset": expected_dataset,
        "manifest_path": rel_path,
        "manifest_exists": True,
        "manifest_sha256": _sha256(path),
        "manifest_ready": manifest_ready,
        "status": status,
        "dataset_name": dataset_name,
        "raw_n_records": raw.get("n_records") if isinstance(raw, Mapping) else None,
        "clean_n_records": clean_n,
        "drop_counts": drops if isinstance(drops, Mapping) else {},
        "records_sha256_declared": records_sha,
        "records_local_sha_matches_manifest": sha_matches,
        "records": records,
    }


def _refseq_build_audit(project_root: str) -> dict[str, object]:
    status_path, status = _latest_status(project_root, "benchmark/refseq_public_build*/status.json")
    if status is None:
        return {
            "status_path": None,
            "status_exists": False,
            "status": "missing",
            "official_corpus_ready": False,
            "raw": _file_audit(project_root, REFSEQ_RAW),
            "records": _file_audit(project_root, REFSEQ_RECORDS),
            "manifest": _manifest_audit(project_root, REFSEQ_MANIFEST, "refseq_human_rna"),
        }
    raw = status.get("raw", {}) if isinstance(status.get("raw"), Mapping) else {}
    records = status.get("records", {}) if isinstance(status.get("records"), Mapping) else {}
    manifest_status = status.get("manifest", {}) if isinstance(status.get("manifest"), Mapping) else {}
    progress = status.get("progress", {}) if isinstance(status.get("progress"), Mapping) else {}
    progress_tail = _progress_tail_from_declared_path(project_root, progress.get("path"))
    latest_progress = progress_tail[-1] if progress_tail else {}
    local_manifest = _manifest_audit(project_root, REFSEQ_MANIFEST, "refseq_human_rna")
    raw_exists = bool(raw.get("exists")) or bool(os.path.exists(_path(project_root, REFSEQ_RAW)))
    records_exists = bool(records.get("exists")) or bool(os.path.exists(_path(project_root, REFSEQ_RECORDS)))
    manifest_exists = bool(manifest_status.get("exists")) or bool(local_manifest.get("manifest_exists"))
    official_ready = bool(raw_exists and records_exists and manifest_exists)
    return {
        "status_path": _rel(project_root, status_path),
        "status_exists": True,
        "status_sha256": _sha256(status_path) if status_path else None,
        "status": status.get("status"),
        "claim_policy": status.get("claim_policy"),
        "official_corpus_ready": official_ready,
        "raw": {
            "path": raw.get("path", REFSEQ_RAW),
            "exists": raw_exists,
            "sha256": raw.get("sha256"),
            "size_bytes": raw.get("size_bytes"),
        },
        "records": {
            "path": records.get("path", REFSEQ_RECORDS),
            "exists": records_exists,
            "sha256": records.get("sha256"),
            "size_bytes": records.get("size_bytes"),
        },
        "manifest": local_manifest,
        "manifest_status": dict(manifest_status),
        "progress": {
            "path": progress.get("path"),
            "n_events": progress.get("n_events"),
            "last_event": latest_progress.get("event", progress.get("last_event")),
            "last_loadavg": latest_progress.get("loadavg", progress.get("last_loadavg")),
            "tail": progress_tail,
        },
    }


def _stage_a_sweep_audit(project_root: str) -> dict[str, object]:
    status_path, status = _latest_status(project_root, "benchmark/stage_a_scalelaw*/status.json")
    if status is None:
        plan_path, plan = _latest_status(project_root, "benchmark/stage_a_scalelaw*/plan.json")
        if plan is None:
            return {
                "status_path": None,
                "plan_ready": False,
                "controlled_sweep_complete": False,
                "status": "missing",
            }
        progress_path = os.path.join(os.path.dirname(plan_path), "progress.jsonl") if plan_path else ""
        progress_tail = _read_jsonl_tail(progress_path)
        last_event = progress_tail[-1] if progress_tail else {}
        return {
            "status_path": None,
            "plan_path": _rel(project_root, plan_path),
            "plan_ready": True,
            "controlled_sweep_complete": False,
            "status": "planned_no_status",
            "axes": plan.get("axes", {}) if isinstance(plan, Mapping) else {},
            "n_runs": plan.get("n_runs") if isinstance(plan, Mapping) else None,
            "n_complete": 0,
            "last_event": last_event.get("event") if isinstance(last_event, Mapping) else None,
            "last_loadavg": last_event.get("loadavg") if isinstance(last_event, Mapping) else None,
        }

    summary = status.get("summary", {}) if isinstance(status.get("summary"), Mapping) else {}
    plan = status.get("plan", {}) if isinstance(status.get("plan"), Mapping) else {}
    progress_tail = _progress_tail_from_declared_path(project_root, status.get("progress_path"))
    latest_progress = progress_tail[-1] if progress_tail else {}
    n_runs = summary.get("n_runs")
    n_complete = summary.get("n_complete")
    complete = bool(isinstance(n_runs, int) and n_runs > 0 and n_complete == n_runs)
    return {
        "status_path": _rel(project_root, status_path),
        "status_sha256": _sha256(status_path) if status_path else None,
        "plan_ready": True,
        "controlled_sweep_complete": complete,
        "status": "complete" if complete else "queued_or_running",
        "claim_policy": status.get("claim_policy"),
        "axes": plan.get("axes", {}),
        "n_runs": n_runs,
        "n_complete": n_complete,
        "n_incomplete": summary.get("n_incomplete"),
        "status_counts": summary.get("status_counts"),
        "last_event": latest_progress.get("event", summary.get("last_event")),
        "last_loadavg": latest_progress.get("loadavg", summary.get("last_loadavg")),
        "n_load_gate_wait_events": summary.get("n_load_gate_wait_events"),
        "progress_tail": progress_tail,
        "ready_for_scale_law_claim": False,
        "source_records_sha256": plan.get("source_records_sha256"),
        "source_record_count": plan.get("source_record_count"),
    }


def _stage_a_downstream_eval_audit(project_root: str) -> dict[str, object]:
    path = _path(project_root, STAGE_A_DOWNSTREAM_EVAL_AUDIT)
    payload = _load_json_if_exists(path)
    if payload is None:
        return {
            "path": STAGE_A_DOWNSTREAM_EVAL_AUDIT,
            "exists": False,
            "status": "missing_stage_a_downstream_eval_readiness",
            "ready_for_stage_a_downstream_eval_claim": False,
            "ready_for_true_scale_law_claim": False,
            "missing_or_incomplete": ["stage_a_downstream_eval_readiness_missing"],
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    return {
        "path": STAGE_A_DOWNSTREAM_EVAL_AUDIT,
        "exists": True,
        "sha256": _sha256(path),
        "status": summary.get("status"),
        "n_runs": summary.get("n_runs"),
        "n_training_complete": summary.get("n_training_complete"),
        "n_downstream_ready": summary.get("n_downstream_ready"),
        "aggregate_report_ready": bool(summary.get("aggregate_report_ready")),
        "trend_report_ready": bool(summary.get("trend_report_ready")),
        "ready_for_stage_a_downstream_eval_claim": bool(
            summary.get("ready_for_stage_a_downstream_eval_claim")
        ),
        "ready_for_true_scale_law_claim": bool(summary.get("ready_for_true_scale_law_claim")),
        "missing_or_incomplete": summary.get("missing_or_incomplete", []),
    }


def _dataset_manifest_contract_audit(project_root: str) -> dict[str, object]:
    path = _path(project_root, DATASET_MANIFEST_AUDIT)
    payload = _load_json_if_exists(path)
    if payload is None:
        return {
            "path": DATASET_MANIFEST_AUDIT,
            "exists": False,
            "all_required_dataset_manifests_complete": False,
            "incomplete_datasets": ["dataset_manifest_audit_missing"],
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    return {
        "path": DATASET_MANIFEST_AUDIT,
        "exists": True,
        "sha256": _sha256(path),
        "all_required_dataset_manifests_complete": bool(
            summary.get("all_required_dataset_manifests_complete")
        ),
        "n_complete": summary.get("n_complete"),
        "n_datasets": summary.get("n_datasets"),
        "incomplete_datasets": summary.get("incomplete_datasets", []),
    }


def _mpra_audit(project_root: str) -> dict[str, object]:
    module_path = _path(project_root, "data/prepare_mpra.py")
    predictor_module_path = _path(project_root, "eval/mpra_te_predictor.py")
    stability_module_path = _path(project_root, "eval/stability_predictor.py")
    manifest_builder_module_path = _path(project_root, "eval/build_downstream_table_manifest.py")
    mpra_input_table = _audit_mpra_table(project_root)
    stability_input_table = _audit_stability_table(project_root)
    found: list[str] = []
    for pattern in MPRA_CANDIDATES:
        found.extend(glob.glob(_path(project_root, pattern)))
    found_rel = sorted({_rel(project_root, path) or path for path in found})
    predictor_reports: list[str] = []
    for pattern in MPRA_PREDICTOR_REPORTS:
        predictor_reports.extend(glob.glob(_path(project_root, pattern)))
    predictor_reports = sorted(set(predictor_reports))
    latest_report_path = max(predictor_reports, key=os.path.getmtime) if predictor_reports else None
    latest_report = _load_json_if_exists(latest_report_path) if latest_report_path else None
    latest_summary = (
        latest_report.get("summary", {})
        if isinstance(latest_report, Mapping) and isinstance(latest_report.get("summary"), Mapping)
        else {}
    )
    te_predictor_audit_ready = bool(latest_summary.get("ready_for_mpra_te_predictor_audit"))
    synthetic_smoke_only = bool(latest_summary.get("synthetic_smoke_only"))
    stability_reports: list[str] = []
    for pattern in STABILITY_PREDICTOR_REPORTS:
        stability_reports.extend(glob.glob(_path(project_root, pattern)))
    stability_reports = sorted(set(stability_reports))
    latest_stability_path = (
        max(stability_reports, key=os.path.getmtime) if stability_reports else None
    )
    latest_stability = (
        _load_json_if_exists(latest_stability_path) if latest_stability_path else None
    )
    latest_stability_summary = (
        latest_stability.get("summary", {})
        if isinstance(latest_stability, Mapping)
        and isinstance(latest_stability.get("summary"), Mapping)
        else {}
    )
    stability_predictor_ready = bool(
        latest_stability_summary.get("ready_for_stability_predictor_audit")
    )
    stability_synthetic_smoke_only = bool(latest_stability_summary.get("synthetic_smoke_only"))
    mpra_table_ready = bool(mpra_input_table.get("ready_for_predictor_input"))
    stability_table_ready = bool(stability_input_table.get("ready_for_predictor_input"))
    downstream_tables_ready = bool(mpra_table_ready and stability_table_ready)
    real_data_ready = bool(downstream_tables_ready or found_rel)
    if downstream_tables_ready and te_predictor_audit_ready and stability_predictor_ready:
        status = "real_te_and_stability_predictor_ready"
    elif mpra_table_ready and te_predictor_audit_ready and not stability_predictor_ready:
        status = "real_te_predictor_ready_stability_missing"
    elif downstream_tables_ready:
        status = "real_te_stability_tables_ready_needs_predictor_audit"
    elif mpra_table_ready or stability_table_ready:
        status = "partial_real_downstream_table_ready_needs_pair_and_predictor_audit"
    elif (latest_report is not None and synthetic_smoke_only) or (
        latest_stability is not None and stability_synthetic_smoke_only
    ):
        status = "synthetic_predictor_smoke_only_no_real_data"
    elif real_data_ready:
        status = "real_data_present_needs_predictor_audit"
    else:
        status = "tooling_only_no_real_data"
    return {
        "tooling_available": os.path.exists(module_path),
        "tooling_path": "data/prepare_mpra.py",
        "predictor_tooling_available": os.path.exists(predictor_module_path),
        "predictor_tooling_path": "eval/mpra_te_predictor.py",
        "stability_tooling_available": os.path.exists(stability_module_path),
        "stability_tooling_path": "eval/stability_predictor.py",
        "manifest_builder_tooling_available": os.path.exists(manifest_builder_module_path),
        "manifest_builder_tooling_path": "eval/build_downstream_table_manifest.py",
        "real_data_artifacts": found_rel,
        "real_data_ready": real_data_ready,
        "mpra_input_table": mpra_input_table,
        "stability_input_table": stability_input_table,
        "mpra_input_table_ready": mpra_table_ready,
        "stability_input_table_ready": stability_table_ready,
        "downstream_input_tables_ready": downstream_tables_ready,
        "predictor_reports": sorted(_rel(project_root, path) or path for path in predictor_reports),
        "latest_predictor_report": _rel(project_root, latest_report_path),
        "te_predictor_audit_ready": te_predictor_audit_ready,
        "te_predictor_synthetic_smoke_only": synthetic_smoke_only,
        "stability_predictor_reports": sorted(
            _rel(project_root, path) or path for path in stability_reports
        ),
        "latest_stability_predictor_report": _rel(project_root, latest_stability_path),
        "stability_predictor_ready": stability_predictor_ready,
        "stability_predictor_synthetic_smoke_only": stability_synthetic_smoke_only,
        "status": status,
        "claim_language": (
            "Real MPRA/TE and stability input tables need schema-valid sequences, "
            "numeric labels, official train/val/test splits, manifests, and held-out "
            "predictor validation before any real TE/stability claim."
        ),
    }


def _split_leakage_audit(project_root: str, refseq_ready: bool) -> dict[str, object]:
    split_files: list[str] = []
    for pattern in (
        "data/**/train.idx",
        "benchmark/**/*/splits/train.idx",
    ):
        split_files.extend(glob.glob(_path(project_root, pattern), recursive=True))
    split_files = sorted(set(split_files))
    leakage_reports = [
        path
        for path in glob.glob(_path(project_root, "benchmark/**/*leakage*.json"), recursive=True)
    ]
    tooling_available = all(
        os.path.exists(_path(project_root, rel))
        for rel in ("data/dedup_split.py", "data/leakage_audit.py")
    )
    cross_corpus_reports = [
        _rel(project_root, path) or path
        for path in leakage_reports
        if "refseq" in os.path.basename(path).lower() or "gencode" in os.path.basename(path).lower()
    ]
    protocol_reports: list[str] = []
    for pattern in FAMILY_LEAKAGE_PROTOCOL_REPORTS:
        protocol_reports.extend(glob.glob(_path(project_root, pattern)))
    protocol_reports = sorted(set(protocol_reports))

    protocol_rows: list[tuple[str, Mapping[str, object], Mapping[str, object]]] = []
    for path in protocol_reports:
        payload = _load_json_if_exists(path)
        summary = (
            payload.get("summary", {})
            if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping)
            else {}
        )
        protocol_rows.append((path, payload or {}, summary))
    real_protocol_rows = [
        row for row in protocol_rows if not bool(row[2].get("synthetic_smoke_only"))
    ]
    preferred_protocol_rows = real_protocol_rows or protocol_rows
    latest_protocol_path = (
        max((row[0] for row in preferred_protocol_rows), key=os.path.getmtime)
        if preferred_protocol_rows
        else None
    )
    latest_protocol_summary = next(
        (summary for path, _payload, summary in protocol_rows if path == latest_protocol_path),
        {},
    )

    real_split_ready = any(
        bool(summary.get("external_records_provided"))
        and bool(summary.get("split_ready"))
        and not bool(summary.get("synthetic_smoke_only"))
        for _path_value, _payload, summary in protocol_rows
    )
    protocol_audit_ready = any(
        bool(summary.get("ready_for_family_leakage_audit"))
        and not bool(summary.get("synthetic_smoke_only"))
        for _path_value, _payload, summary in protocol_rows
    )
    protocol_smoke_only = bool(protocol_rows and not real_protocol_rows)
    cross_corpus_protocol_reports = [
        _rel(project_root, path) or path
        for path, _payload, summary in protocol_rows
        if "refseq" in path.lower()
        and bool(summary.get("external_reference_provided"))
        and not bool(summary.get("synthetic_smoke_only"))
    ]
    cross_corpus_reports = sorted(set(cross_corpus_reports + cross_corpus_protocol_reports))
    ready = bool(refseq_ready and split_files and cross_corpus_reports and protocol_audit_ready)
    if ready:
        status = "family_split_and_leakage_audit_present"
    elif not refseq_ready:
        if real_split_ready:
            status = "blocked_on_refseq_records_real_gencode_split_ready"
        elif protocol_smoke_only:
            status = "blocked_on_refseq_records_protocol_smoke_ready"
        else:
            status = "blocked_on_refseq_records"
    elif protocol_smoke_only:
        status = "protocol_smoke_only_real_audit_missing"
    elif real_split_ready and not protocol_audit_ready:
        status = "family_split_ready_cross_corpus_reference_missing"
    else:
        status = "tooling_ready_audit_missing"
    return {
        "tooling_available": tooling_available,
        "split_tool_path": "data/dedup_split.py",
        "leakage_tool_path": "data/leakage_audit.py",
        "split_files": sorted(_rel(project_root, path) or path for path in split_files),
        "leakage_reports": sorted(_rel(project_root, path) or path for path in leakage_reports),
        "cross_corpus_reports": sorted(cross_corpus_reports),
        "protocol_reports": sorted(_rel(project_root, path) or path for path in protocol_reports),
        "real_protocol_reports": sorted(
            _rel(project_root, path) or path for path, _payload, _summary in real_protocol_rows
        ),
        "latest_protocol_report": _rel(project_root, latest_protocol_path),
        "family_split_protocol_ready": real_split_ready,
        "protocol_audit_ready": protocol_audit_ready,
        "protocol_synthetic_smoke_only": protocol_smoke_only,
        "family_leakage_ready": ready,
        "status": status,
    }


def build_data_scaleup_readiness(project_root: str) -> dict[str, object]:
    project_root = os.path.abspath(project_root)
    gencode = _manifest_audit(project_root, GENCODE_MANIFEST, "gencode_human_transcripts")
    refseq_build = _refseq_build_audit(project_root)
    stage_a = _stage_a_sweep_audit(project_root)
    stage_a_downstream = _stage_a_downstream_eval_audit(project_root)
    dataset_manifest_contract = _dataset_manifest_contract_audit(project_root)
    mpra = _mpra_audit(project_root)
    split_leakage = _split_leakage_audit(project_root, bool(refseq_build.get("official_corpus_ready")))

    refseq_manifest = refseq_build.get("manifest", {})
    refseq_manifest_ready = (
        isinstance(refseq_manifest, Mapping) and bool(refseq_manifest.get("manifest_ready"))
    )
    manifest_rows = [
        {
            "dataset": "gencode_human_transcripts",
            "manifest_exists": gencode.get("manifest_exists"),
            "manifest_ready": gencode.get("manifest_ready"),
            "records_local_exists": (
                gencode.get("records", {}).get("exists")
                if isinstance(gencode.get("records"), Mapping)
                else False
            ),
            "status": gencode.get("status"),
        },
        {
            "dataset": "refseq_human_rna",
            "manifest_exists": refseq_manifest.get("manifest_exists") if isinstance(refseq_manifest, Mapping) else False,
            "manifest_ready": refseq_manifest_ready,
            "records_local_exists": refseq_build.get("records", {}).get("exists") if isinstance(refseq_build.get("records"), Mapping) else False,
            "status": refseq_build.get("status"),
        },
    ]
    missing_or_incomplete: list[str] = []
    if not gencode.get("manifest_ready"):
        missing_or_incomplete.append("gencode_manifest")
    if not refseq_build.get("official_corpus_ready"):
        missing_or_incomplete.append("refseq_official_corpus")
    if not stage_a.get("controlled_sweep_complete"):
        missing_or_incomplete.append("stage_a_data_model_step_sweep")
    if not stage_a_downstream.get("ready_for_stage_a_downstream_eval_claim"):
        missing_or_incomplete.append("stage_a_downstream_evaluation")
    if not (
        mpra.get("downstream_input_tables_ready")
        and mpra.get("te_predictor_audit_ready")
        and mpra.get("stability_predictor_ready")
    ):
        missing_or_incomplete.append("real_mpra_te_stability_data")
    if not split_leakage.get("family_leakage_ready"):
        missing_or_incomplete.append("family_disjoint_split_and_leakage_audit")
    if not all(row.get("manifest_ready") for row in manifest_rows) or not dataset_manifest_contract.get(
        "all_required_dataset_manifests_complete"
    ):
        missing_or_incomplete.append("all_dataset_manifests")

    rows = [
        {
            "area": "GENCODE base corpus",
            "status": gencode.get("status"),
            "ready": bool(gencode.get("manifest_ready")),
            "evidence": f"clean_n={gencode.get('clean_n_records')}; records_sha={gencode.get('records_sha256_declared')}",
            "claim_language": "GENCODE manifest/SHA evidence is present; local records availability is reported separately.",
        },
        {
            "area": "RefSeq corpus scale-up",
            "status": refseq_build.get("status"),
            "ready": bool(refseq_build.get("official_corpus_ready")),
            "evidence": (
                f"raw={refseq_build.get('raw', {}).get('exists') if isinstance(refseq_build.get('raw'), Mapping) else None}; "
                f"records={refseq_build.get('records', {}).get('exists') if isinstance(refseq_build.get('records'), Mapping) else None}; "
                f"manifest={refseq_manifest_ready}; "
                f"last_event={refseq_build.get('progress', {}).get('last_event') if isinstance(refseq_build.get('progress'), Mapping) else None}"
            ),
            "claim_language": "RefSeq parser/build queue readiness only until official raw, records, and manifest exist.",
        },
        {
            "area": "Stage A data/model/step sweep",
            "status": stage_a.get("status"),
            "ready": bool(stage_a.get("controlled_sweep_complete")),
            "evidence": (
                f"runs={stage_a.get('n_complete')}/{stage_a.get('n_runs')}; "
                f"last_event={stage_a.get('last_event')}; last_loadavg={stage_a.get('last_loadavg')}"
            ),
            "claim_language": "Controlled axes are queue evidence until all runs and downstream audits complete.",
        },
        {
            "area": "Stage A downstream evaluation",
            "status": stage_a_downstream.get("status"),
            "ready": bool(stage_a_downstream.get("ready_for_stage_a_downstream_eval_claim")),
            "evidence": (
                f"training={stage_a_downstream.get('n_training_complete')}/"
                f"{stage_a_downstream.get('n_runs')}; "
                f"downstream={stage_a_downstream.get('n_downstream_ready')}/"
                f"{stage_a_downstream.get('n_runs')}; "
                f"aggregate={stage_a_downstream.get('aggregate_report_ready')}; "
                f"trend={stage_a_downstream.get('trend_report_ready')}"
            ),
            "claim_language": "Stage A checkpoints need proposal-ranking, T1-T7 aggregate, runtime, and trend audits before any scale-law claim.",
        },
        {
            "area": "MPRA/TE/stability data",
            "status": mpra.get("status"),
            "ready": bool(
                mpra.get("downstream_input_tables_ready")
                and mpra.get("te_predictor_audit_ready")
                and mpra.get("stability_predictor_ready")
            ),
            "evidence": (
                f"tooling={mpra.get('tooling_available')}; "
                f"predictor_tooling={mpra.get('predictor_tooling_available')}; "
                f"stability_tooling={mpra.get('stability_tooling_available')}; "
                f"manifest_builder={mpra.get('manifest_builder_tooling_available')}; "
                f"mpra_table={mpra.get('mpra_input_table_ready')}; "
                f"stability_table={mpra.get('stability_input_table_ready')}; "
                f"real_artifacts={len(mpra.get('real_data_artifacts', []))}; "
                f"te_predictor_audit={mpra.get('te_predictor_audit_ready')}; "
                f"stability_predictor_audit={mpra.get('stability_predictor_ready')}; "
                f"synthetic_smoke=({mpra.get('te_predictor_synthetic_smoke_only')}, "
                f"{mpra.get('stability_predictor_synthetic_smoke_only')})"
            ),
            "claim_language": "No real TE/stability predictor claim without external data and validation artifacts.",
        },
        {
            "area": "Family-disjoint split and leakage",
            "status": split_leakage.get("status"),
            "ready": bool(split_leakage.get("family_leakage_ready")),
            "evidence": (
                f"split_files={len(split_leakage.get('split_files', []))}; "
                f"cross_corpus_reports={len(split_leakage.get('cross_corpus_reports', []))}; "
                f"protocol_reports={len(split_leakage.get('protocol_reports', []))}; "
                f"real_protocol_reports={len(split_leakage.get('real_protocol_reports', []))}; "
                f"split_protocol_ready={split_leakage.get('family_split_protocol_ready')}; "
                f"protocol_smoke={split_leakage.get('protocol_synthetic_smoke_only')}"
            ),
            "claim_language": "Leakage tooling exists, but GENCODE/RefSeq family-disjoint evidence awaits RefSeq records.",
        },
        {
            "area": "Dataset manifests",
            "status": (
                "complete"
                if dataset_manifest_contract.get("all_required_dataset_manifests_complete")
                else "incomplete"
            ),
            "ready": bool(dataset_manifest_contract.get("all_required_dataset_manifests_complete")),
            "evidence": (
                "; ".join(f"{row['dataset']}={row['status']}" for row in manifest_rows)
                + f"; contract={dataset_manifest_contract.get('n_complete')}/"
                f"{dataset_manifest_contract.get('n_datasets')}; "
                f"incomplete={dataset_manifest_contract.get('incomplete_datasets')}"
            ),
            "claim_language": "Every data version needs source URL, SHA256, record counts, drop stats, and split stats.",
        },
    ]
    return {
        "artifact_kind": "data_scaleup_readiness",
        "project_root": project_root,
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "ready_for_data_scale_claim": False,
            "ready_for_refseq_scaleup_claim": False,
            "ready_for_real_te_or_stability_claim": False,
            "ready_for_family_disjoint_leakage_claim": bool(split_leakage.get("family_leakage_ready")),
            "ready_for_true_scale_law_claim": False,
            "all_data_scaleup_ready": False,
            "gencode_manifest_ready": bool(gencode.get("manifest_ready")),
            "gencode_records_local_exists": (
                gencode.get("records", {}).get("exists")
                if isinstance(gencode.get("records"), Mapping)
                else False
            ),
            "refseq_official_corpus_ready": bool(refseq_build.get("official_corpus_ready")),
            "refseq_build_status": refseq_build.get("status"),
            "stage_a_controlled_sweep_complete": bool(stage_a.get("controlled_sweep_complete")),
            "stage_a_controlled_sweep_status": stage_a.get("status"),
            "stage_a_downstream_eval_ready": bool(
                stage_a_downstream.get("ready_for_stage_a_downstream_eval_claim")
            ),
            "stage_a_downstream_eval_status": stage_a_downstream.get("status"),
            "stage_a_downstream_training_complete": stage_a_downstream.get(
                "n_training_complete"
            ),
            "stage_a_downstream_ready_runs": stage_a_downstream.get("n_downstream_ready"),
            "mpra_real_data_ready": bool(mpra.get("real_data_ready")),
            "downstream_manifest_builder_tooling_available": bool(
                mpra.get("manifest_builder_tooling_available")
            ),
            "mpra_input_table_ready": bool(mpra.get("mpra_input_table_ready")),
            "stability_input_table_ready": bool(mpra.get("stability_input_table_ready")),
            "downstream_input_tables_ready": bool(mpra.get("downstream_input_tables_ready")),
            "mpra_te_predictor_audit_ready": bool(mpra.get("te_predictor_audit_ready")),
            "mpra_te_predictor_synthetic_smoke_only": bool(mpra.get("te_predictor_synthetic_smoke_only")),
            "stability_predictor_audit_ready": bool(mpra.get("stability_predictor_ready")),
            "stability_predictor_synthetic_smoke_only": bool(
                mpra.get("stability_predictor_synthetic_smoke_only")
            ),
            "family_leakage_ready": bool(split_leakage.get("family_leakage_ready")),
            "family_split_protocol_ready": bool(
                split_leakage.get("family_split_protocol_ready")
            ),
            "family_leakage_protocol_real_report_present": bool(
                split_leakage.get("real_protocol_reports")
            ),
            "family_leakage_protocol_audit_ready": bool(
                split_leakage.get("protocol_audit_ready")
            ),
            "family_leakage_protocol_synthetic_smoke_only": bool(
                split_leakage.get("protocol_synthetic_smoke_only")
            ),
            "dataset_manifest_contract_ready": bool(
                dataset_manifest_contract.get("all_required_dataset_manifests_complete")
            ),
            "n_missing_or_incomplete": len(missing_or_incomplete),
            "missing_or_incomplete": missing_or_incomplete,
        },
        "gencode_manifest_audit": gencode,
        "refseq_build_audit": refseq_build,
        "stage_a_sweep_audit": stage_a,
        "stage_a_downstream_eval_audit": stage_a_downstream,
        "mpra_te_stability_audit": mpra,
        "family_split_leakage_audit": split_leakage,
        "manifest_audit": {
            "rows": manifest_rows,
            "contract_audit": dataset_manifest_contract,
        },
        "rows": rows,
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Data Scale-Up Readiness\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Ready for data scale claim: `{summary.get('ready_for_data_scale_claim')}`; "
            f"RefSeq corpus ready: `{summary.get('refseq_official_corpus_ready')}`; "
            f"real TE/stability ready: `{summary.get('ready_for_real_te_or_stability_claim')}`; "
            f"family leakage ready: `{summary.get('family_leakage_ready')}`\n"
        )
        fh.write(
            f"- GENCODE manifest ready: `{summary.get('gencode_manifest_ready')}`; "
            f"GENCODE local records exist: `{summary.get('gencode_records_local_exists')}`; "
            f"Stage A sweep status: `{summary.get('stage_a_controlled_sweep_status')}`\n"
        )
        fh.write(
            f"- Missing or incomplete: `{summary.get('missing_or_incomplete')}`\n\n"
        )
        fh.write("| Area | Status | Ready | Evidence | Claim language |\n")
        fh.write("|---|---|---|---|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('area')} | {row.get('status')} | `{row.get('ready')}` | "
                f"{row.get('evidence')} | {row.get('claim_language')} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/data_scaleup_readiness.json")
    parser.add_argument("--out-md", default="docs/data_scaleup_readiness.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_data_scaleup_readiness(project_root)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_data_scaleup_readiness",
    "write_report_json",
    "write_report_markdown",
    "main",
]
