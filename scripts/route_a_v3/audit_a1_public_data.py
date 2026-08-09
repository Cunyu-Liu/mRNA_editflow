#!/usr/bin/env python3
"""Build a non-sensitive A1 gap inventory for ordinary public datasets.

This collector is deliberately conservative.  It summarizes counts and field
coverage from legacy ordinary canonical records, but it cannot qualify a study
or support a scientific claim.  Qualification requires newly built V3 records,
paper-faithful transforms, group/leakage/power audits, exposure evidence and a
license decision.  Restricted or sealed paths are rejected before any read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


DELTA_LABEL_KEYS: Mapping[str, Sequence[str]] = {
    "GSE149487": ("te_log_fold_change", "transcript_log_fold_change"),
    "ENCSR854RUF": (
        "log2FoldChange_Skew_GM12878",
        "log2FoldChange_Skew_HEK293FT",
        "log2FoldChange_Skew_HEPG2",
        "log2FoldChange_Skew_HMEC",
        "log2FoldChange_Skew_K562",
        "log2FoldChange_Skew_SKNSH",
    ),
    "GSE232572": ("log2fc_activity",),
    "GSE186455": ("n2a_log2fc_activity", "vglut_log2fc_activity"),
}

GENE_KEYS = ("gene", "gene_id", "gene_symbol", "genes")
EXPLICIT_GROUP_KEYS = (
    "biological_source_group_id",
    "biological_parent_id",
    "mother",
    "mother_id",
    "wt_id",
    "transcript_id",
    "transcript_accession",
    "locus_id",
    "design_family_id",
)
RAW_LOCATOR_KEYS = ("raw_record_locator", "raw_row", "row_index", "source_row")
REPLICATE_KEY = re.compile(r"(?:^|_)rep(?:licate)?\d+$", re.IGNORECASE)
SE_KEY = re.compile(r"(?:^|_)(?:se|stderr|std_error)(?:$|_)|lfcse", re.IGNORECASE)


class OrdinaryScopeError(RuntimeError):
    """Raised before I/O when an input escapes the ordinary-public scope."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sequence(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper().replace("T", "U")
    if not normalized or any(base not in "ACGUN" for base in normalized):
        return None
    return normalized


def _sequence_digest(value: Any) -> Optional[bytes]:
    sequence = _canonical_sequence(value)
    if sequence is None:
        return None
    return hashlib.sha256(sequence.encode("ascii")).digest()


def _safe_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or None
    if isinstance(value, (list, tuple, set)):
        values = sorted({str(item).strip() for item in value if str(item).strip()})
        return "|".join(values) if values else None
    return None


def ensure_ordinary_path(path: Path, forbidden_tokens: Iterable[str]) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve(strict=False)
    lowered = str(resolved).lower()
    matched = sorted({token for token in forbidden_tokens if token.lower() in lowered})
    if matched:
        raise OrdinaryScopeError(
            "A1 ordinary-public audit rejected path before read; forbidden token(s): "
            + ",".join(matched)
        )
    return resolved


def load_protocol(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("contract_id") != "mrna_xeditflow_route_a_v3":
        raise ValueError("unexpected A1 protocol contract_id")
    scope = protocol.get("scope") or {}
    included = list(scope.get("included_dataset_ids") or [])
    excluded = set(scope.get("excluded_dataset_ids") or [])
    if not included or len(included) != len(set(included)):
        raise ValueError("included_dataset_ids must be non-empty and unique")
    if set(included) & excluded:
        raise ValueError("included and excluded dataset ids overlap")
    if scope.get("training_allowed") is not False:
        raise ValueError("A1 audit protocol must keep training disabled")
    if scope.get("model_selection_allowed") is not False:
        raise ValueError("A1 audit protocol must keep model selection disabled")
    if (protocol.get("legacy_canonical_policy") or {}).get("may_auto_qualify_study") is not False:
        raise ValueError("legacy canonical records may not auto-qualify a study")
    return protocol


def _manifest_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = manifest.get("files")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, Mapping)]
    samples = manifest.get("samples")
    if isinstance(samples, list):
        return [entry for entry in samples if isinstance(entry, Mapping)]
    return []


def summarize_p0_manifest(
    dataset_id: str,
    p0_root: Path,
    *,
    verify_file_hashes: bool,
) -> Dict[str, Any]:
    dataset_dir = p0_root / dataset_id
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return {
            "manifest_path": str(manifest_path),
            "manifest_present": False,
            "manifest_sha256": None,
            "declared_files": 0,
            "present_files": 0,
            "files_with_declared_sha256": 0,
            "verified_file_hashes": 0,
            "hash_verification_requested": verify_file_hashes,
            "hash_mismatches": [],
            "missing_files": [],
            "unlisted_payload_files": [],
            "quarantined_extra_files": [],
            "license_field_present": False,
            "status": "MISSING_BLOCKED",
        }
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = _manifest_entries(manifest)
    present = 0
    declared_sha = 0
    verified = 0
    declared_names: set[str] = set()
    mismatches: list[str] = []
    missing: list[str] = []
    for entry in entries:
        name = entry.get("name") or entry.get("filename")
        if not isinstance(name, str) or not name:
            missing.append("<ENTRY_WITHOUT_FILENAME>")
            continue
        declared_names.add(name)
        file_path = dataset_dir / name
        if not file_path.is_file():
            missing.append(name)
            continue
        present += 1
        expected = entry.get("sha256")
        if isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected):
            declared_sha += 1
            if verify_file_hashes:
                observed = sha256_file(file_path)
                if observed == expected:
                    verified += 1
                else:
                    mismatches.append(name)
    disk_names = {
        child.name
        for child in dataset_dir.iterdir()
        if child.is_file() and child.name != "manifest.json"
    }
    extra_names = sorted(disk_names - declared_names)
    quarantined_extra_files = sorted(
        name
        for name in extra_names
        if ".corrupt.bak" in name or name.endswith(".part") or name.endswith(".quarantine")
    )
    unlisted_payload_files = sorted(set(extra_names) - set(quarantined_extra_files))
    license_present = bool(manifest.get("license")) or any(bool(e.get("license")) for e in entries)
    complete = (
        bool(entries)
        and present == len(entries)
        and declared_sha == len(entries)
        and not mismatches
        and not unlisted_payload_files
    )
    if verify_file_hashes:
        complete = complete and verified == len(entries)
    return {
        "manifest_path": str(manifest_path),
        "manifest_present": True,
        "manifest_sha256": sha256_file(manifest_path),
        "declared_files": len(entries),
        "present_files": present,
        "files_with_declared_sha256": declared_sha,
        "verified_file_hashes": verified,
        "hash_verification_requested": verify_file_hashes,
        "hash_mismatches": sorted(mismatches),
        "missing_files": sorted(missing),
        "unlisted_payload_files": unlisted_payload_files,
        "quarantined_extra_files": quarantined_extra_files,
        "license_field_present": license_present,
        "status": "COMPLETE" if complete else "INCOMPLETE_BLOCKED",
    }


def _new_accumulator(track_pools: bool) -> MutableMapping[str, Any]:
    return {
        "nominal_rows": 0,
        "records_with_valid_source": 0,
        "records_with_valid_candidate": 0,
        "records_with_both_sequences": 0,
        "records_with_nonempty_labels": 0,
        "records_with_verified_edit_script": 0,
        "records_with_explicit_group_field": 0,
        "records_with_gene_group": 0,
        "records_with_raw_record_locator": 0,
        "records_with_paper_faithful_transform": 0,
        "records_with_replicate_labels": 0,
        "records_with_standard_error_label": 0,
        "source_equals_candidate": 0,
        "source_hashes": set(),
        "candidate_hashes": set(),
        "gene_groups": set(),
        "source_to_candidates": defaultdict(set) if track_pools else None,
        "edit_count_strata": Counter(),
        "regions": Counter(),
        "label_key_rows": Counter(),
        "label_key_nonnull_rows": Counter(),
        "source_files": Counter(),
        "record_types": Counter(),
        "legacy_data_roles": Counter(),
        "delta_sign_balance": Counter(),
    }


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _update_accumulator(dataset_id: str, acc: MutableMapping[str, Any], record: Mapping[str, Any]) -> None:
    acc["nominal_rows"] += 1
    source_digest = _sequence_digest(record.get("source_sequence"))
    candidate_digest = _sequence_digest(record.get("candidate_sequence"))
    if source_digest is not None:
        acc["records_with_valid_source"] += 1
        acc["source_hashes"].add(source_digest)
    if candidate_digest is not None:
        acc["records_with_valid_candidate"] += 1
        acc["candidate_hashes"].add(candidate_digest)
    if source_digest is not None and candidate_digest is not None:
        acc["records_with_both_sequences"] += 1
        if source_digest == candidate_digest:
            acc["source_equals_candidate"] += 1
        pools = acc["source_to_candidates"]
        if pools is not None:
            pools[source_digest].add(candidate_digest)

    labels = record.get("labels") if isinstance(record.get("labels"), Mapping) else {}
    if labels:
        acc["records_with_nonempty_labels"] += 1
    replicate_nonnull = 0
    se_nonnull = 0
    for key, value in labels.items():
        key = str(key)
        acc["label_key_rows"][key] += 1
        if value is not None:
            acc["label_key_nonnull_rows"][key] += 1
            if REPLICATE_KEY.search(key):
                replicate_nonnull += 1
            if SE_KEY.search(key):
                se_nonnull += 1
    if replicate_nonnull >= 2:
        acc["records_with_replicate_labels"] += 1
    if se_nonnull or _finite_number(record.get("standard_error")) is not None:
        acc["records_with_standard_error_label"] += 1
    for key in DELTA_LABEL_KEYS.get(dataset_id, ()):
        number = _finite_number(labels.get(key))
        if number is None:
            acc["delta_sign_balance"]["missing_or_nonfinite"] += 1
        elif number > 0:
            acc["delta_sign_balance"]["positive"] += 1
        elif number < 0:
            acc["delta_sign_balance"]["negative"] += 1
        else:
            acc["delta_sign_balance"]["exact_zero"] += 1

    edit_script = record.get("edit_script")
    if isinstance(edit_script, list):
        edit_count = len(edit_script)
    else:
        edit_count = record.get("edit_distance")
    if isinstance(edit_count, int) and edit_count >= 0:
        acc["edit_count_strata"][str(edit_count)] += 1
    else:
        acc["edit_count_strata"]["UNKNOWN"] += 1
    if record.get("edit_script_verified") is True:
        acc["records_with_verified_edit_script"] += 1

    region = _safe_scalar(record.get("region")) or "UNKNOWN"
    acc["regions"][region] += 1
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    if any(_safe_scalar(metadata.get(key)) for key in EXPLICIT_GROUP_KEYS):
        acc["records_with_explicit_group_field"] += 1
    if any(_safe_scalar(metadata.get(key)) for key in RAW_LOCATOR_KEYS):
        acc["records_with_raw_record_locator"] += 1
    if isinstance(record.get("paper_faithful_transform"), Mapping):
        acc["records_with_paper_faithful_transform"] += 1
    gene_value = next((_safe_scalar(metadata.get(key)) for key in GENE_KEYS if _safe_scalar(metadata.get(key))), None)
    if gene_value:
        acc["records_with_gene_group"] += 1
        acc["gene_groups"].add(gene_value)
    for output_key, metadata_key in (
        ("source_files", "source_file"),
        ("record_types", "record_type"),
        ("legacy_data_roles", "data_role"),
    ):
        value = _safe_scalar(metadata.get(metadata_key))
        if value:
            acc[output_key][value] += 1


def scan_legacy_canonical(path: Path, included_ids: Sequence[str]) -> Dict[str, MutableMapping[str, Any]]:
    included = set(included_ids)
    accumulators = {
        dataset_id: _new_accumulator(track_pools=dataset_id != "GSE145046")
        for dataset_id in included_ids
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at canonical line {line_number}: {exc}") from exc
            dataset_id = record.get("accession")
            if dataset_id in included:
                _update_accumulator(str(dataset_id), accumulators[str(dataset_id)], record)
    return accumulators


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _pool_summary(pools: Optional[Mapping[bytes, set[bytes]]]) -> Dict[str, Any]:
    if pools is None:
        return {
            "status": "NOT_REPRESENTED_BY_LEGACY_SOURCE_FIELDS",
            "one_candidate_pools": 0,
            "two_candidate_pairwise_pools": 0,
            "ndcg_eligible_pools_ge_3": 0,
            "maximum_candidates_in_one_pool": 0,
        }
    sizes = [len(candidates) for candidates in pools.values()]
    return {
        "status": "STRUCTURAL_PROXY_ONLY_NOT_SPLIT_OR_ENDPOINT_ADJUDICATED",
        "one_candidate_pools": sum(size == 1 for size in sizes),
        "two_candidate_pairwise_pools": sum(size == 2 for size in sizes),
        "ndcg_eligible_pools_ge_3": sum(size >= 3 for size in sizes),
        "maximum_candidates_in_one_pool": max(sizes, default=0),
    }


def _study_blockers(
    dataset_id: str,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[str]:
    blockers = list((protocol.get("legacy_canonical_policy") or {}).get("required_before_qualification") or [])
    if summary["nominal_rows"] == 0:
        blockers.append("NO_LEGACY_CANONICAL_ROWS")
    if summary["records_with_nonempty_labels"] == 0:
        blockers.append("NO_LABEL_BEARING_ROWS")
    if summary["records_with_explicit_group_field"] == 0:
        blockers.append("NO_EXPLICIT_BIOLOGICAL_GROUP_FIELD")
    if summary["records_with_replicate_labels"] == 0 and summary["records_with_standard_error_label"] == 0:
        blockers.append("NO_REPLICATE_OR_STANDARD_ERROR_COVERAGE")
    if manifest.get("status") != "COMPLETE":
        blockers.append("P0_MANIFEST_OR_FILE_INTEGRITY_INCOMPLETE")
    if not manifest.get("license_field_present"):
        blockers.append("LICENSE_AND_REDISTRIBUTION_NOT_BOUND_IN_INPUT_MANIFEST")
    study_rule = ((protocol.get("study_rules") or {}).get(dataset_id) or {})
    recovery = study_rule.get("required_recovery")
    if recovery:
        blockers.append(str(recovery))
    if dataset_id == "GSE145046":
        blockers.extend(
            [
                "LEGACY_CANONICAL_CONSUMES_INPUT_SUPPORT_NOT_30_SAMPLE_LABEL_COMPLETE_JOIN",
                "DENSE_SCAFFOLD_AND_SOURCE_ANCHOR_NOT_BOUND",
                "FULL_CONTEXT_SEQUENCE_NOT_BOUND",
            ]
        )
    if dataset_id == "GSE114002":
        blockers.extend(
            [
                "RANDOM_AND_NATURAL_SUBSETS_NOT_V3_SEPARATED",
                "SHORT_SOURCE_MERGE_KEYS_NOT_ALL_BOUND",
                "CHECKPOINT_LEVEL_FOUNDATION_EXPOSURE_NOT_CLOSED",
            ]
        )
    if dataset_id == "GSE217518":
        blockers.append("FIVE_UTR_AND_THREE_UTR_SUBSETS_NOT_V3_SEPARATED")
    return sorted(set(blockers))


def finalize_study(
    dataset_id: str,
    acc: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    nominal_rows = int(acc["nominal_rows"])
    pool_summary = _pool_summary(acc["source_to_candidates"])
    summary: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "intended_grade": ((protocol.get("study_rules") or {}).get(dataset_id) or {}).get("intended_grade"),
        "qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "nominal_rows": nominal_rows,
        "distinct_candidates": len(acc["candidate_hashes"]),
        "biological_source_groups": {
            "explicit_v3_groups": 0,
            "distinct_source_sequence_proxy": len(acc["source_hashes"]),
            "proxy_is_qualification_sufficient": False,
        },
        "gene_groups": {
            "distinct_explicit_gene_values": len(acc["gene_groups"]),
            "records_with_gene_group": int(acc["records_with_gene_group"]),
        },
        "study_groups": 1 if nominal_rows else 0,
        "eligible_multi_candidate_pools": pool_summary,
        "edit_count_strata": _counter_dict(acc["edit_count_strata"]),
        "replicate_and_se_coverage": {
            "records_with_two_or_more_replicate_labels": int(acc["records_with_replicate_labels"]),
            "records_with_standard_error_label": int(acc["records_with_standard_error_label"]),
            "coverage_is_v3_adjudicated": False,
        },
        "beneficial_and_noise_zone_balance": {
            "legacy_explicit_delta_sign_counts": _counter_dict(acc["delta_sign_balance"]),
            "beneficial_direction_verified": False,
            "noise_equivalence_margin_frozen": False,
            "status": "NOT_ADJUDICATED",
        },
        "post_dedup_effective_n": {
            "distinct_source_sequence_proxy": len(acc["source_hashes"]),
            "group_bootstrap_power_simulation_status": "NOT_RUN",
            "qualification_sufficient": False,
        },
        "foundation_exposure": {
            "status": "AUDIT_PENDING",
            "known_policy_declaration": ((protocol.get("study_rules") or {}).get(dataset_id) or {}).get("known_exposure"),
            "checkpoint_level_binding": False,
        },
        "license_and_redistribution_status": {
            "status": "UNKNOWN_BLOCKED",
            "input_manifest_has_license_field": bool(manifest.get("license_field_present")),
            "redistribution_allowed": None,
        },
        "legacy_field_coverage": {
            "records_with_valid_source": int(acc["records_with_valid_source"]),
            "records_with_valid_candidate": int(acc["records_with_valid_candidate"]),
            "records_with_both_sequences": int(acc["records_with_both_sequences"]),
            "source_equals_candidate": int(acc["source_equals_candidate"]),
            "records_with_nonempty_labels": int(acc["records_with_nonempty_labels"]),
            "records_with_verified_edit_script": int(acc["records_with_verified_edit_script"]),
            "records_with_explicit_group_field": int(acc["records_with_explicit_group_field"]),
            "records_with_raw_record_locator": int(acc["records_with_raw_record_locator"]),
            "records_with_paper_faithful_transform": int(acc["records_with_paper_faithful_transform"]),
            "regions": _counter_dict(acc["regions"]),
            "label_key_rows": _counter_dict(acc["label_key_rows"]),
            "label_key_nonnull_rows": _counter_dict(acc["label_key_nonnull_rows"]),
            "source_files": _counter_dict(acc["source_files"]),
            "record_types": _counter_dict(acc["record_types"]),
            "legacy_data_roles": _counter_dict(acc["legacy_data_roles"]),
        },
        "p0_manifest": dict(manifest),
    }
    summary["blockers"] = _study_blockers(dataset_id, acc, manifest, protocol)
    return summary


def build_report(
    *,
    protocol_path: Path,
    canonical_records: Path,
    p0_root: Path,
    expected_canonical_sha256: Optional[str] = None,
    verify_dataset_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    forbidden_tokens = list((protocol.get("scope") or {}).get("forbidden_path_tokens") or [])
    protocol_path = ensure_ordinary_path(protocol_path, forbidden_tokens)
    canonical_records = ensure_ordinary_path(canonical_records, forbidden_tokens)
    p0_root = ensure_ordinary_path(p0_root, forbidden_tokens)
    if not canonical_records.is_file():
        raise FileNotFoundError(canonical_records)
    if not p0_root.is_dir():
        raise FileNotFoundError(p0_root)

    canonical_sha256 = sha256_file(canonical_records)
    if expected_canonical_sha256 and canonical_sha256 != expected_canonical_sha256:
        raise ValueError(
            f"canonical SHA-256 mismatch: expected {expected_canonical_sha256}, observed {canonical_sha256}"
        )
    included_ids = list(protocol["scope"]["included_dataset_ids"])
    verify_dataset_ids = set(verify_dataset_ids or set())
    unknown_verify_ids = verify_dataset_ids - set(included_ids)
    if unknown_verify_ids:
        raise ValueError(
            "hash verification requested for dataset(s) outside the ordinary allowlist: "
            + ",".join(sorted(unknown_verify_ids))
        )
    manifests = {
        dataset_id: summarize_p0_manifest(
            dataset_id,
            p0_root,
            verify_file_hashes=dataset_id in verify_dataset_ids,
        )
        for dataset_id in included_ids
    }
    accumulators = scan_legacy_canonical(canonical_records, included_ids)
    studies = [
        finalize_study(dataset_id, accumulators[dataset_id], manifests[dataset_id], protocol)
        for dataset_id in included_ids
    ]
    return {
        "contract_id": "mrna_xeditflow_route_a_v3",
        "schema_version": "3.0.0",
        "report_id": "A1_ORDINARY_PUBLIC_DATA_GAP_INVENTORY_V1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "protocol_id": protocol["protocol_id"],
            "status": protocol["status"],
        },
        "inputs": {
            "legacy_canonical_records": {
                "path": str(canonical_records),
                "sha256": canonical_sha256,
                "purpose": "GAP_INVENTORY_ONLY",
            },
            "p0_root": str(p0_root),
            "file_hash_verification_dataset_ids": sorted(verify_dataset_ids),
        },
        "ordinary_scope": {
            "included_dataset_ids": included_ids,
            "excluded_dataset_ids": list(protocol["scope"]["excluded_dataset_ids"]),
            "sealed_contact": False,
            "training_started": False,
            "gpu_work_started": False,
        },
        "studies": studies,
        "gate": {
            "minimum_independent_ordinary_studies": protocol["gate"]["minimum_independent_ordinary_studies"],
            "minimum_qualified_a1_studies": protocol["gate"]["minimum_qualified_a1_studies"],
            "minimum_qualified_a2_dense_studies": protocol["gate"]["minimum_qualified_a2_dense_studies"],
            "qualified_independent_ordinary_studies": 0,
            "qualified_a1_studies": 0,
            "qualified_a2_dense_studies": 0,
            "decision": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "next_phase_authorized": False,
        },
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "interpretation": (
            "This artifact is a legacy-data gap inventory, not a V3 data freeze, "
            "study qualification, training artifact, model result, or scientific conclusion."
        ),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/route_a_v3_a1_qualification.json"),
    )
    parser.add_argument("--canonical-records", type=Path, required=True)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--expected-canonical-sha256")
    parser.add_argument(
        "--verify-file-hashes-for",
        action="append",
        default=[],
        metavar="DATASET_ID",
        help="Repeat for selected ordinary datasets; avoids rehashing unrelated very large assets.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    report = build_report(
        protocol_path=protocol_path,
        canonical_records=args.canonical_records,
        p0_root=args.p0_root,
        expected_canonical_sha256=args.expected_canonical_sha256,
        verify_dataset_ids=set(args.verify_file_hashes_for),
    )
    _atomic_write_json(args.out, report)
    print(
        json.dumps(
            {
                "report": str(args.out),
                "evidence_status": report["evidence_status"],
                "qualified_independent_ordinary_studies": 0,
                "qualified_a1_studies": 0,
                "qualified_a2_dense_studies": 0,
                "next_phase_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
