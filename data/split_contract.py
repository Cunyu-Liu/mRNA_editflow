"""Immutable, fail-closed train/validation/test split contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl

SPLIT_ROLES = ("train", "val", "test")


class SplitContractError(ValueError):
    """Base class for split-contract verification failures."""


class SplitSchemaError(SplitContractError):
    """The manifest is missing or contains invalid schema fields."""


class SplitHashError(SplitContractError):
    """A manifested artifact or identifier digest changed."""


class SplitIndexError(SplitContractError):
    """A role index file is malformed, duplicated, or out of range."""


class SplitOverlapError(SplitContractError):
    """Role indices or exact sequences overlap across roles."""


class SplitRoleError(SplitContractError):
    """A caller requested an unknown or scientifically invalid role."""


class TrainValidationOverlapError(SplitOverlapError):
    """Train and validation inputs share identifiers, families, or sequences."""


@dataclass(frozen=True)
class VerifiedRole:
    name: str
    idx_path: str
    sha256: str
    count: int
    selected_id_digest: str
    indices: Tuple[int, ...]


@dataclass(frozen=True)
class VerifiedSplitContract:
    manifest_path: str
    manifest_sha256: str
    schema_version: int
    dataset_id: str
    records_path: str
    records_sha256: str
    records_count: int
    transcript_id_digest: str
    records_content_digest: str
    roles: Mapping[str, VerifiedRole]
    excluded: Optional[VerifiedRole]
    excluded_reason: Optional[str]
    cluster_assignment_path: Optional[str]
    cluster_assignment_sha256: Optional[str]
    cluster_assignments: Tuple[int, ...]
    leakage_report_path: str
    leakage_report_sha256: str
    family_disjoint: bool
    near_neighbor_threshold_passed: bool
    paper_eligible: bool
    block_reasons: Tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcript_id_digest(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for transcript_id in ids:
        digest.update(str(transcript_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def records_content_digest(records: Sequence[MRNARecord]) -> str:
    """Digest canonical in-memory records so public APIs cannot swap sequences."""
    encoded = json.dumps(
        [record.to_dict() for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sequence_hash(sequence: str) -> str:
    """Stable sequence identity used to reject train/validation leakage."""
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def _family_clusters(records: Sequence[MRNARecord]) -> set[str]:
    """Return supplied family-cluster identities, ignoring absent metadata."""
    clusters: set[str] = set()
    for record in records:
        metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
        for key in ("family_cluster", "family_cluster_id", "cluster_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                clusters.add(str(value))
                break
    return clusters


def assert_train_validation_disjoint(
    train_records: Sequence[MRNARecord],
    val_records: Sequence[MRNARecord],
    *,
    split_contract: Optional["VerifiedSplitContract"] = None,
) -> None:
    """Fail closed on transcript, exact-sequence, or supplied-family overlap."""
    train_ids = {record.transcript_id for record in train_records}
    val_ids = {record.transcript_id for record in val_records}
    overlap = train_ids & val_ids
    if overlap:
        raise TrainValidationOverlapError(
            "train/val transcript id overlap: " + sorted(overlap)[0]
        )
    train_hashes = {sequence_hash(record.seq) for record in train_records}
    val_hashes = {sequence_hash(record.seq) for record in val_records}
    if train_hashes & val_hashes:
        raise TrainValidationOverlapError("train/val sequence hash overlap detected")
    cluster_overlap = _family_clusters(train_records) & _family_clusters(val_records)
    if cluster_overlap:
        raise TrainValidationOverlapError(
            "train/val family cluster overlap: " + sorted(cluster_overlap)[0]
        )
    if split_contract is not None:
        if not split_contract.family_disjoint:
            raise TrainValidationOverlapError(
                "split contract does not certify family-disjoint train/val roles"
            )
        if split_contract.cluster_assignments:
            train_clusters = {
                split_contract.cluster_assignments[index]
                for index in split_contract.roles["train"].indices
            }
            val_clusters = {
                split_contract.cluster_assignments[index]
                for index in split_contract.roles["val"].indices
            }
            if train_clusters & val_clusters:
                raise TrainValidationOverlapError(
                    "split-contract train/val family cluster overlap detected"
                )


def verify_supplied_role_records(
    records: Sequence[MRNARecord],
    contract: "VerifiedSplitContract",
    role: str,
) -> list[MRNARecord]:
    """Verify a role-specific JSONL against the immutable full split contract."""
    universe = load_records_jsonl(contract.records_path)
    expected = select_role_records(universe, contract, role)
    if records_content_digest(records) != records_content_digest(expected):
        raise SplitHashError(
            f"provided {role} records do not exactly match the verified split role"
        )
    return list(records)


def _resolve_path(manifest_dir: Path, raw: object, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SplitSchemaError(f"{field} must be a non-empty path string")
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    root = manifest_dir.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SplitSchemaError(
            f"{field} relative path escapes the manifest directory: {raw!r}"
        ) from exc
    return resolved


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SplitSchemaError(f"{field} must be an object")
    return value


def _require_sha(value: object, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise SplitSchemaError(f"{field} must be a 64-character SHA-256")
    return text.lower()


def _read_indices(path: Path, role: str) -> Tuple[int, ...]:
    values = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = int(text)
            except ValueError as exc:
                raise SplitIndexError(f"{role} index line {line_no} is not an integer") from exc
            values.append(value)
    if len(values) != len(set(values)):
        raise SplitIndexError(f"{role} index file contains duplicate indices")
    return tuple(values)


def _read_cluster_assignments(path: Path) -> Tuple[int, ...]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.suffix.lower() == ".json":
                payload = json.load(fh)
                if not isinstance(payload, list):
                    raise SplitIndexError("cluster assignment JSON must be a list")
                values = [int(value) for value in payload]
            else:
                values = [int(line.strip()) for line in fh if line.strip()]
    except (ValueError, json.JSONDecodeError) as exc:
        raise SplitIndexError("cluster assignments must contain integer cluster ids") from exc
    if any(value < 0 for value in values):
        raise SplitIndexError("cluster assignment ids must be non-negative")
    return tuple(values)


def _verify_role(
    name: str,
    payload: Mapping[str, object],
    manifest_dir: Path,
    records: Sequence[MRNARecord],
) -> VerifiedRole:
    idx_path = _resolve_path(manifest_dir, payload.get("idx_path"), f"split.roles.{name}.idx_path")
    expected_sha = _require_sha(payload.get("sha256"), f"split.roles.{name}.sha256")
    if not idx_path.is_file():
        raise SplitIndexError(f"{name} index file does not exist: {idx_path}")
    actual_sha = sha256_file(idx_path)
    if actual_sha != expected_sha:
        raise SplitHashError(f"{name} index SHA mismatch: expected {expected_sha}, got {actual_sha}")
    indices = _read_indices(idx_path, name)
    try:
        expected_count = int(payload.get("count"))
    except (TypeError, ValueError) as exc:
        raise SplitSchemaError(f"split.roles.{name}.count must be an integer") from exc
    if expected_count != len(indices):
        raise SplitIndexError(
            f"{name} count mismatch: manifest {expected_count}, index file {len(indices)}"
        )
    bad = [idx for idx in indices if idx < 0 or idx >= len(records)]
    if bad:
        raise SplitIndexError(f"{name} contains out-of-range index {bad[0]}")
    selected_digest = transcript_id_digest([records[idx].transcript_id for idx in indices])
    expected_digest = _require_sha(
        payload.get("selected_id_digest"), f"split.roles.{name}.selected_id_digest"
    )
    if selected_digest != expected_digest:
        raise SplitHashError(f"{name} selected transcript identifier digest mismatch")
    return VerifiedRole(
        name=name,
        idx_path=str(idx_path),
        sha256=actual_sha,
        count=len(indices),
        selected_id_digest=selected_digest,
        indices=indices,
    )


def load_and_verify_split_manifest(
    path: str,
    records_path: Optional[str] = None,
) -> VerifiedSplitContract:
    """Load and fully verify a split manifest before scientific work starts."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise SplitSchemaError(f"split manifest does not exist: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SplitSchemaError(f"split manifest is not valid JSON: {exc}") from exc
    root = _require_mapping(payload, "manifest")
    if root.get("schema_version") != 1:
        raise SplitSchemaError("schema_version must equal 1")
    dataset_id = root.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise SplitSchemaError("dataset_id must be a non-empty string")
    records_meta = _require_mapping(root.get("records"), "records")
    manifest_dir = manifest_path.parent
    resolved_records = (
        Path(records_path).resolve()
        if records_path is not None
        else _resolve_path(manifest_dir, records_meta.get("path"), "records.path")
    )
    expected_records_sha = _require_sha(records_meta.get("sha256"), "records.sha256")
    if not resolved_records.is_file():
        raise SplitHashError(f"records file does not exist: {resolved_records}")
    actual_records_sha = sha256_file(resolved_records)
    if actual_records_sha != expected_records_sha:
        raise SplitHashError(
            f"records SHA mismatch: expected {expected_records_sha}, got {actual_records_sha}"
        )
    records = load_records_jsonl(str(resolved_records))
    try:
        expected_count = int(records_meta.get("count"))
    except (TypeError, ValueError) as exc:
        raise SplitSchemaError("records.count must be an integer") from exc
    if expected_count != len(records):
        raise SplitIndexError(
            f"records count mismatch: manifest {expected_count}, file {len(records)}"
        )
    ids = [record.transcript_id for record in records]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise SplitIndexError("records require unique, non-empty transcript identifiers")
    full_id_digest = transcript_id_digest(ids)
    expected_id_digest = _require_sha(
        records_meta.get("transcript_id_digest"), "records.transcript_id_digest"
    )
    if full_id_digest != expected_id_digest:
        raise SplitHashError("full transcript identifier digest mismatch")

    split = _require_mapping(root.get("split"), "split")
    algorithm = split.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise SplitSchemaError("split.algorithm must be a non-empty string")
    try:
        int(split.get("seed"))
        family_threshold = float(split.get("family_threshold"))
    except (TypeError, ValueError) as exc:
        raise SplitSchemaError("split.seed and split.family_threshold are required") from exc
    if not 0.0 <= family_threshold <= 1.0:
        raise SplitSchemaError("split.family_threshold must be in [0, 1]")
    near_neighbor = _require_mapping(split.get("near_neighbor"), "split.near_neighbor")
    try:
        near_k = int(near_neighbor.get("k"))
        near_jaccard = float(near_neighbor.get("jaccard"))
        near_containment = float(near_neighbor.get("containment"))
    except (TypeError, ValueError) as exc:
        raise SplitSchemaError(
            "split.near_neighbor requires integer k plus jaccard and containment"
        ) from exc
    if near_k <= 0 or not 0.0 <= near_jaccard <= 1.0 or not 0.0 <= near_containment <= 1.0:
        raise SplitSchemaError("split.near_neighbor thresholds are out of range")
    role_payloads = _require_mapping(split.get("roles"), "split.roles")
    if set(role_payloads) != set(SPLIT_ROLES):
        raise SplitSchemaError("split.roles must contain exactly train, val, and test")
    roles: Dict[str, VerifiedRole] = {}
    for role in SPLIT_ROLES:
        roles[role] = _verify_role(
            role,
            _require_mapping(role_payloads.get(role), f"split.roles.{role}"),
            manifest_dir,
            records,
        )
    excluded_payload = split.get("excluded")
    excluded = None
    excluded_reason = None
    if excluded_payload is not None:
        excluded_map = _require_mapping(excluded_payload, "split.excluded")
        if not excluded_map.get("reason"):
            raise SplitSchemaError("split.excluded requires a reason")
        excluded_reason = str(excluded_map.get("reason"))
        excluded = _verify_role("excluded", excluded_map, manifest_dir, records)

    cluster_assignment_path: Optional[Path] = None
    cluster_assignment_sha: Optional[str] = None
    cluster_assignments: Tuple[int, ...] = ()
    cluster_payload = split.get("cluster_assignments")
    if cluster_payload is not None:
        cluster_meta = _require_mapping(cluster_payload, "split.cluster_assignments")
        cluster_assignment_path = _resolve_path(
            manifest_dir, cluster_meta.get("path"), "split.cluster_assignments.path"
        )
        cluster_assignment_sha = _require_sha(
            cluster_meta.get("sha256"), "split.cluster_assignments.sha256"
        )
        if not cluster_assignment_path.is_file():
            raise SplitIndexError(
                f"cluster assignment file does not exist: {cluster_assignment_path}"
            )
        if sha256_file(cluster_assignment_path) != cluster_assignment_sha:
            raise SplitHashError("cluster assignment SHA mismatch")
        cluster_assignments = _read_cluster_assignments(cluster_assignment_path)
        try:
            cluster_count = int(cluster_meta.get("count"))
        except (TypeError, ValueError) as exc:
            raise SplitSchemaError("split.cluster_assignments.count must be an integer") from exc
        if cluster_count != len(cluster_assignments) or len(cluster_assignments) != len(records):
            raise SplitIndexError("cluster assignments must cover the records universe exactly")
        expected_assignment_digest = _require_sha(
            cluster_meta.get("assignment_digest"),
            "split.cluster_assignments.assignment_digest",
        )
        actual_assignment_digest = hashlib.sha256(
            json.dumps(cluster_assignments, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_assignment_digest != expected_assignment_digest:
            raise SplitHashError("cluster assignment digest mismatch")

    occupied: Dict[int, str] = {}
    for role in list(roles.values()) + ([excluded] if excluded is not None else []):
        assert role is not None
        for idx in role.indices:
            if idx in occupied:
                raise SplitOverlapError(
                    f"record index {idx} overlaps roles {occupied[idx]} and {role.name}"
                )
            occupied[idx] = role.name
    if len(occupied) != len(records):
        missing = sorted(set(range(len(records))) - set(occupied))
        raise SplitIndexError(f"split does not cover the records universe; first missing index {missing[0]}")
    role_sequences = {
        name: {records[idx].seq for idx in role.indices} for name, role in roles.items()
    }
    for pos, left in enumerate(SPLIT_ROLES):
        for right in SPLIT_ROLES[pos + 1:]:
            if role_sequences[left] & role_sequences[right]:
                raise SplitOverlapError(f"exact sequence overlap detected across {left}/{right}")

    computed_cluster_disjoint: Optional[bool] = None
    if cluster_assignments:
        role_clusters = {
            name: {cluster_assignments[idx] for idx in role.indices}
            for name, role in roles.items()
        }
        computed_cluster_disjoint = True
        for pos, left in enumerate(SPLIT_ROLES):
            for right in SPLIT_ROLES[pos + 1:]:
                if role_clusters[left] & role_clusters[right]:
                    computed_cluster_disjoint = False
                    break
            if computed_cluster_disjoint is False:
                break

    audits = _require_mapping(root.get("audits"), "audits")
    family_disjoint = audits.get("family_disjoint") is True
    if int(audits.get("exact_cross_role_matches", -1)) != 0:
        raise SplitOverlapError("audits.exact_cross_role_matches must equal zero")
    near_passed = audits.get("near_neighbor_threshold_passed") is True
    leakage_path = _resolve_path(
        manifest_dir, audits.get("leakage_report_path"), "audits.leakage_report_path"
    )
    leakage_sha = _require_sha(
        audits.get("leakage_report_sha256"), "audits.leakage_report_sha256"
    )
    if not leakage_path.is_file() or sha256_file(leakage_path) != leakage_sha:
        raise SplitHashError("leakage report SHA mismatch")
    try:
        with open(leakage_path, "r", encoding="utf-8") as fh:
            leakage_report = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SplitSchemaError("leakage report must be valid JSON") from exc
    leakage_report = _require_mapping(leakage_report, "leakage_report")
    report_split = leakage_report.get("split", {})
    report_split = report_split if isinstance(report_split, Mapping) else {}
    report_summary = leakage_report.get("summary", {})
    report_summary = report_summary if isinstance(report_summary, Mapping) else {}
    reported_cluster_disjoint = report_split.get("cluster_disjoint")
    if reported_cluster_disjoint is not None and bool(reported_cluster_disjoint) != family_disjoint:
        raise SplitOverlapError("leakage report cluster-disjoint result disagrees with manifest")
    if computed_cluster_disjoint is not None and computed_cluster_disjoint != family_disjoint:
        raise SplitOverlapError("cluster assignments disagree with audits.family_disjoint")
    reported_exact = report_summary.get(
        "leakage_exact_match_count", report_summary.get("exact_match_count")
    )
    if reported_exact is not None and int(reported_exact) != 0:
        raise SplitOverlapError("leakage report contains exact cross-role matches")
    block_reasons_raw = root.get("block_reasons", [])
    if not isinstance(block_reasons_raw, list) or not all(
        isinstance(item, str) and item for item in block_reasons_raw
    ):
        raise SplitSchemaError("block_reasons must be a list of non-empty strings")
    paper_eligible = root.get("paper_eligible") is True
    if paper_eligible and (not family_disjoint or not near_passed or block_reasons_raw):
        raise SplitSchemaError(
            "paper_eligible=true requires family and near-neighbor gates with no block reasons"
        )
    if paper_eligible and not cluster_assignments:
        raise SplitSchemaError(
            "paper_eligible=true requires immutable cluster assignments"
        )
    if paper_eligible:
        reported_flagged = report_summary.get("leakage_flagged_fraction")
        reported_near_pass = report_summary.get("near_neighbor_threshold_passed")
        if reported_near_pass is not True and (
            reported_flagged is None or float(reported_flagged) > 0.0
        ):
            raise SplitSchemaError(
                "paper_eligible=true requires leakage-report evidence for the near-neighbor gate"
            )

    return VerifiedSplitContract(
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        schema_version=1,
        dataset_id=dataset_id,
        records_path=str(resolved_records),
        records_sha256=actual_records_sha,
        records_count=len(records),
        transcript_id_digest=full_id_digest,
        records_content_digest=records_content_digest(records),
        roles=MappingProxyType(roles),
        excluded=excluded,
        excluded_reason=excluded_reason,
        cluster_assignment_path=(
            str(cluster_assignment_path) if cluster_assignment_path is not None else None
        ),
        cluster_assignment_sha256=cluster_assignment_sha,
        cluster_assignments=cluster_assignments,
        leakage_report_path=str(leakage_path),
        leakage_report_sha256=leakage_sha,
        family_disjoint=family_disjoint,
        near_neighbor_threshold_passed=near_passed,
        paper_eligible=paper_eligible,
        block_reasons=tuple(block_reasons_raw),
    )


def select_role_records(
    records: Sequence[MRNARecord],
    contract: VerifiedSplitContract,
    role: str,
) -> list[MRNARecord]:
    if role not in SPLIT_ROLES:
        raise SplitRoleError(f"unknown split role {role!r}")
    if len(records) != contract.records_count:
        raise SplitHashError("provided records do not match contract count")
    if transcript_id_digest([record.transcript_id for record in records]) != contract.transcript_id_digest:
        raise SplitHashError("provided records do not match contract transcript order")
    if records_content_digest(records) != contract.records_content_digest:
        raise SplitHashError("provided records do not match verified record content")
    selected = [records[idx] for idx in contract.roles[role].indices]
    if transcript_id_digest([record.transcript_id for record in selected]) != contract.roles[role].selected_id_digest:
        raise SplitHashError("selected records do not match role identifier digest")
    return selected


def build_split_provenance(contract: VerifiedSplitContract, role: str) -> dict:
    if role not in SPLIT_ROLES:
        raise SplitRoleError(f"unknown split role {role!r}")
    selected = contract.roles[role]
    return {
        "dataset_id": contract.dataset_id,
        "records_path": contract.records_path,
        "records_sha256": contract.records_sha256,
        "records_count": contract.records_count,
        "split_manifest_sha256": contract.manifest_sha256,
        "split_manifest_path": contract.manifest_path,
        "split_schema_version": contract.schema_version,
        "split_role": role,
        "role_idx_sha256": selected.sha256,
        "role_idx_path": selected.idx_path,
        "selected_count": selected.count,
        "selected_transcript_id_digest": selected.selected_id_digest,
        "cluster_assignment_path": contract.cluster_assignment_path,
        "cluster_assignment_sha256": contract.cluster_assignment_sha256,
        "selected_cluster_digest": (
            hashlib.sha256(
                json.dumps(
                    [contract.cluster_assignments[idx] for idx in selected.indices],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if contract.cluster_assignments else None
        ),
        "paper_eligible": contract.paper_eligible,
        "block_reasons": list(contract.block_reasons),
    }


def build_split_manifest(
    *,
    dataset_id: str,
    records_path: str,
    role_idx_paths: Mapping[str, str],
    leakage_report_path: str,
    algorithm: str,
    seed: int,
    family_threshold: float,
    family_disjoint: bool,
    exact_cross_role_matches: int,
    near_neighbor_threshold_passed: bool,
    near_neighbor_k: int = 15,
    near_neighbor_jaccard: float = 0.8,
    near_neighbor_containment: float = 0.95,
    cluster_assignment_path: Optional[str] = None,
    excluded_idx_path: Optional[str] = None,
    excluded_reason: Optional[str] = None,
    paper_eligible: bool = False,
    block_reasons: Sequence[str] = (),
) -> dict:
    """Build manifest content; writing is deliberately left to the caller."""
    records = load_records_jsonl(records_path)
    ids = [record.transcript_id for record in records]
    roles = {}
    for role in SPLIT_ROLES:
        if role not in role_idx_paths:
            raise SplitSchemaError(f"missing {role} index path")
        idx_path = Path(role_idx_paths[role]).resolve()
        indices = _read_indices(idx_path, role)
        roles[role] = {
            "idx_path": str(idx_path),
            "sha256": sha256_file(idx_path),
            "count": len(indices),
            "selected_id_digest": transcript_id_digest([ids[idx] for idx in indices]),
        }
    split_payload: dict[str, object] = {
        "algorithm": algorithm,
        "seed": int(seed),
        "family_threshold": float(family_threshold),
        "near_neighbor": {
            "k": int(near_neighbor_k),
            "jaccard": float(near_neighbor_jaccard),
            "containment": float(near_neighbor_containment),
        },
        "roles": roles,
    }
    if (excluded_idx_path is None) != (excluded_reason is None):
        raise SplitSchemaError(
            "excluded_idx_path and excluded_reason must be provided together"
        )
    if excluded_idx_path is not None:
        if not str(excluded_reason).strip():
            raise SplitSchemaError("excluded_reason must be non-empty")
        excluded_path = Path(excluded_idx_path).resolve()
        excluded_indices = _read_indices(excluded_path, "excluded")
        split_payload["excluded"] = {
            "idx_path": str(excluded_path),
            "sha256": sha256_file(excluded_path),
            "count": len(excluded_indices),
            "selected_id_digest": transcript_id_digest(
                [ids[idx] for idx in excluded_indices]
            ),
            "reason": str(excluded_reason),
        }
    if cluster_assignment_path is not None:
        assignment_path = Path(cluster_assignment_path).resolve()
        assignments = _read_cluster_assignments(assignment_path)
        split_payload["cluster_assignments"] = {
            "path": str(assignment_path),
            "sha256": sha256_file(assignment_path),
            "count": len(assignments),
            "assignment_digest": hashlib.sha256(
                json.dumps(assignments, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    elif paper_eligible:
        raise SplitSchemaError("paper-eligible manifests require cluster_assignment_path")
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "records": {
            "path": str(Path(records_path).resolve()),
            "sha256": sha256_file(records_path),
            "count": len(records),
            "transcript_id_digest": transcript_id_digest(ids),
        },
        "split": split_payload,
        "audits": {
            "family_disjoint": bool(family_disjoint),
            "exact_cross_role_matches": int(exact_cross_role_matches),
            "near_neighbor_threshold_passed": bool(near_neighbor_threshold_passed),
            "leakage_report_path": str(Path(leakage_report_path).resolve()),
            "leakage_report_sha256": sha256_file(leakage_report_path),
        },
        "paper_eligible": bool(paper_eligible),
        "block_reasons": list(block_reasons),
    }


__all__ = [
    "SPLIT_ROLES",
    "SplitContractError",
    "SplitSchemaError",
    "SplitHashError",
    "SplitIndexError",
    "SplitOverlapError",
    "SplitRoleError",
    "TrainValidationOverlapError",
    "VerifiedRole",
    "VerifiedSplitContract",
    "sha256_file",
    "transcript_id_digest",
    "records_content_digest",
    "sequence_hash",
    "assert_train_validation_disjoint",
    "verify_supplied_role_records",
    "load_and_verify_split_manifest",
    "select_role_records",
    "build_split_provenance",
    "build_split_manifest",
]
