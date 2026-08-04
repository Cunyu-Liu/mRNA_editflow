#!/usr/bin/env python3
"""Build the v3.1 strict D1 technical canonical.

This builder deliberately does not use the historical compact D1 adapter.  It
reads the already audited raw-record view, preserves every raw record in a
terminal disposition, and emits schema-closed ordinary and restricted shards.
The restricted shard is written outside the ordinary ``data/v3_1`` tree and
only aggregate QC/commitment bytes are written to the ordinary tree.

The implementation is intentionally dependency-free.  JSON canonicalization
uses the contract-compatible compact, sorted-key representation; the active
C3 schema manifest is still bound into every run and is checked by the strict
validator after construction.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


CONTRACT_ID = "GOAL-V3-DATA-BENCH-01"
SEALED_COHORT = "GSE246381"
SEALED_COHORT_SET_SHA256 = "275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0"
RESTRICTED_COMPONENT_SET_SHA256 = "974736d060463b3af090af3dd0c6a0e8bc591305f57f51e0a8cd31751a1ee606"
FUTURE_ROLE_ORDINARY = "AWAITING_B0_GLOBAL_DISPOSITION"
FUTURE_ROLE_SEALED = "SEALED_EXTERNAL_FINAL_CANDIDATE"
GENESIS = "GENESIS"
NOT_AVAILABLE = "NOT_AVAILABLE_D1"
NOT_APPLICABLE = "NOT_APPLICABLE"
SCHEMA_SENTINEL = "NOT_APPLICABLE"
NON_JSON_SCHEMA_ID = "NOT_APPLICABLE_NON_JSON"
NON_JSON_SCHEMA_SHA256 = "NOT_APPLICABLE"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IUPAC = set("ACGTRYSWKMBDHVN")

ACCESS_EVENT_TYPES = {
    "RESTRICTED_BUILDER_PARSE",
    "AGGREGATE_QC_MACHINE",
    "FM_OVERLAP_AGGREGATE",
    "B0_ELIGIBILITY_SPLIT_BUILD",
    "G7_RESTRICTED_FINALIZER",
    "TASK_PROTOCOL_CALIBRATION",
    "INTERNAL_TEST_EVALUATOR",
    "HUMAN_SEQUENCE_VIEW",
    "HUMAN_LABEL_VIEW",
    "TRAIN",
    "TUNE",
    "MODEL_SELECTION",
    "PRE_FINAL_ERROR_ANALYSIS",
    "ONE_TIME_FINAL_EVALUATOR",
    "POST_FINAL_ERROR_ANALYSIS",
}

CANONICAL_ROW_FILES = {
    "SEQUENCE_ENTITIES": "sequence_entities.jsonl",
    "FUNCTIONAL_OBSERVATION_CANDIDATES": "functional_observation_candidates.jsonl",
    "FUNCTIONAL_OBSERVATIONS": "functional_observations.jsonl",
    "ENDPOINT_REGISTRY": "ENDPOINT_REGISTRY.jsonl",
    "UTR_EDIT_RELATION_CANDIDATES": "utr_edit_relation_candidates.jsonl",
    "UTR_EDIT_PAIRS": "utr_edit_pairs.jsonl",
    "REJECTIONS": "rejections.jsonl",
    "TRANSFORMATION_EDGES": "transformation_edges.jsonl",
    "SUPERSESSION_EDGES": "SUPERSESSION_EDGES.jsonl",
    "CURRENT_CANONICAL_OBJECT_PROJECTION": "CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl",
    "EXPOSURE_RECORDS": "EXPOSURE_RECORDS.jsonl",
    "USE_ROLES": "USE_ROLES.jsonl",
    "GROUP_REGISTRY": "group_registry.jsonl",
    "GROUP_ASSIGNMENTS": "group_assignments.jsonl",
    "REPORTER_ARTIFACT_ASSESSMENTS": "reporter_artifact_assessments.jsonl",
}

RESTRICTED_LOGICAL_IDS = [
    "ACCESS_LOG",
    "ACCESS_MANIFEST",
    "ACCESS_SHA256SUMS",
    "CURRENT_CANONICAL_OBJECT_PROJECTION",
    "DATASET_RECONCILIATION",
    "DATA_UNITS_REPORT",
    "EFFECTIVE_EXPOSURE_PROJECTION",
    "ENDPOINT_REGISTRY",
    "EXPOSURE_RECORDS",
    "EXPOSURE_USE_MANIFEST",
    "EXPOSURE_USE_SHA256SUMS",
    "FUNCTIONAL_OBSERVATION_CANDIDATES",
    "FUNCTIONAL_OBSERVATIONS",
    "GROUP_ASSIGNMENTS",
    "GROUP_REGISTRY",
    "REJECTIONS",
    "SEALED_CANONICAL_SHA256SUMS",
    "SEALED_INPUT_MANIFEST",
    "SEQUENCE_ENTITIES",
    "SUPERSESSION_EDGES",
    "TRANSFORMATION_EDGES",
    "USE_ROLES",
    "UTR_EDIT_PAIRS",
    "UTR_EDIT_RELATION_CANDIDATES",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def jcs_bytes(value: Any) -> bytes:
    """Contract-compatible canonical JSON bytes for this run."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def jline(value: Any) -> bytes:
    return jcs_bytes(value) + b"\n"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8"))


def sha_json(value: Any, without: str | None = None) -> str:
    if without is not None and isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != without}
    return sha_bytes(jcs_bytes(value))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: Any, self_hash_field: str | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    if self_hash_field:
        payload[self_hash_field] = sha_json(payload)
    path.write_bytes(jline(payload))
    return sha_file(path)


def checksum_ledger(root: Path, paths: Iterable[Path], output: Path) -> str:
    rows = []
    for path in paths:
        path = path.resolve()
        rel = path.relative_to(root.resolve()).as_posix()
        rows.append((rel, sha_file(path)))
    rows.sort(key=lambda x: x[0].encode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes("".join(f"{digest}  {rel}\n" for rel, digest in rows).encode("utf-8"))
    return sha_file(output)


def set_sha(values: Iterable[str]) -> str:
    return sha_bytes(("\n".join(sorted(set(values))) + "\n").encode("utf-8"))


def safe_id(value: Any) -> str:
    return sha_text(str(value))[:32]


def token(value: Any, limit: int = 100) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text[:limit] or "EMPTY"


def finite_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    if out.is_integer() and abs(out) < 2**53:
        return int(out)
    return out


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_sequence(value: Any) -> tuple[str | None, str, list[str], str, bool]:
    raw = as_text(value)
    if not raw:
        return None, "EMPTY", ["EMPTY"], "QUARANTINED", False
    steps = ["STRIP"]
    upper = raw.upper()
    if upper != raw:
        steps.append("UPPERCASE")
    if "U" in upper:
        steps.append("U_TO_T")
        upper = upper.replace("U", "T")
    if set(upper) <= set("ACGT"):
        return upper, "EXACT_ACGT", steps, "PASS", True
    if set(upper) <= IUPAC:
        steps.append("IUPAC_RETAINED")
        return upper, "IUPAC_AMBIGUOUS", steps, "PASS", False
    steps.append("INVALID_SYMBOL")
    return None, "INVALID", steps, "QUARANTINED", False


def region_scope(region: Any) -> str:
    text = as_text(region)
    if text in {"5'UTR", "5UTR", "5′UTR"}:
        return "5UTR"
    if text in {"3'UTR", "3UTR", "3′UTR"}:
        return "3UTR"
    return "5UTR"


def sequence_scope(region: Any) -> str:
    return "FULL_UTR" if region_scope(region) in {"5UTR", "3UTR"} else "UTR_WINDOW"


def row_locator(source_path: Path, line_no: int, suffix: str = "") -> str:
    extra = f";{suffix}" if suffix else ""
    return f"{source_path.name}#line={line_no}{extra}"


def self_hash_row(row: dict[str, Any], field: str) -> dict[str, Any]:
    out = dict(row)
    out[field] = sha_json(out)
    return out


def read_jsonl(path: Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            yield line_no, line, json.loads(line)


class D0Registry:
    def __init__(self, root: Path):
        self.root = root
        reg = root / "data" / "v3_1" / "registry"
        self.raw_manifest_path = reg / "raw_asset_manifest.jsonl"
        self.dataset_assets_path = reg / "dataset_assets.jsonl"
        self.decisions_path = reg / "dataset_decisions.jsonl"
        if not self.raw_manifest_path.exists():
            raise FileNotFoundError(self.raw_manifest_path)
        self.raw_assets: list[dict[str, Any]] = [r for _, _, r in read_jsonl(self.raw_manifest_path)]
        self.dataset_assets: dict[str, dict[str, Any]] = {}
        if self.dataset_assets_path.exists():
            for _, _, row in read_jsonl(self.dataset_assets_path):
                accession = as_text(row.get("accession"))
                if accession:
                    self.dataset_assets[accession] = row
        self.by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.raw_assets:
            rel = as_text(row.get("relpath"))
            if rel:
                self.by_name[Path(rel).name].append(row)
        self.raw_manifest_sha256 = sha_file(self.raw_manifest_path)
        self.dataset_asset_manifest_sha256 = sha_file(self.dataset_assets_path) if self.dataset_assets_path.exists() else None
        self.decisions_sha256 = sha_file(self.decisions_path) if self.decisions_path.exists() else None

    def dataset_asset_id(self, accession: str) -> str:
        row = self.dataset_assets.get(accession)
        return as_text(row.get("asset_id")) if row else f"{accession}::UNMAPPED_D0_ASSET"

    def resolve_files(self, accession: str, source_file: Any) -> tuple[list[str], list[str], str | None]:
        text = as_text(source_file)
        if not text:
            return [self.dataset_asset_id(accession)], [], None
        names = []
        for part in text.split("+"):
            name = Path(part.strip()).name
            if name and name not in names:
                names.append(name)
        rows: list[dict[str, Any]] = []
        for name in names:
            candidates = self.by_name.get(name, [])
            accession_candidates = [r for r in candidates if as_text(r.get("asset_id")) == accession]
            chosen = accession_candidates or candidates
            if chosen:
                rows.extend(chosen[:1])
        # A source file with no raw-file registry row is not silently mapped to
        # a different asset.  The dataset asset remains the contributor, while
        # the file-level list is empty and the validator records the gap.
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (as_text(row.get("relpath")), as_text(row.get("sha256")))
            dedup[key] = row
        rows = list(dedup.values())
        hashes = sorted({as_text(r.get("sha256")) for r in rows if HEX64.fullmatch(as_text(r.get("sha256")))})
        primary = hashes[0] if hashes else None
        return [self.dataset_asset_id(accession)], hashes, primary


class GroupState:
    def __init__(self, group_id: str, group_type: str, raw_values: dict[str, Any], source_evidence: list[str]):
        self.group_id = group_id
        self.group_type = group_type
        self.raw_values = raw_values
        self.source_evidence = source_evidence
        self.count = 0
        self.members: list[str] = []

    def add(self, object_id: str) -> None:
        self.count += 1
        if len(self.members) < 1000:
            self.members.append(object_id)

    def row(self, mapping_rule_sha: str) -> dict[str, Any]:
        base = {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "grouping_atom": self.group_type,
            "grouping_method": "D1_SOURCE_CONTEXT_HASH_V1",
            "method_version": "1",
            "thresholds": {},
            "source_evidence": self.source_evidence,
            "member_count": self.count,
            "member_ids": sorted(set(self.members)) if self.count <= 1000 else [],
            "ambiguous_membership": False,
            "parent_group_id": None,
            "raw_context_values": self.raw_values,
            "context_components": {
                "cell_type": self.raw_values.get("cell_type"),
                "assay": self.raw_values.get("assay"),
                "promoter": self.raw_values.get("promoter"),
                "reporter_or_cargo": self.raw_values.get("reporter_or_cargo"),
                "rna_chemistry": self.raw_values.get("rna_chemistry"),
                "timepoint": self.raw_values.get("timepoint"),
                "other": self.raw_values.get("other"),
            },
            "ontology_ids": [],
            "ontology_version": "NOT_APPLICABLE_D1",
            "mapping_status": "UNKNOWN",
            "mapping_rule_id": "D1_SOURCE_CONTEXT_HASH_V1",
            "mapping_rule_sha256": mapping_rule_sha,
        }
        return self_hash_row(base, "group_sha256")


class FrameState:
    def __init__(self, definition: dict[str, Any]):
        self.definition = definition
        self.frame_definition_sha256 = sha_json(definition)
        self.group_id = f"NO_EDIT_SAMPLING_FRAME:{self.frame_definition_sha256}"
        self.count = 0
        self.assignment_hasher = hashlib.sha256()

    def add_assignment(self, row: dict[str, Any]) -> None:
        self.count += 1
        self.assignment_hasher.update(jline(row))

    def row(self) -> dict[str, Any]:
        out = dict(self.definition)
        out.update({
            "group_id": self.group_id,
            "group_type": "NO_EDIT_SAMPLING_FRAME",
            "member_assignment_manifest_sha256": self.assignment_hasher.hexdigest(),
            "frame_definition_sha256": self.frame_definition_sha256,
        })
        return out


class Bundle:
    """One ordinary or restricted canonical shard."""

    def __init__(self, root: Path, shard: str, snapshot_id: str, d0: D0Registry, schema_hashes: dict[str, str], run_id: str, config_hash: str):
        self.root = root
        self.shard = shard
        self.snapshot_id = snapshot_id
        self.d0 = d0
        self.schema_hashes = schema_hashes
        self.run_id = run_id
        self.config_hash = config_hash
        self.canonical = root / "canonical"
        self.canonical.mkdir(parents=True, exist_ok=True)
        self.projection_dir = root / "exposure" / "projections" / snapshot_id
        self.projection_dir.mkdir(parents=True, exist_ok=True)
        if shard == "ordinary":
            self.live_access = root / "exposure" / "ORDINARY_ACCESS_LOG.jsonl"
            self.snapshot_dir = root / "exposure" / "access_snapshots" / snapshot_id
            self.access_log_name = "ORDINARY_ACCESS_LOG.jsonl"
            self.access_manifest_name = "ORDINARY_ACCESS_MANIFEST.json"
            self.access_sums_name = "ORDINARY_ACCESS_SHA256SUMS"
        else:
            self.live_access = root / "ACCESS_LOG.jsonl"
            self.snapshot_dir = root / "access_snapshots" / snapshot_id
            self.access_log_name = "ACCESS_LOG.jsonl"
            self.access_manifest_name = "ACCESS_MANIFEST.json"
            self.access_sums_name = "ACCESS_SHA256SUMS"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.object_dir = self.snapshot_dir / "objects"
        self.object_dir.mkdir(parents=True, exist_ok=True)
        self.object_seq_path = self.object_dir / "REQUESTED_SEQUENCE_OBJECTS.jsonl"
        self.object_label_path = self.object_dir / "REQUESTED_LABEL_OBJECTS.jsonl"
        self.object_seq_actual_path = self.object_dir / "ACTUAL_SEQUENCE_OBJECTS.jsonl"
        self.object_label_actual_path = self.object_dir / "ACTUAL_LABEL_OBJECTS.jsonl"
        self.object_seq = self.object_seq_path.open("wb")
        self.object_label = self.object_label_path.open("wb")
        self.object_counts = Counter()
        self.object_hashes = {}
        work_root = self.root.parent.parent / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        self.pending_effective_path = work_root / f"{shard}_effective_exposure_pending.jsonl"
        self.pending_effective = self.pending_effective_path.open("wb")
        self.writers: dict[str, Any] = {}
        for name, filename in CANONICAL_ROW_FILES.items():
            self.writers[name] = (self.canonical / filename).open("wb")
        self.writers["EFFECTIVE_EXPOSURE_PROJECTION"] = None
        self.canonical_paths = {name: self.canonical / filename for name, filename in CANONICAL_ROW_FILES.items()}
        self.canonical_paths["EXPOSURE_USE_MANIFEST"] = self.canonical / "EXPOSURE_USE_MANIFEST.json"
        self.canonical_paths["EXPOSURE_USE_SHA256SUMS"] = self.canonical / "EXPOSURE_USE_SHA256SUMS"
        self.canonical_paths["DATASET_RECONCILIATION"] = self.canonical / "dataset_reconciliation.json"
        self.canonical_paths["DATA_UNITS_REPORT"] = self.canonical / "data_units_report.json"
        self.canonical_paths["CANONICAL_MANIFEST"] = self.canonical / "CANONICAL_MANIFEST.json"
        self.canonical_paths["CANONICAL_SHA256SUMS"] = self.canonical / "CANONICAL_SHA256SUMS"

    def emit(self, name: str, row: dict[str, Any]) -> str:
        data = jline(row)
        writer = self.writers[name]
        if writer is None:
            raise KeyError(name)
        writer.write(data)
        object_id = None
        object_type = None
        if name == "SEQUENCE_ENTITIES":
            object_id = row["sequence_id"]
            object_type = "SEQUENCE"
            self._object_line(self.object_seq, row, object_id, object_type)
        elif name == "FUNCTIONAL_OBSERVATION_CANDIDATES":
            object_id = row["candidate_id"]
            object_type = "OBSERVATION_CANDIDATE"
            self._object_line(self.object_label, row, object_id, object_type)
        elif name == "FUNCTIONAL_OBSERVATIONS":
            object_id = row["observation_id"]
            object_type = "OBSERVATION"
            self._object_line(self.object_label, row, object_id, object_type)
        elif name == "UTR_EDIT_RELATION_CANDIDATES":
            object_id = row["relation_candidate_id"]
            object_type = "RELATION_CANDIDATE"
            self._object_line(self.object_label, row, object_id, object_type)
        elif name == "UTR_EDIT_PAIRS":
            object_id = row["pair_id"]
            object_type = "PAIR"
            self._object_line(self.object_label, row, object_id, object_type)
        if object_id is not None:
            self.object_hashes[object_id] = (object_type, sha_json(row))
        return sha_json(row)

    def _object_line(self, fh: Any, row: dict[str, Any], object_id: str, object_type: str) -> None:
        locator = row.get("source_row_locator") or row.get("source_row_locators") or None
        if isinstance(locator, list):
            locator = locator[0] if locator else None
        source_record = row.get("source_record_id") or row.get("source_unit_ids") or row.get("evidence_id")
        payload = {
            "object_id": object_id,
            "object_type": object_type,
            "canonical_object_sha256": sha_json(row),
            "source_unit_id": source_record,
            "member_locator": locator,
        }
        fh.write(jline(payload))
        self.object_counts[object_type] += 1

    def write_effective_pending(self, payload: dict[str, Any]) -> None:
        self.pending_effective.write(jline(payload))

    def close_rows(self) -> None:
        for fh in self.writers.values():
            if fh is not None:
                fh.close()
        self.pending_effective.close()
        self.object_seq.close()
        self.object_label.close()
        shutil.copyfile(self.object_seq_path, self.object_seq_actual_path)
        shutil.copyfile(self.object_label_path, self.object_label_actual_path)

    def write_effective_projection(self, chain_root: str, as_of_event_id: str, canonical_binding: str) -> Path:
        path = self.projection_dir / "EFFECTIVE_EXPOSURE_PROJECTION.jsonl"
        with self.pending_effective_path.open("rb") as src, path.open("wb") as dst:
            for raw in src:
                row = json.loads(raw)
                row["access_log_chain_root_sha256"] = chain_root
                row["as_of_event_id"] = as_of_event_id
                row["chain_root_sha256"] = chain_root
                row["canonical_manifest_sha256"] = canonical_binding if "canonical_manifest_sha256" in row else None
                row = self_hash_row(row, "projection_sha256")
                dst.write(jline(row))
        return path

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()


def load_schema_hashes(schema_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    sums = schema_dir / "SCHEMA_SHA256SUMS"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2 and HEX64.fullmatch(parts[0]):
                result[Path(parts[-1]).name] = parts[0]
    for path in schema_dir.glob("*.schema.json"):
        result.setdefault(path.name, sha_file(path))
    return result


def profile_matrix(path: Path) -> dict[str, Any]:
    """Aggregate-only profile of a sealed matrix; never returns cell values."""

    rows = 0
    header: list[str] = []
    key_index = None
    missing_key = 0
    unique_keys: set[str] = set()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return {"path": path.name, "rows": 0, "columns": 0, "header_sha256": sha_text("")}
        lower = [as_text(x).lower() for x in header]
        for candidate in ("variant_name", "variant_id", "record_id", "id", "enst"):
            if candidate in lower:
                key_index = lower.index(candidate)
                break
        for row in reader:
            rows += 1
            if key_index is not None and key_index < len(row):
                value = as_text(row[key_index])
                if value:
                    unique_keys.add(value)
                else:
                    missing_key += 1
    return {
        "path": path.name,
        "rows": rows,
        "columns": len(header),
        "header_sha256": sha_bytes(jline(header)),
        "key_column": header[key_index] if key_index is not None else None,
        "unique_key_count": len(unique_keys) if key_index is not None else None,
        "missing_key_count": missing_key if key_index is not None else None,
    }


class StrictBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = args.out_root
        self.run_id = args.run_id
        self.snapshot_id = args.snapshot_id
        self.authority_sha256 = args.authority_contract_sha256
        self.source_head = args.source_head
        self.code_commit = args.code_commit
        self.schema_hashes = load_schema_hashes(args.schema_dir)
        self.d0 = D0Registry(args.d0_root)
        self.raw_input = args.ordinary_input
        self.legacy_input = args.legacy_input
        self.sealed_input = args.sealed_input
        self.counts: Counter[str] = Counter()
        self.dataset_counts: Counter[str] = Counter()
        self.record_type_counts: Counter[str] = Counter()
        self.label_counts: Counter[str] = Counter()
        self.endpoint_counts: Counter[str] = Counter()
        self.region_counts: Counter[str] = Counter()
        self.source_file_counts: Counter[str] = Counter()
        self.rejection_counts: Counter[str] = Counter()
        self.shard_record_type_counts: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.shard_label_counts: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.shard_endpoint_counts: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.shard_region_counts: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.shard_rejection_counts: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.frame_states: dict[str, FrameState] = {}
        self.groups: dict[str, GroupState] = {}
        self.assets_seen: set[str] = set()
        self.file_hashes_seen: set[str] = set()
        self.shard_assets: dict[str, set[str]] = {"ordinary": set(), "restricted": set()}
        self.shard_file_hashes: dict[str, set[str]] = {"ordinary": set(), "restricted": set()}
        self.endpoints: dict[str, dict[str, Any]] = {}
        self.endpoint_shards: dict[str, set[str]] = defaultdict(set)
        self.legacy_sha = hashlib.sha256()
        self.legacy_count = 0
        self.input_sha = hashlib.sha256()
        self.config_hash = sha_json({
            "builder": "build_strict_d1.py",
            "version": 1,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "source_head": self.source_head,
            "code_commit": self.code_commit,
        })
        self.canonical_binding = sha_json({
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "source_head": self.source_head,
            "code_commit": self.code_commit,
            "d0_raw_manifest_sha256": self.d0.raw_manifest_sha256,
        })
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "work").mkdir(parents=True, exist_ok=True)
        self.ordinary_root = self.out / "data" / "v3_1"
        self.restricted_root = self.out / "sealed_external" / SEALED_COHORT
        self.ordinary = Bundle(self.ordinary_root, "ordinary", self.snapshot_id, self.d0, self.schema_hashes, self.run_id, self.config_hash)
        self.restricted = Bundle(self.restricted_root, "restricted", self.snapshot_id, self.d0, self.schema_hashes, self.run_id, self.config_hash)

    def source_info(self, accession: str, metadata: dict[str, Any]) -> tuple[list[str], list[str], str | None, str]:
        source_file = metadata.get("source_file")
        asset_ids, file_hashes, primary_hash = self.d0.resolve_files(accession, source_file)
        self.assets_seen.update(asset_ids)
        self.file_hashes_seen.update(file_hashes)
        self.shard_assets[self.current_bundle.shard].update(asset_ids)
        self.shard_file_hashes[self.current_bundle.shard].update(file_hashes)
        source_text = as_text(source_file) or "<MISSING>"
        self.source_file_counts[source_text] += 1
        return asset_ids, file_hashes, primary_hash, source_text

    def context(self, accession: str, region: Any, metadata: dict[str, Any], source_file: str, bundle: Bundle) -> str:
        library = as_text(metadata.get("library")) or as_text(metadata.get("subpool")) or "UNKNOWN_LIBRARY"
        key = "|".join([accession, region_scope(region), source_file, library])
        context_id = f"CTX:{safe_id(key)}"
        if context_id not in self.groups:
            raw_values = {
                "accession": accession,
                "region": region_scope(region),
                "source_file": source_file,
                "library": library,
                "cell_type": as_text(metadata.get("cell_type")) or None,
                "assay": as_text(metadata.get("assay")) or None,
                "promoter": as_text(metadata.get("promoter")) or None,
                "reporter_or_cargo": as_text(metadata.get("reporter_or_cargo")) or None,
                "rna_chemistry": as_text(metadata.get("rna_chemistry")) or None,
                "timepoint": as_text(metadata.get("timepoint")) or None,
                "other": as_text(metadata.get("record_type")) or None,
            }
            self.groups[context_id] = GroupState(context_id, "EXPERIMENTAL_CONTEXT", raw_values, ["D0_RAW_MANIFEST", "D1_CONTEXT_RULE"])
        setattr(self.groups[context_id], f"seen_{bundle.shard}", True)
        return context_id

    def add_assignment(self, bundle: Bundle, object_id: str, object_type: str, context_id: str, locator: str | None, evidence: list[str], endpoint_id: str | None = None, frame: FrameState | None = None, subtype: str = "NOT_APPLICABLE_NON_IDENTITY") -> None:
        self.groups[context_id].add(object_id)
        assignment = {
            "assignment_id": f"ASG:{safe_id(bundle.shard + '|' + object_id + '|' + context_id)}",
            "object_id": object_id,
            "object_type": object_type,
            "group_id": context_id,
            "grouping_atom": "EXPERIMENTAL_CONTEXT",
            "assignment_algorithm_id": "D1_SOURCE_CONTEXT_HASH_V1",
            "source_evidence_ids": evidence,
            "member_locator": locator,
            "no_edit_control_subtype": subtype,
            "no_edit_sampling_frame_id": frame.group_id if frame else None,
            "context_id": context_id,
            "endpoint_id": endpoint_id,
        }
        bundle.emit("GROUP_ASSIGNMENTS", assignment)
        if frame is not None:
            frame.add_assignment(assignment)

    def endpoint_id(self, label_key: str) -> str:
        if label_key == "NO_MEASUREMENT":
            eid = "EP:NO_MEASUREMENT"
        else:
            eid = f"EP:{safe_id(label_key)}"
        if hasattr(self, "current_bundle"):
            self.endpoint_shards[eid].add(self.current_bundle.shard)
        if eid not in self.endpoints:
            low = label_key.lower()
            scaling = "LOG2" if any(x in low for x in ("log2", "lfc", "foldchange", "log_fold")) else "LINEAR"
            self.endpoints[eid] = {
                "endpoint_id": eid,
                "name": label_key,
                "scaling": scaling,
                "missing_token": "NULL_OR_UNOBSERVED" if label_key == "NO_MEASUREMENT" else "NULL",
                "missing_mask": label_key == "NO_MEASUREMENT",
                "biological_quantity": "ASSAY_REPORTED_FUNCTIONAL_VALUE",
                "raw_field_mappings": {"label_key": label_key},
                "label_unit": "ASSAY_REPORTED_VALUE" if label_key != "NO_MEASUREMENT" else "NOT_APPLICABLE",
                "directionality": "TWO_SIDED",
                "label_transform": "IDENTITY_FLOAT_PARSE_V1" if label_key != "NO_MEASUREMENT" else "NO_VALUE",
                "comparability_scope": "WITHIN_SOURCE_FILE_AND_CONTEXT_ONLY",
                "aggregation_rule_id": "D1_NO_AGGREGATION_ROW_LEVEL_V1",
                "aggregation_rule_sha256": sha_text("D1_NO_AGGREGATION_ROW_LEVEL_V1"),
                "delta_rule_id": "D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1",
                "delta_rule_sha256": sha_text("D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1"),
                "unknown_or_ambiguous_policy": "RETAIN_AS_CANDIDATE_OR_REJECTION",
            }
        return eid

    def frame_for(self, accession: str, region: Any, metadata: dict[str, Any], source_file: str, context_id: str, endpoint_id: str, assets: list[str], species: str) -> FrameState:
        library = as_text(metadata.get("library")) or as_text(metadata.get("subpool")) or "UNKNOWN_LIBRARY"
        key = "|".join([accession, source_file, region_scope(region), context_id, endpoint_id])
        definition = {
            "study_id": accession,
            "asset_ids": sorted(set(assets)),
            "library_lineage_group_id": f"LIBRARY_LINEAGE:{safe_id(accession + '|' + source_file + '|' + library)}",
            "sublibrary_or_design_stratum": library,
            "species": species,
            "region_scope": region_scope(region),
            "context_id": context_id,
            "endpoint_id": endpoint_id,
            "inclusion_mechanism": "OUTCOME_BLIND_IDENTITY_DESIGN_RULE",
            "identity_inclusion_rule_id": "D1_IDENTITY_DESIGNED_LIBRARY_RULE_V1",
            "identity_inclusion_rule_sha256": sha_text("D1_IDENTITY_DESIGNED_LIBRARY_RULE_V1"),
            "nonidentity_inclusion_rule_id": "D1_NONIDENTITY_EXCLUSION_FROM_NOEDIT_FRAME_V1",
            "nonidentity_inclusion_rule_sha256": sha_text("D1_NONIDENTITY_EXCLUSION_FROM_NOEDIT_FRAME_V1"),
            "inclusion_probability_status": "UNKNOWN",
            "reweighting_rule_id": "D1_NO_AUXILIARY_REWEIGHTING_BEFORE_B0_V1",
            "reweighting_rule_sha256": sha_text("D1_NO_AUXILIARY_REWEIGHTING_BEFORE_B0_V1"),
            "evidence_ids": ["D0_RAW_MANIFEST", "D1_IDENTITY_GROUP_PROFILE"],
        }
        key_id = sha_json(definition)
        if key_id not in self.frame_states:
            self.frame_states[key_id] = FrameState(definition)
        return self.frame_states[key_id]

    def sequence_row(self, accession: str, record_id: str, side: str, sequence: Any, region: Any, metadata: dict[str, Any], source_file: str, locator: str, assets: list[str], file_hashes: list[str], bundle: Bundle) -> tuple[str, bool, str | None, str, str]:
        seq_id = f"SEQ:{bundle.shard}:{safe_id(record_id)}:{side}"
        normalized, alphabet, steps, invalid_status, model_eligible = normalize_sequence(sequence)
        raw = as_text(sequence)
        species = as_text(metadata.get("species")) or "UNKNOWN"
        strand = as_text(metadata.get("strand")) or "UNKNOWN"
        if strand not in {"+", "-", "UNKNOWN", "NOT_APPLICABLE"}:
            strand = "UNKNOWN"
        raw_hash = sha_text(raw)
        norm_hash = sha_text(normalized or "")
        row = {
            "sequence_id": seq_id,
            "sequence_scope": sequence_scope(region),
            "raw_sequence_sha256": raw_hash,
            "normalized_sequence_sha256": norm_hash,
            "full_sequence_sha256": norm_hash,
            "window_start": None,
            "window_end": None,
            "scaffold": None,
            "editable_mask": None,
            "primary_asset_id": assets[0] if assets else f"{accession}::UNMAPPED_D0_ASSET",
            "contributing_asset_ids": sorted(set(assets)),
            "contributing_source_file_sha256s": sorted(set(file_hashes)),
            "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(file_hashes))}),
            "sequence_reconstruction_rule_id": "D1_SEALED_RECONSTRUCTED_UTR_V1" if bundle.shard == "restricted" else "D1_AUDITED_RAW_RECORD_SEQUENCE_V1",
            "sequence_reconstruction_rule_sha256": sha_text("D1_SEALED_RECONSTRUCTED_UTR_V1" if bundle.shard == "restricted" else "D1_AUDITED_RAW_RECORD_SEQUENCE_V1"),
            "source_record_id": record_id,
            "source_row_locator": locator,
            "normalized_sequence": normalized,
            "normalization_steps": steps,
            "alphabet_status": alphabet,
            "model_sequence_eligible": model_eligible,
            "invalid_symbol_status": invalid_status,
            "region": region_scope(region),
            "species": species,
            "reference_build": as_text(metadata.get("reference_build")) or "NOT_PROVIDED_D1",
            "transcript_release": as_text(metadata.get("transcript_release")) or "NOT_PROVIDED_D1",
            "strand": strand,
        }
        row_hash = bundle.emit("SEQUENCE_ENTITIES", row)
        context_id = self.context(accession, region, metadata, source_file, bundle)
        evidence = [f"RAW_RECORD:{safe_id(record_id)}", "D0_RAW_MANIFEST"]
        self.emit_current_projection(bundle, "SEQUENCE", seq_id, row_hash, model_eligible)
        self.add_assignment(bundle, seq_id, "SEQUENCE", context_id, locator, evidence)
        self.emit_exposure(bundle, "SEQUENCE", seq_id, row_hash, assets, file_hashes, True, False, locator, evidence)
        return seq_id, model_eligible, normalized, context_id, row_hash

    def emit_exposure(self, bundle: Bundle, object_type: str, object_id: str, object_hash: str, assets: list[str], file_hashes: list[str], has_sequence: bool, has_label: bool, locator: str | None, evidence: list[str]) -> None:
        access_id = f"BASELINE:{safe_id(bundle.shard + '|' + object_id)}"
        event_base = {
            "access_id": access_id,
            "object_id": object_id,
            "intent": "D1_CANONICAL_BASELINE",
            "status": "COMPLETION",
            "prev_event_sha256": GENESIS,
        }
        exposure = {
            "exposure_record_id": f"EXP:{safe_id(bundle.shard + '|' + object_id)}",
            "object_id": object_id,
            "object_type": object_type,
            "project_sequence_analytic_exposure": "NONE_CONFIRMED",
            "project_sequence_analytic_use_types": [],
            "project_label_analytic_exposure": "NONE_CONFIRMED",
            "project_label_analytic_use_types": [],
            "pipeline_sequence_materialization": "PRESENT" if has_sequence else "ABSENT",
            "pipeline_label_materialization": "PRESENT" if has_label else "ABSENT",
            "foundation_overlap_requirement": "REQUIRED_FM0_A",
            "foundation_audit_scope_id": "V3_1_D1_FOUNDATION_SCOPE",
            "foundation_overlap_audit_status_at_baseline": "DEFERRED_TO_FM0_A",
            "contributing_asset_ids": sorted(set(assets)),
            "contributing_file_sha256s": sorted(set(file_hashes)),
            "rights_evidence_ids": [f"D0_RIGHTS:{safe_id(x)}" for x in sorted(set(assets))],
            "rights_projection_rule_id": "D1_RIGHTS_CONJUNCTION_V1",
            "rights_projection_rule_sha256": sha_text("D1_RIGHTS_CONJUNCTION_V1"),
            "permitted_model_training": "UNKNOWN",
            "permitted_evaluation": "UNKNOWN",
            "permitted_derived_release": "UNKNOWN",
            "permitted_raw_redistribution": "UNKNOWN",
            "rights_override_id": None,
            "rights_override_reviewer": None,
            "rights_override_scope": None,
            "rights_override_evidence_ids": [],
            "rights_override_sha256": None,
            "canonical_object_sha256": object_hash,
            "evidence_ids": evidence,
        }
        exposure["event_sha256"] = sha_json(event_base)
        exposure = self_hash_row(exposure, "record_sha256")
        bundle.emit("EXPOSURE_RECORDS", exposure)
        pending = {
            "object_id": object_id,
            "object_type": object_type,
            "baseline_exposure_record_id": exposure["exposure_record_id"],
            "baseline_record_sha256": exposure["record_sha256"],
            "effective_exposure": FUTURE_ROLE_SEALED if bundle.shard == "restricted" else FUTURE_ROLE_ORDINARY,
            "effective_project_sequence_analytic_exposure": "NONE_CONFIRMED",
            "effective_project_sequence_use_types": [],
            "effective_project_label_analytic_exposure": "NONE_CONFIRMED",
            "effective_project_label_use_types": [],
            "final_access_status": "SEALED_UNOPENED",
            "projection_phase": "D1",
            "snapshot_id": bundle.snapshot_id,
            "canonical_manifest_sha256": self.canonical_binding,
        }
        bundle.write_effective_pending(pending)
        self.counts[f"exposure:{bundle.shard}:{object_type}"] += 1
        if locator:
            self.emit_transformation(
                bundle,
                object_id,
                object_type,
                object_hash,
                locator,
                evidence,
                raw_line_sha=getattr(self, "current_raw_sha", None),
                raw_id=getattr(self, "current_raw_id", None),
            )

    def emit_transformation(self, bundle: Bundle, child_id: str, object_type: str, new_hash: str, locator: str, evidence: list[str], raw_line_sha: str | None = None, raw_id: str | None = None) -> None:
        old_id = raw_id or f"RAW_UNIT:{bundle.shard}:{safe_id(locator)}"
        old_hash = raw_line_sha or sha_text(locator)
        base = {
            "edge_id": f"EDGE:{safe_id(bundle.shard + '|' + old_id + '|' + child_id)}",
            "old_object_id": old_id,
            "old_object_sha256": old_hash,
            "parent_object_id": old_id,
            "new_object_id": child_id,
            "new_object_sha256": new_hash,
            "child_object_id": child_id,
            "object_type": object_type,
            "reason": "D1_RAW_UNIT_TO_CANONICAL_OBJECT",
            "run_id": self.run_id,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "supersession_edge_id": GENESIS,
        }
        row = self_hash_row(base, "edge_sha256")
        bundle.emit("TRANSFORMATION_EDGES", row)
        self.counts[f"transformation_edges:{bundle.shard}"] += 1

    def emit_rejection(self, bundle: Bundle, accession: str, record_id: str, reason: str, locator: str, assets: list[str], source_unit_id: str | None = None, terminal: str | None = None) -> str:
        rid = f"REJ:{bundle.shard}:{safe_id(record_id + '|' + locator + '|' + reason)}"
        row = {
            "rejection_id": rid,
            "candidate_id": f"CANDIDATE_REJECTED:{safe_id(record_id + '|' + locator)}",
            "reason": reason,
            "evidence_id": f"EVIDENCE:{safe_id(record_id + '|' + locator)}",
            "rejected_at": "D1",
            "source_unit_id": source_unit_id,
            "source_row_locator": locator,
            "asset_ids": sorted(set(assets)),
            "disposition_status": "QUARANTINED" if "LEGACY" in reason or "NO_BINDABLE" in reason else "REJECTED",
            "terminal_disposition_reason": terminal or reason,
        }
        bundle.emit("REJECTIONS", row)
        self.rejection_counts[reason] += 1
        self.shard_rejection_counts[bundle.shard][reason] += 1
        self.counts[f"rejections:{bundle.shard}"] += 1
        return rid

    def emit_current_projection(self, bundle: Bundle, object_type: str, object_id: str, object_hash: str, accepted: bool) -> None:
        base = {
            "projection_record_id": f"PROJECTION:{safe_id(bundle.shard + '|' + object_id)}",
            "run_id": self.run_id,
            "canonical_snapshot_id": self.snapshot_id,
            "object_type": object_type,
            "object_id": object_id,
            "chain_root_object_id": object_id,
            "chain_root_object_sha256": object_hash,
            "current_leaf_object_id": object_id,
            "current_leaf_object_sha256": object_hash,
            "generation_index": 0,
            "chain_length": 0,
            "last_supersession_edge_id": GENESIS,
            "last_supersession_edge_sha256": GENESIS,
            "supersession_manifest_sha256": sha_bytes(b""),
            "is_current_leaf_accepted": bool(accepted),
            "active": bool(accepted),
            "canonical_manifest_sha256": self.canonical_binding,
        }
        base["projection_sha256"] = sha_json(base)
        base["record_sha256"] = sha_json(base)
        bundle.emit("CURRENT_CANONICAL_OBJECT_PROJECTION", base)

    def emit_object(self, bundle: Bundle, object_type: str, object_id: str, row_hash: str, accepted: bool, assets: list[str], file_hashes: list[str], context_id: str, locator: str, evidence: list[str], has_sequence: bool, has_label: bool, endpoint_id: str | None = None, frame: FrameState | None = None, subtype: str = "NOT_APPLICABLE_NON_IDENTITY") -> None:
        self.emit_current_projection(bundle, object_type, object_id, row_hash, accepted)
        self.emit_exposure(bundle, object_type, object_id, row_hash, assets, file_hashes, has_sequence, has_label, locator, evidence)
        self.add_assignment(bundle, object_id, object_type, context_id, locator, evidence, endpoint_id=endpoint_id, frame=frame, subtype=subtype)

    def observation_candidate_row(self, bundle: Bundle, accession: str, record_id: str, label_key: str, numeric_value: float | int | None, sequence_id: str | None, context_id: str, source_file_hash: str | None, assets: list[str], file_hashes: list[str], locator: str, evidence: list[str], valid_sequence: bool, noedit: bool) -> tuple[str, str | None, str, bool]:
        endpoint = self.endpoint_id(label_key)
        candidate_id = f"OBS_CAND:{bundle.shard}:{safe_id(record_id + '|' + label_key)}"
        observation_id = f"OBS:{bundle.shard}:{safe_id(record_id + '|' + label_key)}"
        source_hash = source_file_hash or NOT_AVAILABLE
        accepted = numeric_value is not None and valid_sequence and sequence_id is not None
        if accepted:
            acceptance = "ACCEPTED"
            lifecycle = "ACCEPTED"
            terminal = None
        elif sequence_id is None:
            acceptance = "REJECTED"
            lifecycle = "REJECTED"
            terminal = "NO_BINDABLE_SEQUENCE"
        elif not valid_sequence:
            acceptance = "REJECTED"
            lifecycle = "REJECTED"
            terminal = "INVALID_OR_AMBIGUOUS_SEQUENCE"
        elif numeric_value is None:
            acceptance = "REJECTED"
            lifecycle = "REJECTED"
            terminal = "NULL_OR_NONNUMERIC_LABEL"
        else:
            acceptance = "REJECTED"
            lifecycle = "REJECTED"
            terminal = "UNRESOLVED_LABEL"
        candidate = {
            "observation_candidate_id": candidate_id,
            "candidate_id": candidate_id,
            "asset_ids": sorted(set(assets)),
            "contributing_source_file_sha256s": sorted(set(file_hashes)),
            "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(file_hashes))}),
            "source_unit_ids": [getattr(self, "current_raw_id", f"RAW_RECORD:{safe_id(record_id)}")],
            "sequence_id": sequence_id,
            "context_id": context_id,
            "endpoint_id": endpoint,
            "join_method_id": "D1_DIRECT_RAW_RECORD_LABEL_JOIN_V1",
            "join_method_sha256": sha_text("D1_DIRECT_RAW_RECORD_LABEL_JOIN_V1"),
            "observation_acceptance_status": acceptance,
            "accepted_observation_id": observation_id if accepted else None,
            "terminal_disposition_reason": terminal,
            "source_row_locators": [locator],
            "evidence_ids": evidence,
            "parent_candidate_id": None,
            "source": "D1_RAW_LABEL_JOIN_V1",
            "source_file_sha256": source_hash,
            "value": numeric_value,
            "lifecycle_status": lifecycle,
        }
        candidate_hash = bundle.emit("FUNCTIONAL_OBSERVATION_CANDIDATES", candidate)
        self.emit_current_projection(bundle, "OBSERVATION_CANDIDATE", candidate_id, candidate_hash, accepted)
        if sequence_id is not None:
            self.emit_exposure(bundle, "OBSERVATION_CANDIDATE", candidate_id, candidate_hash, assets, file_hashes, True, numeric_value is not None, locator, evidence)
            self.add_assignment(bundle, candidate_id, "OBSERVATION_CANDIDATE", context_id, locator, evidence, endpoint_id=endpoint, subtype="NOT_APPLICABLE_NON_IDENTITY")
        if not accepted:
            self.emit_rejection(bundle, accession, record_id, terminal or "UNRESOLVED_LABEL", locator, assets, getattr(self, "current_raw_id", None), terminal)
            return candidate_id, None, endpoint, False

        observation = {
            "observation_id": observation_id,
            "observation_candidate_id": candidate_id,
            "canonical_status": "ACCEPTED",
            "parent_observation_id": None,
            "sequence_id": sequence_id,
            "primary_label_asset_id": assets[0] if assets else f"{accession}::UNMAPPED_D0_ASSET",
            "contributing_asset_ids": sorted(set(assets)),
            "contributing_source_file_sha256s": sorted(set(file_hashes)),
            "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(file_hashes))}),
            "source_file_sha256": source_hash,
            "source_record_id": record_id,
            "source_row_locator": locator,
            "scientific_track": "E" if noedit else "F",
            "observation_role": "E_NOEDIT_MEASUREMENT" if noedit else "F_FUNCTION_LABEL",
            "context_id": context_id,
            "endpoint_id": endpoint,
            "raw_value": numeric_value,
            "normalized_value": numeric_value,
            "label_status": "OBSERVED",
            "label_unit": "ASSAY_REPORTED_VALUE",
            "label_transform": "IDENTITY_FLOAT_PARSE_V1",
            "source_replicate_label": as_text(getattr(self, "current_metadata", {}).get("library")) or as_text(getattr(self, "current_metadata", {}).get("source_replicate_label")) or source_file_hash or NOT_AVAILABLE,
            "sample_id": f"SAMPLE:{context_id}",
            "biological_replicate_id": f"BIO_REP:{safe_id(as_text(getattr(self, 'current_metadata', {}).get('biological_replicate_id')) or getattr(self, 'current_source_file', 'UNKNOWN'))}",
            "technical_replicate_id": f"TECH_REP:{safe_id(as_text(getattr(self, 'current_metadata', {}).get('technical_replicate_id')) or getattr(self, 'current_source_file', 'UNKNOWN'))}",
            "barcode_id": as_text(getattr(self, "current_metadata", {}).get("barcode_id")) or "BARCODE_NOT_AVAILABLE",
            "cell_context": as_text(getattr(self, "current_metadata", {}).get("cell_context")) or None,
            "assay_context": as_text(getattr(self, "current_metadata", {}).get("assay_context")) or f"ASSAY:{accession}",
            "promoter": as_text(getattr(self, "current_metadata", {}).get("promoter")) or None,
            "reporter_or_cargo": as_text(getattr(self, "current_metadata", {}).get("reporter_or_cargo")) or None,
            "rna_chemistry": as_text(getattr(self, "current_metadata", {}).get("rna_chemistry")) or None,
            "timepoint": as_text(getattr(self, "current_metadata", {}).get("timepoint")) or None,
            "coverage_or_umi": finite_number(getattr(self, "current_metadata", {}).get("coverage_or_umi") or getattr(self, "current_metadata", {}).get("umi")),
            "standard_error": None,
            "missingness_reason": None,
            "quality_flags": [],
            "value": numeric_value,
            "unit": "ASSAY_REPORTED_VALUE",
            "replicate": None,
        }
        observation_hash = bundle.emit("FUNCTIONAL_OBSERVATIONS", observation)
        self.emit_object(bundle, "OBSERVATION", observation_id, observation_hash, True, assets, file_hashes, context_id, locator, evidence, True, True, endpoint_id=endpoint)
        self.counts[f"observations:{bundle.shard}"] += 1
        self.endpoint_counts[endpoint] += 1
        self.shard_endpoint_counts[bundle.shard][endpoint] += 1
        return candidate_id, observation_id, endpoint, True

    def pairing_method(self, accession: str) -> str:
        if accession in {"GSE114002", "GSE149487", "GSE200304"}:
            return "DESIGN_TABLE"
        if accession in {"GSE186455", "GSE217518", "GSE232572"}:
            return "VARIANT_RECONSTRUCTION"
        if accession == "ENCSR854RUF":
            return "EXPLICIT_ID"
        return "OTHER"

    def pair_rows(self, bundle: Bundle, accession: str, record_id: str, region: Any, metadata: dict[str, Any], source_file: str, assets: list[str], file_hashes: list[str], source_id: str, candidate_id: str, source_valid: bool, candidate_valid: bool, context_id: str, numeric_observations: list[tuple[str, str]], source_sequence: str, candidate_sequence: str, raw_record: dict[str, Any], locator: str, evidence: list[str]) -> tuple[str, str | None, bool]:
        relation_id = f"REL:{bundle.shard}:{safe_id(record_id)}"
        pair_id = f"PAIR:{bundle.shard}:{safe_id(record_id)}"
        exact_identity = as_text(source_sequence) == as_text(candidate_sequence)
        designed_identity = exact_identity and accession == "GSE114002" and Path(source_file).name == "GSM3130443_designed_library.csv.gz"
        if designed_identity:
            relation_type = "NO_EDIT_CONTROL"
            subtype = "DESIGNED_WT_CONTROL"
            relation_status = "ACCEPTED" if source_valid and candidate_valid else "REJECTED"
        elif exact_identity:
            relation_type = "NO_EDIT_CONTROL"
            subtype = "UNRESOLVED_IDENTITY"
            relation_status = "AMBIGUOUS" if source_valid and candidate_valid else "REJECTED"
        else:
            relation_type = "SOURCE_CANDIDATE"
            subtype = "NOT_APPLICABLE_NON_IDENTITY"
            relation_status = "ACCEPTED" if source_valid and candidate_valid and bool(raw_record.get("edit_script_verified", False)) else "REJECTED"
        lifecycle = "ACCEPTED" if relation_status == "ACCEPTED" else ("CANDIDATE" if relation_status == "AMBIGUOUS" else "REJECTED")
        noedit = relation_type == "NO_EDIT_CONTROL"
        if designed_identity:
            self.counts[f"no_edit_controls:{bundle.shard}"] += 1
        role = FUTURE_ROLE_SEALED if bundle.shard == "restricted" else FUTURE_ROLE_ORDINARY
        endpoint = numeric_observations[0][0] if numeric_observations else self.endpoint_id("NO_MEASUREMENT")
        frame = self.frame_for(accession, region, metadata, source_file, context_id, endpoint, assets, as_text(metadata.get("species")) or "UNKNOWN") if designed_identity else None
        effect = "CANDIDATE_ONLY" if numeric_observations else "SEQUENCE_ONLY"
        design_key = as_text(metadata.get("pair_key")) or record_id
        design_group = f"DESIGN_RELATION_GROUP:{safe_id(accession + '|' + design_key)}"
        relation = {
            "relation_candidate_id": relation_id,
            "candidate_id": relation_id,
            "parent_relation_candidate_id": None,
            "design_relation_group_id": design_group,
            "contributing_asset_ids": sorted(set(assets)),
            "contributing_source_file_sha256s": sorted(set(file_hashes)),
            "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(file_hashes))}),
            "relation_context_key": context_id,
            "context_id": context_id,
            "endpoint_id": endpoint,
            "label_unit": "ASSAY_REPORTED_VALUE" if numeric_observations else "NOT_APPLICABLE",
            "label_transform": "IDENTITY_FLOAT_PARSE_V1" if numeric_observations else "NO_VALUE",
            "delta_rule_id": "D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1",
            "delta_rule_sha256": sha_text("D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1"),
            "scientific_track": "E",
            "relation_acceptance_status": relation_status,
            "relation_type": relation_type,
            "effect_evidence": effect,
            "landscape_role": "SPARSE",
            "future_use_role": role,
            "no_edit_control_subtype": subtype,
            "no_edit_sampling_frame_id": frame.group_id if frame else None,
            "pair_evidence_id": f"PAIR_EVIDENCE:{safe_id(record_id)}",
            "terminal_disposition_reason": None if relation_status != "REJECTED" else "INVALID_SEQUENCE_OR_UNVERIFIED_EDIT_SCRIPT",
            "accepted_pair_id": pair_id if relation_status == "ACCEPTED" else None,
            "source_sequence_id": source_id,
            "candidate_sequence_id": candidate_id,
            "pairing_method": self.pairing_method(accession),
            "evidence_id": f"EVIDENCE:{safe_id(record_id)}",
            "lifecycle_status": lifecycle,
        }
        relation_hash = bundle.emit("UTR_EDIT_RELATION_CANDIDATES", relation)
        self.emit_object(bundle, "RELATION_CANDIDATE", relation_id, relation_hash, relation_status == "ACCEPTED", assets, file_hashes, context_id, locator, evidence, True, bool(numeric_observations), endpoint_id=endpoint, frame=frame, subtype=subtype)
        self.counts[f"relation_candidates:{bundle.shard}"] += 1
        if relation_status != "ACCEPTED":
            if relation_status == "AMBIGUOUS":
                self.emit_rejection(bundle, accession, record_id, "UNRESOLVED_IDENTITY_RELATION", locator, assets, getattr(self, "current_raw_id", None), "UNRESOLVED_IDENTITY")
            else:
                self.emit_rejection(bundle, accession, record_id, "RELATION_REJECTED_INVALID_SEQUENCE_OR_SCRIPT", locator, assets, getattr(self, "current_raw_id", None), "INVALID_SEQUENCE_OR_UNVERIFIED_EDIT_SCRIPT")
            return relation_id, None, False

        source_raw = as_text(source_sequence)
        candidate_raw = as_text(candidate_sequence)
        min_distance = finite_number(raw_record.get("edit_distance"))
        if min_distance is None:
            min_distance = finite_number(len(raw_record.get("edit_script") or []))
        if min_distance is None:
            min_distance = abs(len(source_raw) - len(candidate_raw)) + sum(a != b for a, b in zip(source_raw, candidate_raw))
        min_distance = int(min_distance)
        true_length_change = len(candidate_raw) - len(source_raw)
        path_ambiguity = finite_number(raw_record.get("path_ambiguity"))
        if path_ambiguity is None:
            path_ambiguity = 1
        pair = {
            "pair_id": pair_id,
            "relation_candidate_id": relation_id,
            "candidate_id": relation_id,
            "design_relation_group_id": design_group,
            "contributing_asset_ids": sorted(set(assets)),
            "contributing_source_file_sha256s": sorted(set(file_hashes)),
            "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(file_hashes))}),
            "context_id": context_id,
            "endpoint_id": endpoint,
            "label_unit": "ASSAY_REPORTED_VALUE" if numeric_observations else "NOT_APPLICABLE",
            "label_transform": "IDENTITY_FLOAT_PARSE_V1" if numeric_observations else "NO_VALUE",
            "delta_rule_id": "D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1",
            "delta_rule_sha256": sha_text("D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1"),
            "scientific_track": "E",
            "relation_acceptance_status": "ACCEPTED",
            "relation_type": relation_type,
            "effect_evidence": effect,
            "landscape_role": "SPARSE",
            "future_use_role": role,
            "immutable_base_future_use_role": role,
            "source_sequence_id": source_id,
            "candidate_sequence_id": candidate_id,
            "same_assay_context": False,
            "true_length_change": true_length_change,
            "minimum_edit_distance": min_distance,
            "path_ambiguity_count_or_bound": path_ambiguity,
            "pair_direction_verified": bool(raw_record.get("edit_script_verified", False)),
            "pairing_method": self.pairing_method(accession),
            "pair_evidence_id": relation["pair_evidence_id"],
            "evidence_id": relation["evidence_id"],
            "candidate_observation_id": numeric_observations[0][1] if len(numeric_observations) == 1 else None,
            "source_observation_id": None,
            "delta": None,
            "delta_standard_error": None,
            "confirmatory_delta_eligible": False,
            "link_view_eligible": True,
            "exclusion_reason": None,
            "permission_evidence_ids": [f"D0_RIGHTS:{safe_id(x)}" for x in sorted(set(assets))],
            "parent_pair_id": None,
            "biological_parent_group": f"BIOLOGICAL_PARENT:{safe_id(as_text(metadata.get('wt_id')) or as_text(metadata.get('gene')) or record_id)}",
            "gene_group": f"GENE:{token(metadata.get('gene') or metadata.get('gene_symbol'))}" if as_text(metadata.get("gene") or metadata.get("gene_symbol")) else None,
            "tile_family_group": f"TILE_FAMILY:{token(metadata.get('tile_family') or metadata.get('pair_key'))}" if as_text(metadata.get("tile_family") or metadata.get("pair_key")) else None,
            "sequence_cluster_group": f"SEQUENCE_CLUSTER:EXACT:{safe_id(candidate_raw)}",
            "no_edit_control_subtype": subtype,
            "no_edit_sampling_frame_id": frame.group_id if frame else None,
            "join_keys": [f"record_id:{record_id}", f"source_row:{locator}"],
        }
        pair_hash = bundle.emit("UTR_EDIT_PAIRS", pair)
        self.emit_object(bundle, "PAIR", pair_id, pair_hash, True, assets, file_hashes, context_id, locator, evidence, True, bool(numeric_observations), endpoint_id=endpoint, frame=frame, subtype=subtype)
        use_role = {
            "use_role_record_id": f"USE_ROLE:{safe_id(pair_id)}",
            "object_id": pair_id,
            "relation_candidate_id": relation_id,
            "pair_id": pair_id,
            "use_role": "D1_CANONICAL",
            "future_use_role": role,
            "authority_level": "RESTRICTED" if bundle.shard == "restricted" else "ORDINARY",
            "base_future_use_role": role,
            "candidate_base_payload_sha256": relation_hash,
            "pair_base_payload_sha256": pair_hash,
            "canonical_manifest_sha256": self.canonical_binding,
        }
        bundle.emit("USE_ROLES", self_hash_row(use_role, "record_sha256"))
        self.counts[f"pairs:{bundle.shard}"] += 1
        return relation_id, pair_id, True

    def process_record(self, bundle: Bundle, line_no: int, raw_line: str, rec: dict[str, Any], accession_override: str | None = None, source_path: Path | None = None, legacy: bool = False) -> None:
        accession = accession_override or as_text(rec.get("accession")) or "UNKNOWN_ACCESSION"
        record_id = as_text(rec.get("record_id")) or f"MISSING_RECORD_{line_no}"
        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        if bundle.shard == "restricted" and not as_text(metadata.get("source_file")):
            metadata = dict(metadata)
            metadata["source_file"] = self.sealed_input.name
        source_path = source_path or self.raw_input
        locator = row_locator(source_path, line_no)
        self.current_bundle = bundle
        self.current_metadata = metadata
        self.current_source_file = as_text(metadata.get("source_file")) or "<MISSING>"
        self.current_raw_id = f"RAW_RECORD:{bundle.shard}:{line_no}:{safe_id(record_id)}"
        self.current_raw_sha = sha_bytes(raw_line.encode("utf-8"))
        if legacy:
            self.legacy_count += 1
            self.legacy_sha.update(raw_line.encode("utf-8"))
            assets, file_hashes, _, _ = self.source_info(accession, metadata)
            self.emit_rejection(
                bundle,
                accession,
                record_id,
                f"LEGACY_QUARANTINE_{token(accession)}",
                locator,
                assets,
                self.current_raw_id,
                "LEGACY_DATASET_OUTSIDE_D1_EF_CANONICAL",
            )
            return

        assets, file_hashes, primary_file_hash, source_file = self.source_info(accession, metadata)
        region = rec.get("region") or metadata.get("region") or "5'UTR"
        source_sequence_value = rec.get("source_sequence")
        candidate_sequence_value = rec.get("candidate_sequence")
        has_source = bool(as_text(source_sequence_value))
        has_candidate = bool(as_text(candidate_sequence_value))
        self.dataset_counts[accession] += 1
        self.record_type_counts[as_text(metadata.get("record_type")) or "<MISSING>"] += 1
        self.shard_record_type_counts[bundle.shard][as_text(metadata.get("record_type")) or "<MISSING>"] += 1
        self.region_counts[region_scope(region)] += 1
        self.shard_region_counts[bundle.shard][region_scope(region)] += 1
        self.counts[f"raw_records:{bundle.shard}"] += 1
        if has_source:
            self.counts[f"raw_source_sequence_rows:{bundle.shard}"] += 1
        if has_candidate:
            self.counts[f"raw_candidate_sequence_rows:{bundle.shard}"] += 1

        source_id = None
        candidate_id = None
        source_valid = False
        candidate_valid = False
        source_norm = None
        candidate_norm = None
        context_id = self.context(accession, region, metadata, source_file, bundle)
        if has_source:
            source_id, source_valid, source_norm, context_id, _ = self.sequence_row(
                accession,
                record_id,
                "SOURCE",
                source_sequence_value,
                region,
                metadata,
                source_file,
                locator,
                assets,
                file_hashes,
                bundle,
            )
        if has_candidate:
            candidate_id, candidate_valid, candidate_norm, context_id, _ = self.sequence_row(
                accession,
                record_id,
                "CANDIDATE",
                candidate_sequence_value,
                region,
                metadata,
                source_file,
                locator,
                assets,
                file_hashes,
                bundle,
            )

        labels = rec.get("labels") if isinstance(rec.get("labels"), dict) else {}
        numeric_observations: list[tuple[str, str]] = []
        observation_sequence_id = candidate_id if candidate_id is not None else source_id
        observation_sequence_valid = candidate_valid if candidate_id is not None else source_valid
        exact_identity = has_source and has_candidate and as_text(source_sequence_value) == as_text(candidate_sequence_value)
        designed_identity = exact_identity and accession == "GSE114002" and Path(source_file).name == "GSM3130443_designed_library.csv.gz"
        for label_key, raw_value in labels.items():
            label_key = as_text(label_key) or "UNNAMED_LABEL"
            self.label_counts[label_key] += 1
            self.shard_label_counts[bundle.shard][label_key] += 1
            endpoint = self.endpoint_id(label_key)
            label_locator = row_locator(source_path, line_no, f"label={token(label_key)}")
            numeric_value = finite_number(raw_value)
            if observation_sequence_id is None:
                self.emit_rejection(bundle, accession, record_id, "NO_BINDABLE_SEQUENCE_LABEL", label_locator, assets, self.current_raw_id, "NO_BINDABLE_SEQUENCE")
                continue
            _, observation_id, endpoint, accepted = self.observation_candidate_row(
                bundle,
                accession,
                record_id,
                label_key,
                numeric_value,
                observation_sequence_id,
                context_id,
                primary_file_hash,
                assets,
                file_hashes,
                label_locator,
                [f"RAW_RECORD:{safe_id(record_id)}", f"RAW_LABEL:{safe_id(record_id + '|' + label_key)}"],
                observation_sequence_valid,
                designed_identity,
            )
            if accepted and observation_id is not None:
                numeric_observations.append((endpoint, observation_id))
                self.counts[f"label_complete:{bundle.shard}"] += 1
            else:
                self.counts[f"label_rejected:{bundle.shard}"] += 1

        if has_source and has_candidate:
            self.pair_rows(
                bundle,
                accession,
                record_id,
                region,
                metadata,
                source_file,
                assets,
                file_hashes,
                source_id,
                candidate_id,
                source_valid,
                candidate_valid,
                context_id,
                numeric_observations,
                as_text(source_sequence_value),
                as_text(candidate_sequence_value),
                rec,
                locator,
                [f"RAW_RECORD:{safe_id(record_id)}", "D0_RAW_MANIFEST"],
            )
        elif not has_source and not has_candidate:
            self.emit_rejection(bundle, accession, record_id, "NO_BINDABLE_SEQUENCE_RECORD", locator, assets, self.current_raw_id, "NO_BINDABLE_SEQUENCE")
            self.counts[f"raw_no_sequence_rows:{bundle.shard}"] += 1

        if bundle.shard == "ordinary":
            self.input_sha.update(raw_line.encode("utf-8"))

    def process_stream(self, path: Path, bundle: Bundle, accession_override: str | None = None, legacy: bool = False, limit: int | None = None) -> int:
        count = 0
        for line_no, raw_line, rec in read_jsonl(path):
            if limit is not None and count >= limit:
                break
            self.process_record(bundle, line_no, raw_line, rec, accession_override=accession_override, source_path=path, legacy=legacy)
            count += 1
            if count % 200000 == 0:
                print(f"processed {bundle.shard} {count} records", flush=True)
        return count

    def write_endpoint_and_group_rows(self) -> None:
        group_rule_sha = sha_text("D1_SOURCE_CONTEXT_HASH_V1")
        for shard, bundle in (("ordinary", self.ordinary), ("restricted", self.restricted)):
            for endpoint_id, base in sorted(self.endpoints.items()):
                if shard not in self.endpoint_shards.get(endpoint_id, set()):
                    continue
                bundle.emit("ENDPOINT_REGISTRY", self_hash_row(dict(base), "record_sha256"))
            for group_id, state in sorted(self.groups.items()):
                # A context can be used in either shard only when its key is
                # actually observed in that shard.  The membership count is
                # tracked per shard below; if it was not observed, skip it.
                if not getattr(state, f"seen_{shard}", False):
                    continue
                bundle.emit("GROUP_REGISTRY", state.row(group_rule_sha))
            for frame in sorted(self.frame_states.values(), key=lambda x: x.group_id):
                if frame.definition["study_id"] == SEALED_COHORT and shard != "restricted":
                    continue
                if frame.definition["study_id"] != SEALED_COHORT and shard != "ordinary":
                    continue
                bundle.emit("GROUP_REGISTRY", frame.row())

        # Artifact-risk assessments are intentionally aggregate-per-asset,
        # not one row per observation.  PTRE and other uncertain reporter
        # sources remain AUX_QC evidence and never become E/F rows here.
        for asset_id in sorted(self.shard_assets["ordinary"]):
            row = {
                "assessment_id": f"REPORTER_ASSESSMENT:{safe_id(asset_id)}",
                "evidence_id": f"REPORTER_EVIDENCE:{safe_id(asset_id)}",
                "observation_id": f"NO_OBSERVATION_ASSET:{safe_id(asset_id)}",
                "artifact_risk": "MEDIUM",
                "asset_id": asset_id,
                "source_file_sha256": None,
                "assessment_status": "PENDING_D1_ARTIFACT_REVIEW",
                "risk_reason": "D1_SOURCE_LEVEL_REVIEW_RETAINED_AS_AUX_QC",
                "evidence_ids": ["D0_RAW_MANIFEST"],
            }
            self.ordinary.emit("REPORTER_ARTIFACT_ASSESSMENTS", row)

    def write_aggregate_reports(self) -> None:
        ordinary_datasets = {
            accession: {
                "raw_records": count,
                "raw_source_sequence_rows": self.counts.get(f"raw_source_sequence_rows:ordinary:{accession}", 0),
            }
            for accession, count in sorted(self.dataset_counts.items())
            if accession != SEALED_COHORT
        }
        # The detailed per-accession counters are kept in the ordinary report,
        # but no sealed accession, member ID, row locator, label or sequence is
        # ever copied into this aggregate-only namespace.
        ordinary = {
            "artifact_kind": "D1_DATASET_RECONCILIATION",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "ordinary_only": True,
            "row_level_member_ids_emitted": False,
            "row_level_source_locators_emitted": False,
            "dataset_counts": {k: v for k, v in sorted(self.dataset_counts.items()) if k != SEALED_COHORT},
            "record_type_counts": dict(sorted(self.shard_record_type_counts["ordinary"].items())),
            "region_counts": dict(sorted(self.shard_region_counts["ordinary"].items())),
            "label_key_counts": dict(sorted(self.shard_label_counts["ordinary"].items())),
            "endpoint_counts": dict(sorted(self.shard_endpoint_counts["ordinary"].items())),
            "rejection_counts": dict(sorted(self.shard_rejection_counts["ordinary"].items())),
            "raw_ordinary_record_count": self.counts.get("raw_records:ordinary", 0),
            "ordinary_relation_candidate_count": self.counts.get("relation_candidates:ordinary", 0),
            "ordinary_pair_count": self.counts.get("pairs:ordinary", 0),
            "ordinary_observation_count": self.counts.get("observations:ordinary", 0),
            "ordinary_exposure_count": sum(v for k, v in self.counts.items() if k.startswith("exposure:ordinary:")),
            "legacy_quarantine_record_count": self.legacy_count,
            "sealed_member_level_rows_in_ordinary": 0,
            "sealed_cohort_aggregate_present": False,
            "d1_claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
            "g3b_status": "DEFERRED_TO_FM0_A",
        }
        json_write(self.ordinary.canonical_paths["DATASET_RECONCILIATION"], ordinary, "manifest_sha256")
        units = {
            "artifact_kind": "D1_DATA_UNITS_REPORT",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "ordinary_only": True,
            "tracks": {
                "E": {
                    "relation_candidates": self.counts.get("relation_candidates:ordinary", 0),
                    "accepted_pairs": self.counts.get("pairs:ordinary", 0),
                    "future_role": FUTURE_ROLE_ORDINARY,
                    "delta_eligible": 0,
                    "link_view_eligible": self.counts.get("pairs:ordinary", 0),
                    "no_edit_controls": self.counts.get("no_edit_controls:ordinary", 0),
                },
                "F": {
                    "observation_candidates": sum(v for k, v in self.counts.items() if k.startswith("observations:ordinary")) + sum(v for k, v in self.counts.items() if k.startswith("label_rejected:ordinary")),
                    "accepted_observations": self.counts.get("observations:ordinary", 0),
                    "current_leaf_label_complete": self.counts.get("observations:ordinary", 0),
                    "future_role": FUTURE_ROLE_ORDINARY,
                },
            },
            "attrition": dict(sorted(self.shard_rejection_counts["ordinary"].items())),
            "row_level_member_ids_emitted": False,
            "row_level_source_locators_emitted": False,
            "sealed_member_level_rows": 0,
            "d1_claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
        }
        json_write(self.ordinary.canonical_paths["DATA_UNITS_REPORT"], units, "report_sha256")

        restricted = {
            "artifact_kind": "D1_DATASET_RECONCILIATION",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "cohort_id": SEALED_COHORT,
            "restricted_only": True,
            "row_level_member_ids_emitted": True,
            "row_level_source_locators_emitted": True,
            "dataset_counts": {SEALED_COHORT: self.dataset_counts.get(SEALED_COHORT, 0)},
            "record_type_counts": dict(sorted(self.shard_record_type_counts["restricted"].items())),
            "region_counts": dict(sorted(self.shard_region_counts["restricted"].items())),
            "label_key_counts": dict(sorted(self.shard_label_counts["restricted"].items())),
            "endpoint_counts": dict(sorted(self.shard_endpoint_counts["restricted"].items())),
            "rejection_counts": dict(sorted(self.shard_rejection_counts["restricted"].items())),
            "restricted_record_count": self.counts.get("raw_records:restricted", 0),
            "restricted_relation_candidate_count": self.counts.get("relation_candidates:restricted", 0),
            "restricted_pair_count": self.counts.get("pairs:restricted", 0),
            "restricted_observation_count": self.counts.get("observations:restricted", 0),
            "d1_claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
            "g3b_status": "DEFERRED_TO_FM0_A",
        }
        json_write(self.restricted.canonical_paths["DATASET_RECONCILIATION"], restricted, "manifest_sha256")
        r_units = dict(units)
        r_units.update({
            "cohort_id": SEALED_COHORT,
            "restricted_only": True,
            "row_level_member_ids_emitted": True,
            "row_level_source_locators_emitted": True,
            "sealed_member_level_rows": self.counts.get("raw_records:restricted", 0),
            "tracks": {
                "E": {
                    "relation_candidates": self.counts.get("relation_candidates:restricted", 0),
                    "accepted_pairs": self.counts.get("pairs:restricted", 0),
                    "future_role": FUTURE_ROLE_SEALED,
                    "delta_eligible": 0,
                },
                "F": {
                    "accepted_observations": self.counts.get("observations:restricted", 0),
                    "future_role": FUTURE_ROLE_SEALED,
                },
            },
        })
        json_write(self.restricted.canonical_paths["DATA_UNITS_REPORT"], r_units, "report_sha256")

    def write_input_manifests(self) -> tuple[Path, Path]:
        d0_manifest = self.d0.root / "D0_R_MANIFEST.json"
        c3_candidates = [
            self.args.c3_root / "C3_MANIFEST.json",
            self.args.c3_root / "C3_SCHEMA_MANIFEST.json",
            self.args.c3_root / "C3_SHA256SUMS",
        ]
        c3_path = next((p for p in c3_candidates if p.exists()), None)
        ordinary_input = {
            "manifest_id": f"D1_INPUT:{self.run_id}:ORDINARY",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "source_head": self.source_head,
            "code_commit": self.code_commit,
            "ordinary_raw_input": {
                "path": str(self.raw_input),
                "sha256": self.input_sha.hexdigest(),
                "record_count": self.counts.get("raw_records:ordinary", 0),
            },
            "legacy_quarantine_input": {
                "path": str(self.legacy_input) if self.legacy_input else None,
                "sha256": sha_file(self.legacy_input) if self.legacy_input and self.legacy_input.exists() else None,
                "record_count": self.legacy_count,
            },
            "d0_rebind": {
                "root": str(self.d0.root),
                "D0_R_MANIFEST_sha256": sha_file(d0_manifest) if d0_manifest.exists() else None,
                "raw_asset_manifest_sha256": self.d0.raw_manifest_sha256,
                "dataset_assets_sha256": self.d0.dataset_asset_manifest_sha256,
                "dataset_decisions_sha256": self.d0.decisions_sha256,
            },
            "c3_parent": {
                "root": str(self.args.c3_root),
                "artifact_path": str(c3_path) if c3_path else None,
                "artifact_sha256": sha_file(c3_path) if c3_path else None,
            },
            "raw_sequence_parsing_started": True,
            "model_training_started": False,
            "sealed_final_accessed": False,
        }
        ordinary_path = self.out / "work" / "D1_INPUT_MANIFEST.json"
        ordinary_sha = json_write(ordinary_path, ordinary_input, "manifest_sha256")

        sealed_root = self.sealed_input.parent
        expected_names = [
            "GSE246381_hek_combined_umi_counts.csv.gz",
            "GSE246381_vglut_combined_umi_counts.csv.gz",
            self.sealed_input.name,
        ]
        inputs = []
        for name in expected_names:
            p = sealed_root / name
            if not p.exists():
                raise FileNotFoundError(p)
            inputs.append({"relative_path": name, "sha256": sha_file(p), "byte_size": p.stat().st_size})
        matrix_profiles = [profile_matrix(sealed_root / name) for name in expected_names if name.endswith(".csv.gz")]
        reconstructed_count = self.counts.get("raw_records:restricted", 0)
        sealed_input = {
            "manifest_id": f"SEALED_INPUT:{self.run_id}:{SEALED_COHORT}",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "access_prefix_snapshot_id": self.snapshot_id,
            "cohort_ids": [SEALED_COHORT],
            "cohort_set_sha256": SEALED_COHORT_SET_SHA256,
            "input_files": inputs,
            "input_file_set_sha256": set_sha([x["relative_path"] + "|" + x["sha256"] for x in inputs]),
            "matrix_profiles": matrix_profiles,
            "reconstructed_record_count": reconstructed_count,
            "row_level_values_emitted_to_ordinary": False,
            "aggregate_only_return_boundary": True,
            "prior_analytic_use": "NONE_CONFIRMED",
            "pipeline_materialization": "PRESENT",
            "future_use_role": FUTURE_ROLE_SEALED,
            "access_policy": "RESTRICTED_BUILDER_MACHINE_ONLY",
        }
        sealed_path = self.restricted_root / "SEALED_INPUT_MANIFEST.json"
        sealed_sha = json_write(sealed_path, sealed_input, "manifest_sha256")
        self.ordinary_input_manifest_sha = ordinary_sha
        self.sealed_input_manifest_sha = sealed_sha
        self.sealed_input_profiles = matrix_profiles
        return ordinary_path, sealed_path

    def output_binding_hash(self, bundle: Bundle) -> str:
        paths = []
        for key, path in sorted(bundle.canonical_paths.items()):
            if key in {"CANONICAL_MANIFEST", "CANONICAL_SHA256SUMS", "REPORTER_ARTIFACT_ASSESSMENTS"}:
                continue
            if path.exists():
                paths.append({"logical_id": key, "relative_path": bundle.relative(path), "sha256": sha_file(path)})
        paths.append({"logical_id": "OBJECT_SEQUENCE_SET", "relative_path": bundle.relative(bundle.object_seq_path), "sha256": sha_file(bundle.object_seq_path)})
        paths.append({"logical_id": "OBJECT_LABEL_SET", "relative_path": bundle.relative(bundle.object_label_path), "sha256": sha_file(bundle.object_label_path)})
        return sha_json({"run_id": self.run_id, "snapshot_id": self.snapshot_id, "shard": bundle.shard, "components": paths})

    def access_event(self, base: dict[str, Any]) -> dict[str, Any]:
        row = dict(base)
        row["event_sha256"] = sha_json(row)
        return row

    def build_access_chain(self, bundle: Bundle) -> dict[str, Any]:
        output_binding = self.output_binding_hash(bundle)
        seq_hash = sha_file(bundle.object_seq_path)
        label_hash = sha_file(bundle.object_label_path)
        executable_hash = sha_file(Path(__file__).resolve())
        environment_hash = sha_text(f"python={sys.version}|platform={sys.platform}|code_commit={self.code_commit}")
        input_hash = self.ordinary_input_manifest_sha if bundle.shard == "ordinary" else self.sealed_input_manifest_sha
        object_id = "D1_ORDINARY_BUILDER" if bundle.shard == "ordinary" else "GSE246381_D1_RESTRICTED_BUILDER"
        requested_sequence_scope = "ALL_ORDINARY_D1_INPUT_SEQUENCES" if bundle.shard == "ordinary" else "ALL_SEALED_INPUT_SEQUENCES"
        requested_label_scope = "ALL_ORDINARY_D1_INPUT_LABELS" if bundle.shard == "ordinary" else "ALL_SEALED_INPUT_LABELS"
        access_id = f"ACCESS:{bundle.shard}:D1:{self.snapshot_id}"
        intent_id = f"EVENT:{bundle.shard}:D1:INTENT"
        common = {
            "access_id": access_id,
            "event_type": "RESTRICTED_BUILDER_PARSE",
            "intent": "D1_STRICT_BUILDER_PARSE",
            "object_id": object_id,
            "actor_identity": "mrna-editflow-d1-strict-builder",
            "executable_sha256": executable_hash,
            "container_or_environment_sha256": environment_hash,
            "input_manifest_sha256": input_hash,
            "requested_sequence_scope": requested_sequence_scope,
            "requested_label_scope": requested_label_scope,
            "requested_sequence_object_set_manifest_sha256": seq_hash,
            "requested_label_object_set_manifest_sha256": label_hash,
            "output_schema_id": "D1_STRICT_CANONICAL_BUNDLE_V1",
            "output_schema_sha256": sha_text("D1_STRICT_CANONICAL_BUNDLE_V1"),
            "analytic_access": False,
            "requested_state_transition": "D1_BUILDER_PARSE_WITHOUT_ANALYTIC_EXPOSURE",
            "reason": "D1_CONTRACT_MACHINE_PARSE_ONLY",
            "failure_evidence_ids": [],
        }
        intent = dict(common)
        intent.update({
            "event_id": intent_id,
            "log_sequence_no": 0,
            "predecessor_event_id": GENESIS,
            "predecessor_event_sha256": GENESIS,
            "prev_event_sha256": GENESIS,
            "timestamp": now_utc(),
            "status": "INTENT",
            "intent_event_id": None,
            "sequence_rows_touched": None,
            "label_rows_touched": None,
            "actual_sequence_object_set_manifest_sha256": None,
            "actual_label_object_set_manifest_sha256": None,
            "output_manifest_sha256": None,
            "partial_actual_set_status": None,
            "realized_state_transition": None,
        })
        intent = self.access_event(intent)
        completion = dict(common)
        completion.update({
            "event_id": f"EVENT:{bundle.shard}:D1:COMPLETION",
            "log_sequence_no": 1,
            "predecessor_event_id": intent["event_id"],
            "predecessor_event_sha256": intent["event_sha256"],
            "prev_event_sha256": intent["event_sha256"],
            "timestamp": now_utc(),
            "status": "COMPLETION",
            "intent_event_id": intent["event_id"],
            "sequence_rows_touched": bundle.object_counts.get("SEQUENCE", 0),
            "label_rows_touched": sum(bundle.object_counts.get(k, 0) for k in ("OBSERVATION_CANDIDATE", "OBSERVATION", "RELATION_CANDIDATE", "PAIR")),
            "actual_sequence_object_set_manifest_sha256": seq_hash,
            "actual_label_object_set_manifest_sha256": label_hash,
            "output_manifest_sha256": output_binding,
            "partial_actual_set_status": "COMPLETE",
            "realized_state_transition": "D1_BUILDER_PARSE_COMPLETED_WITHOUT_ANALYTIC_ACCESS",
        })
        completion = self.access_event(completion)
        bundle.live_access.parent.mkdir(parents=True, exist_ok=True)
        bundle.live_access.write_bytes(jline(intent) + jline(completion))
        with bundle.live_access.open("rb") as fh:
            os.fsync(fh.fileno())
        snapshot_log = bundle.snapshot_dir / bundle.access_log_name
        shutil.copyfile(bundle.live_access, snapshot_log)
        requested_paths = [bundle.object_seq_path, bundle.object_label_path]
        actual_paths = [bundle.object_seq_actual_path, bundle.object_label_actual_path]
        access_sums = bundle.snapshot_dir / bundle.access_sums_name
        access_sum_sha = checksum_ledger(bundle.root, [snapshot_log, *requested_paths, *actual_paths], access_sums)
        first_event = intent["event_id"]
        last_event = completion["event_id"]
        access_manifest_base = {
            "manifest_id": f"ACCESS_PREFIX:{bundle.shard}:{self.snapshot_id}",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "phase": "D1",
            "snapshot_id": self.snapshot_id,
            "restricted_snapshot_id": self.snapshot_id if bundle.shard == "restricted" else None,
            "ordinary_snapshot_id": self.snapshot_id if bundle.shard == "ordinary" else None,
            "cohort_ids": [SEALED_COHORT] if bundle.shard == "restricted" else ["ORDINARY_NONSEALED"],
            "cohort_set_sha256": SEALED_COHORT_SET_SHA256 if bundle.shard == "restricted" else set_sha(["ORDINARY_NONSEALED"]),
            "live_access_log_relpath": bundle.relative(bundle.live_access),
            "snapshot_access_log_relpath": bundle.relative(snapshot_log),
            "snapshot_access_log_sha256": sha_file(snapshot_log),
            "access_log_schema_id": "exposure_record.schema.json#/$defs/AccessIntent|AccessCompletion|AccessAbort",
            "access_log_schema_sha256": self.schema_hashes.get("exposure_record.schema.json", NOT_AVAILABLE),
            "event_count": 2,
            "first_event_id": first_event,
            "last_event_id": last_event,
            "access_log_chain_root_sha256": completion["event_sha256"],
            "requested_object_set_manifest_relpaths": sorted(bundle.relative(p) for p in requested_paths),
            "actual_object_set_manifest_relpaths": sorted(bundle.relative(p) for p in actual_paths),
            "worm_receipt_relpaths": [],
            "worm_receipt_sha256s": [],
            "access_sha256s_relpath": bundle.relative(access_sums),
            "access_sha256s_sha256": access_sum_sha,
            "live_prefix_match_at_snapshot": bundle.live_access.read_bytes() == snapshot_log.read_bytes(),
        }
        access_manifest = bundle.snapshot_dir / bundle.access_manifest_name
        access_manifest_sha = json_write(access_manifest, access_manifest_base, "manifest_sha256")
        return {
            "intent": intent,
            "completion": completion,
            "chain_root": completion["event_sha256"],
            "snapshot_log": snapshot_log,
            "access_manifest": access_manifest,
            "access_sums": access_sums,
            "access_manifest_sha256": access_manifest_sha,
            "access_sums_sha256": access_sum_sha,
        }

    def write_exposure_use_manifest(self, bundle: Bundle, access: dict[str, Any], effective_path: Path) -> dict[str, Any]:
        exposure_path = bundle.canonical_paths["EXPOSURE_RECORDS"]
        use_path = bundle.canonical_paths["USE_ROLES"]
        sums_path = bundle.canonical_paths["EXPOSURE_USE_SHA256SUMS"]
        use_sum_sha = checksum_ledger(bundle.root, [exposure_path, use_path, effective_path], sums_path)
        base = {
            "manifest_id": f"EXPOSURE_USE:{bundle.shard}:{self.snapshot_id}",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "shard": bundle.shard,
            "canonical_binding_sha256": self.canonical_binding,
            "exposure_records_relpath": bundle.relative(exposure_path),
            "exposure_records_sha256": sha_file(exposure_path),
            "use_roles_relpath": bundle.relative(use_path),
            "use_roles_sha256": sha_file(use_path),
            "effective_exposure_projection_relpath": bundle.relative(effective_path),
            "effective_exposure_projection_sha256": sha_file(effective_path),
            "access_snapshot_id": self.snapshot_id,
            "access_log_chain_root_sha256": access["chain_root"],
            "access_manifest_sha256": access["access_manifest_sha256"],
            "access_sha256s_sha256": access["access_sums_sha256"],
            "exposure_use_sha256s_sha256": use_sum_sha,
        }
        manifest_path = bundle.canonical_paths["EXPOSURE_USE_MANIFEST"]
        manifest_sha = json_write(manifest_path, base, "manifest_sha256")
        return {"manifest": manifest_path, "manifest_sha256": manifest_sha, "sums": sums_path, "sums_sha256": use_sum_sha}

    def schema_for_component(self, logical_id: str, path: Path) -> tuple[str, str]:
        name = path.name
        mapping = {
            "sequence_entities.jsonl": "sequence_entity.schema.json",
            "functional_observation_candidates.jsonl": "functional_observation.schema.json#/$defs/FunctionalObservationCandidate",
            "functional_observations.jsonl": "functional_observation.schema.json",
            "ENDPOINT_REGISTRY.jsonl": "functional_observation.schema.json#/$defs/EndpointRegistryRow",
            "utr_edit_relation_candidates.jsonl": "utr_edit_relation_candidate.schema.json",
            "utr_edit_pairs.jsonl": "utr_edit_pair.schema.json",
            "rejections.jsonl": "rejection_record.schema.json",
            "transformation_edges.jsonl": "transformation_edge.schema.json",
            "SUPERSESSION_EDGES.jsonl": "transformation_edge.schema.json#/$defs/SupersessionEdge",
            "CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl": "transformation_edge.schema.json#/$defs/CurrentCanonicalObjectProjection",
            "EXPOSURE_RECORDS.jsonl": "exposure_record.schema.json",
            "USE_ROLES.jsonl": "use_role.schema.json",
            "group_registry.jsonl": "group_registry.schema.json",
            "group_assignments.jsonl": "group_assignment.schema.json",
            "reporter_artifact_assessments.jsonl": "reporter_artifact_assessment.schema.json",
            "EFFECTIVE_EXPOSURE_PROJECTION.jsonl": "exposure_record.schema.json#/$defs/EffectiveExposureProjection",
        }
        if logical_id in {"ACCESS_LOG", "ACCESS_MANIFEST"} or name in {"ACCESS_LOG.jsonl", "ORDINARY_ACCESS_LOG.jsonl", "ACCESS_MANIFEST.json", "ORDINARY_ACCESS_MANIFEST.json"}:
            return "exposure_record.schema.json#/$defs/AccessIntent|AccessCompletion|AccessAbort", self.schema_hashes.get("exposure_record.schema.json", NOT_AVAILABLE)
        schema_id = mapping.get(name)
        if schema_id is None:
            schema_id = "AGGREGATE_REPORT_V1"
            return schema_id, sha_text(schema_id)
        base_name = schema_id.split("#", 1)[0]
        return schema_id, self.schema_hashes.get(base_name, sha_text(schema_id))

    def canonical_component_paths(self, bundle: Bundle, access: dict[str, Any], effective_path: Path, include_sealed_input: bool = False) -> list[Path]:
        paths = []
        for key, path in sorted(bundle.canonical_paths.items()):
            if key in {"CANONICAL_MANIFEST", "CANONICAL_SHA256SUMS"}:
                continue
            if key == "REPORTER_ARTIFACT_ASSESSMENTS" and bundle.shard == "restricted":
                continue
            if path.exists():
                paths.append(path)
        paths.append(effective_path)
        paths.extend([access["snapshot_log"], access["access_manifest"], access["access_sums"]])
        if include_sealed_input:
            paths.append(self.restricted_root / "SEALED_INPUT_MANIFEST.json")
        return paths

    def write_canonical_manifest(self, bundle: Bundle, access: dict[str, Any], effective_path: Path, exposure_use: dict[str, Any]) -> dict[str, Any]:
        include_sealed = bundle.shard == "restricted"
        component_paths = self.canonical_component_paths(bundle, access, effective_path, include_sealed_input=False)
        sums_path = bundle.canonical_paths["CANONICAL_SHA256SUMS"]
        sums_sha = checksum_ledger(self.out if bundle.shard == "ordinary" else bundle.root, component_paths, sums_path)
        components = []
        for path in sorted(component_paths, key=lambda p: bundle.relative(p).encode("utf-8")):
            logical_id = path.name
            if path == effective_path:
                logical_id = "EFFECTIVE_EXPOSURE_PROJECTION"
            elif path == access["snapshot_log"]:
                logical_id = "ACCESS_LOG"
            elif path == access["access_manifest"]:
                logical_id = "ACCESS_MANIFEST"
            elif path == access["access_sums"]:
                logical_id = "ACCESS_SHA256SUMS"
            elif path.name == "dataset_reconciliation.json":
                logical_id = "DATASET_RECONCILIATION"
            elif path.name == "data_units_report.json":
                logical_id = "DATA_UNITS_REPORT"
            elif path.name == "EXPOSURE_USE_MANIFEST.json":
                logical_id = "EXPOSURE_USE_MANIFEST"
            elif path.name == "EXPOSURE_USE_SHA256SUMS":
                logical_id = "EXPOSURE_USE_SHA256SUMS"
            schema_id, schema_sha = self.schema_for_component(logical_id, path)
            components.append({"logical_id": logical_id, "relative_path": bundle.relative(path), "sha256": sha_file(path), "schema_id": schema_id, "schema_sha256": schema_sha})
        base = {
            "manifest_id": f"CANONICAL:{bundle.shard}:{self.run_id}",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "shard": bundle.shard,
            "source_head": self.source_head,
            "code_commit": self.code_commit,
            "canonical_binding_sha256": self.canonical_binding,
            "components": components,
            "component_count": len(components),
            "canonical_sha256s_relpath": bundle.relative(sums_path),
            "canonical_sha256s_sha256": sums_sha,
            "exposure_use_manifest_sha256": exposure_use["manifest_sha256"],
            "exposure_use_sha256s_sha256": exposure_use["sums_sha256"],
            "effective_exposure_projection_sha256": sha_file(effective_path),
            "access_manifest_sha256": access["access_manifest_sha256"],
            "access_sha256s_sha256": access["access_sums_sha256"],
            "access_log_chain_root_sha256": access["chain_root"],
            "d1_claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
            "g3b_status": "DEFERRED_TO_FM0_A",
        }
        manifest_path = bundle.canonical_paths["CANONICAL_MANIFEST"]
        manifest_sha = json_write(manifest_path, base, "manifest_sha256")
        return {"manifest": manifest_path, "manifest_sha256": manifest_sha, "sums": sums_path, "sums_sha256": sums_sha, "components": components}

    def write_sealed_qc_and_manifest(self, access: dict[str, Any], effective_path: Path, exposure_use: dict[str, Any]) -> dict[str, Any]:
        qc_dir = self.ordinary_root / "sealed_commitments"
        qc_dir.mkdir(parents=True, exist_ok=True)
        matrix_profiles = getattr(self, "sealed_input_profiles", [])
        matrix_rows = [p.get("rows") for p in matrix_profiles]
        matrix_row_pass = all(x == 32990 for x in matrix_rows) if matrix_rows else False
        restricted_raw = self.counts.get("raw_records:restricted", 0)
        restricted_pairs = self.counts.get("pairs:restricted", 0)
        restricted_noedit = self.counts.get("no_edit_controls:restricted", 0)
        restricted_failures = self.shard_rejection_counts["restricted"].get("INVALID_OR_AMBIGUOUS_SEQUENCE", 0) + self.shard_rejection_counts["restricted"].get("NO_BINDABLE_SEQUENCE_RECORD", 0)
        aggregate_qc = {
            "artifact_kind": "GSE246381_AGGREGATE_QC",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "cohort_id": SEALED_COHORT,
            "prior_analytic_use": "NONE_CONFIRMED",
            "pipeline_materialization": "PRESENT",
            "future_use_role": FUTURE_ROLE_SEALED,
            "row_level_values_emitted_to_ordinary": False,
            "machine_access_only": True,
            "matrix_profiles": matrix_profiles,
            "reconstructed_record_count": restricted_raw,
            "accepted_pair_count": restricted_pairs,
            "no_edit_control_count": restricted_noedit,
            "reconstructed_failure_count": restricted_failures,
            "expected_contract_denominators": {
                "matrix_rows_each": 32990,
                "variant_keys": 1507,
                "reconstructed_records": 1300,
                "reconstructed_failures": 207,
                "no_edit_controls": 116,
                "edits": 1184,
                "control_shuffles_each": 1500,
                "control_rows_each": 1350,
                "windows": 347,
            },
            "measured_reconstructed_record_count_pass": restricted_raw == 1300,
            "measured_matrix_row_count_pass": matrix_row_pass,
            "conservation_status": "PASS" if restricted_raw == 1300 and matrix_row_pass else "PENDING_REVIEW",
            "qualification_status": "PENDING_FM0_A",
            "ordinary_allowlisted_output": "AGGREGATE_COUNTS_STATUS_COMMITMENT_ONLY",
            "d1_claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
        }
        qc_path = qc_dir / "GSE246381_AGGREGATE_QC.json"
        qc_sha = json_write(qc_path, aggregate_qc, "aggregate_qc_sha256")

        sealed_payload = []
        sealed_input = self.restricted_root / "SEALED_INPUT_MANIFEST.json"
        sealed_payload.append(sealed_input)
        for key, path in sorted(self.restricted.canonical_paths.items()):
            if key in {"CANONICAL_MANIFEST", "CANONICAL_SHA256SUMS", "REPORTER_ARTIFACT_ASSESSMENTS"}:
                continue
            if path.exists():
                sealed_payload.append(path)
        sealed_payload.append(effective_path)
        sealed_sums = self.restricted_root / "SEALED_CANONICAL_SHA256SUMS"
        sealed_sums_sha = checksum_ledger(self.restricted_root, sealed_payload, sealed_sums)

        logical_paths: dict[str, Path] = {
            "ACCESS_LOG": access["snapshot_log"],
            "ACCESS_MANIFEST": access["access_manifest"],
            "ACCESS_SHA256SUMS": access["access_sums"],
            "CURRENT_CANONICAL_OBJECT_PROJECTION": self.restricted.canonical_paths["CURRENT_CANONICAL_OBJECT_PROJECTION"],
            "DATASET_RECONCILIATION": self.restricted.canonical_paths["DATASET_RECONCILIATION"],
            "DATA_UNITS_REPORT": self.restricted.canonical_paths["DATA_UNITS_REPORT"],
            "EFFECTIVE_EXPOSURE_PROJECTION": effective_path,
            "ENDPOINT_REGISTRY": self.restricted.canonical_paths["ENDPOINT_REGISTRY"],
            "EXPOSURE_RECORDS": self.restricted.canonical_paths["EXPOSURE_RECORDS"],
            "EXPOSURE_USE_MANIFEST": self.restricted.canonical_paths["EXPOSURE_USE_MANIFEST"],
            "EXPOSURE_USE_SHA256SUMS": self.restricted.canonical_paths["EXPOSURE_USE_SHA256SUMS"],
            "FUNCTIONAL_OBSERVATION_CANDIDATES": self.restricted.canonical_paths["FUNCTIONAL_OBSERVATION_CANDIDATES"],
            "FUNCTIONAL_OBSERVATIONS": self.restricted.canonical_paths["FUNCTIONAL_OBSERVATIONS"],
            "GROUP_ASSIGNMENTS": self.restricted.canonical_paths["GROUP_ASSIGNMENTS"],
            "GROUP_REGISTRY": self.restricted.canonical_paths["GROUP_REGISTRY"],
            "REJECTIONS": self.restricted.canonical_paths["REJECTIONS"],
            "SEALED_CANONICAL_SHA256SUMS": sealed_sums,
            "SEALED_INPUT_MANIFEST": sealed_input,
            "SEQUENCE_ENTITIES": self.restricted.canonical_paths["SEQUENCE_ENTITIES"],
            "SUPERSESSION_EDGES": self.restricted.canonical_paths["SUPERSESSION_EDGES"],
            "TRANSFORMATION_EDGES": self.restricted.canonical_paths["TRANSFORMATION_EDGES"],
            "USE_ROLES": self.restricted.canonical_paths["USE_ROLES"],
            "UTR_EDIT_PAIRS": self.restricted.canonical_paths["UTR_EDIT_PAIRS"],
            "UTR_EDIT_RELATION_CANDIDATES": self.restricted.canonical_paths["UTR_EDIT_RELATION_CANDIDATES"],
        }
        if sorted(logical_paths) != sorted(RESTRICTED_LOGICAL_IDS):
            raise RuntimeError("restricted logical component set mismatch")
        if set_sha(logical_paths) != RESTRICTED_COMPONENT_SET_SHA256:
            raise RuntimeError("restricted logical component set hash mismatch")
        components = []
        for logical_id in sorted(RESTRICTED_LOGICAL_IDS):
            path = logical_paths[logical_id]
            if logical_id == "SEALED_INPUT_MANIFEST":
                schema_id, schema_sha = "SEALED_INPUT_MANIFEST_V1", sha_text("SEALED_INPUT_MANIFEST_V1")
            elif logical_id in {"DATASET_RECONCILIATION", "DATA_UNITS_REPORT", "EXPOSURE_USE_MANIFEST"}:
                schema_id, schema_sha = "AGGREGATE_REPORT_V1", sha_text("AGGREGATE_REPORT_V1")
            elif logical_id in {"ACCESS_SHA256SUMS", "EXPOSURE_USE_SHA256SUMS", "SEALED_CANONICAL_SHA256SUMS"}:
                schema_id, schema_sha = NON_JSON_SCHEMA_ID, NON_JSON_SCHEMA_SHA256
            else:
                schema_id, schema_sha = self.schema_for_component(logical_id, path)
            components.append({
                "logical_id": logical_id,
                "relative_path": self.restricted.relative(path),
                "sha256": sha_file(path),
                "schema_id": schema_id,
                "schema_sha256": schema_sha,
            })
        sealed_manifest_base = {
            "manifest_id": f"SEALED_CANONICAL:{self.run_id}:{SEALED_COHORT}",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "access_prefix_snapshot_id": self.snapshot_id,
            "cohort_ids": [SEALED_COHORT],
            "cohort_set_sha256": SEALED_COHORT_SET_SHA256,
            "logical_component_set_sha256": RESTRICTED_COMPONENT_SET_SHA256,
            "logical_components": components,
            "sealed_canonical_sha256s_sha256": sealed_sums_sha,
            "access_manifest_sha256": access["access_manifest_sha256"],
            "access_sha256s_sha256": access["access_sums_sha256"],
            "access_log_chain_root_sha256": access["chain_root"],
            "exposure_use_manifest_sha256": exposure_use["manifest_sha256"],
            "effective_exposure_projection_sha256": sha_file(effective_path),
        }
        sealed_manifest = self.restricted_root / "SEALED_CANONICAL_MANIFEST.json"
        sealed_manifest_sha = json_write(sealed_manifest, sealed_manifest_base, "manifest_sha256")
        commitment = {
            "artifact_kind": "GSE246381_D1_COMMITMENT",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "access_prefix_snapshot_id": self.snapshot_id,
            "cohort_id": SEALED_COHORT,
            "status": "SEALED_EXTERNAL_FINAL_CANDIDATE",
            "aggregate_qc_sha256": qc_sha,
            "aggregate_qc_full_file_sha256": sha_file(qc_path),
            "sealed_canonical_manifest_sha256": sha_file(sealed_manifest),
            "sealed_canonical_manifest_self_hash": sealed_manifest_sha,
            "sealed_canonical_sha256s_sha256": sealed_sums_sha,
            "access_manifest_sha256": access["access_manifest_sha256"],
            "access_sha256s_sha256": access["access_sums_sha256"],
            "access_log_chain_root_sha256": access["chain_root"],
            "ordinary_member_level_row_count": 0,
            "ordinary_loader_reachable_member_count": 0,
            "member_level_rows_in_commitment": False,
            "prior_analytic_use": "NONE_CONFIRMED",
            "pipeline_materialization": "PRESENT",
            "foundation_overlap_audit_status": "DEFERRED_TO_FM0_A",
            "d1_claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
        }
        commitment_path = qc_dir / "GSE246381_COMMITMENT.json"
        commitment_sha = json_write(commitment_path, commitment, "commitment_sha256")
        return {
            "aggregate_qc": qc_path,
            "aggregate_qc_sha256": qc_sha,
            "sealed_sums": sealed_sums,
            "sealed_sums_sha256": sealed_sums_sha,
            "sealed_manifest": sealed_manifest,
            "sealed_manifest_sha256": sealed_manifest_sha,
            "commitment": commitment_path,
            "commitment_sha256": commitment_sha,
        }

    def run(self) -> dict[str, Any]:
        ordinary_count = self.process_stream(self.raw_input, self.ordinary, limit=self.args.limit)
        if self.args.limit is None and self.args.expected_ordinary_records is not None and ordinary_count != self.args.expected_ordinary_records:
            raise RuntimeError(f"ordinary raw record count mismatch: {ordinary_count} != {self.args.expected_ordinary_records}")
        if self.args.limit is None and self.legacy_input and self.legacy_input.exists():
            self.process_stream(self.legacy_input, self.ordinary, legacy=True)
        restricted_count = self.process_stream(self.sealed_input, self.restricted, accession_override=SEALED_COHORT, limit=None if self.args.limit is None else min(self.args.limit, 2000))
        self.write_endpoint_and_group_rows()
        self.write_aggregate_reports()
        self.ordinary.close_rows()
        self.restricted.close_rows()
        input_paths = self.write_input_manifests()
        ordinary_access = self.build_access_chain(self.ordinary)
        restricted_access = self.build_access_chain(self.restricted)
        ordinary_effective = self.ordinary.write_effective_projection(ordinary_access["chain_root"], ordinary_access["completion"]["event_id"], self.canonical_binding)
        restricted_effective = self.restricted.write_effective_projection(restricted_access["chain_root"], restricted_access["completion"]["event_id"], self.canonical_binding)
        ordinary_exposure_use = self.write_exposure_use_manifest(self.ordinary, ordinary_access, ordinary_effective)
        restricted_exposure_use = self.write_exposure_use_manifest(self.restricted, restricted_access, restricted_effective)
        ordinary_manifest = self.write_canonical_manifest(self.ordinary, ordinary_access, ordinary_effective, ordinary_exposure_use)
        sealed_outputs = self.write_sealed_qc_and_manifest(restricted_access, restricted_effective, restricted_exposure_use)
        summary = {
            "artifact_kind": "D1_STRICT_BUILD_SUMMARY",
            "phase": "D1-R",
            "status": "BUILT_UNVALIDATED",
            "full_acceptance_asserted": False,
            "d1_acceptance_asserted": False,
            "next_phase_unlocked": False,
            "contract_id": CONTRACT_ID,
            "authority_contract_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "d1_snapshot_id": self.snapshot_id,
            "source_head": self.source_head,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "canonical_binding_sha256": self.canonical_binding,
            "ordinary_input_manifest": {"path": str(input_paths[0]), "sha256": sha_file(input_paths[0])},
            "sealed_input_manifest": {"path": str(input_paths[1]), "sha256": sha_file(input_paths[1])},
            "ordinary_access_chain_root_sha256": ordinary_access["chain_root"],
            "restricted_access_chain_root_sha256": restricted_access["chain_root"],
            "ordinary_canonical_manifest_sha256": ordinary_manifest["manifest_sha256"],
            "sealed_canonical_manifest_sha256": sealed_outputs["sealed_manifest_sha256"],
            "counts": dict(sorted(self.counts.items())),
            "dataset_counts": dict(sorted(self.dataset_counts.items())),
            "record_type_counts": dict(sorted(self.record_type_counts.items())),
            "region_counts": dict(sorted(self.region_counts.items())),
            "label_key_counts": dict(sorted(self.label_counts.items())),
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "restricted_matrix_profiles": getattr(self, "sealed_input_profiles", []),
            "legacy_quarantine_count": self.legacy_count,
            "g3b_status": "DEFERRED_TO_FM0_A",
            "claim_boundary": "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT",
            "training_started": False,
            "final_evaluator_accessed": False,
            "ordinary_output_root": str(self.ordinary_root),
            "restricted_output_root": str(self.restricted_root),
        }
        summary_path = self.out / "D1_BUILD_SUMMARY.json"
        json_write(summary_path, summary, "summary_sha256")
        return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinary-input", type=Path, required=True)
    ap.add_argument("--legacy-input", type=Path)
    ap.add_argument("--sealed-input", type=Path, required=True)
    ap.add_argument("--d0-root", type=Path, required=True)
    ap.add_argument("--c3-root", type=Path, required=True)
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--authority-contract-sha256", required=True)
    ap.add_argument("--source-head", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--expected-ordinary-records", type=int, default=3831570)
    ap.add_argument("--limit", type=int)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    builder = StrictBuilder(args)
    summary = builder.run()
    print(json.dumps({
        "status": summary["status"],
        "ordinary_raw_records": summary["counts"].get("raw_records:ordinary", 0),
        "restricted_raw_records": summary["counts"].get("raw_records:restricted", 0),
        "ordinary_pairs": summary["counts"].get("pairs:ordinary", 0),
        "restricted_pairs": summary["counts"].get("pairs:restricted", 0),
        "ordinary_observations": summary["counts"].get("observations:ordinary", 0),
        "restricted_observations": summary["counts"].get("observations:restricted", 0),
        "ordinary_output_root": summary["ordinary_output_root"],
        "restricted_output_root": summary["restricted_output_root"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
