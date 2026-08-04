#!/usr/bin/env python3
"""Fail-closed validator for the v3.1 strict D1 canonical bundle.

The validator is deliberately metadata-only in its reports.  It may read the
restricted raw files and canonical sequence/label payloads on the remote
machine, but it never prints row values, sequences, labels, or member IDs.
It validates the active C3 JSON schemas, canonical JSONL bytes, row hashes,
foreign keys, candidate/pair bijections, raw-input conservation, D0 lineage,
current projections, access chains, checksum ledgers, and the restricted
24-component sealed manifest.

The implementation uses SQLite as an on-disk index so a full D1 run does not
require loading millions of identifiers or canonical rows into Python memory.
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
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


CONTRACT_ID = "GOAL-V3-DATA-BENCH-01"
CLAIM_BOUNDARY = "D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT"
G3B_STATUS = "DEFERRED_TO_FM0_A"
SEALED_COHORT = "GSE246381"
SEALED_MARKER = "gse246381"
SEALED_COMPONENT_SET_SHA256 = "974736d060463b3af090af3dd0c6a0e8bc591305f57f51e0a8cd31751a1ee606"
SEALED_COHORT_SET_SHA256 = "275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0"
FUTURE_ROLE_ORDINARY = "AWAITING_B0_GLOBAL_DISPOSITION"
FUTURE_ROLE_SEALED = "SEALED_EXTERNAL_FINAL_CANDIDATE"
GENESIS = "GENESIS"
NOT_AVAILABLE = "NOT_AVAILABLE_D1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IUPAC = set("ACGTRYSWKMBDHVN")
NON_JSON_SCHEMA_ID = "NOT_APPLICABLE_NON_JSON"
NON_JSON_SCHEMA_SHA256 = "NOT_APPLICABLE"

CANONICAL_FILES = {
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

REQUIRED_CANONICAL_FILES = [*CANONICAL_FILES.values()]
RESTRICTED_LOGICAL_IDS = {
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
}

SCHEMA_SPECS = {
    "sequence_entities.jsonl": ("sequence_entity.schema.json", None),
    "functional_observation_candidates.jsonl": ("functional_observation.schema.json", "FunctionalObservationCandidate"),
    "functional_observations.jsonl": ("functional_observation.schema.json", None),
    "ENDPOINT_REGISTRY.jsonl": ("functional_observation.schema.json", "EndpointRegistryRow"),
    "utr_edit_relation_candidates.jsonl": ("utr_edit_relation_candidate.schema.json", None),
    "utr_edit_pairs.jsonl": ("utr_edit_pair.schema.json", None),
    "rejections.jsonl": ("rejection_record.schema.json", None),
    "transformation_edges.jsonl": ("transformation_edge.schema.json", None),
    "SUPERSESSION_EDGES.jsonl": ("transformation_edge.schema.json", "SupersessionEdge"),
    "CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl": ("transformation_edge.schema.json", "CurrentCanonicalObjectProjection"),
    "EXPOSURE_RECORDS.jsonl": ("exposure_record.schema.json", None),
    "USE_ROLES.jsonl": ("use_role.schema.json", None),
    "group_assignments.jsonl": ("group_assignment.schema.json", None),
    "reporter_artifact_assessments.jsonl": ("reporter_artifact_assessment.schema.json", None),
}

OBJECT_SPECS = {
    "sequence_entities.jsonl": ("SEQUENCE", "sequence_id"),
    "functional_observation_candidates.jsonl": ("OBSERVATION_CANDIDATE", "candidate_id"),
    "functional_observations.jsonl": ("OBSERVATION", "observation_id"),
    "utr_edit_relation_candidates.jsonl": ("RELATION_CANDIDATE", "relation_candidate_id"),
    "utr_edit_pairs.jsonl": ("PAIR", "pair_id"),
}

ACCESS_EVENT_DEF = {
    "INTENT": "AccessIntent",
    "COMPLETION": "AccessCompletion",
    "ABORT": "AccessAbort",
}


def jcs_bytes(value: Any) -> bytes:
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


def canonical_json_write(path: Path, value: dict[str, Any], self_field: str | None = None) -> str:
    payload = dict(value)
    if self_field:
        payload[self_field] = sha_json(payload)
    path.write_bytes(jline(payload))
    return sha_file(path)


def set_sha(values: Iterable[str]) -> str:
    return sha_bytes(("\n".join(sorted(set(values))) + "\n").encode("utf-8"))


def as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def safe_id(value: Any) -> str:
    return sha_text(str(value))[:32]


def token(value: Any, limit: int = 100) -> str:
    text = as_text(value)
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


def region_scope(value: Any) -> str:
    text = as_text(value)
    if text in {"5'UTR", "5UTR", "5′UTR"}:
        return "5UTR"
    if text in {"3'UTR", "3UTR", "3′UTR"}:
        return "3UTR"
    return "5UTR"


def sequence_scope(value: Any) -> str:
    return "FULL_UTR" if region_scope(value) in {"5UTR", "3UTR"} else "UTR_WINDOW"


def row_locator(path: Path, line_no: int, suffix: str = "") -> str:
    extra = f";{suffix}" if suffix else ""
    return f"{path.name}#line={line_no}{extra}"


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))


def contains_marker(value: Any, marker: str = SEALED_MARKER) -> bool:
    if isinstance(value, str):
        return marker in value.lower()
    if isinstance(value, dict):
        return any(contains_marker(k, marker) or contains_marker(v, marker) for k, v in value.items())
    if isinstance(value, list):
        return any(contains_marker(v, marker) for v in value)
    return False


class Errors:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []

    def add(self, code: str, artifact: str | None = None, line: int | None = None, field: str | None = None) -> None:
        key = code if artifact is None else f"{code}:{artifact}"
        self.counts[key] += 1
        if len(self.examples) < 100:
            item: dict[str, Any] = {"code": code}
            if artifact:
                item["artifact"] = artifact
            if line is not None:
                item["line"] = line
            if field:
                item["field"] = field
            self.examples.append(item)


class SchemaBook:
    def __init__(self, schema_dir: Path, errors: Errors) -> None:
        self.schema_dir = schema_dir
        self.errors = errors
        self.raw: dict[str, dict[str, Any]] = {}
        self.validators: dict[tuple[str, str | None], Any] = {}
        self.hashes: dict[str, str] = {}
        try:
            from jsonschema.validators import validator_for  # type: ignore
            self.validator_for = validator_for
        except Exception:
            self.validator_for = None
            self.errors.add("RUNTIME_MISSING_JSONSCHEMA")
        for path in sorted(schema_dir.glob("*.json")):
            try:
                self.raw[path.name] = json_load(path)
                self.hashes[path.name] = sha_file(path)
            except Exception:
                self.errors.add("SCHEMA_READ_ERROR", path.name)

    def validator(self, filename: str, definition: str | None) -> Any | None:
        key = (filename, definition)
        if key in self.validators:
            return self.validators[key]
        schema = self.raw.get(filename)
        if schema is None:
            self.errors.add("SCHEMA_MISSING", filename)
            return None
        if definition is not None:
            schema = schema.get("$defs", {}).get(definition)
            if schema is None:
                self.errors.add("SCHEMA_DEFINITION_MISSING", f"{filename}#/$defs/{definition}")
                return None
        if self.validator_for is None:
            return None
        try:
            cls = self.validator_for(schema)
            self.validators[key] = cls(schema)
            return self.validators[key]
        except Exception:
            self.errors.add("SCHEMA_COMPILE_ERROR", f"{filename}#/$defs/{definition}" if definition else filename)
            return None

    def validate(self, row: dict[str, Any], filename: str, definition: str | None, artifact: str, line: int) -> None:
        validator = self.validator(filename, definition)
        if validator is None:
            return
        try:
            iterator = validator.iter_errors(row)
            for error in iterator:
                path = ".".join(str(x) for x in error.path)
                self.errors.add(f"SCHEMA_{error.validator.upper()}", artifact, line, path or None)
        except Exception:
            self.errors.add("SCHEMA_RUNTIME_ERROR", artifact, line)


class D0Info:
    def __init__(self, root: Path, errors: Errors) -> None:
        self.root = root
        self.errors = errors
        reg = root / "data" / "v3_1" / "registry"
        self.raw_manifest = reg / "raw_asset_manifest.jsonl"
        self.dataset_assets_path = reg / "dataset_assets.jsonl"
        self.decisions_path = reg / "dataset_decisions.jsonl"
        self.raw_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.dataset_assets: dict[str, dict[str, Any]] = {}
        if not self.raw_manifest.exists():
            errors.add("D0_RAW_MANIFEST_MISSING")
        else:
            for row in self.iter_jsonl(self.raw_manifest):
                rel = as_text(row.get("relpath"))
                if rel:
                    self.raw_by_name[Path(rel).name].append(row)
        if self.dataset_assets_path.exists():
            for row in self.iter_jsonl(self.dataset_assets_path):
                accession = as_text(row.get("accession"))
                if accession:
                    self.dataset_assets[accession] = row
        self.raw_manifest_sha256 = sha_file(self.raw_manifest) if self.raw_manifest.exists() else None
        self.dataset_assets_sha256 = sha_file(self.dataset_assets_path) if self.dataset_assets_path.exists() else None
        self.decisions_sha256 = sha_file(self.decisions_path) if self.decisions_path.exists() else None

    def iter_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        for line in path.open("r", encoding="utf-8"):
            if line.strip():
                yield json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))

    def asset_id(self, accession: str) -> str:
        row = self.dataset_assets.get(accession)
        return as_text(row.get("asset_id")) if row else f"{accession}::UNMAPPED_D0_ASSET"

    def resolve(self, accession: str, source_file: Any) -> tuple[list[str], list[str], str | None]:
        text = as_text(source_file)
        if not text:
            return [self.asset_id(accession)], [], None
        names: list[str] = []
        for part in text.split("+"):
            name = Path(part.strip()).name
            if name and name not in names:
                names.append(name)
        chosen_rows: list[dict[str, Any]] = []
        for name in names:
            rows = self.raw_by_name.get(name, [])
            accession_rows = [r for r in rows if as_text(r.get("asset_id")) == accession]
            chosen_rows.extend((accession_rows or rows)[:1])
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in chosen_rows:
            dedup[(as_text(row.get("relpath")), as_text(row.get("sha256")))] = row
        hashes = sorted({as_text(r.get("sha256")) for r in dedup.values() if HEX64.fullmatch(as_text(r.get("sha256")))})
        return [self.asset_id(accession)], hashes, hashes[0] if hashes else None


class Store:
    def __init__(self, path: Path, errors: Errors) -> None:
        self.errors = errors
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, object_type TEXT NOT NULL,
              row_sha256 TEXT NOT NULL, PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS sequences(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, row_sha256 TEXT NOT NULL,
              raw_hash TEXT, norm_hash TEXT, full_hash TEXT, original_length INTEGER,
              region_scope TEXT, source_record_id TEXT, source_row_locator TEXT,
              primary_asset_id TEXT,
              assets_json TEXT, files_json TEXT, contributor_hash TEXT,
              model_eligible INTEGER, alphabet_status TEXT, invalid_status TEXT,
              normalized_hash TEXT, PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS obs_candidates(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, row_sha256 TEXT NOT NULL,
              sequence_id TEXT, context_id TEXT, endpoint_id TEXT,
              accepted_observation_id TEXT, lifecycle_status TEXT, acceptance_status TEXT,
              terminal_reason TEXT, source_file_sha256 TEXT, contributor_hash TEXT,
              assets_json TEXT, files_json TEXT, source_locator TEXT, payload_hash TEXT,
              PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS observations(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, row_sha256 TEXT NOT NULL,
              candidate_id TEXT, sequence_id TEXT, context_id TEXT, endpoint_id TEXT,
              scientific_track TEXT, observation_role TEXT, source_file_sha256 TEXT,
              contributor_hash TEXT, assets_json TEXT, files_json TEXT,
              source_locator TEXT, payload_hash TEXT, PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS relations(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, row_sha256 TEXT NOT NULL,
              accepted_pair_id TEXT, source_sequence_id TEXT, candidate_sequence_id TEXT,
              context_id TEXT, endpoint_id TEXT, design_group TEXT, relation_type TEXT,
              relation_status TEXT, lifecycle_status TEXT, effect_evidence TEXT,
              landscape_role TEXT, future_use_role TEXT, no_edit_subtype TEXT,
              frame_id TEXT, scientific_track TEXT, pairing_method TEXT,
              label_unit TEXT, label_transform TEXT, delta_rule_id TEXT,
              delta_rule_sha256 TEXT, pair_evidence_id TEXT, contributor_hash TEXT,
              assets_json TEXT, files_json TEXT, source_locator TEXT, payload_hash TEXT,
              PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS pairs(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, row_sha256 TEXT NOT NULL,
              relation_id TEXT, source_sequence_id TEXT, candidate_sequence_id TEXT,
              context_id TEXT, endpoint_id TEXT, design_group TEXT, relation_type TEXT,
              relation_status TEXT, effect_evidence TEXT, landscape_role TEXT,
              future_use_role TEXT, immutable_role TEXT, no_edit_subtype TEXT,
              frame_id TEXT, scientific_track TEXT, pairing_method TEXT,
              label_unit TEXT, label_transform TEXT, delta_rule_id TEXT,
              delta_rule_sha256 TEXT, pair_evidence_id TEXT, contributor_hash TEXT,
              assets_json TEXT, files_json TEXT, source_locator TEXT, payload_hash TEXT,
              PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS projections(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, object_type TEXT,
              current_leaf_id TEXT, current_leaf_hash TEXT, accepted INTEGER,
              generation_index INTEGER, row_sha256 TEXT, PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS exposures(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, object_type TEXT,
              canonical_hash TEXT, record_sha256 TEXT, row_sha256 TEXT, PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS use_roles(
              object_id TEXT NOT NULL, shard TEXT NOT NULL, relation_id TEXT, pair_id TEXT,
              future_role TEXT, authority_level TEXT, row_sha256 TEXT, PRIMARY KEY(object_id, shard));
            CREATE TABLE IF NOT EXISTS groups(
              group_id TEXT NOT NULL, shard TEXT NOT NULL, group_type TEXT,
              group_sha256 TEXT, member_count INTEGER, member_ids_json TEXT,
              frame_hash TEXT, frame_definition_sha256 TEXT, study_id TEXT,
              PRIMARY KEY(group_id, shard));
            CREATE TABLE IF NOT EXISTS assignments(
              assignment_id TEXT NOT NULL, shard TEXT NOT NULL, object_id TEXT,
              object_type TEXT, group_id TEXT, context_id TEXT, endpoint_id TEXT,
              frame_id TEXT, member_locator TEXT, assignment_order INTEGER,
              PRIMARY KEY(assignment_id, shard));
            CREATE TABLE IF NOT EXISTS edges(
              edge_id TEXT NOT NULL, shard TEXT NOT NULL, old_id TEXT, new_id TEXT,
              new_type TEXT, old_hash TEXT, new_hash TEXT, row_sha256 TEXT,
              PRIMARY KEY(edge_id, shard));
            CREATE TABLE IF NOT EXISTS rejections(
              rejection_id TEXT NOT NULL, shard TEXT NOT NULL, reason TEXT,
              source_locator TEXT, candidate_id TEXT, row_sha256 TEXT,
              PRIMARY KEY(rejection_id, shard));
            CREATE TABLE IF NOT EXISTS object_set_rows(
              shard TEXT NOT NULL, set_kind TEXT NOT NULL, object_id TEXT NOT NULL,
              object_type TEXT NOT NULL, object_sha256 TEXT NOT NULL,
              PRIMARY KEY(shard, set_kind, object_id));
            CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(shard, object_type);
            CREATE INDEX IF NOT EXISTS idx_assign_group ON assignments(shard, group_id);
            CREATE INDEX IF NOT EXISTS idx_rej_locator ON rejections(shard, source_locator, reason);
            """
        )
        self.conn.commit()

    def insert(self, table: str, values: dict[str, Any], key: tuple[str, str]) -> None:
        cols = list(values)
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        try:
            self.conn.execute(sql, [values[c] for c in cols])
        except sqlite3.IntegrityError:
            self.errors.add("DUPLICATE_KEY", f"{table}:{key[0]}:{key[1]}")

    def one(self, table: str, where: str, args: tuple[Any, ...]) -> sqlite3.Row | None:
        self.conn.row_factory = sqlite3.Row
        return self.conn.execute(f"SELECT * FROM {table} WHERE {where}", args).fetchone()

    def count(self, table: str, shard: str) -> int:
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE shard=?", (shard,)).fetchone()[0])

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


class Validator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = args.run_root.resolve()
        self.ordinary_root = self.run_root / "data" / "v3_1"
        self.restricted_root = self.run_root / "sealed_external" / SEALED_COHORT
        self.errors = Errors()
        self.schema = SchemaBook(args.schema_dir.resolve(), self.errors)
        self.d0 = D0Info(args.d0_root.resolve(), self.errors)
        self.summary: dict[str, Any] = {}
        self.input_manifest: dict[str, Any] = {}
        self.store = Store(self.run_root / "work" / "D1_VALIDATION.sqlite", self.errors)
        self.counters: Counter[str] = Counter()
        self.raw_stats: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.raw_hashes: dict[str, str] = {}
        self.raw_paths: dict[str, Path] = {}
        self.frame_hashers: dict[tuple[str, str], hashlib._Hash] = {}
        self.assignment_ord: Counter[str] = Counter()
        self.group_rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.expected_rejections: dict[str, Counter[str]] = {"ordinary": Counter(), "restricted": Counter()}
        self.expected_noedit: Counter[str] = Counter()

    def require(self, path: Path, artifact: str) -> bool:
        if not path.exists():
            self.errors.add("MISSING_REQUIRED_ARTIFACT", artifact)
            return False
        return True

    def read_jsonl(
        self,
        path: Path,
        artifact: str,
        schema_spec: tuple[str, str | None] | None,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
        marker_scan: bool = False,
    ) -> int:
        if not self.require(path, artifact):
            return 0
        count = 0
        with path.open("r", encoding="utf-8", newline="") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
                except Exception:
                    self.errors.add("INVALID_JSON_OR_NONFINITE", artifact, line_no)
                    continue
                if not isinstance(row, dict):
                    self.errors.add("JSONL_ROW_NOT_OBJECT", artifact, line_no)
                    continue
                if line != jline(row).decode("utf-8"):
                    self.errors.add("NON_CANONICAL_JSONL_BYTES", artifact, line_no)
                if marker_scan and contains_marker(row):
                    self.errors.add("SEALED_MARKER_LEAK", artifact, line_no)
                if schema_spec:
                    self.schema.validate(row, schema_spec[0], schema_spec[1], artifact, line_no)
                if callback:
                    callback(line_no, row)
        self.counters[f"rows:{artifact}"] = count
        return count

    def load_metadata(self) -> None:
        summary_path = self.run_root / "D1_BUILD_SUMMARY.json"
        input_path = self.run_root / "work" / "D1_INPUT_MANIFEST.json"
        if self.require(summary_path, "D1_BUILD_SUMMARY.json"):
            try:
                self.summary = json_load(summary_path)
                if self.summary.get("summary_sha256") != sha_json({k: v for k, v in self.summary.items() if k != "summary_sha256"}):
                    self.errors.add("SELF_HASH_MISMATCH", "D1_BUILD_SUMMARY.json")
            except Exception:
                self.errors.add("INVALID_JSON", "D1_BUILD_SUMMARY.json")
        if self.require(input_path, "work/D1_INPUT_MANIFEST.json"):
            try:
                self.input_manifest = json_load(input_path)
                if self.input_manifest.get("manifest_sha256") != sha_json({k: v for k, v in self.input_manifest.items() if k != "manifest_sha256"}):
                    self.errors.add("SELF_HASH_MISMATCH", "work/D1_INPUT_MANIFEST.json")
            except Exception:
                self.errors.add("INVALID_JSON", "work/D1_INPUT_MANIFEST.json")
        expected = self.args.authority_contract_sha256
        for artifact, obj in [("D1_BUILD_SUMMARY.json", self.summary), ("work/D1_INPUT_MANIFEST.json", self.input_manifest)]:
            if obj.get("contract_id") != CONTRACT_ID:
                self.errors.add("CONTRACT_ID_MISMATCH", artifact)
            if obj.get("contract_sha256", obj.get("authority_contract_sha256")) != expected:
                self.errors.add("CONTRACT_HASH_MISMATCH", artifact)
        if self.summary.get("claim_boundary") != CLAIM_BOUNDARY:
            self.errors.add("CLAIM_BOUNDARY_MISMATCH", "D1_BUILD_SUMMARY.json")
        if self.summary.get("g3b_status") != G3B_STATUS:
            self.errors.add("G3B_STATUS_MISMATCH", "D1_BUILD_SUMMARY.json")
        if self.summary.get("training_started") is not False or self.summary.get("final_evaluator_accessed") is not False:
            self.errors.add("UNAUTHORIZED_ANALYTIC_STATE", "D1_BUILD_SUMMARY.json")
        ordinary_path = self.input_manifest.get("ordinary_raw_input", {}).get("path")
        self.raw_paths["ordinary"] = Path(ordinary_path) if ordinary_path else Path()
        restricted_path = self.input_manifest.get("sealed_raw_input", {}).get("path")
        self.raw_paths["restricted"] = Path(restricted_path) if restricted_path else Path(self.args.sealed_input)
        if not self.raw_paths["restricted"].is_file():
            self.raw_paths["restricted"] = Path(self.args.sealed_input)
        legacy = self.input_manifest.get("legacy_quarantine_input", {}).get("path")
        self.raw_paths["legacy"] = Path(legacy) if legacy else Path()

    def verify_input_manifest(self) -> None:
        ordinary = self.input_manifest.get("ordinary_raw_input", {})
        if self.raw_paths.get("ordinary") and self.raw_paths["ordinary"].exists():
            actual_sha, count = self.hash_jsonl_file(self.raw_paths["ordinary"], ordinary.get("record_count"))
            self.raw_hashes["ordinary"] = actual_sha
            if actual_sha != ordinary.get("sha256"):
                self.errors.add("RAW_INPUT_HASH_MISMATCH", "ordinary_raw_input")
            if count != ordinary.get("record_count"):
                self.errors.add("RAW_INPUT_COUNT_MISMATCH", "ordinary_raw_input")
        else:
            self.errors.add("RAW_INPUT_MISSING", "ordinary_raw_input")
        legacy = self.input_manifest.get("legacy_quarantine_input", {})
        if legacy.get("path") and self.raw_paths.get("legacy") and self.raw_paths["legacy"].exists():
            actual_sha, count = self.hash_jsonl_file(self.raw_paths["legacy"])
            ordinary_count = ordinary.get("record_count")
            full_ordinary = ordinary_count == self.args.expected_ordinary_records
            if actual_sha != legacy.get("sha256") or (full_ordinary and count != legacy.get("record_count")):
                self.errors.add("LEGACY_INPUT_BINDING_MISMATCH", "legacy_quarantine_input")
        elif legacy.get("record_count", 0) != 0:
            self.errors.add("LEGACY_INPUT_MISSING", "legacy_quarantine_input")
        d0 = self.input_manifest.get("d0_rebind", {})
        if self.d0.raw_manifest_sha256 != d0.get("raw_asset_manifest_sha256"):
            self.errors.add("D0_RAW_MANIFEST_HASH_MISMATCH")
        if self.d0.dataset_assets_sha256 != d0.get("dataset_assets_sha256"):
            self.errors.add("D0_DATASET_ASSETS_HASH_MISMATCH")
        if self.d0.decisions_sha256 != d0.get("dataset_decisions_sha256"):
            self.errors.add("D0_DATASET_DECISIONS_HASH_MISMATCH")
        c3 = self.input_manifest.get("c3_parent", {})
        c3_path = Path(c3.get("artifact_path", ""))
        if c3_path.exists() and sha_file(c3_path) != c3.get("artifact_sha256"):
            self.errors.add("C3_PARENT_HASH_MISMATCH")
        elif not c3_path.exists():
            self.errors.add("C3_PARENT_MISSING")

    def hash_jsonl_file(self, path: Path, limit: int | None = None) -> tuple[str, int]:
        h = hashlib.sha256()
        count = 0
        with path.open("rb") as fh:
            for raw in fh:
                h.update(raw)
                if raw.strip():
                    count += 1
                    if limit is not None and count >= limit:
                        break
        return h.hexdigest(), count

    def object_insert(self, shard: str, row: dict[str, Any], object_type: str, id_field: str) -> tuple[str, str]:
        object_id = as_text(row.get(id_field))
        row_hash = sha_json(row)
        if not object_id:
            self.errors.add("OBJECT_ID_MISSING", f"{shard}:{object_type}")
        self.store.insert("objects", {"object_id": object_id, "shard": shard, "object_type": object_type, "row_sha256": row_hash}, (object_id, shard))
        return object_id, row_hash

    def load_sequence(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        object_id, row_hash = self.object_insert(shard, row, "SEQUENCE", "sequence_id")
        self.store.insert("sequences", {
            "object_id": object_id, "shard": shard, "row_sha256": row_hash,
            "raw_hash": row.get("raw_sequence_sha256"), "norm_hash": row.get("normalized_sequence_sha256"),
            "full_hash": row.get("full_sequence_sha256"), "original_length": row.get("original_length"),
            "region_scope": row.get("region_scope"), "source_record_id": row.get("source_record_id"),
            "source_row_locator": row.get("source_row_locator"), "primary_asset_id": row.get("primary_asset_id"),
            "assets_json": json.dumps(row.get("contributing_asset_ids"), sort_keys=True, separators=(",", ":")),
            "files_json": json.dumps(row.get("contributing_source_file_sha256s"), sort_keys=True, separators=(",", ":")),
            "contributor_hash": row.get("contributor_set_sha256"), "model_eligible": int(bool(row.get("model_sequence_eligible"))),
            "alphabet_status": row.get("alphabet_status"), "invalid_status": row.get("invalid_symbol_status"),
            "normalized_hash": sha_text(row.get("normalized_sequence") or ""),
        }, (object_id, shard))

    def load_obs_candidate(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        object_id, row_hash = self.object_insert(shard, row, "OBSERVATION_CANDIDATE", "candidate_id")
        payload_hash = sha_json({
            "sequence_id": row.get("sequence_id"),
            "context_id": row.get("context_id"),
            "endpoint_id": row.get("endpoint_id"),
            "contributing_asset_ids": row.get("asset_ids"),
            "contributing_source_file_sha256s": row.get("contributing_source_file_sha256s"),
            "contributor_set_sha256": row.get("contributor_set_sha256"),
            "source_file_sha256": row.get("source_file_sha256"),
            "value": row.get("value"),
        })
        self.store.insert("obs_candidates", {
            "object_id": object_id, "shard": shard, "row_sha256": row_hash, "sequence_id": row.get("sequence_id"),
            "context_id": row.get("context_id"), "endpoint_id": row.get("endpoint_id"),
            "accepted_observation_id": row.get("accepted_observation_id"), "lifecycle_status": row.get("lifecycle_status"),
            "acceptance_status": row.get("observation_acceptance_status"), "terminal_reason": row.get("terminal_disposition_reason"),
            "source_file_sha256": row.get("source_file_sha256"), "contributor_hash": row.get("contributor_set_sha256"),
            "assets_json": json.dumps(row.get("asset_ids"), sort_keys=True, separators=(",", ":")),
            "files_json": json.dumps(row.get("contributing_source_file_sha256s"), sort_keys=True, separators=(",", ":")),
            "source_locator": (row.get("source_row_locators") or [None])[0], "payload_hash": payload_hash,
        }, (object_id, shard))

    def load_observation(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        object_id, row_hash = self.object_insert(shard, row, "OBSERVATION", "observation_id")
        payload_hash = sha_json({k: row.get(k) for k in ("sequence_id", "context_id", "endpoint_id", "contributing_asset_ids", "contributing_source_file_sha256s", "contributor_set_sha256", "source_file_sha256", "value")})
        self.store.insert("observations", {
            "object_id": object_id, "shard": shard, "row_sha256": row_hash, "candidate_id": row.get("observation_candidate_id"),
            "sequence_id": row.get("sequence_id"), "context_id": row.get("context_id"), "endpoint_id": row.get("endpoint_id"),
            "scientific_track": row.get("scientific_track"), "observation_role": row.get("observation_role"),
            "source_file_sha256": row.get("source_file_sha256"), "contributor_hash": row.get("contributor_set_sha256"),
            "assets_json": json.dumps(row.get("contributing_asset_ids"), sort_keys=True, separators=(",", ":")),
            "files_json": json.dumps(row.get("contributing_source_file_sha256s"), sort_keys=True, separators=(",", ":")),
            "source_locator": row.get("source_row_locator"), "payload_hash": payload_hash,
        }, (object_id, shard))

    def relation_payload_hash(self, row: dict[str, Any], pair: bool = False) -> str:
        keys = ["context_id", "endpoint_id", "design_relation_group_id", "effect_evidence", "landscape_role", "future_use_role", "no_edit_control_subtype", "no_edit_sampling_frame_id", "scientific_track", "relation_type", "pairing_method", "label_unit", "label_transform", "delta_rule_id", "delta_rule_sha256", "pair_evidence_id", "source_sequence_id", "candidate_sequence_id", "contributing_asset_ids", "contributing_source_file_sha256s", "contributor_set_sha256"]
        if pair:
            keys.append("immutable_base_future_use_role")
        return sha_json({k: row.get(k) for k in keys})

    def load_relation(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        object_id, row_hash = self.object_insert(shard, row, "RELATION_CANDIDATE", "relation_candidate_id")
        self.store.insert("relations", {
            "object_id": object_id, "shard": shard, "row_sha256": row_hash,
            "accepted_pair_id": row.get("accepted_pair_id"), "source_sequence_id": row.get("source_sequence_id"),
            "candidate_sequence_id": row.get("candidate_sequence_id"), "context_id": row.get("context_id"),
            "endpoint_id": row.get("endpoint_id"), "design_group": row.get("design_relation_group_id"),
            "relation_type": row.get("relation_type"), "relation_status": row.get("relation_acceptance_status"),
            "lifecycle_status": row.get("lifecycle_status"), "effect_evidence": row.get("effect_evidence"),
            "landscape_role": row.get("landscape_role"), "future_use_role": row.get("future_use_role"),
            "no_edit_subtype": row.get("no_edit_control_subtype"), "frame_id": row.get("no_edit_sampling_frame_id"),
            "scientific_track": row.get("scientific_track"), "pairing_method": row.get("pairing_method"),
            "label_unit": row.get("label_unit"), "label_transform": row.get("label_transform"),
            "delta_rule_id": row.get("delta_rule_id"), "delta_rule_sha256": row.get("delta_rule_sha256"),
            "pair_evidence_id": row.get("pair_evidence_id"), "contributor_hash": row.get("contributor_set_sha256"),
            "assets_json": json.dumps(row.get("contributing_asset_ids"), sort_keys=True, separators=(",", ":")),
            "files_json": json.dumps(row.get("contributing_source_file_sha256s"), sort_keys=True, separators=(",", ":")),
            "source_locator": None, "payload_hash": self.relation_payload_hash(row),
        }, (object_id, shard))

    def load_pair(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        object_id, row_hash = self.object_insert(shard, row, "PAIR", "pair_id")
        self.store.insert("pairs", {
            "object_id": object_id, "shard": shard, "row_sha256": row_hash, "relation_id": row.get("relation_candidate_id"),
            "source_sequence_id": row.get("source_sequence_id"), "candidate_sequence_id": row.get("candidate_sequence_id"),
            "context_id": row.get("context_id"), "endpoint_id": row.get("endpoint_id"), "design_group": row.get("design_relation_group_id"),
            "relation_type": row.get("relation_type"), "relation_status": row.get("relation_acceptance_status"),
            "effect_evidence": row.get("effect_evidence"), "landscape_role": row.get("landscape_role"),
            "future_use_role": row.get("future_use_role"), "immutable_role": row.get("immutable_base_future_use_role"),
            "no_edit_subtype": row.get("no_edit_control_subtype"), "frame_id": row.get("no_edit_sampling_frame_id"),
            "scientific_track": row.get("scientific_track"), "pairing_method": row.get("pairing_method"),
            "label_unit": row.get("label_unit"), "label_transform": row.get("label_transform"),
            "delta_rule_id": row.get("delta_rule_id"), "delta_rule_sha256": row.get("delta_rule_sha256"),
            "pair_evidence_id": row.get("pair_evidence_id"), "contributor_hash": row.get("contributor_set_sha256"),
            "assets_json": json.dumps(row.get("contributing_asset_ids"), sort_keys=True, separators=(",", ":")),
            "files_json": json.dumps(row.get("contributing_source_file_sha256s"), sort_keys=True, separators=(",", ":")),
            "source_locator": None, "payload_hash": self.relation_payload_hash(row, pair=False),
        }, (object_id, shard))

    def load_projection(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        expected_projection = sha_json({k: v for k, v in row.items() if k not in {"projection_sha256", "record_sha256"}})
        expected_record = sha_json({k: v for k, v in row.items() if k != "record_sha256"})
        if row.get("projection_sha256") != expected_projection:
            self.errors.add("PROJECTION_HASH_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("record_sha256") != expected_record:
            self.errors.add("RECORD_HASH_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("current_leaf_object_id") != row.get("object_id") or row.get("chain_root_object_id") != row.get("object_id"):
            self.errors.add("CURRENT_LEAF_ID_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("current_leaf_object_sha256") != row.get("chain_root_object_sha256"):
            self.errors.add("CURRENT_LEAF_HASH_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("generation_index") != 0 or row.get("chain_length") != 0:
            self.errors.add("UNEXPECTED_SUPERSESSION_GENERATION", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("last_supersession_edge_id") != GENESIS or row.get("last_supersession_edge_sha256") != GENESIS:
            self.errors.add("UNEXPECTED_SUPERSESSION_EDGE", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("supersession_manifest_sha256") != sha_bytes(b""):
            self.errors.add("EMPTY_SUPERSESSION_HASH_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("active") != row.get("is_current_leaf_accepted"):
            self.errors.add("CURRENT_ACTIVE_FLAG_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        if row.get("canonical_manifest_sha256") != self.summary.get("canonical_binding_sha256"):
            self.errors.add("CANONICAL_BINDING_MISMATCH", f"{shard}:CURRENT_CANONICAL_OBJECT_PROJECTION", line_no)
        self.store.insert("projections", {
            "object_id": row.get("object_id"), "shard": shard, "object_type": row.get("object_type"),
            "current_leaf_id": row.get("current_leaf_object_id"), "current_leaf_hash": row.get("current_leaf_object_sha256"),
            "accepted": int(bool(row.get("is_current_leaf_accepted"))), "generation_index": row.get("generation_index"),
            "row_sha256": sha_json(row),
        }, (as_text(row.get("object_id")), shard))

    def load_exposure(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        if row.get("record_sha256") != sha_json(row, "record_sha256"):
            self.errors.add("RECORD_HASH_MISMATCH", f"{shard}:EXPOSURE_RECORDS", line_no)
        self.store.insert("exposures", {
            "object_id": row.get("object_id"), "shard": shard, "object_type": row.get("object_type"),
            "canonical_hash": row.get("canonical_object_sha256"), "record_sha256": row.get("record_sha256"),
            "row_sha256": sha_json(row),
        }, (as_text(row.get("object_id")), shard))

    def load_use_role(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        if row.get("record_sha256") != sha_json(row, "record_sha256"):
            self.errors.add("RECORD_HASH_MISMATCH", f"{shard}:USE_ROLES", line_no)
        self.store.insert("use_roles", {
            "object_id": row.get("object_id"), "shard": shard, "relation_id": row.get("relation_candidate_id"),
            "pair_id": row.get("pair_id"), "future_role": row.get("future_use_role"),
            "authority_level": row.get("authority_level"), "row_sha256": sha_json(row),
        }, (as_text(row.get("object_id")), shard))

    def load_edge(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        if row.get("edge_sha256") != sha_json(row, "edge_sha256"):
            self.errors.add("EDGE_HASH_MISMATCH", f"{shard}:TRANSFORMATION_EDGES", line_no)
        self.store.insert("edges", {
            "edge_id": row.get("edge_id"), "shard": shard, "old_id": row.get("old_object_id"),
            "new_id": row.get("new_object_id"), "new_type": row.get("object_type"),
            "old_hash": row.get("old_object_sha256"), "new_hash": row.get("new_object_sha256"),
            "row_sha256": sha_json(row),
        }, (as_text(row.get("edge_id")), shard))

    def load_rejection(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        self.store.insert("rejections", {
            "rejection_id": row.get("rejection_id"), "shard": shard, "reason": row.get("reason"),
            "source_locator": row.get("source_row_locator"), "candidate_id": row.get("candidate_id"),
            "row_sha256": sha_json(row),
        }, (as_text(row.get("rejection_id")), shard))

    def load_group(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        if row.get("group_type") == "NO_EDIT_SAMPLING_FRAME" and "study_id" in row:
            definition = {k: v for k, v in row.items() if k not in {"group_id", "group_type", "member_assignment_manifest_sha256", "frame_definition_sha256"}}
            if row.get("frame_definition_sha256") != sha_json(definition):
                self.errors.add("FRAME_DEFINITION_HASH_MISMATCH", f"{shard}:GROUP_REGISTRY", line_no)
            self.frame_hashers[(shard, as_text(row.get("group_id")))] = hashlib.sha256()
            self.store.insert("groups", {
                "group_id": row.get("group_id"), "shard": shard, "group_type": row.get("group_type"),
                "group_sha256": None, "member_count": None, "member_ids_json": None, "frame_hash": row.get("member_assignment_manifest_sha256"),
                "frame_definition_sha256": row.get("frame_definition_sha256"), "study_id": row.get("study_id"),
            }, (as_text(row.get("group_id")), shard))
        else:
            if row.get("group_sha256") != sha_json(row, "group_sha256"):
                self.errors.add("GROUP_HASH_MISMATCH", f"{shard}:GROUP_REGISTRY", line_no)
            self.store.insert("groups", {
                "group_id": row.get("group_id"), "shard": shard, "group_type": row.get("group_type"),
                "group_sha256": row.get("group_sha256"), "member_count": row.get("member_count"),
                "member_ids_json": json.dumps(row.get("member_ids"), sort_keys=True, separators=(",", ":")),
                "frame_hash": None, "frame_definition_sha256": None, "study_id": None,
            }, (as_text(row.get("group_id")), shard))
        self.group_rows[(shard, as_text(row.get("group_id")))] = row

    def load_assignment(self, shard: str, line_no: int, row: dict[str, Any]) -> None:
        order = self.assignment_ord[shard]
        self.assignment_ord[shard] += 1
        frame_id = row.get("no_edit_sampling_frame_id")
        if frame_id and (shard, as_text(frame_id)) in self.frame_hashers:
            self.frame_hashers[(shard, as_text(frame_id))].update(jline(row))
        self.store.insert("assignments", {
            "assignment_id": row.get("assignment_id"), "shard": shard, "object_id": row.get("object_id"),
            "object_type": row.get("object_type"), "group_id": row.get("group_id"), "context_id": row.get("context_id"),
            "endpoint_id": row.get("endpoint_id"), "frame_id": frame_id, "member_locator": row.get("member_locator"),
            "assignment_order": order,
        }, (as_text(row.get("assignment_id")), shard))

    def load_shard(self, shard: str, root: Path) -> None:
        marker = shard == "ordinary"
        callbacks: dict[str, Callable[[int, dict[str, Any]], None]] = {
            "sequence_entities.jsonl": lambda n, r: self.load_sequence(shard, n, r),
            "functional_observation_candidates.jsonl": lambda n, r: self.load_obs_candidate(shard, n, r),
            "functional_observations.jsonl": lambda n, r: self.load_observation(shard, n, r),
            "utr_edit_relation_candidates.jsonl": lambda n, r: self.load_relation(shard, n, r),
            "utr_edit_pairs.jsonl": lambda n, r: self.load_pair(shard, n, r),
            "CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl": lambda n, r: self.load_projection(shard, n, r),
            "EXPOSURE_RECORDS.jsonl": lambda n, r: self.load_exposure(shard, n, r),
            "USE_ROLES.jsonl": lambda n, r: self.load_use_role(shard, n, r),
            "rejections.jsonl": lambda n, r: self.load_rejection(shard, n, r),
            "transformation_edges.jsonl": lambda n, r: self.load_edge(shard, n, r),
            "group_registry.jsonl": lambda n, r: self.load_group(shard, n, r),
            "group_assignments.jsonl": lambda n, r: self.load_assignment(shard, n, r),
        }
        for logical, filename in CANONICAL_FILES.items():
            if shard == "restricted" and filename == "reporter_artifact_assessments.jsonl":
                continue
            path = root / "canonical" / filename
            artifact = f"{shard}/canonical/{filename}"
            spec = SCHEMA_SPECS.get(filename)
            if filename == "group_registry.jsonl":
                self.read_group_registry(shard, path, artifact, marker)
            else:
                self.read_jsonl(path, artifact, spec, callbacks.get(filename), marker)
        self.store.conn.commit()

    def read_group_registry(self, shard: str, path: Path, artifact: str, marker: bool) -> None:
        if not self.require(path, artifact):
            return
        count = 0
        with path.open("r", encoding="utf-8", newline="") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
                except Exception:
                    self.errors.add("INVALID_JSON_OR_NONFINITE", artifact, line_no)
                    continue
                if line != jline(row).decode("utf-8"):
                    self.errors.add("NON_CANONICAL_JSONL_BYTES", artifact, line_no)
                if marker and contains_marker(row):
                    self.errors.add("SEALED_MARKER_LEAK", artifact, line_no)
                definition = "NoEditSamplingFrameRow" if row.get("group_type") == "NO_EDIT_SAMPLING_FRAME" and "study_id" in row else None
                self.schema.validate(row, "group_registry.schema.json", definition, artifact, line_no)
                self.load_group(shard, line_no, row)
        self.counters[f"rows:{artifact}"] = count

    def scalar(self, query: str, args: tuple[Any, ...] = ()) -> int:
        return int(self.store.conn.execute(query, args).fetchone()[0])

    def verify_exact_object_coverage(self, shard: str) -> None:
        # Every emitted scientific canonical object has exactly one current
        # projection and one immutable baseline exposure record.
        missing_projection = self.scalar(
            "SELECT COUNT(*) FROM objects o LEFT JOIN projections p ON p.object_id=o.object_id AND p.shard=o.shard WHERE o.shard=? AND p.object_id IS NULL",
            (shard,),
        )
        orphan_projection = self.scalar(
            "SELECT COUNT(*) FROM projections p LEFT JOIN objects o ON o.object_id=p.object_id AND o.shard=p.shard WHERE p.shard=? AND o.object_id IS NULL",
            (shard,),
        )
        missing_exposure = self.scalar(
            "SELECT COUNT(*) FROM objects o LEFT JOIN exposures e ON e.object_id=o.object_id AND e.shard=o.shard WHERE o.shard=? AND e.object_id IS NULL",
            (shard,),
        )
        orphan_exposure = self.scalar(
            "SELECT COUNT(*) FROM exposures e LEFT JOIN objects o ON o.object_id=e.object_id AND o.shard=e.shard WHERE e.shard=? AND o.object_id IS NULL",
            (shard,),
        )
        for code, count in [
            ("MISSING_CURRENT_PROJECTION", missing_projection),
            ("ORPHAN_CURRENT_PROJECTION", orphan_projection),
            ("MISSING_EXPOSURE_RECORD", missing_exposure),
            ("ORPHAN_EXPOSURE_RECORD", orphan_exposure),
        ]:
            if count:
                self.errors.add(code, shard, count)
        bad_exposure_hash = self.scalar(
            "SELECT COUNT(*) FROM exposures e JOIN objects o ON o.object_id=e.object_id AND o.shard=e.shard WHERE e.shard=? AND (e.object_type != o.object_type OR e.canonical_hash != o.row_sha256)",
            (shard,),
        )
        if bad_exposure_hash:
            self.errors.add("EXPOSURE_OBJECT_HASH_MISMATCH", shard, bad_exposure_hash)
        bad_projection = self.scalar(
            "SELECT COUNT(*) FROM projections p JOIN objects o ON o.object_id=p.object_id AND o.shard=p.shard WHERE p.shard=? AND (p.object_type != o.object_type OR p.current_leaf_id != o.object_id OR p.current_leaf_hash != o.row_sha256)",
            (shard,),
        )
        if bad_projection:
            self.errors.add("PROJECTION_OBJECT_HASH_MISMATCH", shard, bad_projection)

        # D1 transformation edges are the raw-unit -> canonical-object edges.
        # Rejection records are terminal dispositions, not scientific object
        # types and therefore intentionally have no transformation edge.
        missing_edges = self.scalar(
            "SELECT COUNT(*) FROM objects o LEFT JOIN edges e ON e.new_id=o.object_id AND e.shard=o.shard WHERE o.shard=? AND e.new_id IS NULL",
            (shard,),
        )
        orphan_edges = self.scalar(
            "SELECT COUNT(*) FROM edges e LEFT JOIN objects o ON o.object_id=e.new_id AND o.shard=e.shard WHERE e.shard=? AND o.object_id IS NULL",
            (shard,),
        )
        bad_edges = self.scalar(
            "SELECT COUNT(*) FROM edges e JOIN objects o ON o.object_id=e.new_id AND o.shard=e.shard WHERE e.shard=? AND (e.new_type != o.object_type OR e.new_hash != o.row_sha256)",
            (shard,),
        )
        for code, count in [("MISSING_TRANSFORMATION_EDGE", missing_edges), ("ORPHAN_TRANSFORMATION_EDGE", orphan_edges), ("TRANSFORMATION_OBJECT_HASH_MISMATCH", bad_edges)]:
            if count:
                self.errors.add(code, shard, count)

        # No supersession is materialized in D1; the explicit empty ledger is
        # bound into every current projection.
        supersession = self._root_for(shard) / "canonical" / "SUPERSESSION_EDGES.jsonl"
        if supersession.exists() and sha_file(supersession) != sha_bytes(b""):
            self.errors.add("SUPERSESSION_LEDGER_NOT_EMPTY", shard)

    def verify_foreign_keys_and_bijections(self, shard: str) -> None:
        seq_tables = ["sequences"]
        for table, fields in [("obs_candidates", ["sequence_id"]), ("observations", ["sequence_id"]), ("relations", ["source_sequence_id", "candidate_sequence_id"]), ("pairs", ["source_sequence_id", "candidate_sequence_id"])]:
            for field in fields:
                bad = self.scalar(
                    f"SELECT COUNT(*) FROM {table} x LEFT JOIN sequences s ON s.object_id=x.{field} AND s.shard=x.shard WHERE x.shard=? AND x.{field} IS NOT NULL AND s.object_id IS NULL",
                    (shard,),
                )
                if bad:
                    self.errors.add("ORPHAN_SEQUENCE_FK", f"{shard}:{table}:{field}", bad)

        accepted_candidates = self.scalar("SELECT COUNT(*) FROM obs_candidates WHERE shard=? AND acceptance_status='ACCEPTED'", (shard,))
        observations = self.scalar("SELECT COUNT(*) FROM observations WHERE shard=?", (shard,))
        if accepted_candidates != observations:
            self.errors.add("OBSERVATION_BIJECTION_COUNT", shard)
        bad = self.scalar(
            """
            SELECT COUNT(*) FROM obs_candidates c
            LEFT JOIN observations o ON o.object_id=c.accepted_observation_id AND o.shard=c.shard
            WHERE c.shard=? AND c.acceptance_status='ACCEPTED' AND
              (c.accepted_observation_id IS NULL OR o.object_id IS NULL OR o.candidate_id != c.object_id OR
               o.sequence_id != c.sequence_id OR o.context_id != c.context_id OR o.endpoint_id != c.endpoint_id OR
               o.source_file_sha256 != c.source_file_sha256 OR o.contributor_hash != c.contributor_hash OR
               o.assets_json != c.assets_json OR o.files_json != c.files_json OR o.payload_hash != c.payload_hash)
            """,
            (shard,),
        )
        if bad:
            self.errors.add("OBSERVATION_BIJECTION_PAYLOAD", shard, bad)
        orphan_obs = self.scalar(
            """
            SELECT COUNT(*) FROM observations o
            LEFT JOIN obs_candidates c ON c.object_id=o.candidate_id AND c.shard=o.shard
            WHERE o.shard=? AND (c.object_id IS NULL OR c.acceptance_status!='ACCEPTED' OR c.accepted_observation_id != o.object_id)
            """,
            (shard,),
        )
        if orphan_obs:
            self.errors.add("ORPHAN_OBSERVATION", shard, orphan_obs)

        accepted_relations = self.scalar("SELECT COUNT(*) FROM relations WHERE shard=? AND relation_status='ACCEPTED'", (shard,))
        pairs = self.scalar("SELECT COUNT(*) FROM pairs WHERE shard=?", (shard,))
        if accepted_relations != pairs:
            self.errors.add("PAIR_BIJECTION_COUNT", shard)
        bad_pair = self.scalar(
            """
            SELECT COUNT(*) FROM relations r
            LEFT JOIN pairs p ON p.object_id=r.accepted_pair_id AND p.shard=r.shard
            WHERE r.shard=? AND r.relation_status='ACCEPTED' AND
              (r.accepted_pair_id IS NULL OR p.object_id IS NULL OR p.relation_id != r.object_id OR
               p.source_sequence_id != r.source_sequence_id OR p.candidate_sequence_id != r.candidate_sequence_id OR
               p.context_id != r.context_id OR p.endpoint_id != r.endpoint_id OR p.design_group != r.design_group OR
               p.relation_type != r.relation_type OR p.relation_status != 'ACCEPTED' OR
               p.effect_evidence != r.effect_evidence OR p.landscape_role != r.landscape_role OR
               p.future_use_role != r.future_use_role OR p.immutable_role != r.future_use_role OR
               p.no_edit_subtype != r.no_edit_subtype OR p.frame_id != r.frame_id OR
               p.scientific_track != r.scientific_track OR p.pairing_method != r.pairing_method OR
               p.label_unit != r.label_unit OR p.label_transform != r.label_transform OR
               p.delta_rule_id != r.delta_rule_id OR p.delta_rule_sha256 != r.delta_rule_sha256 OR
               p.pair_evidence_id != r.pair_evidence_id OR p.contributor_hash != r.contributor_hash OR
               p.assets_json != r.assets_json OR p.files_json != r.files_json OR p.payload_hash != r.payload_hash)
            """,
            (shard,),
        )
        if bad_pair:
            self.errors.add("PAIR_BIJECTION_PAYLOAD", shard, bad_pair)
        orphan_pairs = self.scalar(
            """
            SELECT COUNT(*) FROM pairs p
            LEFT JOIN relations r ON r.object_id=p.relation_id AND r.shard=p.shard
            WHERE p.shard=? AND (r.object_id IS NULL OR r.relation_status!='ACCEPTED' OR r.accepted_pair_id != p.object_id)
            """,
            (shard,),
        )
        if orphan_pairs:
            self.errors.add("ORPHAN_PAIR", shard, orphan_pairs)

        # Use roles are D1 pair-only, with no role rows for restricted rejected
        # candidates or observations.
        bad_use = self.scalar(
            "SELECT COUNT(*) FROM use_roles u LEFT JOIN pairs p ON p.object_id=u.object_id AND p.shard=u.shard WHERE u.shard=? AND (p.object_id IS NULL OR u.pair_id != p.object_id OR u.relation_id != p.relation_id)",
            (shard,),
        )
        extra_use = self.scalar(
            "SELECT COUNT(*) FROM pairs p LEFT JOIN use_roles u ON u.object_id=p.object_id AND u.shard=p.shard WHERE p.shard=? AND u.object_id IS NULL",
            (shard,),
        )
        if bad_use:
            self.errors.add("USE_ROLE_FK_OR_PAYLOAD", shard, bad_use)
        if extra_use:
            self.errors.add("MISSING_USE_ROLE", shard, extra_use)
        expected_role = FUTURE_ROLE_ORDINARY if shard == "ordinary" else FUTURE_ROLE_SEALED
        bad_role = self.scalar("SELECT COUNT(*) FROM use_roles WHERE shard=? AND (future_role != ? OR authority_level != ?)", (shard, expected_role, "ORDINARY" if shard == "ordinary" else "RESTRICTED"))
        if bad_role:
            self.errors.add("USE_ROLE_POLICY_MISMATCH", shard, bad_role)

    def verify_groups(self, shard: str) -> None:
        bad_assignment_object = self.scalar(
            "SELECT COUNT(*) FROM assignments a LEFT JOIN objects o ON o.object_id=a.object_id AND o.shard=a.shard WHERE a.shard=? AND o.object_id IS NULL",
            (shard,),
        )
        bad_assignment_group = self.scalar(
            "SELECT COUNT(*) FROM assignments a LEFT JOIN groups g ON g.group_id=a.group_id AND g.shard=a.shard WHERE a.shard=? AND g.group_id IS NULL",
            (shard,),
        )
        missing_assignment = self.scalar(
            "SELECT COUNT(*) FROM objects o LEFT JOIN assignments a ON a.object_id=o.object_id AND a.shard=o.shard WHERE o.shard=? AND a.object_id IS NULL",
            (shard,),
        )
        duplicate_assignment = self.scalar(
            "SELECT COUNT(*) FROM (SELECT object_id FROM assignments WHERE shard=? GROUP BY object_id HAVING COUNT(*) != 1)",
            (shard,),
        )
        for code, count in [("ASSIGNMENT_OBJECT_FK", bad_assignment_object), ("ASSIGNMENT_GROUP_FK", bad_assignment_group), ("MISSING_OBJECT_ASSIGNMENT", missing_assignment), ("ASSIGNMENT_CARDINALITY", duplicate_assignment)]:
            if count:
                self.errors.add(code, shard, count)
        for (row_shard, group_id), row in self.group_rows.items():
            if row_shard != shard:
                continue
            if row.get("group_type") == "NO_EDIT_SAMPLING_FRAME":
                expected_hash = self.frame_hashers.get((shard, group_id))
                if expected_hash is not None and row.get("member_assignment_manifest_sha256") != expected_hash.hexdigest():
                    self.errors.add("FRAME_ASSIGNMENT_MANIFEST_MISMATCH", shard)
                continue
            actual_count = self.scalar("SELECT COUNT(*) FROM assignments WHERE shard=? AND group_id=?", (shard, group_id))
            if actual_count != row.get("member_count"):
                self.errors.add("GROUP_MEMBER_COUNT_MISMATCH", shard)
            member_ids = row.get("member_ids")
            if member_ids is None:
                member_ids = json.loads(row.get("member_ids_json") or "[]")
            if actual_count <= 1000:
                actual = [r[0] for r in self.store.conn.execute("SELECT object_id FROM assignments WHERE shard=? AND group_id=? ORDER BY object_id", (shard, group_id))]
                if actual != sorted(set(member_ids)):
                    self.errors.add("GROUP_MEMBER_SET_MISMATCH", shard)
            elif member_ids:
                self.errors.add("GROUP_MEMBER_CAP_MISMATCH", shard)

        bad_frame_link = self.scalar(
            """
            SELECT COUNT(*) FROM assignments a
            JOIN objects o ON o.object_id=a.object_id AND o.shard=a.shard
            LEFT JOIN groups g ON g.group_id=a.frame_id AND g.shard=a.shard
            WHERE a.shard=? AND a.frame_id IS NOT NULL AND g.group_id IS NULL
            """,
            (shard,),
        )
        if bad_frame_link:
            self.errors.add("NO_EDIT_FRAME_FK", shard, bad_frame_link)
        # Every accepted designed identity relation/pair carries the same
        # frame ID; nonidentity rows must carry the explicit nonidentity enum.
        bad_subtype = self.scalar(
            "SELECT COUNT(*) FROM relations WHERE shard=? AND relation_type!='NO_EDIT_CONTROL' AND no_edit_subtype!='NOT_APPLICABLE_NON_IDENTITY'",
            (shard,),
        )
        bad_pair_subtype = self.scalar(
            "SELECT COUNT(*) FROM pairs WHERE shard=? AND relation_type!='NO_EDIT_CONTROL' AND no_edit_subtype!='NOT_APPLICABLE_NON_IDENTITY'",
            (shard,),
        )
        if bad_subtype or bad_pair_subtype:
            self.errors.add("NONIDENTITY_NOEDIT_SUBTYPE", shard, bad_subtype + bad_pair_subtype)

    def _root_for(self, shard: str) -> Path:
        return self.ordinary_root if shard == "ordinary" else self.restricted_root

    def context_id(self, accession: str, region: Any, metadata: dict[str, Any], source_file: str) -> str:
        library = as_text(metadata.get("library")) or as_text(metadata.get("subpool")) or "UNKNOWN_LIBRARY"
        return f"CTX:{safe_id('|'.join([accession, region_scope(region), source_file, library]))}"

    def endpoint_id(self, label_key: str) -> str:
        return "EP:NO_MEASUREMENT" if label_key == "NO_MEASUREMENT" else f"EP:{safe_id(label_key)}"

    def frame_id(self, accession: str, region: Any, metadata: dict[str, Any], source_file: str, context: str, endpoint: str, assets: list[str]) -> str:
        library = as_text(metadata.get("library")) or as_text(metadata.get("subpool")) or "UNKNOWN_LIBRARY"
        definition = {
            "study_id": accession,
            "asset_ids": sorted(set(assets)),
            "library_lineage_group_id": f"LIBRARY_LINEAGE:{safe_id(accession + '|' + source_file + '|' + library)}",
            "sublibrary_or_design_stratum": library,
            "species": as_text(metadata.get("species")) or "UNKNOWN",
            "region_scope": region_scope(region),
            "context_id": context,
            "endpoint_id": endpoint,
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
        return f"NO_EDIT_SAMPLING_FRAME:{sha_json(definition)}"

    def compare(self, actual: Any, expected: Any, code: str, artifact: str) -> None:
        if actual != expected:
            self.errors.add(code, artifact)

    def sequence_for_raw(self, shard: str, accession: str, record_id: str, side: str, value: Any, region: Any, metadata: dict[str, Any], source_file: str, locator: str, assets: list[str], files: list[str], primary: str | None, raw_line: str) -> tuple[str, bool]:
        seq_id = f"SEQ:{shard}:{safe_id(record_id)}:{side}"
        normalized, alphabet, steps, invalid, eligible = normalize_sequence(value)
        row = self.store.one("sequences", "object_id=? AND shard=?", (seq_id, shard))
        artifact = f"{shard}:raw_sequence:{side}"
        if row is None:
            self.errors.add("MISSING_SEQUENCE_ENTITY", artifact)
            return seq_id, eligible
        self.compare(row["raw_hash"], sha_text(as_text(value)), "SEQUENCE_RAW_HASH_MISMATCH", artifact)
        self.compare(row["norm_hash"], sha_text(normalized or ""), "SEQUENCE_NORMALIZED_HASH_MISMATCH", artifact)
        self.compare(row["full_hash"], sha_text(normalized or ""), "SEQUENCE_FULL_HASH_MISMATCH", artifact)
        self.compare(row["normalized_hash"], row["norm_hash"], "SEQUENCE_NORMALIZED_PAYLOAD_HASH_MISMATCH", artifact)
        self.compare(row["original_length"], len(as_text(value)), "SEQUENCE_LENGTH_MISMATCH", artifact)
        self.compare(row["region_scope"], region_scope(region), "SEQUENCE_REGION_SCOPE_MISMATCH", artifact)
        self.compare(row["source_record_id"], record_id, "SEQUENCE_SOURCE_RECORD_MISMATCH", artifact)
        self.compare(row["source_row_locator"], locator, "SEQUENCE_SOURCE_LOCATOR_MISMATCH", artifact)
        self.compare(row["assets_json"], json.dumps(sorted(set(assets)), separators=(",", ":")), "SEQUENCE_ASSET_LINEAGE_MISMATCH", artifact)
        self.compare(row["files_json"], json.dumps(sorted(set(files)), separators=(",", ":")), "SEQUENCE_FILE_LINEAGE_MISMATCH", artifact)
        self.compare(row["contributor_hash"], sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}), "SEQUENCE_CONTRIBUTOR_HASH_MISMATCH", artifact)
        self.compare(row["primary_asset_id"], assets[0] if assets else f"{accession}::UNMAPPED_D0_ASSET", "SEQUENCE_PRIMARY_ASSET_MISMATCH", artifact)
        self.compare(row["model_eligible"], int(eligible), "SEQUENCE_ELIGIBILITY_MISMATCH", artifact)
        self.compare(row["alphabet_status"], alphabet, "SEQUENCE_ALPHABET_STATUS_MISMATCH", artifact)
        self.compare(row["invalid_status"], invalid, "SEQUENCE_INVALID_STATUS_MISMATCH", artifact)
        edge = self.store.one("edges", "new_id=? AND shard=?", (seq_id, shard))
        if edge is None or edge["old_id"] != f"RAW_RECORD:{shard}:{self.current_line}:{safe_id(record_id)}" or edge["old_hash"] != sha_bytes(raw_line.encode("utf-8")):
            self.errors.add("SEQUENCE_TRANSFORMATION_LINEAGE_MISMATCH", artifact)
        return seq_id, eligible

    def expected_contributors(self, accession: str, metadata: dict[str, Any], restricted: bool) -> tuple[str, list[str], list[str], str | None]:
        source_value = metadata.get("source_file")
        if restricted and not as_text(source_value):
            source_value = Path(self.args.sealed_input).name
        assets, files, primary = self.d0.resolve(accession, source_value)
        return as_text(source_value) or "<MISSING>", assets, files, primary

    def reject_expected(self, shard: str, reason: str, locator: str) -> None:
        self.expected_rejections[shard][reason] += 1
        if self.store.one("rejections", "shard=? AND source_locator=? AND reason=?", (shard, locator, reason)) is None:
            self.errors.add("MISSING_REJECTION_DISPOSITION", f"{shard}:{reason}")

    def raw_record(self, shard: str, line_no: int, raw_line: str, rec: dict[str, Any], accession_override: str | None = None, path: Path | None = None) -> None:
        restricted = shard == "restricted"
        accession = accession_override or as_text(rec.get("accession")) or "UNKNOWN_ACCESSION"
        record_id = as_text(rec.get("record_id")) or f"MISSING_RECORD_{line_no}"
        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        if restricted and not as_text(metadata.get("source_file")):
            metadata = dict(metadata)
            metadata["source_file"] = Path(self.args.sealed_input).name
        source_path = path or self.raw_paths[shard]
        locator = row_locator(source_path, line_no)
        region = rec.get("region") or metadata.get("region") or "5'UTR"
        source_value = rec.get("source_sequence")
        candidate_value = rec.get("candidate_sequence")
        has_source = bool(as_text(source_value))
        has_candidate = bool(as_text(candidate_value))
        source_file, assets, files, primary = self.expected_contributors(accession, metadata, restricted)
        context = self.context_id(accession, region, metadata, source_file)
        stats = self.raw_stats[shard]
        stats["raw_records"] += 1
        stats[f"dataset:{accession}"] += 1
        stats[f"record_type:{as_text(metadata.get('record_type')) or '<MISSING>'}"] += 1
        stats[f"region:{region_scope(region)}"] += 1
        if has_source:
            stats["source_sequence_rows"] += 1
        if has_candidate:
            stats["candidate_sequence_rows"] += 1
        source_id = None
        candidate_id = None
        source_valid = False
        candidate_valid = False
        if has_source:
            source_id, source_valid = self.sequence_for_raw(shard, accession, record_id, "SOURCE", source_value, region, metadata, source_file, locator, assets, files, primary, raw_line)
        if has_candidate:
            candidate_id, candidate_valid = self.sequence_for_raw(shard, accession, record_id, "CANDIDATE", candidate_value, region, metadata, source_file, locator, assets, files, primary, raw_line)

        labels = rec.get("labels") if isinstance(rec.get("labels"), dict) else {}
        exact_identity = has_source and has_candidate and as_text(source_value) == as_text(candidate_value)
        designed_identity = exact_identity and accession == "GSE114002" and Path(source_file).name == "GSM3130443_designed_library.csv.gz"
        accepted_observation_ids: list[tuple[str, str]] = []
        for label_key_raw, raw_value in labels.items():
            label_key = as_text(label_key_raw) or "UNNAMED_LABEL"
            stats["labels"] += 1
            stats[f"label:{label_key}"] += 1
            endpoint = self.endpoint_id(label_key)
            label_locator = row_locator(source_path, line_no, f"label={token(label_key)}")
            numeric = finite_number(raw_value)
            observation_sequence_id = candidate_id if candidate_id is not None else source_id
            observation_valid = candidate_valid if candidate_id is not None else source_valid
            candidate_id_for_label = f"OBS_CAND:{shard}:{safe_id(record_id + '|' + label_key)}"
            observation_id = f"OBS:{shard}:{safe_id(record_id + '|' + label_key)}"
            if observation_sequence_id is None:
                stats["label_rejected"] += 1
                self.reject_expected(shard, "NO_BINDABLE_SEQUENCE_LABEL", label_locator)
                if self.store.one("obs_candidates", "object_id=? AND shard=?", (candidate_id_for_label, shard)) is not None:
                    self.errors.add("UNEXPECTED_NOSEQUENCE_OBSERVATION_CANDIDATE", shard)
                continue
            accepted = numeric is not None and observation_valid
            if accepted:
                acceptance, lifecycle, terminal = "ACCEPTED", "ACCEPTED", None
            elif not observation_valid:
                acceptance, lifecycle, terminal = "REJECTED", "REJECTED", "INVALID_OR_AMBIGUOUS_SEQUENCE"
            elif numeric is None:
                acceptance, lifecycle, terminal = "REJECTED", "REJECTED", "NULL_OR_NONNUMERIC_LABEL"
            else:
                acceptance, lifecycle, terminal = "REJECTED", "REJECTED", "UNRESOLVED_LABEL"
            c = self.store.one("obs_candidates", "object_id=? AND shard=?", (candidate_id_for_label, shard))
            if c is None:
                self.errors.add("MISSING_OBSERVATION_CANDIDATE", shard)
            else:
                expected_source_hash = primary or NOT_AVAILABLE
                expected_payload = sha_json({"sequence_id": observation_sequence_id, "context_id": context, "endpoint_id": endpoint, "contributing_asset_ids": sorted(set(assets)), "contributing_source_file_sha256s": sorted(set(files)), "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}), "source_file_sha256": expected_source_hash, "value": numeric})
                for actual, expected, code in [
                    (c["sequence_id"], observation_sequence_id, "OBS_CAND_SEQUENCE_MISMATCH"),
                    (c["context_id"], context, "OBS_CAND_CONTEXT_MISMATCH"),
                    (c["endpoint_id"], endpoint, "OBS_CAND_ENDPOINT_MISMATCH"),
                    (c["accepted_observation_id"], observation_id if accepted else None, "OBS_CAND_ACCEPTED_ID_MISMATCH"),
                    (c["acceptance_status"], acceptance, "OBS_CAND_ACCEPTANCE_MISMATCH"),
                    (c["lifecycle_status"], lifecycle, "OBS_CAND_LIFECYCLE_MISMATCH"),
                    (c["terminal_reason"], terminal, "OBS_CAND_TERMINAL_MISMATCH"),
                    (c["source_file_sha256"], expected_source_hash, "OBS_CAND_SOURCE_FILE_MISMATCH"),
                    (c["contributor_hash"], sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}), "OBS_CAND_CONTRIBUTOR_HASH_MISMATCH"),
                    (c["assets_json"], json.dumps(sorted(set(assets)), separators=(",", ":")), "OBS_CAND_ASSET_MISMATCH"),
                    (c["files_json"], json.dumps(sorted(set(files)), separators=(",", ":")), "OBS_CAND_FILE_MISMATCH"),
                    (c["source_locator"], label_locator, "OBS_CAND_LOCATOR_MISMATCH"),
                    (c["payload_hash"], expected_payload, "OBS_CAND_PAYLOAD_MISMATCH"),
                ]:
                    self.compare(actual, expected, code, shard)
                edge = self.store.one("edges", "new_id=? AND shard=?", (candidate_id_for_label, shard))
                if edge is None or edge["old_id"] != f"RAW_RECORD:{shard}:{line_no}:{safe_id(record_id)}" or edge["old_hash"] != sha_bytes(raw_line.encode("utf-8")):
                    self.errors.add("OBS_CAND_TRANSFORMATION_LINEAGE_MISMATCH", shard)
            if accepted:
                accepted_observation_ids.append((endpoint, observation_id))
                stats["label_complete"] += 1
                o = self.store.one("observations", "object_id=? AND shard=?", (observation_id, shard))
                if o is None:
                    self.errors.add("MISSING_FUNCTIONAL_OBSERVATION", shard)
                else:
                    expected_source_hash = primary or NOT_AVAILABLE
                    expected_payload = sha_json({"sequence_id": observation_sequence_id, "context_id": context, "endpoint_id": endpoint, "contributing_asset_ids": sorted(set(assets)), "contributing_source_file_sha256s": sorted(set(files)), "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}), "source_file_sha256": expected_source_hash, "value": numeric})
                    for actual, expected, code in [
                        (o["candidate_id"], candidate_id_for_label, "OBSERVATION_CANDIDATE_ID_MISMATCH"),
                        (o["sequence_id"], observation_sequence_id, "OBSERVATION_SEQUENCE_MISMATCH"),
                        (o["context_id"], context, "OBSERVATION_CONTEXT_MISMATCH"),
                        (o["endpoint_id"], endpoint, "OBSERVATION_ENDPOINT_MISMATCH"),
                        (o["scientific_track"], "E" if designed_identity else "F", "OBSERVATION_TRACK_MISMATCH"),
                        (o["observation_role"], "E_NOEDIT_MEASUREMENT" if designed_identity else "F_FUNCTION_LABEL", "OBSERVATION_ROLE_MISMATCH"),
                        (o["source_file_sha256"], expected_source_hash, "OBSERVATION_SOURCE_FILE_MISMATCH"),
                        (o["contributor_hash"], sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}), "OBSERVATION_CONTRIBUTOR_HASH_MISMATCH"),
                        (o["assets_json"], json.dumps(sorted(set(assets)), separators=(",", ":")), "OBSERVATION_ASSET_MISMATCH"),
                        (o["files_json"], json.dumps(sorted(set(files)), separators=(",", ":")), "OBSERVATION_FILE_MISMATCH"),
                        (o["source_locator"], label_locator, "OBSERVATION_LOCATOR_MISMATCH"),
                        (o["payload_hash"], expected_payload, "OBSERVATION_PAYLOAD_MISMATCH"),
                    ]:
                        self.compare(actual, expected, code, shard)
                    edge = self.store.one("edges", "new_id=? AND shard=?", (observation_id, shard))
                    if edge is None or edge["old_id"] != f"RAW_RECORD:{shard}:{line_no}:{safe_id(record_id)}" or edge["old_hash"] != sha_bytes(raw_line.encode("utf-8")):
                        self.errors.add("OBSERVATION_TRANSFORMATION_LINEAGE_MISMATCH", shard)
            else:
                stats["label_rejected"] += 1
                reason = "NO_BINDABLE_SEQUENCE_LABEL" if observation_sequence_id is None else ("INVALID_OR_AMBIGUOUS_SEQUENCE" if not observation_valid else ("NULL_OR_NONNUMERIC_LABEL" if numeric is None else "UNRESOLVED_LABEL"))
                self.reject_expected(shard, reason, label_locator)

        if has_source and has_candidate:
            stats["relation_candidates"] += 1
            relation_id = f"REL:{shard}:{safe_id(record_id)}"
            pair_id = f"PAIR:{shard}:{safe_id(record_id)}"
            edit_verified = bool(rec.get("edit_script_verified", False))
            if designed_identity:
                relation_type, subtype = "NO_EDIT_CONTROL", "DESIGNED_WT_CONTROL"
                relation_status = "ACCEPTED" if source_valid and candidate_valid else "REJECTED"
            elif exact_identity:
                relation_type, subtype = "NO_EDIT_CONTROL", "UNRESOLVED_IDENTITY"
                relation_status = "AMBIGUOUS" if source_valid and candidate_valid else "REJECTED"
            else:
                relation_type, subtype = "SOURCE_CANDIDATE", "NOT_APPLICABLE_NON_IDENTITY"
                relation_status = "ACCEPTED" if source_valid and candidate_valid and edit_verified else "REJECTED"
            lifecycle = "ACCEPTED" if relation_status == "ACCEPTED" else ("CANDIDATE" if relation_status == "AMBIGUOUS" else "REJECTED")
            endpoint = accepted_observation_ids[0][0] if accepted_observation_ids else "EP:NO_MEASUREMENT"
            frame = self.frame_id(accession, region, metadata, source_file, context, endpoint, assets) if designed_identity else None
            design_key = as_text(metadata.get("pair_key")) or record_id
            design_group = f"DESIGN_RELATION_GROUP:{safe_id(accession + '|' + design_key)}"
            role = FUTURE_ROLE_SEALED if restricted else FUTURE_ROLE_ORDINARY
            r = self.store.one("relations", "object_id=? AND shard=?", (relation_id, shard))
            if r is None:
                self.errors.add("MISSING_RELATION_CANDIDATE", shard)
            else:
                expected = {
                    "accepted_pair_id": pair_id if relation_status == "ACCEPTED" else None,
                    "source_sequence_id": source_id, "candidate_sequence_id": candidate_id, "context_id": context,
                    "endpoint_id": endpoint, "design_group": design_group, "relation_type": relation_type,
                    "relation_status": relation_status, "lifecycle_status": lifecycle, "effect_evidence": "CANDIDATE_ONLY" if accepted_observation_ids else "SEQUENCE_ONLY",
                    "landscape_role": "SPARSE", "future_use_role": role, "no_edit_subtype": subtype, "frame_id": frame,
                    "scientific_track": "E", "pairing_method": self.pairing_method(accession), "label_unit": "ASSAY_REPORTED_VALUE" if accepted_observation_ids else "NOT_APPLICABLE",
                    "label_transform": "IDENTITY_FLOAT_PARSE_V1" if accepted_observation_ids else "NO_VALUE", "delta_rule_id": "D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1",
                    "delta_rule_sha256": sha_text("D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1"), "pair_evidence_id": f"PAIR_EVIDENCE:{safe_id(record_id)}",
                    "contributor_hash": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}),
                    "assets_json": json.dumps(sorted(set(assets)), separators=(",", ":")), "files_json": json.dumps(sorted(set(files)), separators=(",", ":")),
                    "payload_hash": self.relation_payload_hash({"context_id": context, "endpoint_id": endpoint, "design_relation_group_id": design_group, "effect_evidence": "CANDIDATE_ONLY" if accepted_observation_ids else "SEQUENCE_ONLY", "landscape_role": "SPARSE", "future_use_role": role, "no_edit_control_subtype": subtype, "no_edit_sampling_frame_id": frame, "scientific_track": "E", "relation_type": relation_type, "pairing_method": self.pairing_method(accession), "label_unit": "ASSAY_REPORTED_VALUE" if accepted_observation_ids else "NOT_APPLICABLE", "label_transform": "IDENTITY_FLOAT_PARSE_V1" if accepted_observation_ids else "NO_VALUE", "delta_rule_id": "D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1", "delta_rule_sha256": sha_text("D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1"), "pair_evidence_id": f"PAIR_EVIDENCE:{safe_id(record_id)}", "source_sequence_id": source_id, "candidate_sequence_id": candidate_id, "contributing_asset_ids": sorted(set(assets)), "contributing_source_file_sha256s": sorted(set(files)), "contributor_set_sha256": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))})}),
                }
                for field, expected_value in expected.items():
                    self.compare(r[field], expected_value, f"RELATION_{field.upper()}_MISMATCH", shard)
                edge = self.store.one("edges", "new_id=? AND shard=?", (relation_id, shard))
                if edge is None or edge["old_id"] != f"RAW_RECORD:{shard}:{line_no}:{safe_id(record_id)}" or edge["old_hash"] != sha_bytes(raw_line.encode("utf-8")):
                    self.errors.add("RELATION_TRANSFORMATION_LINEAGE_MISMATCH", shard)
            if relation_status == "ACCEPTED":
                stats["pairs"] += 1
                p = self.store.one("pairs", "object_id=? AND shard=?", (pair_id, shard))
                if p is None:
                    self.errors.add("MISSING_ACCEPTED_PAIR", shard)
                else:
                    for field, expected_value in {
                        "relation_id": relation_id, "source_sequence_id": source_id, "candidate_sequence_id": candidate_id,
                        "context_id": context, "endpoint_id": endpoint, "design_group": design_group, "relation_type": relation_type,
                        "relation_status": "ACCEPTED", "effect_evidence": "CANDIDATE_ONLY" if accepted_observation_ids else "SEQUENCE_ONLY",
                        "landscape_role": "SPARSE", "future_use_role": role, "immutable_role": role, "no_edit_subtype": subtype,
                        "frame_id": frame, "scientific_track": "E", "pairing_method": self.pairing_method(accession),
                        "label_unit": "ASSAY_REPORTED_VALUE" if accepted_observation_ids else "NOT_APPLICABLE", "label_transform": "IDENTITY_FLOAT_PARSE_V1" if accepted_observation_ids else "NO_VALUE",
                        "delta_rule_id": "D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1", "delta_rule_sha256": sha_text("D1_NO_CONFIRMATORY_DELTA_UNTIL_B0_JOIN_V1"),
                        "pair_evidence_id": f"PAIR_EVIDENCE:{safe_id(record_id)}", "contributor_hash": sha_json({"asset_ids": sorted(set(assets)), "source_file_sha256s": sorted(set(files))}),
                        "assets_json": json.dumps(sorted(set(assets)), separators=(",", ":")), "files_json": json.dumps(sorted(set(files)), separators=(",", ":")),
                    }.items():
                        self.compare(p[field], expected_value, f"PAIR_{field.upper()}_MISMATCH", shard)
                    if p["payload_hash"] != r["payload_hash"]:
                        self.errors.add("PAIR_RELATION_PAYLOAD_HASH_MISMATCH", shard)
                    edge = self.store.one("edges", "new_id=? AND shard=?", (pair_id, shard))
                    if edge is None or edge["old_id"] != f"RAW_RECORD:{shard}:{line_no}:{safe_id(record_id)}" or edge["old_hash"] != sha_bytes(raw_line.encode("utf-8")):
                        self.errors.add("PAIR_TRANSFORMATION_LINEAGE_MISMATCH", shard)
            else:
                reason = "UNRESOLVED_IDENTITY_RELATION" if relation_status == "AMBIGUOUS" else "RELATION_REJECTED_INVALID_SEQUENCE_OR_SCRIPT"
                self.reject_expected(shard, reason, locator)
        elif not has_source and not has_candidate:
            stats["no_sequence_records"] += 1
            self.reject_expected(shard, "NO_BINDABLE_SEQUENCE_RECORD", locator)

    def pairing_method(self, accession: str) -> str:
        if accession in {"GSE114002", "GSE149487", "GSE200304"}:
            return "DESIGN_TABLE"
        if accession in {"GSE186455", "GSE217518", "GSE232572"}:
            return "VARIANT_RECONSTRUCTION"
        if accession == "ENCSR854RUF":
            return "EXPLICIT_ID"
        return "OTHER"

    def validate_raw_inputs(self) -> None:
        self.current_line = 0
        for shard, path, accession in [("ordinary", self.raw_paths.get("ordinary"), None), ("restricted", self.raw_paths.get("restricted"), SEALED_COHORT)]:
            if not path or not path.is_file():
                self.errors.add("RAW_INPUT_MISSING", shard)
                continue
            target_count = self.input_manifest.get("ordinary_raw_input", {}).get("record_count") if shard == "ordinary" else self.summary.get("counts", {}).get("raw_records:restricted")
            seen = 0
            with path.open("r", encoding="utf-8", newline="") as fh:
                for line_no, raw_line in enumerate(fh, 1):
                    if not raw_line.strip():
                        continue
                    self.current_line = line_no
                    try:
                        rec = json.loads(raw_line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
                    except Exception:
                        self.errors.add("RAW_INPUT_INVALID_JSON", shard, line_no)
                        continue
                    if not isinstance(rec, dict):
                        self.errors.add("RAW_INPUT_ROW_NOT_OBJECT", shard, line_no)
                        continue
                    self.raw_record(shard, line_no, raw_line, rec, accession, path)
                    seen += 1
                    if target_count is not None and seen >= int(target_count):
                        break
            if target_count is not None and seen != int(target_count):
                self.errors.add("RAW_INPUT_PREFIX_SHORT", shard)
        legacy = self.raw_paths.get("legacy")
        if legacy and legacy.is_file() and self.input_manifest.get("legacy_quarantine_input", {}).get("record_count", 0):
            with legacy.open("r", encoding="utf-8", newline="") as fh:
                for line_no, raw_line in enumerate(fh, 1):
                    if not raw_line.strip():
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except Exception:
                        self.errors.add("LEGACY_INPUT_INVALID_JSON", "legacy", line_no)
                        continue
                    accession = as_text(rec.get("accession")) or "UNKNOWN_ACCESSION"
                    reason = f"LEGACY_QUARANTINE_{token(accession)}"
                    locator = row_locator(legacy, line_no)
                    self.expected_rejections["ordinary"][reason] += 1
                    self.reject_expected("ordinary", reason, locator)
                    self.raw_stats["ordinary"]["legacy_records"] += 1

        # Full D1 is intentionally not inferred from a small smoke input.
        if self.raw_stats["ordinary"]["raw_records"] != self.args.expected_ordinary_records:
            self.errors.add("ORDINARY_INPUT_NOT_FULL_EXPECTED_SET")
        if self.raw_stats["restricted"]["raw_records"] != self.args.expected_restricted_records:
            self.errors.add("RESTRICTED_INPUT_NOT_FULL_EXPECTED_SET")

    def verify_raw_counts(self) -> None:
        for shard in ("ordinary", "restricted"):
            s = self.raw_stats[shard]
            expected = {
                "sequences": s["source_sequence_rows"] + s["candidate_sequence_rows"],
                "obs_candidates": s["labels"] - s["no_sequence_records"] * 0,  # corrected below from row-level table
                "observations": s["label_complete"],
                "relations": s["relation_candidates"],
                "pairs": s["pairs"],
                "rejections": sum(self.expected_rejections[shard].values()),
            }
            # A label with no bindable sequence has no observation-candidate
            # row; all other labels do.
            no_sequence_labels = self.expected_rejections[shard].get("NO_BINDABLE_SEQUENCE_LABEL", 0)
            expected["obs_candidates"] = s["labels"] - no_sequence_labels
            expected["transformations"] = expected["sequences"] + expected["obs_candidates"] + expected["observations"] + expected["relations"] + expected["pairs"]
            actual = {
                "sequences": self.store.count("sequences", shard),
                "obs_candidates": self.store.count("obs_candidates", shard),
                "observations": self.store.count("observations", shard),
                "relations": self.store.count("relations", shard),
                "pairs": self.store.count("pairs", shard),
                "rejections": self.store.count("rejections", shard),
                "transformations": self.store.count("edges", shard),
            }
            self.counters.update({f"{shard}:{k}": v for k, v in actual.items()})
            for key, value in expected.items():
                if actual.get(key) != value:
                    self.errors.add("RAW_OUTPUT_CONSERVATION_MISMATCH", f"{shard}:{key}")
            for reason, value in self.expected_rejections[shard].items():
                actual_reason = self.scalar("SELECT COUNT(*) FROM rejections WHERE shard=? AND reason=?", (shard, reason))
                if actual_reason != value:
                    self.errors.add("REJECTION_REASON_CONSERVATION_MISMATCH", f"{shard}:{reason}")
            summary_counts = self.summary.get("counts", {})
            pairs = {
                f"raw_records:{shard}": s["raw_records"],
                f"raw_source_sequence_rows:{shard}": s["source_sequence_rows"],
                f"raw_candidate_sequence_rows:{shard}": s["candidate_sequence_rows"],
                f"relation_candidates:{shard}": s["relation_candidates"],
                f"pairs:{shard}": s["pairs"],
                f"observations:{shard}": s["label_complete"],
                f"rejections:{shard}": sum(self.expected_rejections[shard].values()),
                f"transformation_edges:{shard}": actual["transformations"],
            }
            for key, value in pairs.items():
                if summary_counts.get(key, 0) != value:
                    self.errors.add("SUMMARY_COUNT_MISMATCH", f"{shard}:{key}")

    def verify_self_json(self, path: Path, field: str, artifact: str, marker_scan: bool = False) -> dict[str, Any]:
        if not self.require(path, artifact):
            return {}
        try:
            value = json_load(path)
        except Exception:
            self.errors.add("INVALID_JSON", artifact)
            return {}
        if not isinstance(value, dict):
            self.errors.add("JSON_ARTIFACT_NOT_OBJECT", artifact)
            return {}
        if value.get(field) != sha_json({k: v for k, v in value.items() if k != field}):
            self.errors.add("SELF_HASH_MISMATCH", artifact)
        if marker_scan and contains_marker(value):
            self.errors.add("SEALED_MARKER_LEAK", artifact)
        return value

    def ledger_text(self, root: Path, paths: Iterable[Path]) -> bytes:
        rows: list[tuple[str, str]] = []
        for path in paths:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
            rows.append((rel, sha_file(path)))
        rows.sort(key=lambda x: x[0].encode("utf-8"))
        return "".join(f"{digest}  {rel}\n" for rel, digest in rows).encode("utf-8")

    def verify_ledger(self, path: Path, root: Path, expected_paths: Iterable[Path], artifact: str) -> None:
        if not self.require(path, artifact):
            return
        try:
            expected = self.ledger_text(root, expected_paths)
        except Exception:
            self.errors.add("LEDGER_PATH_ERROR", artifact)
            return
        if path.read_bytes() != expected:
            self.errors.add("CHECKSUM_LEDGER_MISMATCH", artifact)

    def schema_ref(self, logical_id: str, path: Path) -> tuple[str, str]:
        if logical_id == "SEALED_INPUT_MANIFEST":
            sid = "SEALED_INPUT_MANIFEST_V1"
            return sid, sha_text(sid)
        for logical_name, filename in CANONICAL_FILES.items():
            if logical_id == filename:
                logical_id = logical_name
                break
        if logical_id in {"DATASET_RECONCILIATION", "DATA_UNITS_REPORT", "EXPOSURE_USE_MANIFEST"}:
            sid = "AGGREGATE_REPORT_V1"
            return sid, sha_text(sid)
        if logical_id in {"ACCESS_SHA256SUMS", "EXPOSURE_USE_SHA256SUMS", "SEALED_CANONICAL_SHA256SUMS"}:
            return NON_JSON_SCHEMA_ID, NON_JSON_SCHEMA_SHA256
        mapping = {
            "SEQUENCE_ENTITIES": ("sequence_entity.schema.json", None),
            "FUNCTIONAL_OBSERVATION_CANDIDATES": ("functional_observation.schema.json", "FunctionalObservationCandidate"),
            "FUNCTIONAL_OBSERVATIONS": ("functional_observation.schema.json", None),
            "ENDPOINT_REGISTRY": ("functional_observation.schema.json", "EndpointRegistryRow"),
            "UTR_EDIT_RELATION_CANDIDATES": ("utr_edit_relation_candidate.schema.json", None),
            "UTR_EDIT_PAIRS": ("utr_edit_pair.schema.json", None),
            "REJECTIONS": ("rejection_record.schema.json", None),
            "TRANSFORMATION_EDGES": ("transformation_edge.schema.json", None),
            "SUPERSESSION_EDGES": ("transformation_edge.schema.json", "SupersessionEdge"),
            "CURRENT_CANONICAL_OBJECT_PROJECTION": ("transformation_edge.schema.json", "CurrentCanonicalObjectProjection"),
            "EXPOSURE_RECORDS": ("exposure_record.schema.json", None),
            "USE_ROLES": ("use_role.schema.json", None),
            "GROUP_REGISTRY": ("group_registry.schema.json", None),
            "GROUP_ASSIGNMENTS": ("group_assignment.schema.json", None),
            "REPORTER_ARTIFACT_ASSESSMENTS": ("reporter_artifact_assessment.schema.json", None),
            "EFFECTIVE_EXPOSURE_PROJECTION": ("exposure_record.schema.json", "EffectiveExposureProjection"),
        }
        if logical_id in {"ACCESS_LOG", "ACCESS_MANIFEST"}:
            return "exposure_record.schema.json#/$defs/AccessIntent|AccessCompletion|AccessAbort", self.schema.hashes.get("exposure_record.schema.json", NOT_AVAILABLE)
        base, definition = mapping.get(logical_id, (None, None))
        if base is None:
            return "AGGREGATE_REPORT_V1", sha_text("AGGREGATE_REPORT_V1")
        sid = f"{base}#/$defs/{definition}" if definition else base
        return sid, self.schema.hashes.get(base, NOT_AVAILABLE)

    def component_entry(self, logical_id: str, path: Path, root: Path) -> dict[str, Any]:
        schema_id, schema_sha = self.schema_ref(logical_id, path)
        return {
            "logical_id": logical_id,
            "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha_file(path),
            "schema_id": schema_id,
            "schema_sha256": schema_sha,
        }

    def access_paths(self, shard: str) -> dict[str, Path]:
        snapshot = as_text(self.summary.get("d1_snapshot_id"))
        root = self._root_for(shard)
        if shard == "ordinary":
            live = root / "exposure" / "ORDINARY_ACCESS_LOG.jsonl"
            snap = root / "exposure" / "access_snapshots" / snapshot
            return {
                "root": root, "live": live, "log": snap / "ORDINARY_ACCESS_LOG.jsonl",
                "manifest": snap / "ORDINARY_ACCESS_MANIFEST.json", "sums": snap / "ORDINARY_ACCESS_SHA256SUMS",
                "seq": snap / "objects" / "REQUESTED_SEQUENCE_OBJECTS.jsonl", "label": snap / "objects" / "REQUESTED_LABEL_OBJECTS.jsonl",
                "seq_actual": snap / "objects" / "ACTUAL_SEQUENCE_OBJECTS.jsonl", "label_actual": snap / "objects" / "ACTUAL_LABEL_OBJECTS.jsonl",
            }
        live = root / "ACCESS_LOG.jsonl"
        snap = root / "access_snapshots" / snapshot
        return {
            "root": root, "live": live, "log": snap / "ACCESS_LOG.jsonl", "manifest": snap / "ACCESS_MANIFEST.json",
            "sums": snap / "ACCESS_SHA256SUMS", "seq": snap / "objects" / "REQUESTED_SEQUENCE_OBJECTS.jsonl",
            "label": snap / "objects" / "REQUESTED_LABEL_OBJECTS.jsonl", "seq_actual": snap / "objects" / "ACTUAL_SEQUENCE_OBJECTS.jsonl",
            "label_actual": snap / "objects" / "ACTUAL_LABEL_OBJECTS.jsonl",
        }

    def output_binding(self, shard: str, paths: dict[str, Path]) -> str:
        root = paths["root"]
        components: list[dict[str, Any]] = []
        for logical, filename in sorted(CANONICAL_FILES.items()):
            if logical == "REPORTER_ARTIFACT_ASSESSMENTS":
                continue
            p = root / "canonical" / filename
            if p.exists():
                components.append({"logical_id": logical, "relative_path": p.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha_file(p)})
        for logical, filename in (("DATASET_RECONCILIATION", "dataset_reconciliation.json"), ("DATA_UNITS_REPORT", "data_units_report.json")):
            p = root / "canonical" / filename
            if p.exists():
                components.append({"logical_id": logical, "relative_path": p.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha_file(p)})
        components.sort(key=lambda x: x["logical_id"])
        components.append({"logical_id": "OBJECT_SEQUENCE_SET", "relative_path": paths["seq"].resolve().relative_to(root.resolve()).as_posix(), "sha256": sha_file(paths["seq"])})
        components.append({"logical_id": "OBJECT_LABEL_SET", "relative_path": paths["label"].resolve().relative_to(root.resolve()).as_posix(), "sha256": sha_file(paths["label"])})
        return sha_json({"run_id": self.summary.get("run_id"), "snapshot_id": self.summary.get("d1_snapshot_id"), "shard": shard, "components": components})

    def verify_object_set(self, shard: str, path: Path, set_kind: str) -> None:
        artifact = f"{shard}:{path.name}"
        if not self.require(path, artifact):
            return
        count = 0
        with path.open("r", encoding="utf-8", newline="") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
                except Exception:
                    self.errors.add("OBJECT_SET_INVALID_JSON", artifact, line_no)
                    continue
                if line != jline(row).decode("utf-8"):
                    self.errors.add("NON_CANONICAL_OBJECT_SET_BYTES", artifact, line_no)
                if set_kind == "SEQUENCE" and row.get("object_type") != "SEQUENCE":
                    self.errors.add("OBJECT_SET_TYPE_MISMATCH", artifact, line_no)
                if set_kind == "LABEL" and row.get("object_type") == "SEQUENCE":
                    self.errors.add("OBJECT_SET_TYPE_MISMATCH", artifact, line_no)
                self.store.insert("object_set_rows", {"shard": shard, "set_kind": set_kind, "object_id": row.get("object_id"), "object_type": row.get("object_type"), "object_sha256": row.get("canonical_object_sha256")}, (as_text(row.get("object_id")), f"{shard}:{set_kind}"))
        self.counters[f"rows:{artifact}"] = count
        if set_kind == "SEQUENCE":
            missing = self.scalar("SELECT COUNT(*) FROM objects o LEFT JOIN object_set_rows s ON s.object_id=o.object_id AND s.shard=o.shard AND s.set_kind=? WHERE o.shard=? AND o.object_type='SEQUENCE' AND s.object_id IS NULL", (set_kind, shard))
            orphan = self.scalar("SELECT COUNT(*) FROM object_set_rows s LEFT JOIN objects o ON o.object_id=s.object_id AND o.shard=s.shard WHERE s.shard=? AND s.set_kind=? AND (o.object_id IS NULL OR o.object_type!='SEQUENCE')", (shard, set_kind))
            bad_hash = self.scalar("SELECT COUNT(*) FROM object_set_rows s JOIN objects o ON o.object_id=s.object_id AND o.shard=s.shard WHERE s.shard=? AND s.set_kind=? AND s.object_sha256!=o.row_sha256", (shard, set_kind))
        else:
            missing = self.scalar("SELECT COUNT(*) FROM objects o LEFT JOIN object_set_rows s ON s.object_id=o.object_id AND s.shard=o.shard AND s.set_kind=? WHERE o.shard=? AND o.object_type IN ('OBSERVATION_CANDIDATE','OBSERVATION','RELATION_CANDIDATE','PAIR') AND s.object_id IS NULL", (set_kind, shard))
            orphan = self.scalar("SELECT COUNT(*) FROM object_set_rows s LEFT JOIN objects o ON o.object_id=s.object_id AND o.shard=s.shard WHERE s.shard=? AND s.set_kind=? AND (o.object_id IS NULL OR o.object_type='SEQUENCE')", (shard, set_kind))
            bad_hash = self.scalar("SELECT COUNT(*) FROM object_set_rows s JOIN objects o ON o.object_id=s.object_id AND o.shard=s.shard WHERE s.shard=? AND s.set_kind=? AND s.object_sha256!=o.row_sha256", (shard, set_kind))
        for code, value in [("OBJECT_SET_MISSING_OBJECT", missing), ("OBJECT_SET_ORPHAN_OBJECT", orphan), ("OBJECT_SET_HASH_MISMATCH", bad_hash)]:
            if value:
                self.errors.add(code, artifact, value)

    def access_events(self, shard: str, paths: dict[str, Path]) -> list[dict[str, Any]]:
        artifact = f"{shard}:ACCESS_LOG"
        if not self.require(paths["log"], artifact):
            return []
        if paths["live"].exists() and paths["live"].read_bytes() != paths["log"].read_bytes():
            self.errors.add("ACCESS_LIVE_PREFIX_MISMATCH", artifact)
        events: list[dict[str, Any]] = []
        with paths["log"].open("r", encoding="utf-8", newline="") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
                except Exception:
                    self.errors.add("INVALID_ACCESS_JSON", artifact, line_no)
                    continue
                if line != jline(row).decode("utf-8"):
                    self.errors.add("NON_CANONICAL_ACCESS_BYTES", artifact, line_no)
                definition = ACCESS_EVENT_DEF.get(row.get("status"))
                if definition is None:
                    self.errors.add("ACCESS_STATUS_UNKNOWN", artifact, line_no)
                else:
                    self.schema.validate(row, "exposure_record.schema.json", definition, artifact, line_no)
                if shard == "ordinary" and contains_marker(row):
                    self.errors.add("SEALED_MARKER_LEAK", artifact, line_no)
                if row.get("event_sha256") != sha_json(row, "event_sha256"):
                    self.errors.add("ACCESS_EVENT_HASH_MISMATCH", artifact, line_no)
                events.append(row)
        if len(events) != 2:
            self.errors.add("ACCESS_EVENT_COUNT_MISMATCH", artifact)
            return events
        for index, row in enumerate(events):
            expected_status = "INTENT" if index == 0 else "COMPLETION"
            if row.get("log_sequence_no") != index or row.get("status") != expected_status:
                self.errors.add("ACCESS_SEQUENCE_MISMATCH", artifact)
            expected_prev = GENESIS if index == 0 else events[index - 1].get("event_sha256")
            if row.get("prev_event_sha256") != expected_prev or row.get("predecessor_event_sha256") != expected_prev:
                self.errors.add("ACCESS_PREDECESSOR_HASH_MISMATCH", artifact)
            expected_pred_id = GENESIS if index == 0 else events[index - 1].get("event_id")
            if row.get("predecessor_event_id") != expected_pred_id:
                self.errors.add("ACCESS_PREDECESSOR_ID_MISMATCH", artifact)
            if row.get("analytic_access") is not False:
                self.errors.add("ACCESS_ANALYTIC_FLAG_NONZERO", artifact)
            if row.get("event_type") != "RESTRICTED_BUILDER_PARSE":
                self.errors.add("ACCESS_EVENT_TYPE_NOT_D1_MACHINE", artifact)
        completion = events[-1]
        if completion.get("intent_event_id") != events[0].get("event_id") or completion.get("partial_actual_set_status") != "COMPLETE":
            self.errors.add("ACCESS_TERMINAL_CLOSURE_MISMATCH", artifact)
        if completion.get("sequence_rows_touched") != self.store.count("sequences", shard):
            self.errors.add("ACCESS_SEQUENCE_COUNTER_MISMATCH", artifact)
        if completion.get("label_rows_touched") != sum(self.store.count(t, shard) for t in ("obs_candidates", "observations", "relations", "pairs")):
            self.errors.add("ACCESS_LABEL_COUNTER_MISMATCH", artifact)
        self.counters[f"access_events:{shard}"] = len(events)
        return events

    def verify_access_bundle(self, shard: str) -> dict[str, Any]:
        paths = self.access_paths(shard)
        events = self.access_events(shard, paths)
        if not events:
            return {"paths": paths, "events": events, "manifest": {}}
        self.verify_object_set(shard, paths["seq"], "SEQUENCE")
        self.verify_object_set(shard, paths["label"], "LABEL")
        if paths["seq"].read_bytes() != paths["seq_actual"].read_bytes():
            self.errors.add("ACCESS_REQUESTED_ACTUAL_SEQUENCE_DRIFT", shard)
        if paths["label"].read_bytes() != paths["label_actual"].read_bytes():
            self.errors.add("ACCESS_REQUESTED_ACTUAL_LABEL_DRIFT", shard)
        for key in ("seq_actual", "label_actual"):
            if not self.require(paths[key], f"{shard}:{key}"):
                continue
        intent, completion = events[0], events[-1]
        if intent.get("requested_sequence_object_set_manifest_sha256") != sha_file(paths["seq"]):
            self.errors.add("ACCESS_REQUESTED_SEQUENCE_HASH_MISMATCH", shard)
        if intent.get("requested_label_object_set_manifest_sha256") != sha_file(paths["label"]):
            self.errors.add("ACCESS_REQUESTED_LABEL_HASH_MISMATCH", shard)
        if completion.get("actual_sequence_object_set_manifest_sha256") != sha_file(paths["seq_actual"]):
            self.errors.add("ACCESS_ACTUAL_SEQUENCE_HASH_MISMATCH", shard)
        if completion.get("actual_label_object_set_manifest_sha256") != sha_file(paths["label_actual"]):
            self.errors.add("ACCESS_ACTUAL_LABEL_HASH_MISMATCH", shard)
        input_manifest = self.run_root / "work" / "D1_INPUT_MANIFEST.json" if shard == "ordinary" else self.restricted_root / "SEALED_INPUT_MANIFEST.json"
        if completion.get("input_manifest_sha256") != sha_file(input_manifest):
            self.errors.add("ACCESS_INPUT_MANIFEST_HASH_MISMATCH", shard)
        if completion.get("output_manifest_sha256") != self.output_binding(shard, paths):
            self.errors.add("ACCESS_OUTPUT_BINDING_MISMATCH", shard)
        builder = Path(self.args.builder_path)
        expected_executable = sha_file(builder) if builder.exists() else None
        expected_environment = sha_text(f"python={sys.version}|platform={sys.platform}|code_commit={self.summary.get('code_commit')}")
        for event in events:
            if expected_executable and event.get("executable_sha256") != expected_executable:
                self.errors.add("ACCESS_EXECUTABLE_HASH_MISMATCH", shard)
            if event.get("container_or_environment_sha256") != expected_environment:
                self.errors.add("ACCESS_ENVIRONMENT_HASH_MISMATCH", shard)
        access_manifest = self.verify_self_json(paths["manifest"], "manifest_sha256", f"{shard}:ACCESS_MANIFEST", shard == "ordinary")
        root = paths["root"]
        if access_manifest:
            expected_rel = lambda p: p.resolve().relative_to(root.resolve()).as_posix()
            fields = {
                "snapshot_access_log_relpath": expected_rel(paths["log"]),
                "live_access_log_relpath": expected_rel(paths["live"]),
                "snapshot_access_log_sha256": sha_file(paths["log"]),
                "access_sha256s_relpath": expected_rel(paths["sums"]),
                "access_sha256s_sha256": sha_file(paths["sums"]),
                "event_count": len(events),
                "first_event_id": events[0].get("event_id"),
                "last_event_id": events[-1].get("event_id"),
                "access_log_chain_root_sha256": events[-1].get("event_sha256"),
                "live_prefix_match_at_snapshot": paths["live"].read_bytes() == paths["log"].read_bytes(),
            }
            for field, value in fields.items():
                self.compare(access_manifest.get(field), value, f"ACCESS_MANIFEST_{field.upper()}_MISMATCH", shard)
            if access_manifest.get("access_log_schema_sha256") != self.schema.hashes.get("exposure_record.schema.json"):
                self.errors.add("ACCESS_SCHEMA_HASH_MISMATCH", shard)
            if access_manifest.get("cohort_set_sha256") != (SEALED_COHORT_SET_SHA256 if shard == "restricted" else set_sha(["ORDINARY_NONSEALED"])):
                self.errors.add("ACCESS_COHORT_SET_HASH_MISMATCH", shard)
        expected_ledger_paths = [paths["log"], paths["seq"], paths["label"], paths["seq_actual"], paths["label_actual"]]
        self.verify_ledger(paths["sums"], root, expected_ledger_paths, f"{shard}:ACCESS_SHA256SUMS")
        self.counters[f"access_chain_root:{shard}"] = events[-1].get("event_sha256", "")
        return {"paths": paths, "events": events, "manifest": access_manifest}

    def read_effective(self, shard: str, access: dict[str, Any]) -> Path:
        snapshot = as_text(self.summary.get("d1_snapshot_id"))
        path = self._root_for(shard) / "exposure" / "projections" / snapshot / "EFFECTIVE_EXPOSURE_PROJECTION.jsonl"
        artifact = f"{shard}:EFFECTIVE_EXPOSURE_PROJECTION"
        chain_root = access.get("events", [{}])[-1].get("event_sha256") if access.get("events") else None
        count = 0
        if self.require(path, artifact):
            with path.open("r", encoding="utf-8", newline="") as fh:
                for line_no, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    count += 1
                    try:
                        row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite:{x}")))
                    except Exception:
                        self.errors.add("INVALID_JSON_OR_NONFINITE", artifact, line_no)
                        continue
                    if line != jline(row).decode("utf-8"):
                        self.errors.add("NON_CANONICAL_JSONL_BYTES", artifact, line_no)
                    self.schema.validate(row, "exposure_record.schema.json", "EffectiveExposureProjection", artifact, line_no)
                    if row.get("projection_sha256") != sha_json(row, "projection_sha256"):
                        self.errors.add("PROJECTION_HASH_MISMATCH", artifact, line_no)
                    for field, value in [("access_log_chain_root_sha256", chain_root), ("chain_root_sha256", chain_root), ("snapshot_id", self.summary.get("d1_snapshot_id")), ("projection_phase", "D1"), ("final_access_status", "SEALED_UNOPENED")]:
                        self.compare(row.get(field), value, f"EFFECTIVE_{field.upper()}_MISMATCH", shard)
                    obj = self.store.one("objects", "object_id=? AND shard=?", (row.get("object_id"), shard))
                    exp = self.store.one("exposures", "object_id=? AND shard=?", (row.get("object_id"), shard))
                    if obj is None or exp is None:
                        self.errors.add("EFFECTIVE_OBJECT_FK_MISMATCH", shard)
                    else:
                        self.compare(row.get("object_type"), obj["object_type"], "EFFECTIVE_OBJECT_TYPE_MISMATCH", shard)
                        self.compare(row.get("baseline_record_sha256"), exp["record_sha256"], "EFFECTIVE_BASELINE_HASH_MISMATCH", shard)
                        self.compare(row.get("baseline_exposure_record_id"), f"EXP:{safe_id(shard + '|' + row.get('object_id'))}", "EFFECTIVE_BASELINE_ID_MISMATCH", shard)
            self.counters[f"rows:{artifact}"] = count
        if count != self.store.count("objects", shard):
            self.errors.add("EFFECTIVE_OBJECT_COUNT_MISMATCH", shard)
        return path

    def verify_exposure_use(self, shard: str, effective_path: Path, access: dict[str, Any]) -> dict[str, Any]:
        root = self._root_for(shard)
        canonical = root / "canonical"
        manifest_path = canonical / "EXPOSURE_USE_MANIFEST.json"
        sums_path = canonical / "EXPOSURE_USE_SHA256SUMS"
        manifest = self.verify_self_json(manifest_path, "manifest_sha256", f"{shard}:EXPOSURE_USE_MANIFEST", shard == "ordinary")
        expected_paths = [canonical / "EXPOSURE_RECORDS.jsonl", canonical / "USE_ROLES.jsonl", effective_path]
        self.verify_ledger(sums_path, root, expected_paths, f"{shard}:EXPOSURE_USE_SHA256SUMS")
        if manifest:
            expected_rel = lambda p: p.resolve().relative_to(root.resolve()).as_posix()
            for field, value in {
                "exposure_records_relpath": expected_rel(expected_paths[0]), "exposure_records_sha256": sha_file(expected_paths[0]),
                "use_roles_relpath": expected_rel(expected_paths[1]), "use_roles_sha256": sha_file(expected_paths[1]),
                "effective_exposure_projection_relpath": expected_rel(effective_path), "effective_exposure_projection_sha256": sha_file(effective_path),
                "access_snapshot_id": self.summary.get("d1_snapshot_id"), "access_log_chain_root_sha256": access.get("events", [{}])[-1].get("event_sha256"),
                "access_manifest_sha256": sha_file(access["paths"]["manifest"]), "access_sha256s_sha256": sha_file(access["paths"]["sums"]),
                "exposure_use_sha256s_sha256": sha_file(sums_path), "canonical_binding_sha256": self.summary.get("canonical_binding_sha256"),
            }.items():
                self.compare(manifest.get(field), value, f"EXPOSURE_USE_{field.upper()}_MISMATCH", shard)
        return manifest

    def verify_reports(self, shard: str, access: dict[str, Any], effective_path: Path) -> None:
        root = self._root_for(shard)
        canonical = root / "canonical"
        dataset = self.verify_self_json(canonical / "dataset_reconciliation.json", "manifest_sha256", f"{shard}:dataset_reconciliation", shard == "ordinary")
        units = self.verify_self_json(canonical / "data_units_report.json", "report_sha256", f"{shard}:data_units_report", shard == "ordinary")
        for artifact, value in [(f"{shard}:dataset_reconciliation", dataset), (f"{shard}:data_units_report", units)]:
            if value:
                for field, expected in [("contract_id", CONTRACT_ID), ("contract_sha256", self.args.authority_contract_sha256), ("run_id", self.summary.get("run_id")), ("d1_claim_boundary", CLAIM_BOUNDARY), ("g3b_status", G3B_STATUS)]:
                    if field in value:
                        self.compare(value.get(field), expected, f"REPORT_{field.upper()}_MISMATCH", artifact)
        stats = self.raw_stats[shard]
        expected_dataset = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("dataset:")}
        expected_record_types = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("record_type:")}
        expected_regions = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("region:")}
        expected_labels = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("label:")}
        if dataset:
            if dataset.get("dataset_counts") != expected_dataset:
                self.errors.add("REPORT_DATASET_COUNT_MISMATCH", shard)
            if dataset.get("record_type_counts") != expected_record_types:
                self.errors.add("REPORT_RECORD_TYPE_COUNT_MISMATCH", shard)
            if dataset.get("region_counts") != expected_regions:
                self.errors.add("REPORT_REGION_COUNT_MISMATCH", shard)
            if dataset.get("label_key_counts") != expected_labels:
                self.errors.add("REPORT_LABEL_COUNT_MISMATCH", shard)
            if shard == "ordinary":
                for field, expected in [("ordinary_only", True), ("row_level_member_ids_emitted", False), ("row_level_source_locators_emitted", False), ("sealed_member_level_rows_in_ordinary", 0), ("sealed_cohort_aggregate_present", False), ("legacy_quarantine_record_count", stats.get("legacy_records", 0))]:
                    self.compare(dataset.get(field), expected, f"ORDINARY_REPORT_{field.upper()}_MISMATCH", shard)
            else:
                for field, expected in [("restricted_only", True), ("row_level_member_ids_emitted", True), ("row_level_source_locators_emitted", True), ("cohort_id", SEALED_COHORT), ("restricted_record_count", stats.get("raw_records", 0))]:
                    self.compare(dataset.get(field), expected, f"RESTRICTED_REPORT_{field.upper()}_MISMATCH", shard)
        if units:
            if units.get("row_level_member_ids_emitted") != (shard == "restricted") or units.get("row_level_source_locators_emitted") != (shard == "restricted"):
                self.errors.add("DATA_UNITS_ACCESS_SCOPE_MISMATCH", shard)
            if units.get("d1_claim_boundary") != CLAIM_BOUNDARY:
                self.errors.add("DATA_UNITS_CLAIM_BOUNDARY_MISMATCH", shard)
        exposure_manifest = self.verify_exposure_use(shard, effective_path, access)
        if exposure_manifest and contains_marker(exposure_manifest) and shard == "ordinary":
            self.errors.add("SEALED_MARKER_LEAK", f"{shard}:EXPOSURE_USE_MANIFEST")

        # Aggregate endpoint rows must be self-hashed and scoped to the shard.
        endpoint_path = canonical / "ENDPOINT_REGISTRY.jsonl"
        if endpoint_path.exists():
            for line_no, line in enumerate(endpoint_path.open("r", encoding="utf-8"), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_sha256") != sha_json(row, "record_sha256"):
                    self.errors.add("ENDPOINT_RECORD_HASH_MISMATCH", shard, line_no)
                observed = self.scalar("SELECT COUNT(*) FROM observations WHERE shard=? AND endpoint_id=?", (shard, row.get("endpoint_id")))
                relation_observed = self.scalar("SELECT COUNT(*) FROM relations WHERE shard=? AND endpoint_id=?", (shard, row.get("endpoint_id")))
                if observed + relation_observed == 0 and row.get("endpoint_id") != "EP:NO_MEASUREMENT":
                    self.errors.add("ORPHAN_ENDPOINT_REGISTRY_ROW", shard, line_no)

    def verify_canonical_manifest(self, shard: str, access: dict[str, Any], effective_path: Path) -> dict[str, Any]:
        if shard != "ordinary":
            return {}
        root = self.ordinary_root
        canonical = root / "canonical"
        manifest_path = canonical / "CANONICAL_MANIFEST.json"
        sums_path = canonical / "CANONICAL_SHA256SUMS"
        manifest = self.verify_self_json(manifest_path, "manifest_sha256", "ordinary:CANONICAL_MANIFEST", True)
        paths: list[Path] = []
        for filename in CANONICAL_FILES.values():
            p = canonical / filename
            if p.exists():
                paths.append(p)
        paths.extend([canonical / "EXPOSURE_USE_MANIFEST.json", canonical / "EXPOSURE_USE_SHA256SUMS", canonical / "dataset_reconciliation.json", canonical / "data_units_report.json", effective_path, access["paths"]["log"], access["paths"]["manifest"], access["paths"]["sums"]])
        self.verify_ledger(sums_path, self.run_root, paths, "ordinary:CANONICAL_SHA256SUMS")
        if not manifest:
            return {}
        entries: list[dict[str, Any]] = []
        for p in paths:
            if p == effective_path:
                logical = "EFFECTIVE_EXPOSURE_PROJECTION"
            elif p == access["paths"]["log"]:
                logical = "ACCESS_LOG"
            elif p == access["paths"]["manifest"]:
                logical = "ACCESS_MANIFEST"
            elif p == access["paths"]["sums"]:
                logical = "ACCESS_SHA256SUMS"
            elif p.name == "dataset_reconciliation.json":
                logical = "DATASET_RECONCILIATION"
            elif p.name == "data_units_report.json":
                logical = "DATA_UNITS_REPORT"
            elif p.name == "EXPOSURE_USE_MANIFEST.json":
                logical = "EXPOSURE_USE_MANIFEST"
            elif p.name == "EXPOSURE_USE_SHA256SUMS":
                logical = "EXPOSURE_USE_SHA256SUMS"
            else:
                logical = p.name
            entries.append(self.component_entry(logical, p, root))
        entries.sort(key=lambda x: x["relative_path"].encode("utf-8"))
        if manifest.get("components") != entries or manifest.get("component_count") != len(entries):
            self.errors.add("CANONICAL_COMPONENT_SET_MISMATCH", "ordinary")
        for field, expected in {
            "contract_id": CONTRACT_ID, "contract_sha256": self.args.authority_contract_sha256, "run_id": self.summary.get("run_id"),
            "d1_snapshot_id": self.summary.get("d1_snapshot_id"), "canonical_binding_sha256": self.summary.get("canonical_binding_sha256"),
            "canonical_sha256s_relpath": sums_path.resolve().relative_to(root.resolve()).as_posix(), "canonical_sha256s_sha256": sha_file(sums_path),
            "exposure_use_manifest_sha256": sha_file(canonical / "EXPOSURE_USE_MANIFEST.json"), "exposure_use_sha256s_sha256": sha_file(canonical / "EXPOSURE_USE_SHA256SUMS"),
            "effective_exposure_projection_sha256": sha_file(effective_path), "access_manifest_sha256": sha_file(access["paths"]["manifest"]),
            "access_sha256s_sha256": sha_file(access["paths"]["sums"]), "access_log_chain_root_sha256": access["events"][-1].get("event_sha256"),
            "d1_claim_boundary": CLAIM_BOUNDARY, "g3b_status": G3B_STATUS,
        }.items():
            self.compare(manifest.get(field), expected, f"CANONICAL_MANIFEST_{field.upper()}_MISMATCH", "ordinary")
        return manifest

    def verify_matrix_profile(self, path: Path, expected: dict[str, Any], artifact: str) -> None:
        if not path.exists():
            self.errors.add("SEALED_INPUT_FILE_MISSING", artifact)
            return
        actual_sha = sha_file(path)
        if actual_sha != expected.get("sha256"):
            self.errors.add("SEALED_INPUT_FILE_HASH_MISMATCH", artifact)
        if path.stat().st_size != expected.get("byte_size"):
            self.errors.add("SEALED_INPUT_FILE_SIZE_MISMATCH", artifact)
        if not path.name.endswith(".csv.gz"):
            return
        rows = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
                header_line = fh.readline()
                if not header_line:
                    self.errors.add("SEALED_MATRIX_EMPTY", artifact)
                    return
                reader = csv.reader(fh)
                header = next(reader)
                header_hash = sha_bytes(jline(header))
                for _ in reader:
                    rows += 1
                if header_hash != expected.get("header_sha256") or len(header) != expected.get("columns") or rows != expected.get("rows"):
                    self.errors.add("SEALED_MATRIX_PROFILE_MISMATCH", artifact)
        except Exception:
            self.errors.add("SEALED_MATRIX_PROFILE_READ_ERROR", artifact)

    def verify_sealed_manifest_and_commitments(self, access: dict[str, Any], effective_path: Path) -> dict[str, Any]:
        root = self.restricted_root
        sealed_input_path = root / "SEALED_INPUT_MANIFEST.json"
        sealed_input = self.verify_self_json(sealed_input_path, "manifest_sha256", "restricted:SEALED_INPUT_MANIFEST")
        if sealed_input:
            if sealed_input.get("cohort_ids") != [SEALED_COHORT] or sealed_input.get("cohort_set_sha256") != SEALED_COHORT_SET_SHA256:
                self.errors.add("SEALED_INPUT_COHORT_BINDING_MISMATCH")
            if sealed_input.get("input_file_set_sha256") != set_sha([f"{x.get('relative_path')}|{x.get('sha256')}" for x in sealed_input.get("input_files", [])]):
                self.errors.add("SEALED_INPUT_FILE_SET_HASH_MISMATCH")
            profiles = {as_text(profile.get("path")): profile for profile in sealed_input.get("matrix_profiles", [])}
            for item in sealed_input.get("input_files", []):
                expected_profile = dict(item)
                expected_profile.update(profiles.get(as_text(item.get("relative_path")), {}))
                self.verify_matrix_profile(Path(self.args.sealed_input).parent / item.get("relative_path", ""), expected_profile, f"restricted:input:{item.get('relative_path')}")
            if sealed_input.get("reconstructed_record_count") != self.raw_stats["restricted"].get("raw_records", 0):
                self.errors.add("SEALED_INPUT_RECORD_COUNT_MISMATCH")

        sealed_sums = root / "SEALED_CANONICAL_SHA256SUMS"
        sealed_payload = [sealed_input_path]
        for filename in [*CANONICAL_FILES.values(), "dataset_reconciliation.json", "data_units_report.json"]:
            p = root / "canonical" / filename
            if filename == "reporter_artifact_assessments.jsonl":
                continue
            if p.exists():
                sealed_payload.append(p)
        sealed_payload.append(effective_path)
        self.verify_ledger(sealed_sums, root, sealed_payload, "restricted:SEALED_CANONICAL_SHA256SUMS")

        snapshot = self.summary.get("d1_snapshot_id")
        lp: dict[str, Path] = {
            "ACCESS_LOG": access["paths"]["log"], "ACCESS_MANIFEST": access["paths"]["manifest"], "ACCESS_SHA256SUMS": access["paths"]["sums"],
            "CURRENT_CANONICAL_OBJECT_PROJECTION": root / "canonical/CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl",
            "DATASET_RECONCILIATION": root / "canonical/dataset_reconciliation.json", "DATA_UNITS_REPORT": root / "canonical/data_units_report.json",
            "EFFECTIVE_EXPOSURE_PROJECTION": effective_path, "ENDPOINT_REGISTRY": root / "canonical/ENDPOINT_REGISTRY.jsonl",
            "EXPOSURE_RECORDS": root / "canonical/EXPOSURE_RECORDS.jsonl", "EXPOSURE_USE_MANIFEST": root / "canonical/EXPOSURE_USE_MANIFEST.json",
            "EXPOSURE_USE_SHA256SUMS": root / "canonical/EXPOSURE_USE_SHA256SUMS", "FUNCTIONAL_OBSERVATION_CANDIDATES": root / "canonical/functional_observation_candidates.jsonl",
            "FUNCTIONAL_OBSERVATIONS": root / "canonical/functional_observations.jsonl", "GROUP_ASSIGNMENTS": root / "canonical/group_assignments.jsonl",
            "GROUP_REGISTRY": root / "canonical/group_registry.jsonl", "REJECTIONS": root / "canonical/rejections.jsonl", "SEALED_CANONICAL_SHA256SUMS": sealed_sums,
            "SEALED_INPUT_MANIFEST": sealed_input_path, "SEQUENCE_ENTITIES": root / "canonical/sequence_entities.jsonl", "SUPERSESSION_EDGES": root / "canonical/SUPERSESSION_EDGES.jsonl",
            "TRANSFORMATION_EDGES": root / "canonical/transformation_edges.jsonl", "USE_ROLES": root / "canonical/USE_ROLES.jsonl", "UTR_EDIT_PAIRS": root / "canonical/utr_edit_pairs.jsonl",
            "UTR_EDIT_RELATION_CANDIDATES": root / "canonical/utr_edit_relation_candidates.jsonl",
        }
        if set(lp) != RESTRICTED_LOGICAL_IDS:
            self.errors.add("RESTRICTED_LOGICAL_COMPONENT_SET_MISMATCH")
        if set_sha(lp.keys()) != SEALED_COMPONENT_SET_SHA256:
            self.errors.add("RESTRICTED_LOGICAL_COMPONENT_SET_HASH_MISMATCH")
        sealed_manifest = self.verify_self_json(root / "SEALED_CANONICAL_MANIFEST.json", "manifest_sha256", "restricted:SEALED_CANONICAL_MANIFEST")
        if sealed_manifest:
            entries = [self.component_entry(logical, path, root) for logical, path in sorted(lp.items())]
            if sealed_manifest.get("logical_components") != entries:
                self.errors.add("SEALED_LOGICAL_COMPONENT_BINDING_MISMATCH")
            for field, expected in {
                "contract_id": CONTRACT_ID, "contract_sha256": self.args.authority_contract_sha256, "run_id": self.summary.get("run_id"),
                "d1_snapshot_id": self.summary.get("d1_snapshot_id"), "access_prefix_snapshot_id": self.summary.get("d1_snapshot_id"),
                "cohort_ids": [SEALED_COHORT], "cohort_set_sha256": SEALED_COHORT_SET_SHA256, "logical_component_set_sha256": SEALED_COMPONENT_SET_SHA256,
                "sealed_canonical_sha256s_sha256": sha_file(sealed_sums), "access_manifest_sha256": sha_file(access["paths"]["manifest"]),
                "access_sha256s_sha256": sha_file(access["paths"]["sums"]), "access_log_chain_root_sha256": access["events"][-1].get("event_sha256"),
                "exposure_use_manifest_sha256": sha_file(root / "canonical/EXPOSURE_USE_MANIFEST.json"), "effective_exposure_projection_sha256": sha_file(effective_path),
            }.items():
                self.compare(sealed_manifest.get(field), expected, f"SEALED_MANIFEST_{field.upper()}_MISMATCH", "restricted")

        qc_path = self.ordinary_root / "sealed_commitments" / "GSE246381_AGGREGATE_QC.json"
        commitment_path = self.ordinary_root / "sealed_commitments" / "GSE246381_COMMITMENT.json"
        qc = self.verify_self_json(qc_path, "aggregate_qc_sha256", "ordinary:GSE246381_AGGREGATE_QC")
        commitment = self.verify_self_json(commitment_path, "commitment_sha256", "ordinary:GSE246381_COMMITMENT")
        if qc:
            for field, expected in [("cohort_id", SEALED_COHORT), ("contract_id", CONTRACT_ID), ("contract_sha256", self.args.authority_contract_sha256), ("run_id", self.summary.get("run_id")), ("d1_snapshot_id", self.summary.get("d1_snapshot_id")), ("reconstructed_record_count", self.raw_stats["restricted"].get("raw_records", 0)), ("accepted_pair_count", self.store.count("pairs", "restricted")), ("machine_access_only", True), ("row_level_values_emitted_to_ordinary", False), ("qualification_status", "PENDING_FM0_A")]:
                self.compare(qc.get(field), expected, f"QC_{field.upper()}_MISMATCH", "ordinary")
            expected_full = self.raw_stats["restricted"].get("raw_records", 0) == self.args.expected_restricted_records and all(p.get("rows") == self.args.expected_matrix_rows for p in self.summary.get("restricted_matrix_profiles", []))
            if qc.get("conservation_status") != ("PASS" if expected_full else "PENDING_REVIEW"):
                self.errors.add("QC_CONSERVATION_STATUS_MISMATCH", "ordinary")
        if commitment:
            for field, expected in [("cohort_id", SEALED_COHORT), ("contract_id", CONTRACT_ID), ("contract_sha256", self.args.authority_contract_sha256), ("run_id", self.summary.get("run_id")), ("d1_snapshot_id", self.summary.get("d1_snapshot_id")), ("sealed_canonical_manifest_sha256", sha_file(root / "SEALED_CANONICAL_MANIFEST.json")), ("sealed_canonical_sha256s_sha256", sha_file(sealed_sums)), ("access_manifest_sha256", sha_file(access["paths"]["manifest"])), ("access_sha256s_sha256", sha_file(access["paths"]["sums"])), ("access_log_chain_root_sha256", access["events"][-1].get("event_sha256")), ("ordinary_member_level_row_count", 0), ("ordinary_loader_reachable_member_count", 0), ("member_level_rows_in_commitment", False), ("prior_analytic_use", "NONE_CONFIRMED"), ("pipeline_materialization", "PRESENT"), ("foundation_overlap_audit_status", "DEFERRED_TO_FM0_A"), ("d1_claim_boundary", CLAIM_BOUNDARY)]:
                self.compare(commitment.get(field), expected, f"COMMITMENT_{field.upper()}_MISMATCH", "ordinary")
        return sealed_manifest

    def scan_ordinary_namespace(self) -> None:
        for root in (self.ordinary_root / "canonical", self.ordinary_root / "exposure"):
            if not root.exists():
                self.errors.add("ORDINARY_NAMESPACE_MISSING", str(root.relative_to(self.run_root)))
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    try:
                        if SEALED_MARKER.encode("utf-8") in path.read_bytes().lower():
                            self.errors.add("SEALED_MARKER_LEAK", path.relative_to(self.run_root).as_posix())
                    except Exception:
                        self.errors.add("ORDINARY_NAMESPACE_READ_ERROR", path.relative_to(self.run_root).as_posix())

    def verify_common_summary_bindings(self) -> None:
        if self.args.expected_source_head and self.summary.get("source_head") != self.args.expected_source_head:
            self.errors.add("SOURCE_HEAD_MISMATCH", "D1_BUILD_SUMMARY.json")
        if self.args.expected_code_commit and self.summary.get("code_commit") != self.args.expected_code_commit:
            self.errors.add("CODE_COMMIT_MISMATCH", "D1_BUILD_SUMMARY.json")
        if self.summary.get("run_id") != self.input_manifest.get("run_id") or self.summary.get("d1_snapshot_id") != self.input_manifest.get("d1_snapshot_id"):
            self.errors.add("RUN_SNAPSHOT_BINDING_MISMATCH")

    def run(self) -> tuple[dict[str, Any], int]:
        self.load_metadata()
        self.verify_common_summary_bindings()
        self.verify_input_manifest()
        self.load_shard("ordinary", self.ordinary_root)
        self.load_shard("restricted", self.restricted_root)
        self.store.conn.commit()
        self.verify_relational_all()
        self.validate_raw_inputs()
        self.verify_raw_counts()
        ordinary_access = self.verify_access_bundle("ordinary")
        restricted_access = self.verify_access_bundle("restricted")
        ordinary_effective = self.read_effective("ordinary", ordinary_access)
        restricted_effective = self.read_effective("restricted", restricted_access)
        self.verify_reports("ordinary", ordinary_access, ordinary_effective)
        self.verify_reports("restricted", restricted_access, restricted_effective)
        self.verify_canonical_manifest("ordinary", ordinary_access, ordinary_effective)
        self.verify_sealed_manifest_and_commitments(restricted_access, restricted_effective)
        self.scan_ordinary_namespace()
        status = "PASS" if not self.errors.counts else "BLOCKED_WITH_EVIDENCE"
        validation_base = {
            "artifact_kind": "D1_STRICT_VALIDATION",
            "phase": "D1-R",
            "status": status,
            "d1_acceptance_asserted": status == "PASS",
            "full_acceptance_asserted": status == "PASS",
            "next_phase_unlocked": status == "PASS",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.args.authority_contract_sha256,
            "run_id": self.summary.get("run_id"),
            "d1_snapshot_id": self.summary.get("d1_snapshot_id"),
            "source_head": self.summary.get("source_head"),
            "code_commit": self.summary.get("code_commit"),
            "claim_boundary": CLAIM_BOUNDARY,
            "g3b_status": G3B_STATUS,
            "raw_stats": {shard: dict(sorted(stats.items())) for shard, stats in self.raw_stats.items()},
            "counters": dict(sorted(self.counters.items())),
            "error_counts": dict(sorted(self.errors.counts.items())),
            "error_examples": self.errors.examples,
            "sealed_row_level_leak_status": "PASS" if not any(k.startswith("SEALED_MARKER_LEAK") for k in self.errors.counts) else "FAIL",
            "analytic_access_asserted_false": True,
            "training_started": False,
            "final_evaluator_accessed": False,
        }
        validation_path = self.run_root / "D1_VALIDATION.json"
        validation_base["validation_sha256"] = sha_json(validation_base)
        validation_path.write_bytes(jline(validation_base))
        status_base = {
            "artifact_kind": "D1_STATUS",
            "phase": "D1-R",
            "status": status,
            "d1_acceptance_asserted": status == "PASS",
            "full_acceptance_asserted": status == "PASS",
            "next_phase_unlocked": status == "PASS",
            "contract_id": CONTRACT_ID,
            "contract_sha256": self.args.authority_contract_sha256,
            "run_id": self.summary.get("run_id"),
            "d1_snapshot_id": self.summary.get("d1_snapshot_id"),
            "source_head": self.summary.get("source_head"),
            "code_commit": self.summary.get("code_commit"),
            "claim_boundary": CLAIM_BOUNDARY,
            "g3b_status": G3B_STATUS,
            "validation_artifact_sha256": sha_file(validation_path),
            "error_count": sum(self.errors.counts.values()),
            "training_started": False,
            "final_evaluator_accessed": False,
        }
        status_base["status_sha256"] = sha_json(status_base)
        (self.run_root / "D1_STATUS.json").write_bytes(jline(status_base))
        return validation_base, 0 if status == "PASS" else 1

    def verify_relational_all(self) -> None:
        for shard in ("ordinary", "restricted"):
            self.verify_exact_object_coverage(shard)
            self.verify_foreign_keys_and_bijections(shard)
            self.verify_groups(shard)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--d0-root", type=Path, required=True)
    parser.add_argument("--sealed-input", type=Path, required=True)
    parser.add_argument("--builder-path", type=Path, required=True)
    parser.add_argument("--authority-contract-sha256", required=True)
    parser.add_argument("--expected-source-head")
    parser.add_argument("--expected-code-commit")
    parser.add_argument("--expected-ordinary-records", type=int, default=3831570)
    parser.add_argument("--expected-restricted-records", type=int, default=1300)
    parser.add_argument("--expected-matrix-rows", type=int, default=32990)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = Validator(args)
    try:
        result, rc = validator.run()
        print(json.dumps({
            "status": result["status"],
            "error_count": sum(result.get("error_counts", {}).values()),
            "ordinary_raw_records": result["raw_stats"].get("ordinary", {}).get("raw_records", 0),
            "restricted_raw_records": result["raw_stats"].get("restricted", {}).get("raw_records", 0),
            "validation": str((args.run_root / "D1_VALIDATION.json").resolve()),
            "status_artifact": str((args.run_root / "D1_STATUS.json").resolve()),
        }, indent=2, sort_keys=True))
        return rc
    finally:
        validator.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
