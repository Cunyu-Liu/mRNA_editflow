"""P3-01: Local-Edit Benchmark record schema (frozen contract p3_benchmark_v1).

Each benchmark sample carries AT LEAST the 15 fields required by the P3-01
spec (prompt lines 879-901) plus provenance/governance fields:

    source_id, cargo_id, cell_context,
    source_sequence, candidate_sequence,
    edit_list, edit_count, edited_region,
    protein_identity,
    measured_or_proxy_source_value, measured_or_proxy_candidate_value,
    delta, data_source, assay_type, confidence

Anti-fabrication invariant (hard-checked by validate()):
    confidence == "unlabeled"  <=>  values/delta are ALL None.
    We never disguise data lacking local labels as local-delta ground truth.

Coordinate system
-----------------
All sequences use the RNA alphabet (A, C, G, U). ``edit_list`` positions are
0-based offsets WITHIN the stored ``source_sequence`` (the edited-region
scope), i.e. applying the edits to ``source_sequence`` must reproduce
``candidate_sequence`` exactly.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "p3_benchmark_v1"

# 15 prompt-required fields + record_id (stable identity) = 16 core fields.
REQUIRED_FIELDS: Tuple[str, ...] = (
    "record_id",
    "source_id",
    "cargo_id",
    "cell_context",
    "source_sequence",
    "candidate_sequence",
    "edit_list",
    "edit_count",
    "edited_region",
    "protein_identity",
    "measured_or_proxy_source_value",
    "measured_or_proxy_candidate_value",
    "delta",
    "data_source",
    "assay_type",
    "confidence",
)

# Governance / provenance extras (allowed: "at least" the required set).
EXTRA_FIELDS: Tuple[str, ...] = (
    "edit_type",
    "task_eligibility",
    "value_qualifier",
    "value_std",
    "family_cluster_id",
    "motif_flags",
    "internal_features",
    "split_role",
)

ALL_FIELDS: Tuple[str, ...] = REQUIRED_FIELDS + EXTRA_FIELDS

CONFIDENCE_LEVELS: Tuple[str, ...] = ("measured", "proxy", "unlabeled")

EDIT_REGIONS: Tuple[str, ...] = (
    "five_utr",
    "cds_first30",
    "cds_first50",
    "cds_remaining",
    "joint_5utr_cds",
)

EDIT_TYPES: Tuple[str, ...] = (
    "wild_type_anchor",          # measured tier only: edit_count == 0 anchor
    "measured_single",           # measured tier: hamming-1 variant
    "measured_double",           # measured tier: hamming-2 variant
    "measured_multi",            # measured tier: hamming 3..10 variant (within budget)
    "all_legal_single",          # generated: exhaustive legal single edit
    "random_double",             # generated: seeded random pair of legal singles
    "structure_guided_double",   # generated: pair picked by local-structure disruption
    "topranked_double",          # generated: pair of top-|delta| singles
    "matched_negative_single",   # generated: single with near-zero predicted |delta|
)

# Region -> owning application task in the frozen p3_task_v2 contract.
TASK_ELIGIBILITY: Dict[str, str] = {
    "five_utr": "task_a_active",
    "cds_first30": "task_b_frozen_fallback",
    "cds_first50": "task_b_frozen_fallback",
    "cds_remaining": "task_b_frozen_fallback",
    "joint_5utr_cds": "task_c_locked_extension",
}

SPLIT_ROLES: Tuple[str, ...] = ("train", "val", "test", "ood")

MAX_EDIT_BUDGET = 10  # p3_task_v2 edit_budgets {1, 3, 5, 10}


class BenchmarkSchemaError(ValueError):
    """Raised when a benchmark record violates the frozen schema."""


def stable_record_id(payload: Dict[str, Any]) -> str:
    """Deterministic content-addressed record id (sha1 of canonical json)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "p3b:" + hashlib.sha1(encoded).hexdigest()[:16]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Edit:
    """One substitution edit. ``pos`` is 0-based within the region scope."""
    region: str
    pos: int
    ref: str
    alt: str

    def to_dict(self) -> Dict[str, Any]:
        return {"region": self.region, "pos": self.pos, "ref": self.ref, "alt": self.alt}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Edit":
        return Edit(region=str(d["region"]), pos=int(d["pos"]),
                    ref=str(d["ref"]), alt=str(d["alt"]))


@dataclass
class BenchmarkRecord:
    """One local-edit benchmark sample (see module docstring)."""

    source_id: str
    cargo_id: str
    cell_context: str
    source_sequence: str
    candidate_sequence: str
    edit_list: List[Dict[str, Any]]
    edit_count: int
    edited_region: str
    protein_identity: bool
    measured_or_proxy_source_value: Optional[float]
    measured_or_proxy_candidate_value: Optional[float]
    delta: Optional[float]
    data_source: str
    assay_type: str
    confidence: str
    record_id: str = ""
    edit_type: str = ""
    task_eligibility: str = ""
    value_qualifier: str = ""
    value_std: Optional[float] = None
    family_cluster_id: Optional[str] = None
    motif_flags: List[str] = field(default_factory=list)
    internal_features: Dict[str, Any] = field(default_factory=dict)
    split_role: Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "cargo_id": self.cargo_id,
            "cell_context": self.cell_context,
            "source_sequence": self.source_sequence,
            "candidate_sequence": self.candidate_sequence,
            "edit_list": self.edit_list,
            "edit_count": self.edit_count,
            "edited_region": self.edited_region,
            "protein_identity": self.protein_identity,
            "measured_or_proxy_source_value": self.measured_or_proxy_source_value,
            "measured_or_proxy_candidate_value": self.measured_or_proxy_candidate_value,
            "delta": self.delta,
            "data_source": self.data_source,
            "assay_type": self.assay_type,
            "confidence": self.confidence,
            "edit_type": self.edit_type,
            "task_eligibility": self.task_eligibility,
            "value_qualifier": self.value_qualifier,
            "value_std": self.value_std,
            "family_cluster_id": self.family_cluster_id,
            "motif_flags": self.motif_flags,
            "internal_features": self.internal_features,
            "split_role": self.split_role,
        }
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BenchmarkRecord":
        return BenchmarkRecord(
            source_id=str(d["source_id"]),
            cargo_id=str(d["cargo_id"]),
            cell_context=str(d["cell_context"]),
            source_sequence=str(d["source_sequence"]),
            candidate_sequence=str(d["candidate_sequence"]),
            edit_list=[dict(e) for e in d["edit_list"]],
            edit_count=int(d["edit_count"]),
            edited_region=str(d["edited_region"]),
            protein_identity=bool(d["protein_identity"]),
            measured_or_proxy_source_value=d.get("measured_or_proxy_source_value"),
            measured_or_proxy_candidate_value=d.get("measured_or_proxy_candidate_value"),
            delta=d.get("delta"),
            data_source=str(d["data_source"]),
            assay_type=str(d["assay_type"]),
            confidence=str(d["confidence"]),
            record_id=str(d.get("record_id", "")),
            edit_type=str(d.get("edit_type", "")),
            task_eligibility=str(d.get("task_eligibility", "")),
            value_qualifier=str(d.get("value_qualifier", "")),
            value_std=d.get("value_std"),
            family_cluster_id=d.get("family_cluster_id"),
            motif_flags=list(d.get("motif_flags", [])),
            internal_features=dict(d.get("internal_features", {})),
            split_role=d.get("split_role"),
        )

    # ------------------------------------------------------------------
    def finalize(self) -> "BenchmarkRecord":
        """Validate, then assign the content-addressed record_id."""
        self.validate()
        payload = self.to_dict()
        payload["record_id"] = ""
        self.record_id = stable_record_id(payload)
        return self

    def validate(self) -> None:
        # Required-field presence / types
        if not self.source_id:
            raise BenchmarkSchemaError("source_id must be non-empty")
        if not self.cargo_id:
            raise BenchmarkSchemaError("cargo_id must be non-empty")
        if not self.cell_context:
            raise BenchmarkSchemaError("cell_context must be non-empty")
        if not self.source_sequence:
            raise BenchmarkSchemaError("source_sequence must be non-empty")
        if not self.candidate_sequence:
            raise BenchmarkSchemaError("candidate_sequence must be non-empty")
        if self.edited_region not in EDIT_REGIONS:
            raise BenchmarkSchemaError(f"edited_region not in {EDIT_REGIONS}: {self.edited_region}")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise BenchmarkSchemaError(f"confidence not in {CONFIDENCE_LEVELS}: {self.confidence}")
        if self.edit_type and self.edit_type not in EDIT_TYPES:
            raise BenchmarkSchemaError(f"edit_type not in {EDIT_TYPES}: {self.edit_type}")
        if self.split_role is not None and self.split_role not in SPLIT_ROLES:
            raise BenchmarkSchemaError(f"split_role not in {SPLIT_ROLES}: {self.split_role}")
        if not self.data_source:
            raise BenchmarkSchemaError("data_source must be non-empty")
        if not self.assay_type:
            raise BenchmarkSchemaError("assay_type must be non-empty")
        if not isinstance(self.protein_identity, bool):
            raise BenchmarkSchemaError("protein_identity must be bool")

        # edit_count consistency
        if self.edit_count != len(self.edit_list):
            raise BenchmarkSchemaError(
                f"edit_count {self.edit_count} != len(edit_list) {len(self.edit_list)}")
        if self.edit_count > MAX_EDIT_BUDGET:
            raise BenchmarkSchemaError(
                f"edit_count {self.edit_count} exceeds max budget {MAX_EDIT_BUDGET}")
        for e in self.edit_list:
            for k in ("region", "pos", "ref", "alt"):
                if k not in e:
                    raise BenchmarkSchemaError(f"edit_list entry missing {k}: {e}")
            if len(str(e["ref"])) != 1 or len(str(e["alt"])) != 1:
                raise BenchmarkSchemaError(f"only single-nt substitutions allowed: {e}")

        # Substitution consistency: applying edits reproduces candidate.
        if len(self.candidate_sequence) != len(self.source_sequence):
            raise BenchmarkSchemaError("substitution edits must preserve length")
        seq = list(self.source_sequence)
        for e in self.edit_list:
            pos = int(e["pos"])
            if not (0 <= pos < len(seq)):
                raise BenchmarkSchemaError(f"edit pos out of range: {e}")
            if seq[pos] != str(e["ref"]):
                raise BenchmarkSchemaError(
                    f"edit ref mismatch at {pos}: source has {seq[pos]}, edit says {e['ref']}")
            seq[pos] = str(e["alt"])
        if "".join(seq) != self.candidate_sequence:
            raise BenchmarkSchemaError("applying edit_list does not reproduce candidate_sequence")

        # Region/task alignment with the frozen p3_task_v2 contract.
        expected_task = TASK_ELIGIBILITY[self.edited_region]
        if self.task_eligibility and self.task_eligibility != expected_task:
            raise BenchmarkSchemaError(
                f"task_eligibility {self.task_eligibility} != region-implied {expected_task}")

        # Anti-fabrication invariant.
        if self.confidence == "unlabeled":
            if (self.measured_or_proxy_source_value is not None
                    or self.measured_or_proxy_candidate_value is not None
                    or self.delta is not None):
                raise BenchmarkSchemaError(
                    "unlabeled records must have null values/delta "
                    "(never disguise unlabeled data as local-delta ground truth)")
        else:
            if (self.measured_or_proxy_source_value is None
                    or self.measured_or_proxy_candidate_value is None
                    or self.delta is None):
                raise BenchmarkSchemaError(
                    f"{self.confidence} records must carry source/candidate values and delta")
            # delta consistency
            expect = (float(self.measured_or_proxy_candidate_value)
                      - float(self.measured_or_proxy_source_value))
            if abs(expect - float(self.delta)) > 1e-6:
                raise BenchmarkSchemaError(
                    f"delta {self.delta} != candidate-source {expect}")


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def write_benchmark_jsonl(records: Iterable[BenchmarkRecord], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
            n += 1
    return n


def read_benchmark_jsonl(path: str) -> Iterator[BenchmarkRecord]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield BenchmarkRecord.from_dict(json.loads(line))


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance; -1 if lengths differ."""
    if len(a) != len(b):
        return -1
    return sum(1 for x, y in zip(a, b) if x != y)


def edits_from_alignment(source: str, candidate: str, region: str) -> List[Dict[str, Any]]:
    """Derive a substitution edit_list from an equal-length source/candidate pair."""
    if len(source) != len(candidate):
        raise BenchmarkSchemaError("source/candidate length mismatch")
    return [
        {"region": region, "pos": i, "ref": s, "alt": c}
        for i, (s, c) in enumerate(zip(source, candidate))
        if s != c
    ]
