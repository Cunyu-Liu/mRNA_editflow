#!/usr/bin/env python
"""B0-01: Canonical schemas for utr_edit_record, edit_script, generation_task.

Frozen, validated dataclasses for the v2 contract B0 phase. These schemas are
the authoritative types consumed by all downstream B0 tasks (split manifests,
leakage audit, evaluation tracks, data card) and by FM0/MK0/EF0/MB0.

The schemas are intentionally compatible with D1 artifacts:
  - data/d1_canonical_records.jsonl
  - data/data_exposure_ledger.jsonl

Each record in d1_canonical_records.jsonl is a serialized UTREditRecord.
A generation_task is a downstream evaluation/decoding specification that
references a record by record_id and constrains the generation behaviour
per v2 contract §10 (closed_measured_pool / heldout_generative /
open_legal_generation).

B0-01 acceptance: schemas defined and tested.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: B0-01
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants — frozen enums from v2 contract
# ---------------------------------------------------------------------------

REGION_VALUES: Tuple[str, ...] = ("5'UTR", "3'UTR")
# CDS / full-length are D_A observational only — not editable regions for v2.
# They appear in canonical records for exposure tracking but never as
# generation_task regions.
REGION_VALUES_OBSERVATIONAL: Tuple[str, ...] = ("CDS", "full-length")
ALL_REGION_VALUES: Tuple[str, ...] = REGION_VALUES + REGION_VALUES_OBSERVATIONAL
# Accept underscore aliases emitted by legacy D1 build code; normalized to
# the canonical hyphenated form on ingestion.
_REGION_ALIASES: Dict[str, str] = {"full_length": "full-length"}

OP_VALUES: Tuple[str, ...] = ("INS", "DEL", "SUB", "STOP")
ALPHABET: Tuple[str, ...] = ("A", "C", "G", "T")

DATA_ROLES: Tuple[str, ...] = ("D_A", "D_C", "D_D", "D_E")
EVIDENCE_GRADES: Tuple[str, ...] = ("E1", "E2", "E4")
EXPOSURE_STATUSES: Tuple[str, ...] = (
    "unexposed",
    "historically_exposed",
    "observational_no_labels",
    "incomplete",
)
RECORD_TYPES: Tuple[str, ...] = ("paired", "observational", "incomplete")

EVAL_TRACKS: Tuple[str, ...] = (
    "closed_measured_pool",
    "heldout_generative",
    "open_legal_generation",
)

ALLOWED_CLAIMS_POOL: Tuple[str, ...] = (
    "edit_effect",
    "generation_grounding",
    "observational_prior",
    "foundation_adaptation",
    "region_representation",
    "generative_denoising",
)

# GSE246381 forbidden wording (v2 contract §4.2 / §14.1)
FORBIDDEN_SEALED_WORDING: Tuple[str, ...] = (
    "sealed",
    "untouched",
    "never-seen_external_test",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised when a record/task fails schema validation."""


# ---------------------------------------------------------------------------
# 1. EditScript — wraps the list of EditOp produced by D1 edit_script_core
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EditOp:
    """A single edit operation in an edit script.

    Attributes:
        op: operation type ("INS", "DEL", "SUB", "STOP")
        pos: 0-indexed position in the current state at time of application
        token: nucleotide for INS/SUB; empty string for DEL/STOP
    """
    op: str
    pos: int
    token: str

    def __post_init__(self) -> None:
        if self.op not in OP_VALUES:
            raise SchemaError(f"EditOp.op must be one of {OP_VALUES}, got {self.op!r}")
        if not isinstance(self.pos, int) or self.pos < 0:
            raise SchemaError(f"EditOp.pos must be a non-negative int, got {self.pos!r}")
        if self.op in ("INS", "SUB"):
            if self.token not in ALPHABET:
                raise SchemaError(
                    f"EditOp.token for {self.op} must be one of {ALPHABET}, "
                    f"got {self.token!r}"
                )
        elif self.op in ("DEL", "STOP"):
            if self.token != "":
                raise SchemaError(
                    f"EditOp.token for {self.op} must be empty string, "
                    f"got {self.token!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "pos": self.pos, "token": self.token}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditOp":
        return cls(op=d["op"], pos=int(d["pos"]), token=d.get("token", ""))


@dataclass(frozen=True)
class EditScript:
    """An ordered edit script (list of EditOp) with verification metadata.

    The canonical D1 form is the minimal-length Levenshtein script with a
    deterministic MATCH > DEL > INS > SUB traceback. STOP is reserved for
    generation and never appears in paired canonical records.

    Attributes:
        ops: ordered list of EditOp
        verified: True iff apply(ops, source) == candidate
        edit_distance: total number of INS+DEL+SUB operations
        n_ins / n_del / n_sub: per-op counts
        path_ambiguity: number of distinct minimal-length edit scripts (>=1)
    """
    ops: Tuple[EditOp, ...]
    verified: bool
    edit_distance: int
    n_ins: int
    n_del: int
    n_sub: int
    path_ambiguity: int

    def __post_init__(self) -> None:
        if not isinstance(self.ops, tuple):
            raise SchemaError("EditScript.ops must be a tuple")
        for op in self.ops:
            if not isinstance(op, EditOp):
                raise SchemaError("EditScript.ops must contain EditOp instances")
        if self.ops and self.ops[-1].op == "STOP":
            raise SchemaError(
                "STOP is reserved for generation; canonical edit scripts must "
                "not end with STOP"
            )
        counts = {"INS": 0, "DEL": 0, "SUB": 0, "STOP": 0}
        for op in self.ops:
            counts[op.op] += 1
        if counts["STOP"] != 0:
            raise SchemaError("STOP is not allowed in canonical edit scripts")
        if counts["INS"] != self.n_ins:
            raise SchemaError(f"n_ins mismatch: {counts['INS']} vs {self.n_ins}")
        if counts["DEL"] != self.n_del:
            raise SchemaError(f"n_del mismatch: {counts['DEL']} vs {self.n_del}")
        if counts["SUB"] != self.n_sub:
            raise SchemaError(f"n_sub mismatch: {counts['SUB']} vs {self.n_sub}")
        if counts["INS"] + counts["DEL"] + counts["SUB"] != self.edit_distance:
            raise SchemaError(
                f"edit_distance mismatch: "
                f"{counts['INS'] + counts['DEL'] + counts['SUB']} vs {self.edit_distance}"
            )
        if self.path_ambiguity < 1:
            raise SchemaError(
                f"path_ambiguity must be >= 1, got {self.path_ambiguity}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ops": [op.to_dict() for op in self.ops],
            "verified": self.verified,
            "edit_distance": self.edit_distance,
            "n_ins": self.n_ins,
            "n_del": self.n_del,
            "n_sub": self.n_sub,
            "path_ambiguity": self.path_ambiguity,
        }


# ---------------------------------------------------------------------------
# 2. UTREditRecord — frozen schema for a canonical record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UTREditRecord:
    """Frozen schema for a canonical UTR edit record.

    This is the authoritative type for every record in
    data/d1_canonical_records.jsonl.

    For paired records (D_C, D_D), source_sequence + candidate_sequence +
    edit_script are all populated and verified.

    For observational records (D_A), source_sequence is None, edit_script is
    empty, and candidate_sequence holds the observed sequence.

    Attributes:
        record_id: globally unique record identifier (accession + local id)
        dataset: short dataset name (e.g. "sample2019", "gse200304")
        accession: GEO/ENCODE accession
        region: one of REGION_VALUES + REGION_VALUES_OBSERVATIONAL
        source_sequence: source UTR (None for observational)
        candidate_sequence: candidate/observed sequence
        edit_script: EditScript (empty for observational)
        labels: dict of endpoint -> numeric value
        metadata: free-form per-dataset metadata (always contains record_type)
    """
    record_id: str
    dataset: str
    accession: str
    region: str
    source_sequence: Optional[str]
    candidate_sequence: str
    edit_script: EditScript
    labels: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not isinstance(self.record_id, str):
            raise SchemaError(f"record_id must be a non-empty str, got {self.record_id!r}")
        if not self.dataset or not isinstance(self.dataset, str):
            raise SchemaError(f"dataset must be a non-empty str, got {self.dataset!r}")
        if not self.accession or not isinstance(self.accession, str):
            raise SchemaError(f"accession must be a non-empty str, got {self.accession!r}")
        if self.region not in ALL_REGION_VALUES:
            raise SchemaError(
                f"region must be one of {ALL_REGION_VALUES}, got {self.region!r}"
            )
        if self.source_sequence is not None:
            if not isinstance(self.source_sequence, str):
                raise SchemaError("source_sequence must be str or None")
            self._validate_alphabet(self.source_sequence, "source_sequence")
        # candidate_sequence may be None ONLY for incomplete records
        if self.candidate_sequence is not None and not isinstance(self.candidate_sequence, str):
            raise SchemaError("candidate_sequence must be a str or None")
        if self.candidate_sequence:
            self._validate_alphabet(self.candidate_sequence, "candidate_sequence")
        if not isinstance(self.edit_script, EditScript):
            raise SchemaError("edit_script must be an EditScript instance")
        if not isinstance(self.labels, dict):
            raise SchemaError("labels must be a dict")
        for k, v in self.labels.items():
            if not isinstance(k, str):
                raise SchemaError(f"label key must be str, got {k!r}")
            if not isinstance(v, (int, float)):
                raise SchemaError(f"label value must be numeric, got {v!r} for {k!r}")
        if not isinstance(self.metadata, dict):
            raise SchemaError("metadata must be a dict")
        if "record_type" not in self.metadata:
            raise SchemaError(
                "metadata must contain 'record_type' (paired, observational, or incomplete)"
            )
        if self.metadata["record_type"] not in RECORD_TYPES:
            raise SchemaError(
                f"metadata.record_type must be one of {RECORD_TYPES}, "
                f"got {self.metadata['record_type']!r}"
            )
        # Cross-field consistency
        rt = self.metadata["record_type"]
        if rt == "paired":
            if self.source_sequence is None:
                raise SchemaError(
                    "paired records must have non-None source_sequence"
                )
            if not self.edit_script.verified:
                raise SchemaError(
                    "paired records must have edit_script.verified == True"
                )
        elif rt == "observational":
            if self.source_sequence is not None:
                raise SchemaError(
                    "observational records must have source_sequence == None"
                )
            if self.edit_script.ops:
                raise SchemaError(
                    "observational records must have empty edit_script.ops"
                )
            if self.edit_script.edit_distance != 0:
                raise SchemaError(
                    "observational records must have edit_distance == 0"
                )
        # incomplete records: no further cross-field constraints (sequences may
        # be missing); used for GSE173083 where only .rdat files exist in GEO

    @staticmethod
    def _validate_alphabet(seq: str, field_name: str) -> None:
        bad = set(seq) - set(ALPHABET)
        if bad:
            raise SchemaError(
                f"{field_name} contains non-ACGT chars: {sorted(bad)}"
            )

    @property
    def is_paired(self) -> bool:
        return self.metadata.get("record_type") == "paired"

    @property
    def is_observational(self) -> bool:
        return self.metadata.get("record_type") == "observational"

    @property
    def is_incomplete(self) -> bool:
        return self.metadata.get("record_type") == "incomplete"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the D1 canonical_records.jsonl row format.

        This MUST round-trip with from_dict.
        """
        return {
            "record_id": self.record_id,
            "dataset": self.dataset,
            "accession": self.accession,
            "region": self.region,
            "source_sequence": self.source_sequence,
            "candidate_sequence": self.candidate_sequence,
            "edit_script": [op.to_dict() for op in self.edit_script.ops],
            "edit_script_verified": self.edit_script.verified,
            "edit_distance": self.edit_script.edit_distance,
            "n_ins": self.edit_script.n_ins,
            "n_del": self.edit_script.n_del,
            "n_sub": self.edit_script.n_sub,
            "path_ambiguity": self.edit_script.path_ambiguity,
            "labels": dict(self.labels),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UTREditRecord":
        """Deserialize from a D1 canonical_records.jsonl row.

        Accepts both the canonical D1 flat form (edit_script as list of dicts
        + edit_script_verified/edit_distance/n_ins/n_del/n_sub/path_ambiguity
        as top-level fields) and the nested EditScript form.

        Backward-compatibility: if `metadata.record_type` is missing (legacy
        D1 paired records produced by `canonical_record()` did not emit it),
        infer it from `source_sequence` presence:
          - source_sequence is not None  -> "paired"
          - source_sequence is None      -> "observational"
        The inferred value is written back into metadata so downstream
        consumers see an explicit record_type.
        """
        # Detect form
        if isinstance(d.get("edit_script"), dict) and "ops" in d["edit_script"]:
            es_dict = d["edit_script"]
            ops = tuple(EditOp.from_dict(op) for op in es_dict["ops"])
            edit_script = EditScript(
                ops=ops,
                verified=es_dict["verified"],
                edit_distance=es_dict["edit_distance"],
                n_ins=es_dict["n_ins"],
                n_del=es_dict["n_del"],
                n_sub=es_dict["n_sub"],
                path_ambiguity=es_dict["path_ambiguity"],
            )
        else:
            # Flat D1 form
            ops = tuple(EditOp.from_dict(op) for op in d.get("edit_script", []))
            edit_script = EditScript(
                ops=ops,
                verified=bool(d.get("edit_script_verified", True)),
                edit_distance=int(d.get("edit_distance", 0)),
                n_ins=int(d.get("n_ins", 0)),
                n_del=int(d.get("n_del", 0)),
                n_sub=int(d.get("n_sub", 0)),
                path_ambiguity=int(d.get("path_ambiguity", 1)),
            )
        metadata = dict(d.get("metadata", {}))
        # Backward-compat: infer record_type for legacy D1 paired records
        if "record_type" not in metadata:
            src = d.get("source_sequence")
            metadata["record_type"] = "paired" if src is not None else "observational"
        # Normalize region aliases (e.g. "full_length" -> "full-length")
        region = d["region"]
        region = _REGION_ALIASES.get(region, region)
        return cls(
            record_id=d["record_id"],
            dataset=d["dataset"],
            accession=d["accession"],
            region=region,
            source_sequence=d.get("source_sequence"),
            candidate_sequence=d.get("candidate_sequence"),
            edit_script=edit_script,
            labels=dict(d.get("labels", {})),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# 3. GenerationTask — frozen schema for an evaluation/decoding task
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationTask:
    """Frozen schema for a generation/evaluation task on a single record.

    Used by B0-04 evaluation tracks. Each task binds a record_id to a track
    and a set of generation constraints (edit budget, target, must-preserve
    motifs, etc.) per v2 contract §10 + §11.

    Attributes:
        task_id: unique task identifier
        record_id: reference to a UTREditRecord (must exist in canonical set)
        track: one of EVAL_TRACKS
        region: one of REGION_VALUES (CDS/full-length not editable)
        edit_budget: max number of INS+DEL+SUB ops allowed (k in {1,3,5})
        target_endpoint: label key to condition on (e.g. "rl", "te", "half_life")
            or None for unconditional generation
        target_direction: "increase" | "decrease" | "preserve" | None
        target_quantile: optional float in (0,1)
        max_length: optional max candidate length
        min_length: optional min candidate length
        must_preserve_motifs: tuple of motif strings that must appear verbatim
            in the candidate
        allowed_claims: claims that may be made on this task's results
        forbidden_claims: claims that must not be made
        notes: optional free-form notes
    """
    task_id: str
    record_id: str
    track: str
    region: str
    edit_budget: int
    target_endpoint: Optional[str] = None
    target_direction: Optional[str] = None
    target_quantile: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    must_preserve_motifs: Tuple[str, ...] = ()
    allowed_claims: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    notes: Optional[str] = None

    _DIRECTIONS: Tuple[str, ...] = ("increase", "decrease", "preserve")

    def __post_init__(self) -> None:
        if not self.task_id or not isinstance(self.task_id, str):
            raise SchemaError(f"task_id must be a non-empty str, got {self.task_id!r}")
        if not self.record_id or not isinstance(self.record_id, str):
            raise SchemaError(f"record_id must be a non-empty str, got {self.record_id!r}")
        if self.track not in EVAL_TRACKS:
            raise SchemaError(
                f"track must be one of {EVAL_TRACKS}, got {self.track!r}"
            )
        if self.region not in REGION_VALUES:
            raise SchemaError(
                f"GenerationTask.region must be one of {REGION_VALUES} "
                f"(CDS/full-length not editable), got {self.region!r}"
            )
        if not isinstance(self.edit_budget, int) or self.edit_budget < 0:
            raise SchemaError(
                f"edit_budget must be a non-negative int, got {self.edit_budget!r}"
            )
        # v2 contract §11: k in {1, 3, 5}
        if self.edit_budget not in (1, 3, 5) and self.edit_budget != 0:
            raise SchemaError(
                f"edit_budget must be in {1, 3, 5} (or 0 for no-op), "
                f"got {self.edit_budget!r}"
            )
        if self.target_direction is not None and self.target_direction not in self._DIRECTIONS:
            raise SchemaError(
                f"target_direction must be one of {self._DIRECTIONS} or None, "
                f"got {self.target_direction!r}"
            )
        if self.target_endpoint is None and self.target_direction is not None:
            raise SchemaError(
                "target_direction requires target_endpoint to be set"
            )
        if self.target_quantile is not None:
            if not (0.0 < self.target_quantile < 1.0):
                raise SchemaError(
                    f"target_quantile must be in (0,1), got {self.target_quantile!r}"
                )
        if self.max_length is not None and self.max_length <= 0:
            raise SchemaError(
                f"max_length must be positive, got {self.max_length!r}"
            )
        if self.min_length is not None and self.min_length < 0:
            raise SchemaError(
                f"min_length must be non-negative, got {self.min_length!r}"
            )
        if (
            self.max_length is not None
            and self.min_length is not None
            and self.max_length < self.min_length
        ):
            raise SchemaError(
                f"max_length ({self.max_length}) < min_length ({self.min_length})"
            )
        for m in self.must_preserve_motifs:
            if not isinstance(m, str) or not m:
                raise SchemaError(
                    f"must_preserve_motifs entries must be non-empty str, got {m!r}"
                )
            bad = set(m) - set(ALPHABET)
            if bad:
                raise SchemaError(
                    f"motif {m!r} contains non-ACGT chars: {sorted(bad)}"
                )
        for c in self.allowed_claims:
            if c not in ALLOWED_CLAIMS_POOL:
                raise SchemaError(
                    f"allowed_claims entry {c!r} not in {ALLOWED_CLAIMS_POOL}"
                )
        # forbidden_claims may include any string (forbidden is open-set)
        if not isinstance(self.forbidden_claims, tuple):
            raise SchemaError("forbidden_claims must be a tuple")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "record_id": self.record_id,
            "track": self.track,
            "region": self.region,
            "edit_budget": self.edit_budget,
            "target_endpoint": self.target_endpoint,
            "target_direction": self.target_direction,
            "target_quantile": self.target_quantile,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "must_preserve_motifs": list(self.must_preserve_motifs),
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationTask":
        return cls(
            task_id=d["task_id"],
            record_id=d["record_id"],
            track=d["track"],
            region=d["region"],
            edit_budget=int(d["edit_budget"]),
            target_endpoint=d.get("target_endpoint"),
            target_direction=d.get("target_direction"),
            target_quantile=d.get("target_quantile"),
            max_length=d.get("max_length"),
            min_length=d.get("min_length"),
            must_preserve_motifs=tuple(d.get("must_preserve_motifs", [])),
            allowed_claims=tuple(d.get("allowed_claims", [])),
            forbidden_claims=tuple(d.get("forbidden_claims", [])),
            notes=d.get("notes"),
        )


# ---------------------------------------------------------------------------
# 4. Top-level validators (used by B0-02/B0-03/B0-04 auditors)
# ---------------------------------------------------------------------------

def validate_utr_edit_record(d: Dict[str, Any]) -> UTREditRecord:
    """Validate a dict as a UTREditRecord; return the frozen instance.

    Raises SchemaError on any violation.
    """
    return UTREditRecord.from_dict(d)


def validate_generation_task(d: Dict[str, Any]) -> GenerationTask:
    """Validate a dict as a GenerationTask; return the frozen instance."""
    return GenerationTask.from_dict(d)


def validate_canonical_records_file(
    path: str,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate every record in a canonical_records.jsonl file.

    Returns a summary dict with counts and any issues. Used by B0-03 leakage
    audit and B0-05 data card.

    Args:
        path: path to canonical_records.jsonl
        max_records: optional cap (for tests)

    Returns:
        {
            "n_total": int,
            "n_paired": int,
            "n_observational": int,
            "n_incomplete": int,
            "n_invalid": int,
            "issues": List[str],  # first 50 issues
            "by_accession": {accession: count},
            "by_region": {region: count},
            "passed": bool,
        }
    """
    n_total = 0
    n_paired = 0
    n_observational = 0
    n_incomplete = 0
    n_invalid = 0
    issues: List[str] = []
    by_accession: Dict[str, int] = {}
    by_region: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rec = UTREditRecord.from_dict(d)
                n_total += 1
                if rec.is_paired:
                    n_paired += 1
                elif rec.is_incomplete:
                    n_incomplete += 1
                else:
                    n_observational += 1
                by_accession[rec.accession] = by_accession.get(rec.accession, 0) + 1
                by_region[rec.region] = by_region.get(rec.region, 0) + 1
            except Exception as e:
                n_invalid += 1
                if len(issues) < 50:
                    issues.append(f"line {line_no}: {type(e).__name__}: {e}")
            if max_records is not None and n_total + n_invalid >= max_records:
                break

    return {
        "n_total": n_total,
        "n_paired": n_paired,
        "n_observational": n_observational,
        "n_incomplete": n_incomplete,
        "n_invalid": n_invalid,
        "issues": issues,
        "by_accession": by_accession,
        "by_region": by_region,
        "passed": n_invalid == 0,
    }


__all__ = [
    "REGION_VALUES",
    "REGION_VALUES_OBSERVATIONAL",
    "ALL_REGION_VALUES",
    "OP_VALUES",
    "ALPHABET",
    "DATA_ROLES",
    "EVIDENCE_GRADES",
    "EXPOSURE_STATUSES",
    "RECORD_TYPES",
    "EVAL_TRACKS",
    "ALLOWED_CLAIMS_POOL",
    "FORBIDDEN_SEALED_WORDING",
    "SchemaError",
    "EditOp",
    "EditScript",
    "UTREditRecord",
    "GenerationTask",
    "validate_utr_edit_record",
    "validate_generation_task",
    "validate_canonical_records_file",
]
