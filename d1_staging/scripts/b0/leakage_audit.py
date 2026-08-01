#!/usr/bin/env python
"""B0-03: Comprehensive leakage audit across all split manifests.

Checks every overlap channel listed in the B0-03 task description:
  exact / reverse / intermediate / cluster / scaffold / gene / study /
  context / barcode / foundation

Acceptance (task_registry_v2.yaml B0-03):
  - final endpoint as train intermediate = 0
  - exposure ledger coverage = 100%

Scope notes (per contract amendment v2.2 DEC-UTR-EF-V2-20260731-B0-FROZEN-REPLAY-SCOPE):
  - path-state scope = frozen D1 canonical edit_script prefixes + declared
    intermediates.
  - Only 2 D_C datasets have paired records (GSE114002 5'UTR, GSE200304
    3'UTR). gene/context/barcode metadata is not present in canonical
    records, so those channels are documented as N/A (not applicable) with
    the reason recorded. foundation overlap is audited only when an FM0-bound
    foundation training-data manifest is supplied; exact sequence-level
    overlap is never inferred from source-level exposure metadata.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: B0-03
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make B0 schemas + D1 edit_script_core importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_D1_DIR = os.path.normpath(os.path.join(HERE, "..", "d1"))
if _D1_DIR not in sys.path:
    sys.path.insert(0, _D1_DIR)

from canonical_schemas import UTREditRecord  # noqa: E402
from edit_script_core import apply_edit_script  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    "5utr_source_disjoint": "split_5utr_source_disjoint.jsonl",
    "3utr_source_disjoint": "split_3utr_source_disjoint.jsonl",
    "study_disjoint": "split_study_disjoint.jsonl",
    "cross_region_transfer": "split_cross_region_transfer.jsonl",
}

# Scaffold patterns in record_ids:
#   GSE114002_NC_000012.12:g.4911352C>T_0  -> NC_000012.12
#   GSE200304_chr2:69461620_G-C            -> chr2
#   GSE114002_8527_1                         -> None (designed library, no scaffold)
_SCAFFOLD_RE_NC = re.compile(r"(NC_\d{6}\.\d+)")
_SCAFFOLD_RE_CHR = re.compile(r"(chr\w+)")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def load_foundation_manifest(path: str) -> Dict[str, Any]:
    """Load the FM0-bound foundation provenance manifest.

    The manifest is deliberately small and source-level: it binds the exact
    checkpoint revision/hash/license to the documented pretraining corpus and
    to an explicit per-accession exposure classification. The model card does
    not provide a dump of every pretraining sequence, so this function must
    not turn source-level evidence into an exact sequence-overlap claim.
    Invalid or missing manifests are returned as structured errors so the
    caller can fail closed while preserving a report.
    """
    p = Path(path)
    if not p.exists():
        return {
            "_load_error": f"foundation manifest not found: {p}",
            "_manifest_path": str(p),
        }
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # pragma: no cover - exercised by integration
        return {
            "_load_error": f"foundation manifest unreadable: {p}: {exc}",
            "_manifest_path": str(p),
        }
    if not isinstance(payload, dict):
        return {
            "_load_error": f"foundation manifest must be a JSON object: {p}",
            "_manifest_path": str(p),
        }
    payload = dict(payload)
    payload["_manifest_path"] = str(p)
    payload["_manifest_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return payload


def _foundation_manifest_errors(manifest: Dict[str, Any]) -> List[str]:
    """Validate the non-negotiable FM0 foundation provenance fields."""
    errors: List[str] = []
    if manifest.get("_load_error"):
        errors.append(str(manifest["_load_error"]))
        return errors
    required = (
        "schema_version",
        "model_id",
        "revision",
        "checkpoint_sha256",
        "license",
        "corpus_sources",
        "dataset_exposure",
        "exposure_assertions_complete",
        "exact_sequence_manifest_available",
    )
    errors.extend(f"missing:{key}" for key in required if key not in manifest)
    if errors:
        return errors
    if manifest["schema_version"] != "fm0-foundation-training-data/v1":
        errors.append("schema_version_mismatch")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"].strip():
        errors.append("invalid:model_id")
    if not isinstance(manifest["revision"], str) or not manifest["revision"].strip():
        errors.append("invalid:revision")
    checkpoint_sha256 = manifest["checkpoint_sha256"]
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        errors.append("invalid:checkpoint_sha256")
    license_info = manifest["license"]
    if not isinstance(license_info, dict) or not license_info.get("type"):
        errors.append("invalid:license")
    if not isinstance(manifest["corpus_sources"], list) or not manifest["corpus_sources"]:
        errors.append("invalid:corpus_sources")
    exposure = manifest["dataset_exposure"]
    if not isinstance(exposure, list) or not exposure:
        errors.append("invalid:dataset_exposure")
    if manifest["exposure_assertions_complete"] is not True:
        errors.append("exposure_assertions_not_complete")
    if not isinstance(manifest["exact_sequence_manifest_available"], bool):
        errors.append("invalid:exact_sequence_manifest_available")
    seen = set()
    if isinstance(exposure, list):
        row_required = (
            "accession",
            "region",
            "historically_exposed_to_model",
            "exposure_type",
            "evidence_grade",
            "labels_exposed_to_model",
        )
        for row in exposure:
            if not isinstance(row, dict):
                errors.append("invalid:dataset_exposure_row")
                continue
            missing = [key for key in row_required if key not in row]
            errors.extend(f"dataset_exposure_missing:{key}" for key in missing)
            if missing:
                continue
            key = (str(row["accession"]), str(row["region"]))
            if key in seen:
                errors.append(f"duplicate:dataset_exposure:{key[0]}:{key[1]}")
            seen.add(key)
    return errors


def _normalize_region(region: Any) -> str:
    return str(region).replace("′", "'").strip()


def _foundation_exposure_lookup(
    manifest: Dict[str, Any],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (str(row["accession"]), _normalize_region(row["region"])): row
        for row in manifest.get("dataset_exposure", [])
        if isinstance(row, dict) and "accession" in row and "region" in row
    }


def load_paired_records_by_id(path: str) -> Dict[str, UTREditRecord]:
    out: Dict[str, UTREditRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = UTREditRecord.from_dict(d)
            if rec.is_paired:
                out[rec.record_id] = rec
    return out


def load_exposure_ledger_ids(path: str) -> Set[str]:
    """Load all record_ids present in the exposure ledger."""
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rid = d.get("record_id")
            if rid:
                ids.add(rid)
    return ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_intermediate_states(rec: UTREditRecord) -> List[str]:
    """Edit_script prefix states (length 1..len(ops)-1); excludes final candidate."""
    ops = list(rec.edit_script.ops) if rec.edit_script else []
    if not ops or rec.source_sequence is None:
        return []
    states: List[str] = []
    for i in range(1, len(ops)):
        states.append(apply_edit_script(rec.source_sequence, ops[:i]))
    return states


def extract_scaffold(record_id: str) -> Optional[str]:
    """Extract chromosomal scaffold from a record_id, if present."""
    m = _SCAFFOLD_RE_NC.search(record_id)
    if m:
        return m.group(1)
    m = _SCAFFOLD_RE_CHR.search(record_id)
    if m:
        return m.group(1)
    return None


def split_by_role(
    entries: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group manifest entries by split role (train/val/test)."""
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for e in entries:
        s = e.get("split")
        if s in out:
            out[s].append(e)
    return out


# ---------------------------------------------------------------------------
# Overlap checks
# ---------------------------------------------------------------------------

def check_exact_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Exact sequence overlap: source or candidate of train appears in test."""
    train_seqs: Set[str] = set()
    for e in train:
        rec = records_by_id.get(e["record_id"])
        if rec is None:
            continue
        if rec.source_sequence:
            train_seqs.add(rec.source_sequence)
        if rec.candidate_sequence:
            train_seqs.add(rec.candidate_sequence)
    test_seqs: Set[str] = set()
    for e in test:
        rec = records_by_id.get(e["record_id"])
        if rec is None:
            continue
        if rec.source_sequence:
            test_seqs.add(rec.source_sequence)
        if rec.candidate_sequence:
            test_seqs.add(rec.candidate_sequence)
    overlap = train_seqs & test_seqs
    return {
        "channel": "exact",
        "applicable": True,
        "n_train_seqs": len(train_seqs),
        "n_test_seqs": len(test_seqs),
        "n_overlap": len(overlap),
        "overlap_examples": sorted(overlap)[:5],
        "pass": len(overlap) == 0,
    }


def check_reverse_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Reverse: candidate(train) ∩ source(test) and source(train) ∩ candidate(test)."""
    train_cand: Set[str] = set()
    train_src: Set[str] = set()
    for e in train:
        rec = records_by_id.get(e["record_id"])
        if rec is None:
            continue
        if rec.candidate_sequence:
            train_cand.add(rec.candidate_sequence)
        if rec.source_sequence:
            train_src.add(rec.source_sequence)
    test_src: Set[str] = set()
    test_cand: Set[str] = set()
    for e in test:
        rec = records_by_id.get(e["record_id"])
        if rec is None:
            continue
        if rec.source_sequence:
            test_src.add(rec.source_sequence)
        if rec.candidate_sequence:
            test_cand.add(rec.candidate_sequence)
    rev = train_cand & test_src
    sym = train_src & test_cand
    return {
        "channel": "reverse",
        "applicable": True,
        "n_reverse": len(rev),
        "n_symmetric": len(sym),
        "n_overlap": len(rev) + len(sym),
        "pass": len(rev) == 0 and len(sym) == 0,
    }


def check_intermediate_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Intermediate: train edit_script prefix states ∩ test sequences (source+candidate).

    This is the 'final endpoint as train intermediate' acceptance check:
    no test endpoint (candidate) may equal a train intermediate state.
    Also checks test intermediates against train endpoints.
    """
    train_inter: Set[str] = set()
    train_endpoints: Set[str] = set()
    for e in train:
        rec = records_by_id.get(e["record_id"])
        if rec is None or rec.source_sequence is None:
            continue
        train_inter.update(compute_intermediate_states(rec))
        if rec.candidate_sequence:
            train_endpoints.add(rec.candidate_sequence)
        if rec.source_sequence:
            train_endpoints.add(rec.source_sequence)
    test_inter: Set[str] = set()
    test_endpoints: Set[str] = set()
    for e in test:
        rec = records_by_id.get(e["record_id"])
        if rec is None or rec.source_sequence is None:
            continue
        test_inter.update(compute_intermediate_states(rec))
        if rec.candidate_sequence:
            test_endpoints.add(rec.candidate_sequence)
        if rec.source_sequence:
            test_endpoints.add(rec.source_sequence)
    # Acceptance: final endpoint (test candidate) as train intermediate = 0
    endpoint_as_train_inter = test_endpoints & train_inter
    # Symmetric: train endpoint as test intermediate
    train_endpoint_as_test_inter = train_endpoints & test_inter
    # Intermediate-intermediate overlap
    inter_inter = train_inter & test_inter
    return {
        "channel": "intermediate",
        "applicable": True,
        "n_train_intermediates": len(train_inter),
        "n_test_intermediates": len(test_inter),
        "n_endpoint_as_train_intermediate": len(endpoint_as_train_inter),
        "endpoint_as_train_intermediate_examples": sorted(endpoint_as_train_inter)[:5],
        "n_train_endpoint_as_test_intermediate": len(train_endpoint_as_test_inter),
        "n_intermediate_intermediate_overlap": len(inter_inter),
        "n_overlap": len(endpoint_as_train_inter)
        + len(train_endpoint_as_test_inter)
        + len(inter_inter),
        "pass": (
            len(endpoint_as_train_inter) == 0
            and len(train_endpoint_as_test_inter) == 0
            and len(inter_inter) == 0
        ),
    }


def check_cluster_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Cluster overlap: guaranteed by construction in B0-02 (mmseqs cluster split).

    For source_disjoint splits, clusters were split as atomic units, so no
    cluster spans train+test. We re-verify by recomputing source-sequence
    exact duplicates as a proxy (exact same source = same cluster at 1.0 id).
    A full re-cluster is not re-run here; B0-02 already enforced it at 0.8 id.
    """
    # Proxy: identical source sequences across train/test (would be same cluster)
    train_src: Set[str] = set()
    for e in train:
        rec = records_by_id.get(e["record_id"])
        if rec and rec.source_sequence:
            train_src.add(rec.source_sequence)
    test_src: Set[str] = set()
    for e in test:
        rec = records_by_id.get(e["record_id"])
        if rec and rec.source_sequence:
            test_src.add(rec.source_sequence)
    proxy_overlap = train_src & test_src
    return {
        "channel": "cluster",
        "applicable": True,
        "method": "proxy via exact source-sequence overlap (full mmseqs re-cluster done in B0-02)",
        "n_proxy_overlap": len(proxy_overlap),
        "n_overlap": len(proxy_overlap),
        "pass": len(proxy_overlap) == 0,
    }


def check_scaffold_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Scaffold overlap: same chromosomal scaffold in train and test.

    For source_disjoint splits within a single study, scaffold overlap is
    EXPECTED (the library spans many scaffolds and the split is by sequence
    cluster, not by scaffold). This channel is informational: it flags
    cross-scaffold transfer readiness, not a hard leak. The hard leak is
    study-level (entire study in one split), checked by check_study_overlap.
    """
    train_scaf: Set[str] = set()
    test_scaf: Set[str] = set()
    n_no_scaffold_train = 0
    n_no_scaffold_test = 0
    for e in train:
        s = extract_scaffold(e["record_id"])
        if s:
            train_scaf.add(s)
        else:
            n_no_scaffold_train += 1
    for e in test:
        s = extract_scaffold(e["record_id"])
        if s:
            test_scaf.add(s)
        else:
            n_no_scaffold_test += 1
    overlap = train_scaf & test_scaf
    return {
        "channel": "scaffold",
        "applicable": True,
        "informational": True,
        "n_train_scaffolds": len(train_scaf),
        "n_test_scaffolds": len(test_scaf),
        "n_overlap": len(overlap),
        "overlap_examples": sorted(overlap)[:5],
        "n_train_no_scaffold": n_no_scaffold_train,
        "n_test_no_scaffold": n_no_scaffold_test,
        "pass": True,  # informational only; not a hard leak for within-study splits
    }


def check_study_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Study overlap: same accession in train and test (hard leak for study_disjoint)."""
    train_acc: Set[str] = set(e["accession"] for e in train)
    test_acc: Set[str] = set(e["accession"] for e in test)
    overlap = train_acc & test_acc
    return {
        "channel": "study",
        "applicable": True,
        "train_accessions": sorted(train_acc),
        "test_accessions": sorted(test_acc),
        "n_overlap": len(overlap),
        "overlap": sorted(overlap),
        "pass": len(overlap) == 0,
    }


def check_gene_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Gene overlap: N/A — canonical records have no gene identifier metadata."""
    return {
        "channel": "gene",
        "applicable": False,
        "reason": "canonical records have no gene identifier metadata; GSE114002 is a designed MPRA library (SNV variants on RefSeq scaffolds, not gene-tagged) and GSE200304 records are variant-keyed without gene symbols",
        "n_overlap": 0,
        "pass": True,
    }


def check_context_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Context overlap: N/A — no experimental-context metadata in canonical records."""
    return {
        "channel": "context",
        "applicable": False,
        "reason": "no experimental-context metadata (cell type / condition / batch) in canonical records; both D_C datasets are single-context MPRA libraries",
        "n_overlap": 0,
        "pass": True,
    }


def check_barcode_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Barcode overlap: N/A — MPRA barcodes are not retained in canonical records."""
    return {
        "channel": "barcode",
        "applicable": False,
        "reason": "MPRA barcodes are not retained in canonical records (D1 canonical form stores source/candidate sequences + edit_script + labels, not raw oligo barcodes)",
        "n_overlap": 0,
        "pass": True,
    }


def check_foundation_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
    foundation_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Audit test exposure against an explicitly bound FM0 corpus manifest.

    The official model card provides documented corpus sources and dataset-level
    exposure provenance, but not a complete dump of every pretraining sequence.
    Therefore this audit classifies known source/accession exposure and
    explicitly records that exact sequence-level overlap is not asserted.
    Missing or invalid manifest coverage fails closed.
    """
    if foundation_manifest is None:
        return {
            "channel": "foundation",
            "applicable": False,
            "status": "PENDING_FM0",
            "reason": "FM0-bound foundation training-data manifest was not supplied; no exposure claim is made",
            "n_overlap": None,
            "pass": True,
        }

    errors = _foundation_manifest_errors(foundation_manifest)
    if errors:
        return {
            "channel": "foundation",
            "applicable": True,
            "status": "FOUNDATION_MANIFEST_INVALID",
            "reason": "FM0 foundation manifest validation failed",
            "validation_errors": errors,
            "manifest_path": foundation_manifest.get("_manifest_path"),
            "manifest_sha256": foundation_manifest.get("_manifest_sha256"),
            "n_overlap": None,
            "pass": False,
        }

    lookup = _foundation_exposure_lookup(foundation_manifest)
    missing_records = 0
    classified = 0
    known_source_overlap = 0
    grade_counts: Dict[str, int] = defaultdict(int)
    for entry in test:
        record = records_by_id.get(entry.get("record_id"))
        if record is None:
            missing_records += 1
            continue
        key = (str(record.accession), _normalize_region(record.region))
        row = lookup.get(key)
        if row is None:
            missing_records += 1
            continue
        classified += 1
        grade_counts[str(row["evidence_grade"])] += 1
        if bool(row["historically_exposed_to_model"]):
            known_source_overlap += 1

    return {
        "channel": "foundation",
        "applicable": True,
        "status": "AUDITED_SOURCE_LEVEL_EXPOSURE",
        "model_id": foundation_manifest["model_id"],
        "revision": foundation_manifest["revision"],
        "manifest_path": foundation_manifest.get("_manifest_path"),
        "manifest_sha256": foundation_manifest.get("_manifest_sha256"),
        "n_test_records": len(test),
        "n_test_records_classified": classified,
        "n_test_records_unclassified": missing_records,
        "n_known_source_overlap": known_source_overlap,
        "n_overlap": None,
        "evidence_grade_counts": dict(sorted(grade_counts.items())),
        "exact_sequence_overlap_status": (
            "NOT_AVAILABLE_NOT_ASSERTED"
            if not foundation_manifest["exact_sequence_manifest_available"]
            else "EXACT_SEQUENCE_MANIFEST_BOUND"
        ),
        "claim_boundary": (
            "Known source/accession exposure is classified; exact sequence-level "
            "overlap is not inferred from the model card."
        ),
        "pass": missing_records == 0,
    }


# ---------------------------------------------------------------------------
# Exposure ledger coverage
# ---------------------------------------------------------------------------

def check_exposure_ledger_coverage(
    all_manifest_records: Set[str],
    exposure_ledger_ids: Set[str],
) -> Dict[str, Any]:
    """Verify every record in all split manifests appears in the exposure ledger."""
    in_ledger = all_manifest_records & exposure_ledger_ids
    missing = all_manifest_records - exposure_ledger_ids
    coverage = len(in_ledger) / len(all_manifest_records) if all_manifest_records else 1.0
    return {
        "n_manifest_records": len(all_manifest_records),
        "n_in_ledger": len(in_ledger),
        "n_missing": len(missing),
        "coverage": coverage,
        "missing_examples": sorted(missing)[:10],
        "pass": coverage == 1.0 and len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# Per-split audit
# ---------------------------------------------------------------------------

def audit_split_leakage(
    split_type: str,
    manifest_path: str,
    records_by_id: Dict[str, UTREditRecord],
    foundation_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all overlap checks for one split."""
    entries = load_manifest(manifest_path)
    by_role = split_by_role(entries)
    train, test = by_role["train"], by_role["test"]

    checks = []
    for check_fn in (
        check_exact_overlap,
        check_reverse_overlap,
        check_intermediate_overlap,
        check_cluster_overlap,
    ):
        checks.append(check_fn(train, test, records_by_id))

    checks.append(check_scaffold_overlap(train, test))
    study_check = check_study_overlap(train, test)
    # study_overlap is a hard gate ONLY for study_disjoint and
    # cross_region_transfer (where train/test must come from different studies).
    # For *_source_disjoint splits, same-study is expected (the split is
    # within a single study by sequence cluster), so study overlap is
    # informational, not a hard gate.
    if split_type.endswith("source_disjoint"):
        study_check["informational"] = True
        study_check["hard_gate"] = False
    else:
        study_check["informational"] = False
        study_check["hard_gate"] = True
    checks.append(study_check)

    for check_fn in (
        check_gene_overlap,
        check_context_overlap,
        check_barcode_overlap,
    ):
        checks.append(check_fn(train, test, records_by_id))
    foundation_check = check_foundation_overlap(
        train, test, records_by_id, foundation_manifest
    )
    checks.append(foundation_check)

    # Hard-gate channels: exact, reverse, intermediate (endpoint-as-train-inter),
    # cluster, and study (only for study_disjoint / cross_region_transfer).
    # Scaffold is informational. gene/context/barcode are N/A; foundation is
    # a bound exposure audit when an FM0 manifest is supplied.
    hard_gate_channels = {"exact", "reverse", "intermediate", "cluster"}
    hard_pass = all(
        c["pass"] for c in checks
        if c["channel"] in hard_gate_channels and c.get("applicable", True)
    )
    # study hard gate (only where applicable)
    if study_check.get("hard_gate", False) and not study_check["pass"]:
        hard_pass = False
    if foundation_manifest is not None and not foundation_check["pass"]:
        hard_pass = False
    # Acceptance: final endpoint as train intermediate = 0
    inter_check = next(c for c in checks if c["channel"] == "intermediate")
    endpoint_as_train_inter = inter_check["n_endpoint_as_train_intermediate"]

    return {
        "split_type": split_type,
        "manifest_path": manifest_path,
        "n_train": len(train),
        "n_test": len(test),
        "checks": checks,
        "acceptance": {
            "endpoint_as_train_intermediate": endpoint_as_train_inter,
        },
        "hard_gate_pass": hard_pass,
        "pass": hard_pass and endpoint_as_train_inter == 0,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_leakage_audit(
    splits_dir: str,
    canonical_records_path: str,
    exposure_ledger_path: str,
    foundation_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    splits_dir = Path(splits_dir)
    foundation_manifest = (
        load_foundation_manifest(foundation_manifest_path)
        if foundation_manifest_path
        else None
    )

    print("Loading canonical records...")
    records_by_id = load_paired_records_by_id(canonical_records_path)
    print(f"  {len(records_by_id)} paired records")

    print("Loading exposure ledger...")
    ledger_ids = load_exposure_ledger_ids(exposure_ledger_path)
    print(f"  {len(ledger_ids)} ledger entries")

    split_results: Dict[str, Any] = {}
    all_manifest_records: Set[str] = set()
    all_pass = True

    for split_type, filename in SPLIT_FILES.items():
        path = str(splits_dir / filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {split_type}")
            split_results[split_type] = {"pass": False, "error": "manifest not found"}
            all_pass = False
            continue
        print(f"\nAuditing {split_type}...")
        res = audit_split_leakage(
            split_type, path, records_by_id, foundation_manifest
        )
        split_results[split_type] = res
        all_pass = all_pass and res["pass"]
        for e in load_manifest(path):
            all_manifest_records.add(e["record_id"])
        status = "PASS" if res["pass"] else "FAIL"
        print(f"  -> {status}  hard_gate={res['hard_gate_pass']} "
              f"endpoint_as_train_inter={res['acceptance']['endpoint_as_train_intermediate']}")

    print("\nChecking exposure ledger coverage...")
    coverage = check_exposure_ledger_coverage(all_manifest_records, ledger_ids)
    print(f"  coverage={coverage['coverage']:.4f} "
          f"({coverage['n_in_ledger']}/{coverage['n_manifest_records']}) "
          f"missing={coverage['n_missing']}")

    # Overall acceptance
    endpoint_zero = all(
        r.get("acceptance", {}).get("endpoint_as_train_intermediate", 0) == 0
        for r in split_results.values() if "acceptance" in r
    )
    coverage_ok = coverage["pass"]
    foundation_checks = [
        check
        for result in split_results.values()
        for check in result.get("checks", [])
        if check.get("channel") == "foundation"
    ]
    foundation_bound = bool(foundation_manifest_path)
    foundation_ok = (
        foundation_bound
        and len(foundation_checks) == len(split_results)
        and all(
            check.get("applicable") and check.get("pass")
            for check in foundation_checks
        )
    )

    return {
        "task": "B0-03",
        "contract": "utr_editflow_contract_v2",
        "split_audit_results": split_results,
        "exposure_ledger_coverage": coverage,
        "acceptance": {
            "endpoint_as_train_intermediate_must_be_zero": endpoint_zero,
            "exposure_ledger_coverage_must_be_100_percent": coverage_ok,
            "foundation_overlap_audit_bound": foundation_bound,
            "foundation_overlap_audit_pass": foundation_ok if foundation_bound else False,
        },
        "overall_pass": all_pass and endpoint_zero and coverage_ok and (
            foundation_ok if foundation_bound else True
        ),
        "foundation_manifest_binding": {
            "bound": foundation_bound,
            "path": foundation_manifest_path,
            "sha256": foundation_manifest.get("_manifest_sha256") if foundation_manifest else None,
            "status": "AUDITED" if foundation_ok else (
                "PENDING_FM0" if not foundation_bound else "FAILED"
            ),
        },
        "channels": {
            "hard_gate_always": ["exact", "reverse", "intermediate", "cluster"],
            "hard_gate_conditional": ["study (hard gate for study_disjoint + cross_region_transfer only; informational for *_source_disjoint)"],
            "informational": ["scaffold"],
            "not_applicable": ["gene", "context", "barcode"],
            "pending": [] if foundation_bound else ["foundation (pending FM0)"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="B0-03: Comprehensive leakage audit")
    parser.add_argument(
        "--splits-dir", default="data/b0_splits",
    )
    parser.add_argument(
        "--canonical-records", default="data/d1_canonical_records.jsonl",
    )
    parser.add_argument(
        "--exposure-ledger", default="data/data_exposure_ledger.jsonl",
    )
    parser.add_argument(
        "--foundation-manifest",
        default=None,
        help="FM0-bound foundation checkpoint/training-data exposure manifest.",
    )
    parser.add_argument(
        "--output", default="data/b0_03_leakage_audit_report.json",
    )
    args = parser.parse_args()

    report = run_leakage_audit(
        args.splits_dir, args.canonical_records, args.exposure_ledger,
        args.foundation_manifest,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== Leakage audit report written to {out_path} ===")
    print(f"Overall pass: {report['overall_pass']}")
    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
