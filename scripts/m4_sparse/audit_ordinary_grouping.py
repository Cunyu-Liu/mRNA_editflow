"""R1 ordinary-only grouping and ranking-eligibility audit.

This audit deliberately does not qualify exact source-sequence matches as
biological source groups.  It reports them as *provisional reconstruction
candidates* alongside missing provenance atoms, study dominance, edit-budget
coverage, and legacy endpoint collisions.  No sealed-final input is accepted.

Example (stdout only, no artifact write)::

    python -m scripts.m4_sparse.audit_ordinary_grouping \
        --dataset artifacts/b0x/effect_dataset.jsonl

To persist a new report, pass ``--out``.  The path is opened in exclusive
creation mode and therefore cannot overwrite prior evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Support both ``python -m scripts...`` and the documented repo-root direct
# invocation ``python scripts/m4_sparse/audit_ordinary_grouping.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.m4_sparse.o0x_search import candidate_identity
from scripts.m4_sparse.dataset import (
    IncompatibleEditEncodingError,
    encode_pair,
)
from scripts.m4_sparse.ordinary_io import (
    assert_ordinary_path,
    assert_ordinary_rows,
    load_ordinary_jsonl,
)


AUDIT_VERSION = "r1_ordinary_grouping_audit_v3"
PRIMARY_EDIT_BUDGETS = {1, 3, 5}
EXPLORATORY_EDIT_BUDGETS = {10}
GROUPING_ATOMS = (
    "biological_parent_group",
    "gene_group",
    "tile_family_group",
    "sequence_cluster_group",
    "design_relation_group_id",
    "context_id",
    "assay_context",
    "source_sequence_id",
    "transcript_id",
)


def _text(value) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_sequence(value) -> str:
    return _text(value).upper().replace("T", "U")


def _context(row: dict) -> str:
    return _text(row.get("context_id") or row.get("assay_context"))


def _context_status(row: dict) -> Tuple[Optional[str], Optional[str]]:
    """Return a context only when aliases are present and non-conflicting."""
    context_id = _text(row.get("context_id"))
    assay_context = _text(row.get("assay_context"))
    if not context_id and not assay_context:
        return None, "context"
    if context_id and assay_context and context_id != assay_context:
        return None, "conflicting_context"
    return context_id or assay_context, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_ordinary_input_path(path: Path) -> None:
    assert_ordinary_path(path, purpose="audit input")


def load_ordinary_rows(path: Path) -> List[dict]:
    return load_ordinary_jsonl(path, require_delta=False)


def _edit_count(row: dict) -> Optional[int]:
    value = row.get("edit_count")
    if value is None and isinstance(row.get("edit_list"), list):
        value = len(row["edit_list"])
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _source_sequence_key(row: dict) -> Optional[Tuple[str, str, str, str, str]]:
    source_sequence = _normalize_sequence(row.get("source_sequence"))
    benchmark = _text(row.get("benchmark"))
    study = _text(row.get("study"))
    endpoint = _text(row.get("endpoint"))
    context, context_error = _context_status(row)
    if (
        not source_sequence
        or not benchmark
        or not study
        or not endpoint
        or context_error is not None
    ):
        return None
    return benchmark, study, endpoint, context, source_sequence


def _source_sequence_key_missing_atoms(row: dict) -> Tuple[str, ...]:
    missing = []
    if not _normalize_sequence(row.get("source_sequence")):
        missing.append("source_sequence")
    if not _text(row.get("benchmark")):
        missing.append("benchmark")
    if not _text(row.get("study")):
        missing.append("study")
    if not _text(row.get("endpoint")):
        missing.append("endpoint")
    _, context_error = _context_status(row)
    if context_error is not None:
        missing.append(context_error)
    return tuple(missing)


def _source_value_signature(row: dict) -> Optional[float]:
    value = row.get("source_value")
    if value is None:
        return None
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def _source_sequence_group_report(rows: Sequence[dict]) -> dict:
    groups = defaultdict(list)
    missing_key_rows = 0
    missing_source_sequence_rows = 0
    missing_key_atom_counts = Counter()
    for row in rows:
        key = _source_sequence_key(row)
        if key is None:
            missing_key_rows += 1
            missing_atoms = _source_sequence_key_missing_atoms(row)
            missing_key_atom_counts.update(missing_atoms)
            missing_source_sequence_rows += int("source_sequence" in missing_atoms)
            continue
        groups[key].append(row)

    size_histogram = Counter()
    distinct_pool_rows = 0
    multi_candidate_groups = 0
    max_candidates = 0
    inconsistent_source_value_groups = 0
    missing_or_nonfinite_source_value_groups = 0
    groups_with_multiple_source_ids = 0
    for group_rows in groups.values():
        identities = {
            identity
            for identity in (candidate_identity(row) for row in group_rows)
            if identity is not None
        }
        n_candidates = len(identities)
        size_histogram[n_candidates] += 1
        max_candidates = max(max_candidates, n_candidates)
        if n_candidates >= 2:
            multi_candidate_groups += 1
            distinct_pool_rows += len(group_rows)
        source_value_signatures = [
            _source_value_signature(row) for row in group_rows
        ]
        source_values = {
            value for value in source_value_signatures if value is not None
        }
        if any(value is None for value in source_value_signatures):
            missing_or_nonfinite_source_value_groups += 1
        if len(source_values) > 1:
            inconsistent_source_value_groups += 1
        source_ids = {_text(row.get("source_id")) for row in group_rows if _text(row.get("source_id"))}
        if len(source_ids) > 1:
            groups_with_multiple_source_ids += 1

    return {
        "group_key_definition": (
            "benchmark|study|endpoint|context_id_or_assay_context|"
            "normalized_source_sequence"
        ),
        "qualification_status": "PROVISIONAL_REQUIRES_BIOLOGICAL_PROVENANCE",
        "warning": (
            "Exact source-sequence equality can conflate unrelated loci, "
            "constructs, transcripts, or design families; it is not by itself "
            "a publication-grade grouping atom."
        ),
        "n_groups": len(groups),
        "n_rows_missing_safe_group_key": missing_key_rows,
        "n_rows_missing_source_sequence": missing_source_sequence_rows,
        "missing_group_key_atom_counts": dict(sorted(missing_key_atom_counts.items())),
        "n_groups_with_at_least_2_distinct_candidates": multi_candidate_groups,
        "n_rows_in_groups_with_at_least_2_distinct_candidates": distinct_pool_rows,
        "max_distinct_candidates": max_candidates,
        "distinct_candidate_count_histogram": {
            str(size): count for size, count in sorted(size_histogram.items())
        },
        "n_groups_with_inconsistent_source_value": inconsistent_source_value_groups,
        "n_groups_with_missing_or_nonfinite_source_value": (
            missing_or_nonfinite_source_value_groups
        ),
        "n_groups_with_multiple_mechanical_source_ids": groups_with_multiple_source_ids,
    }


def _r2_incompatibility_code(message: str) -> str:
    rules = (
        ("non-empty source and candidate", "EMPTY_SEQUENCE"),
        ("equal-length sequences only", "UNEQUAL_LENGTH"),
        ("supports SUB edits only", "NON_SUB_EDIT"),
        ("is not an object", "INVALID_EDIT_OBJECT"),
        ("invalid position", "INVALID_EDIT_POSITION"),
        ("has no position", "MISSING_EDIT_POSITION"),
        ("outside source length", "EDIT_POSITION_OUT_OF_RANGE"),
        ("repeats source position", "DUPLICATE_EDIT_POSITION"),
        ("non-ACGU target token", "INVALID_TARGET_TOKEN"),
        ("disagrees with candidate sequence", "TARGET_SEQUENCE_MISMATCH"),
        ("contains a no-op SUB", "NOOP_SUB"),
        ("does not exactly match", "INCOMPLETE_EDIT_SCRIPT"),
        ("cannot represent edit span", "EDIT_SPAN_EXCEEDS_SINGLE_WINDOW"),
    )
    for fragment, code in rules:
        if fragment in message:
            return code
    return "UNCLASSIFIED_R2_ENCODING_ERROR"


def _r2_input_coverage(rows: Sequence[dict]) -> dict:
    """Aggregate prospective R2 encoder coverage without retaining row data."""
    reason_counts = Counter()
    benchmark_totals = Counter()
    benchmark_compatible = Counter()
    study_totals = Counter()
    study_compatible = Counter()
    benchmark_reasons = defaultdict(Counter)
    study_reasons = defaultdict(Counter)
    compatible = 0
    compatible_zero_edit = 0
    zero_edit_abs_delta = []
    for row in rows:
        benchmark = _text(row.get("benchmark")) or "__MISSING_BENCHMARK__"
        study = _text(row.get("study")) or "__MISSING_STUDY__"
        benchmark_totals[benchmark] += 1
        study_totals[study] += 1
        try:
            encode_pair(
                row.get("source_sequence"), row.get("candidate_sequence"),
                row.get("edit_list"),
            )
        except IncompatibleEditEncodingError as exc:
            reason = _r2_incompatibility_code(str(exc))
            reason_counts[reason] += 1
            benchmark_reasons[benchmark][reason] += 1
            study_reasons[study][reason] += 1
            continue
        compatible += 1
        benchmark_compatible[benchmark] += 1
        study_compatible[study] += 1
        if not (row.get("edit_list") or []):
            compatible_zero_edit += 1
            zero_edit_abs_delta.append(abs(float(row["delta"])))
    total = len(rows)

    def _stratified_summary(totals, compatible_counts, reasons):
        summary = {}
        for key in sorted(totals):
            n_total = int(totals[key])
            n_compatible = int(compatible_counts[key])
            summary[key] = {
                "n_records": n_total,
                "n_compatible": n_compatible,
                "n_incompatible": n_total - n_compatible,
                "compatible_fraction": n_compatible / n_total if n_total else None,
                "incompatibility_reason_counts": dict(
                    sorted(reasons[key].items())),
            }
        return summary

    return {
        "input_schema_version": (
            "shared_edit_centered_window_with_padding_masks_v1"),
        "status": (
            "COMPATIBLE" if compatible == total
            else "PARTIAL_COVERAGE_REQUIRES_EXPLICIT_EXCLUSION_OR_NEW_ENCODER"),
        "n_records": total,
        "n_compatible": compatible,
        "n_incompatible": total - compatible,
        "compatible_fraction": compatible / total if total else None,
        "incompatibility_reason_counts": dict(sorted(reason_counts.items())),
        "n_compatible_zero_edit_controls": compatible_zero_edit,
        "zero_edit_control_abs_delta": {
            "mean": (float(np.mean(zero_edit_abs_delta))
                     if zero_edit_abs_delta else None),
            "median": (float(np.median(zero_edit_abs_delta))
                       if zero_edit_abs_delta else None),
        },
        "by_benchmark": _stratified_summary(
            benchmark_totals, benchmark_compatible, benchmark_reasons),
        "by_study": _stratified_summary(
            study_totals, study_compatible, study_reasons),
        "policy": (
            "R2 training must bind an explicit inclusion manifest; incompatible "
            "rows cannot be silently dropped, and zero-edit controls must be "
            "reported separately from editable candidates."),
    }


def _study_dominance(rows: Sequence[dict]) -> dict:
    by_benchmark = defaultdict(Counter)
    for row in rows:
        benchmark = _text(row.get("benchmark")) or "__MISSING_BENCHMARK__"
        study = _text(row.get("study")) or "__MISSING_STUDY__"
        by_benchmark[benchmark][study] += 1

    report = {}
    for benchmark, counts in sorted(by_benchmark.items()):
        total = sum(counts.values())
        shares = {
            study: count / total for study, count in sorted(counts.items())
        } if total else {}
        dominant_study = max(counts, key=counts.get) if counts else None
        max_share = shares.get(dominant_study) if dominant_study is not None else None
        report[benchmark] = {
            "n_records": total,
            "study_counts": dict(sorted(counts.items())),
            "study_shares": shares,
            "dominant_study": dominant_study,
            "dominant_study_share": max_share,
            "reference_max_share": 0.70,
            "exceeds_reference_max_share": (
                bool(max_share > 0.70) if max_share is not None else None
            ),
        }
    return report


def _edit_budget_report(rows: Sequence[dict]) -> dict:
    counts = Counter()
    raw_counts = Counter()
    for row in rows:
        count = _edit_count(row)
        raw_counts["MISSING" if count is None else str(count)] += 1
        if count is None:
            counts["missing"] += 1
            continue
        if count > max(PRIMARY_EDIT_BUDGETS):
            counts["above_primary_max"] += 1
        if count in PRIMARY_EDIT_BUDGETS:
            counts["primary_exact"] += 1
        elif count in EXPLORATORY_EDIT_BUDGETS:
            counts["exploratory_exact"] += 1
        else:
            counts["outside_declared_exact_budgets"] += 1
    return {
        "primary_exact_budgets": sorted(PRIMARY_EDIT_BUDGETS),
        "exploratory_exact_budgets": sorted(EXPLORATORY_EDIT_BUDGETS),
        "classification_counts": dict(sorted(counts.items())),
        "edit_count_histogram": dict(sorted(
            raw_counts.items(),
            key=lambda item: (item[0] == "MISSING", int(item[0]) if item[0].isdigit() else 0),
        )),
    }


def _missing_grouping_atoms(rows: Sequence[dict]) -> dict:
    n = len(rows)
    report = {}
    for field in GROUPING_ATOMS:
        missing = sum(not _text(row.get(field)) for row in rows)
        report[field] = {
            "missing_count": missing,
            "missing_fraction": (missing / n if n else None),
        }
    return report


def _legacy_source_pool_report(rows: Sequence[dict]) -> dict:
    by_legacy_source = defaultdict(list)
    for row in rows:
        source_id = _text(row.get("source_id"))
        if not source_id:
            continue
        by_legacy_source[(
            _text(row.get("benchmark")),
            _text(row.get("study")),
            source_id,
        )].append(row)

    endpoint_mixed = 0
    duplicate_only_mixed = 0
    for group_rows in by_legacy_source.values():
        contexts = {
            (_text(row.get("endpoint")), _context(row)) for row in group_rows
        }
        if len(contexts) <= 1:
            continue
        endpoint_mixed += 1
        identities = {
            identity
            for identity in (candidate_identity(row) for row in group_rows)
            if identity is not None
        }
        if len(identities) <= 1:
            duplicate_only_mixed += 1
    return {
        "historical_key_definition": "benchmark|study|source_id",
        "n_legacy_source_groups": len(by_legacy_source),
        "n_groups_spanning_multiple_endpoint_contexts": endpoint_mixed,
        "n_endpoint_mixed_groups_with_at_most_1_distinct_candidate": duplicate_only_mixed,
    }


def audit_rows(rows: Sequence[dict], source: str = "in_memory") -> dict:
    """Build a deterministic ordinary-only audit from already-loaded rows."""
    assert_ordinary_rows(rows, purpose="grouping audit")

    delta_defined = [row for row in rows if row.get("delta") is not None]
    benchmark_counts = Counter(
        _text(row.get("benchmark")) or "__MISSING_BENCHMARK__"
        for row in delta_defined
    )
    study_counts = Counter(
        _text(row.get("study")) or "__MISSING_STUDY__"
        for row in delta_defined
    )
    endpoint_counts = Counter(
        _text(row.get("endpoint")) or "__MISSING_ENDPOINT__"
        for row in delta_defined
    )
    return {
        "audit_version": AUDIT_VERSION,
        "status": "ORDINARY_ONLY_GROUPING_DIAGNOSTIC",
        "source": source,
        "sealed_rows_seen": 0,
        "publication_grouping_qualified": False,
        "n_records": len(rows),
        "n_delta_defined_records": len(delta_defined),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "study_counts": dict(sorted(study_counts.items())),
        "endpoint_counts": dict(sorted(endpoint_counts.items())),
        "source_sequence_candidate_groups": _source_sequence_group_report(delta_defined),
        "r2_input_coverage": _r2_input_coverage(delta_defined),
        "legacy_source_id_pool_risk": _legacy_source_pool_report(delta_defined),
        "study_dominance": _study_dominance(delta_defined),
        "edit_budget": _edit_budget_report(delta_defined),
        "missing_grouping_atoms": _missing_grouping_atoms(delta_defined),
        "required_next_step": (
            "Bind provisional exact-source groups to assay/context, biological "
            "parent, locus/transcript/design family and sequence-cluster "
            "provenance before using them as O0/G1 headline pools."
        ),
    }


def write_json_exclusive(path: Path, payload: dict) -> None:
    """Persist a new audit without overwriting an earlier report."""
    assert_ordinary_path(path, purpose="audit output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_ordinary_rows(args.dataset)
    report = audit_rows(rows, source=str(args.dataset))
    report["source_sha256"] = sha256_file(args.dataset)
    if args.out is not None:
        write_json_exclusive(args.out, report)
        print("wrote", args.out)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
