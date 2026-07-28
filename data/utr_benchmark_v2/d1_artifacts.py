"""Deterministic D1 contract-artifact projections from frozen dataset stores.

The functions in this module are deliberately pure.  The production builder
uses them to write the five D1 contract artifacts and the acceptance validator
uses them again after reopening the frozen per-dataset manifests and stores.
This makes a same-shape replacement report fail unless its content is exactly
the content implied by those frozen inputs.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from data.utr_benchmark_v2.edit_script import (
    apply_edit_script,
    canonicalize_edit_script,
)
from data.utr_benchmark_v2.records import ABSOLUTE_PAIR_TYPES, MEASURED_PAIR_TYPES


LIBRARY_REQUIRED_FIELDS = (
    "library_design",
    "proposal_distribution",
    "source_selection",
    "candidate_selection",
    "positive_negative_balance",
    "edit_type_coverage",
    "position_coverage",
    "gc_coverage",
    "length_coverage",
    "motif_coverage",
    "known_ascertainment_bias",
)

LIBRARY_REQUIRED_AUDITS = (
    "candidate_proposal_distribution",
    "effect_direction_stratification",
    "study_scaffold_holdout",
    "deterministic_condition_permutation",
    "beneficial_only_selection_sensitivity",
    "library_id_shortcut",
)

TRIMER_PANEL = tuple("".join(parts) for parts in itertools.product("ACGU", repeat=3))

REPRODUCTION_COLUMNS = (
    "dataset_id",
    "status",
    "paper_eligible",
    "total_input_rows",
    "accepted_intervention_records",
    "accepted_absolute_records",
    "accepted_input_rows",
    "rejected_rows",
    "roundtrip_passed",
    "roundtrip_total",
    "label_reproduction_status",
    "reason_code",
)


def _blocked(reason_code: str, *, status: str = "BLOCKED") -> dict[str, Any]:
    return {"status": status, "reason_code": reason_code}


def _computed(**payload: Any) -> dict[str, Any]:
    return {"status": "COMPUTED", **payload}


def _documented(value: Any, source: str) -> dict[str, Any]:
    return {"status": "DOCUMENTED", "value": value, "source": source}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _record_effect(record: Mapping[str, Any]) -> float | None:
    delta = _finite(record.get("delta_raw"))
    if delta is not None:
        return delta
    source = _finite(record.get("source_value_raw"))
    candidate = _finite(record.get("candidate_value_raw"))
    if source is None or candidate is None:
        return None
    return candidate - source


def _effect_direction(record: Mapping[str, Any]) -> str:
    effect = _record_effect(record)
    if effect is None:
        return "unknown"
    if effect > 0:
        return "positive"
    if effect < 0:
        return "negative"
    return "zero"


def _count(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _candidate_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(record) for record in result.get("candidate_records", [])]


def _label_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(record) for record in result.get("label_records", [])]


def recompute_roundtrip(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervention = [
        record
        for record in records
        if record.get("pair_type") not in ABSOLUTE_PAIR_TYPES
    ]
    passed = 0
    failures: list[str] = []
    for index, record in enumerate(intervention):
        record_id = str(record.get("record_id", f"index:{index}"))
        try:
            candidate = apply_edit_script(
                str(record["source_sequence"]),
                record["edit_script"],
            )
        except Exception:
            failures.append(record_id)
            continue
        if candidate != record.get("candidate_sequence"):
            failures.append(record_id)
        else:
            passed += 1
    return {
        "intervention_records": len(intervention),
        "roundtrip_passed": passed,
        "fraction": passed / len(intervention) if intervention else None,
        "failure_record_ids": failures,
    }


def recompute_ambiguity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervention = [
        record
        for record in records
        if record.get("pair_type") not in ABSOLUTE_PAIR_TYPES
    ]
    counts: list[int] = []
    scopes: set[str] = set()
    observed = 0
    for record in intervention:
        canonical = canonicalize_edit_script(
            str(record["source_sequence"]),
            str(record["candidate_sequence"]),
        )
        count = canonical["equivalent_minimal_script_count"]
        counts.append(count)
        scope = canonical["count_scope"]
        if scope:
            scopes.add(str(scope))
        observed += bool(record.get("trajectory_observed"))
    return {
        "records": len(intervention),
        "ambiguous_records": sum(count > 1 for count in counts),
        "max_equivalent_minimal_script_count": max(counts, default=0),
        "constructed_paths_marked_observed": observed,
        "count_scopes": sorted(scopes),
        "records_with_quantified_ambiguity": len(counts),
    }


def recompute_action_coverage(
    records: Sequence[Mapping[str, Any]],
    rejected_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measured = [
        record
        for record in records
        if record.get("pair_type") in MEASURED_PAIR_TYPES
        and record.get("pair_type") not in ABSOLUTE_PAIR_TYPES
    ]
    presence = {action: 0 for action in ("SUB", "INS", "DEL")}
    instances = {action: 0 for action in ("SUB", "INS", "DEL")}
    edit_counts: Counter[str] = Counter()
    combinations: Counter[str] = Counter()
    pair_types: Counter[str] = Counter()
    for record in measured:
        actions = [str(action) for action in record.get("edit_types", [])]
        for action in set(actions):
            if action in presence:
                presence[action] += 1
        for action in actions:
            if action in instances:
                instances[action] += 1
        edit_counts[str(record.get("edit_count"))] += 1
        combinations["+".join(sorted(set(actions)))] += 1
        pair_types[str(record.get("pair_type"))] += 1
    indel = [
        record
        for record in measured
        if {"INS", "DEL"} & set(record.get("edit_types", []))
    ]
    missing_reasons = {
        "MISSING_OR_NONFINITE_LABEL",
        "MISSING_PAIRED_PROCESSED_LABEL",
    }
    return {
        "measured_endpoint_pair_records": len(measured),
        "record_counts_by_action_presence": presence,
        "canonical_action_instance_counts": instances,
        "edit_count_distribution": dict(sorted(edit_counts.items())),
        "action_combination_distribution": dict(sorted(combinations.items())),
        "pair_type_counts": dict(sorted(pair_types.items())),
        "indel_endpoint_pairs": len(indel),
        "indel_endpoint_pairs_with_both_finite_values": sum(
            _finite(record.get("source_value_raw")) is not None
            and _finite(record.get("candidate_value_raw")) is not None
            for record in indel
        ),
        "missing_or_nonfinite_label_rejections": sum(
            str(record.get("reason_code")) in missing_reasons
            for record in rejected_records
        ),
        "observed_trajectory_records": sum(
            bool(record.get("trajectory_observed")) for record in measured
        ),
        "interpretation": (
            "action presence is derived from measured source/candidate "
            "endpoints; canonical action order is constructed and was not "
            "observed experimentally"
        ),
    }


def _proposal_distribution(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if not candidates:
        return _blocked(reason)
    return _computed(
        label_free=True,
        candidate_records=len(candidates),
        pair_type_counts=_count(
            [record.get("pair_type", "UNKNOWN") for record in candidates]
        ),
        edit_count_distribution=_count(
            [record.get("edit_count", "UNKNOWN") for record in candidates]
        ),
        action_combination_distribution=_count(
            [
                "+".join(sorted(set(record.get("edit_types", [])))) or "NONE"
                for record in candidates
            ]
        ),
        interpretation="observed_library_proposal_frequency_not_biological_desirability",
    )


def _source_selection(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if not candidates:
        return _blocked(reason)
    source_ids = [
        str(record["source_id"])
        for record in candidates
        if record.get("source_id") is not None
    ]
    source_groups = [
        str(record["source_group"])
        for record in candidates
        if record.get("source_group") is not None
    ]
    return _computed(
        candidate_records=len(candidates),
        records_with_source=len(source_ids),
        absolute_or_unanchored_records=len(candidates) - len(source_ids),
        distinct_source_ids=len(set(source_ids)),
        distinct_source_groups=len(set(source_groups)),
        source_group_missing_records=(
            len(candidates)
            - sum(record.get("source_group") is not None for record in candidates)
        ),
        selection_interpretation="descriptive_ascertainment_only",
    )


def _candidate_selection(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if not candidates:
        return _blocked(reason)
    sequences = [str(record.get("candidate_sequence", "")) for record in candidates]
    sequence_hashes = [
        hashlib.sha256(sequence.encode("utf-8")).hexdigest() for sequence in sequences
    ]
    return _computed(
        candidate_records=len(candidates),
        distinct_candidate_sequence_hashes=len(set(sequence_hashes)),
        duplicate_candidate_sequence_instances=(
            len(sequence_hashes) - len(set(sequence_hashes))
        ),
        labels_used=False,
        selection_interpretation="descriptive_ascertainment_only",
    )


def _direction_balance(
    labels: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if not labels:
        return _blocked(reason)
    counts = Counter(_effect_direction(record) for record in labels)
    finite_total = counts["positive"] + counts["negative"] + counts["zero"]
    if finite_total == 0:
        return _blocked("NO_FINITE_PAIRED_EFFECTS")
    return _computed(
        direction_counts={
            key: counts[key] for key in ("negative", "zero", "positive", "unknown")
        },
        finite_effect_records=finite_total,
        unknown_effect_records=counts["unknown"],
        positive_fraction_among_finite=counts["positive"] / finite_total,
        negative_fraction_among_finite=counts["negative"] / finite_total,
        interpretation="descriptive_effect_direction_stratification_only",
    )


def _edit_type_coverage(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    intervention = [
        record
        for record in candidates
        if record.get("pair_type") not in ABSOLUTE_PAIR_TYPES
    ]
    if not intervention:
        return _blocked(reason)
    return _computed(
        intervention_records=len(intervention),
        record_counts_by_action_presence={
            action: sum(
                action in set(record.get("edit_types", [])) for record in intervention
            )
            for action in ("SUB", "INS", "DEL")
        },
        action_instance_counts={
            action: sum(
                list(record.get("edit_types", [])).count(action)
                for record in intervention
            )
            for action in ("SUB", "INS", "DEL")
        },
    )


def _position_coverage(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    positions: list[int] = []
    normalized_bins: Counter[str] = Counter()
    for record in candidates:
        source_length = record.get("source_length")
        if not isinstance(source_length, int) or source_length <= 0:
            continue
        for position in record.get("edit_positions", []):
            if not isinstance(position, int):
                continue
            positions.append(position)
            bin_index = min(9, max(0, int(10 * position / source_length)))
            normalized_bins[f"{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}"] += 1
    if not positions:
        return _blocked(reason)
    return _computed(
        edit_position_instances=len(positions),
        minimum_position=min(positions),
        maximum_position=max(positions),
        normalized_decile_counts=dict(sorted(normalized_bins.items())),
        coordinate_system="zero_based_dynamic_state_canonical_script",
    )


def _gc_coverage(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    values: list[float] = []
    bins: Counter[str] = Counter()
    for record in candidates:
        sequence = str(record.get("candidate_sequence", ""))
        if not sequence:
            continue
        fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
        values.append(fraction)
        index = min(9, max(0, int(fraction * 10)))
        bins[f"{index / 10:.1f}-{(index + 1) / 10:.1f}"] += 1
    if not values:
        return _blocked(reason)
    return _computed(
        candidate_records=len(values),
        minimum=min(values),
        maximum=max(values),
        median=statistics.median(values),
        decile_counts=dict(sorted(bins.items())),
    )


def _length_coverage(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    source_lengths = [
        int(record["source_length"])
        for record in candidates
        if isinstance(record.get("source_length"), int)
    ]
    candidate_lengths = [
        int(record["candidate_length"])
        for record in candidates
        if isinstance(record.get("candidate_length"), int)
    ]
    if not candidate_lengths:
        return _blocked(reason)
    return _computed(
        candidate_records=len(candidate_lengths),
        candidate_length_distribution=_count(candidate_lengths),
        candidate_length_minimum=min(candidate_lengths),
        candidate_length_maximum=max(candidate_lengths),
        candidate_length_median=statistics.median(candidate_lengths),
        source_length_distribution=_count(source_lengths),
        variable_candidate_length=len(set(candidate_lengths)) > 1,
    )


def _motif_coverage(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    sequences = [
        str(record.get("candidate_sequence", ""))
        for record in candidates
        if record.get("candidate_sequence")
    ]
    if not sequences:
        return _blocked(reason)
    counts = {trimer: 0 for trimer in TRIMER_PANEL}
    for sequence in sequences:
        for index in range(max(0, len(sequence) - 2)):
            trimer = sequence[index : index + 3]
            if trimer in counts:
                counts[trimer] += 1
    present = [trimer for trimer in TRIMER_PANEL if counts[trimer] > 0]
    missing = [trimer for trimer in TRIMER_PANEL if counts[trimer] == 0]
    return _computed(
        panel="all_64_RNA_trimers",
        label_free=True,
        candidate_records=len(sequences),
        trimer_instance_counts=counts,
        present_trimers=present,
        missing_trimers=missing,
        biological_effect_claimed=False,
        interpretation="sequence_coverage_only_not_motif_function",
    )


def _holdout_feasibility(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if not candidates:
        return _blocked(reason)
    studies = {
        str(record.get("study_group"))
        for record in candidates
        if record.get("study_group") is not None
    }
    scaffolds = {
        str(record.get("scaffold_group"))
        for record in candidates
        if record.get("scaffold_group") is not None
    }
    return _computed(
        distinct_study_groups=len(studies),
        distinct_scaffold_groups=len(scaffolds),
        study_holdout_feasible=len(studies) >= 2,
        scaffold_holdout_feasible=len(scaffolds) >= 2,
        infeasibility_reason_codes=sorted(
            reason
            for reason, condition in (
                ("INSUFFICIENT_DISTINCT_STUDY_GROUPS", len(studies) < 2),
                ("INSUFFICIENT_DISTINCT_SCAFFOLD_GROUPS", len(scaffolds) < 2),
            )
            if condition
        ),
        labels_used=False,
    )


def _condition_permutation(
    dataset_id: str,
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    conditions = sorted(
        {
            str(
                record.get("context_group")
                or record.get("context_id")
                or record.get("assay_id")
            )
            for record in candidates
            if record.get("context_group")
            or record.get("context_id")
            or record.get("assay_id")
        }
    )
    if not candidates:
        return _blocked(reason)
    if len(conditions) < 2:
        return _blocked(
            "INSUFFICIENT_DISTINCT_CONDITIONS_FOR_PERMUTATION",
            status="NOT_APPLICABLE",
        )
    permuted = sorted(
        conditions,
        key=lambda condition: hashlib.sha256(
            f"{dataset_id}\x1fcondition_permutation_v1\x1f{condition}".encode("utf-8")
        ).hexdigest(),
    )
    if permuted == conditions:
        permuted = permuted[1:] + permuted[:1]
    mapping = dict(zip(conditions, permuted))
    mapping_sha256 = hashlib.sha256(
        "\n".join(f"{source}\t{target}" for source, target in mapping.items()).encode(
            "utf-8"
        )
    ).hexdigest()
    return _computed(
        algorithm="sha256_rank_then_deterministic_rotation_if_identity",
        algorithm_version="condition_permutation_v1",
        distinct_conditions=len(conditions),
        mapping=mapping,
        mapping_sha256=mapping_sha256,
        fixed_points=sum(source == target for source, target in mapping.items()),
        labels_used=False,
        interpretation="ascertainment_sensitivity_design_only",
    )


def _beneficial_only_sensitivity(
    labels: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    finite = [record for record in labels if _record_effect(record) is not None]
    if not finite:
        return _blocked("NO_FINITE_PAIRED_EFFECTS" if labels else reason)
    positive = [record for record in finite if (_record_effect(record) or 0.0) > 0]
    return _computed(
        finite_effect_records=len(finite),
        beneficial_only_records=len(positive),
        retained_fraction=len(positive) / len(finite),
        all_effect_action_combinations=_count(
            [
                "+".join(sorted(set(record.get("edit_types", [])))) or "NONE"
                for record in finite
            ]
        ),
        beneficial_only_action_combinations=_count(
            [
                "+".join(sorted(set(record.get("edit_types", [])))) or "NONE"
                for record in positive
            ]
        ),
        selection_changes_observed_proposal_distribution=True,
        biological_desirability_claimed=False,
    )


def _library_id_shortcut(
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if not candidates:
        return _blocked(reason)
    identifiers = [
        str(
            record.get("library_batch")
            or record.get("barcode_batch")
            or record.get("scaffold_group")
        )
        for record in candidates
        if record.get("library_batch")
        or record.get("barcode_batch")
        or record.get("scaffold_group")
    ]
    if not identifiers:
        return _blocked("NO_LIBRARY_OR_BATCH_IDENTIFIER_IN_FROZEN_RECORDS")
    counts = _count(identifiers)
    return _computed(
        records_with_library_identifier=len(identifiers),
        records_without_library_identifier=len(candidates) - len(identifiers),
        distinct_library_identifiers=len(counts),
        records_by_library_identifier=counts,
        between_library_shortcut_audit_feasible=len(counts) >= 2,
        reason_codes=(
            []
            if len(counts) >= 2
            else ["SINGLE_LIBRARY_IDENTIFIER_NO_BETWEEN_LIBRARY_CONTRAST"]
        ),
        labels_used=False,
        interpretation="identifier_support_audit_not_predictive_performance",
    )


def build_library_dataset_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id = str(result["dataset_id"])
    candidates = _candidate_records(result)
    labels = _label_records(result)
    status = str(result.get("status", "UNKNOWN"))
    reason = str(
        result.get("reason_code")
        or ("NO_ADMITTED_CANONICAL_RECORDS" if not candidates else "NOT_APPLICABLE")
    )
    policy = result.get("policy", {})
    design = result.get("library_ascertainment") or policy.get("library_ascertainment")
    if design and str(design).upper() not in {"UNKNOWN", "UNRESOLVED", "TBD"}:
        library_design = _documented(
            design,
            "frozen_dataset_manifest.library_ascertainment",
        )
        known_bias = _documented(
            design,
            "frozen_dataset_manifest.library_ascertainment",
        )
    else:
        library_design = _blocked(reason)
        known_bias = _blocked(reason)

    proposal = _proposal_distribution(candidates, reason)
    direction = _direction_balance(labels, reason)
    entry = {
        "status": status,
        "reason_code": result.get("reason_code"),
        "library_design": library_design,
        "proposal_distribution": proposal,
        "source_selection": _source_selection(candidates, reason),
        "candidate_selection": _candidate_selection(candidates, reason),
        "positive_negative_balance": direction,
        "edit_type_coverage": _edit_type_coverage(candidates, reason),
        "position_coverage": _position_coverage(candidates, reason),
        "gc_coverage": _gc_coverage(candidates, reason),
        "length_coverage": _length_coverage(candidates, reason),
        "motif_coverage": _motif_coverage(candidates, reason),
        "known_ascertainment_bias": known_bias,
        "executed_audits": {
            "candidate_proposal_distribution": proposal,
            "effect_direction_stratification": direction,
            "study_scaffold_holdout": _holdout_feasibility(candidates, reason),
            "deterministic_condition_permutation": _condition_permutation(
                dataset_id, candidates, reason
            ),
            "beneficial_only_selection_sensitivity": (
                _beneficial_only_sensitivity(labels, reason)
            ),
            "library_id_shortcut": _library_id_shortcut(candidates, reason),
        },
        "accounting": dict(result.get("accounting", {})),
        "paper_count_reconciliation": result.get("paper_count_reconciliation"),
        "extraction_audit": result.get("extraction_audit"),
        "rejection_reason_counts": _count(
            [
                record.get("reason_code", "UNKNOWN")
                for record in result.get("rejected_records", [])
            ]
        ),
        "absolute_sequences_are_interventions": False,
        "claim_scope": "descriptive_ascertainment_only",
        "biological_desirability_claimed": False,
        "observed_variant_frequency_interpreted_as_biological_desirability": False,
    }
    return entry


def build_required_artifact_payloads(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda result: str(result["dataset_id"]))
    exposure_rows = []
    ambiguity_datasets: dict[str, Any] = {}
    coverage_datasets: dict[str, Any] = {}
    count_scopes: set[str] = set()
    reproduction_rows: list[dict[str, str]] = []
    for result in ordered:
        dataset_id = str(result["dataset_id"])
        policy = result.get("policy", {})
        provenance = result.get("input_provenance", {})
        exposure_rows.append(
            {
                "dataset_id": dataset_id,
                "status": result.get("status"),
                "historical_exposure": policy.get(
                    "historical_exposure",
                    policy.get("role", "UNKNOWN"),
                ),
                "exposure_grade": policy.get(
                    "exposure_grade", "NOT_APPLICABLE_BLOCKED"
                ),
                "allowed_uses": list(result.get("allowed_uses", [])),
                "forbidden_uses": list(result.get("forbidden_uses", [])),
                "read_final_labels": result.get(
                    "read_final_labels", policy.get("read_final_labels", True)
                ),
                "provenance_complete": bool(
                    provenance.get("provenance_audit", {}).get("complete")
                ),
                "reason_code": result.get("reason_code"),
            }
        )
        labels = _label_records(result)
        rejected = [dict(record) for record in result.get("rejected_records", [])]
        ambiguity = recompute_ambiguity(labels)
        ambiguity_datasets[dataset_id] = ambiguity
        count_scopes.update(ambiguity["count_scopes"])
        coverage_datasets[dataset_id] = recompute_action_coverage(labels, rejected)
        accounting = result.get("accounting", {})
        roundtrip = recompute_roundtrip(labels)
        reproduction_rows.append(
            {
                "dataset_id": dataset_id,
                "status": str(result.get("status", "")),
                "paper_eligible": str(bool(result.get("paper_eligible"))).lower(),
                "total_input_rows": str(accounting.get("total_input_rows", "")),
                "accepted_intervention_records": str(
                    accounting.get("accepted_intervention_rows", "")
                ),
                "accepted_absolute_records": str(
                    accounting.get("accepted_absolute_rows", "")
                ),
                "accepted_input_rows": str(accounting.get("accepted_input_rows", "")),
                "rejected_rows": str(accounting.get("rejected_rows", "")),
                "roundtrip_passed": str(roundtrip["roundtrip_passed"]),
                "roundtrip_total": str(roundtrip["intervention_records"]),
                "label_reproduction_status": str(
                    result.get("label_reproduction", {}).get("status", "")
                ),
                "reason_code": str(result.get("reason_code") or ""),
            }
        )

    coverage = {
        "schema_version": "d1_measured_action_coverage_v2",
        "datasets": coverage_datasets,
        "aggregate_record_counts_by_action_presence": {
            action: sum(
                report["record_counts_by_action_presence"][action]
                for report in coverage_datasets.values()
            )
            for action in ("SUB", "INS", "DEL")
        },
        "aggregate_canonical_action_instance_counts": {
            action: sum(
                report["canonical_action_instance_counts"][action]
                for report in coverage_datasets.values()
            )
            for action in ("SUB", "INS", "DEL")
        },
        "measured_insertion_records": sum(
            report["record_counts_by_action_presence"]["INS"]
            for report in coverage_datasets.values()
        ),
        "observed_trajectory_records": sum(
            report["observed_trajectory_records"]
            for report in coverage_datasets.values()
        ),
    }
    return {
        "data/data_exposure_ledger.jsonl": exposure_rows,
        "data/library_ascertainment_report.json": {
            "schema_version": "d1_library_ascertainment_v2",
            "required_dataset_fields": list(LIBRARY_REQUIRED_FIELDS),
            "required_executed_audits": list(LIBRARY_REQUIRED_AUDITS),
            "claim_scope": "descriptive_ascertainment_only",
            "biological_desirability_claimed": False,
            "datasets": {
                str(result["dataset_id"]): build_library_dataset_entry(result)
                for result in ordered
            },
        },
        "data/edit_script_ambiguity_report.json": {
            "schema_version": "d1_edit_script_ambiguity_v2",
            "count_scope": sorted(count_scopes),
            "count_scope_definition": (
                "equivalent_minimal_script_count counts minimum-cost character "
                "alignments only; it does not claim observed edit paths"
            ),
            "datasets": ambiguity_datasets,
            "constructed_paths_marked_observed": sum(
                report["constructed_paths_marked_observed"]
                for report in ambiguity_datasets.values()
            ),
        },
        "data/measured_action_coverage_report.json": coverage,
        "reports/data_reproduction/summary.csv": reproduction_rows,
    }
