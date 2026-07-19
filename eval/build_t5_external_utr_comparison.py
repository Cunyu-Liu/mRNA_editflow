"""Build the T5 5'UTR comparison for MEF, local search, and UTRGAN.

The report separates three different evidence types:

* the current MEF full-length constrained model result;
* a 5'UTR-only oracle-guided local-search ceiling with edit budget three; and
* measured official UTRGAN outputs scored by the same offline proxy oracle.

The local-search and UTRGAN rows can form a descriptive, fixed-region table.
They are not a model-only head-to-head: local search directly optimizes the
evaluation oracle, while UTRGAN is de novo and is not conditioned on each
source row. Consequently this module never enables a superiority claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from glob import glob
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.metrics import edit_distance
from mrna_editflow.eval.oracle import score_utr
from mrna_editflow.eval.run_eval import bootstrap_ci, paired_permutation_pvalue


CLAIM_POLICY = (
    "This artifact supports a descriptive T5 5'UTR comparison under a shared "
    "offline proxy oracle. The MEF local-search row is oracle-guided rather "
    "than model-only, and UTRGAN candidates are de novo rather than "
    "source-conditioned. Do not use this table to claim MEF or UTRGAN "
    "superiority."
)

DEFAULT_PATHS = {
    "input_pack": "benchmark/external_sota/input_pack_t5_head1024/summary.json",
    "utr_inputs": "benchmark/external_sota/input_pack_t5_head1024/utr5_inputs.jsonl",
    "real_run_audit": "docs/external_sota_real_run_audit.json",
    "utrgan_summary": "benchmark/external_sota/real_runs_t5_head1024/UTRGAN/summary.json",
    "utrgan_outputs": "benchmark/external_sota/real_runs_t5_head1024/UTRGAN/utr5_outputs.jsonl",
    "utrgan_paper10000_summary": "benchmark/external_sota/real_runs_t5_head1024/UTRGAN_paper10000/summary.json",
    "utrgan_paper10000_outputs": "benchmark/external_sota/real_runs_t5_head1024/UTRGAN_paper10000/utr5_outputs.jsonl",
    "utailor_summary": "benchmark/external_sota/real_runs_t5_head1024/UTailoR/summary.json",
    "utailor_outputs": "benchmark/external_sota/real_runs_t5_head1024/UTailoR/utr5_outputs.jsonl",
    "mef_model_summary": "benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json",
    "mef_model_seed_glob": "benchmark/multiseed_t5_public_head1024_mo_pareto_top64/seed_*/eval_summary.json",
    "mef_shared_sources": "benchmark/multiseed_t5_public_head1024_sources.jsonl",
    "mef_utr5_model_summary": "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/multiseed_summary.json",
    "mef_utr5_model_seed_glob": "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/**/seed_*/eval_summary.json",
    "mef_utr5_model_candidate_glob": "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/**/seed_*/candidates.jsonl",
    "mef_utr5_model_sources": "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/sources.jsonl",
    "mef_utr_teacher_summary": "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/multiseed_summary.json",
    "mef_utr_teacher_seed_glob": "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/**/seed_*/eval_summary.json",
    "mef_utr_teacher_candidate_glob": "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/**/seed_*/candidates.jsonl",
    "mef_utr_teacher_sources": "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/sources.jsonl",
    "mef_sequential_utr_teacher_summary": "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/multiseed_summary.json",
    "mef_sequential_utr_teacher_seed_glob": "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/**/seed_*/eval_summary.json",
    "mef_sequential_utr_teacher_candidate_glob": "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/**/seed_*/candidates.jsonl",
    "mef_sequential_utr_teacher_sources": "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/sources.jsonl",
    "mef_utailor_budget5_summary": "benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64/multiseed_summary.json",
    "mef_utailor_budget5_seed_glob": "benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64/**/seed_*/eval_summary.json",
    "mef_utailor_budget5_candidate_glob": "benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64/**/seed_*/candidates.jsonl",
    "mef_utailor_budget5_sources": "benchmark/utailor_strict_25_100_sources.jsonl",
    "utailor_strict_subset_summary": "benchmark/utailor_strict_25_100_sources.summary.json",
    "mef_utr_local_search": "benchmark/utr_local_search_head1024.json",
}


def _load_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain an object")
            rows.append(dict(payload))
    return rows


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _num(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Sequence[object]) -> Optional[float]:
    finite = [value for value in (_num(item) for item in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _nested(payload: Mapping[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_rna(value: object) -> str:
    return "".join(str(value or "").upper().replace("T", "U").split())


def _feature_row(utr: str, cds: str) -> dict[str, float]:
    score = score_utr(utr, cds[:12])
    features = _mapping(score.get("features"))
    return {
        "te_proxy": float(score["ensemble_te"]),
        "uaug_count": float(features.get("uaug_count", 0.0)),
        "kozak_score": float(features.get("kozak", 0.0)),
        "start_accessibility_proxy": float(
            features.get("start_accessibility", 0.0)
        ),
    }


def _model_seed_mean(
    seed_payloads: Sequence[Mapping[str, object]],
    *keys: str,
) -> Optional[float]:
    return _mean([_nested(payload, *keys) for payload in seed_payloads])


def _native_row(input_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scored = [
        _feature_row(
            _normalise_rna(row.get("native_five_utr")),
            _normalise_rna(row.get("fixed_cds_context")),
        )
        for row in input_rows
    ]
    utrs = [_normalise_rna(row.get("native_five_utr")) for row in input_rows]
    return {
        "method": "native_source",
        "status": "measured_reference" if input_rows else "missing",
        "evidence_type": "input_reference",
        "n": len(input_rows),
        "mean_te_proxy": _mean([row["te_proxy"] for row in scored]),
        "mean_te_proxy_delta_vs_native": 0.0 if input_rows else None,
        "mean_uaug_count": _mean([row["uaug_count"] for row in scored]),
        "mean_kozak_score": _mean([row["kozak_score"] for row in scored]),
        "mean_start_accessibility_proxy": _mean(
            [row["start_accessibility_proxy"] for row in scored]
        ),
        "mean_utr_edit_distance_vs_native": 0.0 if input_rows else None,
        "mean_normalized_utr_edit_distance_vs_native": 0.0 if input_rows else None,
        "mean_full_transcript_edit_distance_vs_native": 0.0 if input_rows else None,
        "mean_utr_length": _mean([len(utr) for utr in utrs]),
        "mean_utr_length_delta": 0.0 if input_rows else None,
        "exact_native_utr_match_fraction": 1.0 if input_rows else None,
        "unique_designed_utr_fraction": (
            float(len(set(utrs)) / len(utrs)) if utrs else None
        ),
        "cds_unchanged_fraction": 1.0 if input_rows else None,
        "three_utr_unchanged_fraction": 1.0 if input_rows else None,
        "protein_identity_exact_1_fraction": 1.0 if input_rows else None,
        "within_edit_budget_fraction": 1.0 if input_rows else None,
        "mean_wall_clock_s": None,
        "utr5_only_protocol": True,
        "model_only": False,
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
    }


def _mef_model_row(
    summary: Optional[Mapping[str, object]],
    seed_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    aggregate = _mapping(summary.get("aggregate") if summary else {})
    protein_identity = _num(
        _nested(aggregate, "mean_protein_identity", "mean")
    )
    within_budget = _num(_nested(aggregate, "within_budget_fraction", "mean"))
    legal = _num(_nested(aggregate, "legal_fraction", "mean"))
    return {
        "method": "MEF_full_length_mo_pareto_top64",
        "status": "measured_internal_model_context" if summary else "missing",
        "evidence_type": "ten_seed_full_length_model_context",
        "n": int(_nested(seed_payloads[0], "n_candidates")) if seed_payloads else None,
        "n_seeds": int(_nested(aggregate, "mean_oracle_te", "n") or 0),
        "mean_te_proxy": _num(_nested(aggregate, "mean_oracle_te", "mean")),
        "mean_te_proxy_delta_vs_native": _num(
            _nested(aggregate, "delta_oracle_te_vs_source", "mean")
        ),
        "mean_uaug_count": _model_seed_mean(
            seed_payloads, "metrics", "kozak_uaug", "mean_uaug_count"
        ),
        "mean_kozak_score": _model_seed_mean(
            seed_payloads, "metrics", "kozak_uaug", "mean_kozak_score"
        ),
        "mean_start_accessibility_proxy": _model_seed_mean(
            seed_payloads, "metrics", "structure", "mean_start_accessibility"
        ),
        "mean_utr_edit_distance_vs_native": None,
        "mean_normalized_utr_edit_distance_vs_native": None,
        "mean_full_transcript_edit_distance_vs_native": _num(
            _nested(aggregate, "mean_edit_distance", "mean")
        ),
        "mean_utr_length": None,
        "mean_utr_length_delta": None,
        "exact_native_utr_match_fraction": None,
        "unique_designed_utr_fraction": None,
        "cds_unchanged_fraction": 1.0 if protein_identity == 1.0 else None,
        "three_utr_unchanged_fraction": None,
        "protein_identity_exact_1_fraction": protein_identity,
        "within_edit_budget_fraction": within_budget,
        "legal_fraction": legal,
        "mean_wall_clock_s": None,
        "utr5_only_protocol": False,
        "model_only": True,
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
    }


def _mef_utr5_model_row(
    summary: Optional[Mapping[str, object]],
    seed_payloads: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    candidate_paths: Sequence[str],
    method: str = "MEF_region_adapter_utr5only_top64",
    evidence_type: str = "ten_seed_utr5_only_model",
    expected_n: int = 1024,
    measured_status: str = "measured_internal_model_10seed_utr5only",
) -> tuple[dict[str, object], list[str]]:
    aggregate = _mapping(summary.get("aggregate") if summary else {})
    config = _mapping(summary.get("config") if summary else {})
    failures: list[str] = []
    candidate_sets = [_load_jsonl(path) for path in candidate_paths]
    expected_ids = [str(row.get("transcript_id")) for row in source_rows]
    fixed_cds: list[float] = []
    fixed_three_utr: list[float] = []
    utr_lengths: list[float] = []
    utr_length_deltas: list[float] = []
    for seed_index, candidates in enumerate(candidate_sets):
        if len(candidates) != len(source_rows):
            failures.append(f"seed_{seed_index}_coverage_mismatch")
            continue
        for source, candidate, expected_id in zip(
            source_rows,
            candidates,
            expected_ids,
        ):
            candidate_id = str(candidate.get("transcript_id", ""))
            if not candidate_id.startswith(expected_id):
                failures.append(f"seed_{seed_index}_transcript_order_mismatch")
            fixed_cds.append(
                float(
                    _normalise_rna(candidate.get("cds"))
                    == _normalise_rna(source.get("cds"))
                )
            )
            fixed_three_utr.append(
                float(
                    _normalise_rna(candidate.get("three_utr"))
                    == _normalise_rna(source.get("three_utr"))
                )
            )
            candidate_utr = _normalise_rna(candidate.get("five_utr"))
            source_utr = _normalise_rna(source.get("five_utr"))
            utr_lengths.append(float(len(candidate_utr)))
            utr_length_deltas.append(float(len(candidate_utr) - len(source_utr)))

    protein_identity = _num(
        _nested(aggregate, "mean_protein_identity", "mean")
    )
    within_budget = _num(_nested(aggregate, "within_budget_fraction", "mean"))
    legal = _num(_nested(aggregate, "legal_fraction", "mean"))
    per_seed = summary.get("per_seed", []) if summary else []
    per_seed = per_seed if isinstance(per_seed, list) else []
    delta_by_seed = [
        float(value)
        for row in per_seed
        if isinstance(row, Mapping)
        and isinstance(row.get("metrics"), Mapping)
        and (
            value := _num(
                _mapping(row.get("metrics")).get(
                    "delta_oracle_te_vs_source"
                )
            )
        )
        is not None
    ]
    seed_p = (
        paired_permutation_pvalue(
            delta_by_seed,
            [0.0] * len(delta_by_seed),
            seed=0,
            n_permutations=2000,
        )
        if len(delta_by_seed) == 10
        else None
    )
    mean_delta = _num(
        _nested(aggregate, "delta_oracle_te_vs_source", "mean")
    )
    if mean_delta is None or seed_p is None:
        delta_signal = "missing"
    elif seed_p < 0.05 and mean_delta > 0:
        delta_signal = "significant_positive"
    elif seed_p < 0.05 and mean_delta < 0:
        delta_signal = "significant_negative"
    elif mean_delta > 0:
        delta_signal = "positive_not_significant"
    elif mean_delta < 0:
        delta_signal = "negative_not_significant"
    else:
        delta_signal = "neutral"
    cds_fixed = _mean(fixed_cds)
    three_utr_fixed = _mean(fixed_three_utr)
    editable_regions = config.get("editable_regions")
    region_contract_ok = editable_regions == ["utr5"]
    if summary and not region_contract_ok:
        failures.append("editable_regions_not_utr5_only")
    complete = bool(
        summary
        and len(source_rows) == expected_n
        and len(seed_payloads) == 10
        and len(candidate_sets) == 10
        and all(len(rows) == expected_n for rows in candidate_sets)
        and region_contract_ok
        and legal == 1.0
        and protein_identity == 1.0
        and within_budget == 1.0
        and cds_fixed == 1.0
        and three_utr_fixed == 1.0
        and not failures
    )
    return {
        "method": method,
        "status": (
            measured_status if complete else "missing_or_invalid"
        ),
        "evidence_type": evidence_type,
        "n": len(source_rows) if source_rows else None,
        "n_seeds": int(_nested(aggregate, "mean_oracle_te", "n") or 0),
        "mean_te_proxy": _num(_nested(aggregate, "mean_oracle_te", "mean")),
        "mean_te_proxy_delta_vs_native": mean_delta,
        "te_proxy_delta_seed_ci_low": _num(
            _nested(aggregate, "delta_oracle_te_vs_source", "low")
        ),
        "te_proxy_delta_seed_ci_high": _num(
            _nested(aggregate, "delta_oracle_te_vs_source", "high")
        ),
        "te_proxy_delta_seed_paired_p_vs_source": seed_p,
        "te_proxy_delta_seed_signal": delta_signal,
        "te_proxy_delta_per_seed": delta_by_seed,
        "mean_uaug_count": _model_seed_mean(
            seed_payloads, "metrics", "kozak_uaug", "mean_uaug_count"
        ),
        "mean_kozak_score": _model_seed_mean(
            seed_payloads, "metrics", "kozak_uaug", "mean_kozak_score"
        ),
        "mean_start_accessibility_proxy": _model_seed_mean(
            seed_payloads, "metrics", "structure", "mean_start_accessibility"
        ),
        "mean_utr_edit_distance_vs_native": _num(
            _nested(aggregate, "mean_edit_distance", "mean")
        ),
        "mean_normalized_utr_edit_distance_vs_native": None,
        "mean_full_transcript_edit_distance_vs_native": _num(
            _nested(aggregate, "mean_edit_distance", "mean")
        ),
        "mean_utr_length": _mean(utr_lengths),
        "mean_utr_length_delta": _mean(utr_length_deltas),
        "exact_native_utr_match_fraction": _num(
            _nested(aggregate, "exact_source_match_fraction", "mean")
        ),
        "unique_designed_utr_fraction": _model_seed_mean(
            seed_payloads, "metrics", "diversity_novelty", "unique_fraction"
        ),
        "cds_unchanged_fraction": cds_fixed,
        "three_utr_unchanged_fraction": three_utr_fixed,
        "protein_identity_exact_1_fraction": protein_identity,
        "within_edit_budget_fraction": within_budget,
        "legal_fraction": legal,
        "edit_budget": config.get("edit_budget"),
        "mean_wall_clock_s": None,
        "utr5_only_protocol": region_contract_ok,
        "model_only": True,
        "source_conditioned": True,
        "n_candidate_rows_audited": sum(len(rows) for rows in candidate_sets),
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
    }, sorted(set(failures))


def _local_search_row(
    payload: Optional[Mapping[str, object]],
    input_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, float]], list[str]]:
    per_record = payload.get("per_record", []) if payload else []
    per_record = per_record if isinstance(per_record, list) else []
    scored: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    designed_utrs: list[str] = []
    for item in per_record:
        if not isinstance(item, Mapping):
            failures.append("non_mapping_per_record_row")
            continue
        transcript_id = str(item.get("transcript_id"))
        source = input_by_id.get(transcript_id)
        if source is None:
            failures.append(f"unknown_transcript_id:{transcript_id}")
            continue
        source_utr = _normalise_rna(item.get("source_five_utr"))
        expected_source = _normalise_rna(source.get("native_five_utr"))
        if source_utr != expected_source:
            failures.append(f"source_utr_mismatch:{transcript_id}")
        designed_utr = _normalise_rna(item.get("optimized_five_utr"))
        candidate = _feature_row(
            designed_utr,
            _normalise_rna(source.get("fixed_cds_context")),
        )
        candidate.update(
            {
                "te_proxy_delta_vs_native": float(item.get("delta_te", 0.0)),
                "utr_edit_distance_vs_native": float(
                    item.get("utr_edit_distance", 0.0)
                ),
                "normalized_utr_edit_distance_vs_native": float(
                    float(item.get("utr_edit_distance", 0.0))
                    / max(len(designed_utr), len(source_utr), 1)
                ),
                "utr_length": float(len(designed_utr)),
                "utr_length_delta": float(len(designed_utr) - len(source_utr)),
                "exact_native_utr_match": float(designed_utr == source_utr),
            }
        )
        scored[transcript_id] = candidate
        designed_utrs.append(designed_utr)

    summary = _mapping(payload.get("summary") if payload else {})
    runtime = _mapping(payload.get("runtime") if payload else {})
    config = _mapping(payload.get("config") if payload else {})
    ids_match = bool(input_by_id) and set(scored) == set(input_by_id)
    if scored and not ids_match:
        failures.append("transcript_id_coverage_mismatch")
    edit_budget = int(config.get("edit_budget", 0) or 0)
    within_budget = (
        _mean(
            [
                float(row["utr_edit_distance_vs_native"] <= edit_budget)
                for row in scored.values()
            ]
        )
        if scored
        else None
    )
    constraints_exact = bool(
        scored
        and summary.get("cds_unchanged_fraction") == 1.0
        and summary.get("three_utr_unchanged_fraction") == 1.0
        and within_budget == 1.0
    )
    row = {
        "method": "MEF_utr5_constrained_local_search_budget3",
        "status": (
            "measured_internal_oracle_guided_ceiling"
            if ids_match and constraints_exact and not failures
            else "missing_or_invalid"
        ),
        "evidence_type": "oracle_guided_local_search_ceiling",
        "n": len(scored),
        "mean_te_proxy": _mean([item["te_proxy"] for item in scored.values()]),
        "mean_te_proxy_delta_vs_native": _mean(
            [item["te_proxy_delta_vs_native"] for item in scored.values()]
        ),
        "mean_uaug_count": _mean(
            [item["uaug_count"] for item in scored.values()]
        ),
        "mean_kozak_score": _mean(
            [item["kozak_score"] for item in scored.values()]
        ),
        "mean_start_accessibility_proxy": _mean(
            [item["start_accessibility_proxy"] for item in scored.values()]
        ),
        "mean_utr_edit_distance_vs_native": _mean(
            [item["utr_edit_distance_vs_native"] for item in scored.values()]
        ),
        "mean_normalized_utr_edit_distance_vs_native": _mean(
            [
                item["normalized_utr_edit_distance_vs_native"]
                for item in scored.values()
            ]
        ),
        "mean_full_transcript_edit_distance_vs_native": _mean(
            [item["utr_edit_distance_vs_native"] for item in scored.values()]
        ),
        "mean_utr_length": _mean(
            [item["utr_length"] for item in scored.values()]
        ),
        "mean_utr_length_delta": _mean(
            [item["utr_length_delta"] for item in scored.values()]
        ),
        "exact_native_utr_match_fraction": _mean(
            [item["exact_native_utr_match"] for item in scored.values()]
        ),
        "unique_designed_utr_fraction": (
            float(len(set(designed_utrs)) / len(designed_utrs))
            if designed_utrs
            else None
        ),
        "cds_unchanged_fraction": _num(summary.get("cds_unchanged_fraction")),
        "three_utr_unchanged_fraction": _num(
            summary.get("three_utr_unchanged_fraction")
        ),
        "protein_identity_exact_1_fraction": 1.0 if constraints_exact else None,
        "within_edit_budget_fraction": within_budget,
        "edit_budget": edit_budget or None,
        "mean_wall_clock_s": _num(runtime.get("mean_wall_clock_s")),
        "utr5_only_protocol": True,
        "model_only": False,
        "shared_proxy_oracle": "mrna_editflow.eval.oracle.score_utr",
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
    }
    return row, scored, sorted(set(failures))


def _utrgan_row(
    summary: Optional[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
    input_by_id: Mapping[str, Mapping[str, object]],
    measured: bool,
    method: str = "UTRGAN_official_budgeted_10_steps",
    measured_status: str = "measured_external_budgeted",
) -> tuple[dict[str, object], dict[str, dict[str, float]], list[str]]:
    failures: list[str] = []
    scored: dict[str, dict[str, float]] = {}
    for item in output_rows:
        transcript_id = str(item.get("transcript_id"))
        if transcript_id not in input_by_id:
            failures.append(f"unknown_transcript_id:{transcript_id}")
            continue
        if transcript_id in scored:
            failures.append(f"duplicate_transcript_id:{transcript_id}")
        scored[transcript_id] = {
            "te_proxy": float(item.get("te_proxy", 0.0)),
            "te_proxy_delta_vs_native": float(
                item.get("te_proxy_delta_vs_native", 0.0)
            ),
            "uaug_count": float(item.get("uaug_count", 0.0)),
            "kozak_score": float(item.get("kozak_score", 0.0)),
            "start_accessibility_proxy": float(
                item.get("start_accessibility_proxy", 0.0)
            ),
            "utr_edit_distance_vs_native": float(
                item.get("utr_edit_distance_vs_native", 0.0)
            ),
            "normalized_utr_edit_distance_vs_native": float(
                item.get("normalized_utr_edit_distance_vs_native", 0.0)
            ),
            "utr_length": float(item.get("designed_utr_length", 0.0)),
            "utr_length_delta": float(item.get("utr_length_delta", 0.0)),
            "exact_native_utr_match": float(
                bool(item.get("exact_native_utr_match"))
            ),
        }
    coverage = bool(input_by_id) and set(scored) == set(input_by_id)
    if scored and not coverage:
        failures.append("transcript_id_coverage_mismatch")
    complete = bool(
        measured
        and summary
        and summary.get("n_inputs") == len(input_by_id)
        and summary.get("n_outputs") == len(input_by_id)
        and summary.get("n_failures") == 0
        and coverage
    )
    row = {
        "method": method,
        "status": measured_status if complete else "missing_or_invalid",
        "evidence_type": "official_external_de_novo_generator",
        "n": len(scored),
        "mean_te_proxy": _num(summary.get("mean_te_proxy") if summary else None),
        "mean_te_proxy_delta_vs_native": _num(
            summary.get("mean_te_proxy_delta_vs_native") if summary else None
        ),
        "mean_uaug_count": _num(
            summary.get("mean_uaug_count") if summary else None
        ),
        "mean_kozak_score": _num(
            summary.get("mean_kozak_score") if summary else None
        ),
        "mean_start_accessibility_proxy": _num(
            summary.get("mean_start_accessibility_proxy") if summary else None
        ),
        "mean_utr_edit_distance_vs_native": _num(
            summary.get("mean_utr_edit_distance_vs_native") if summary else None
        ),
        "mean_normalized_utr_edit_distance_vs_native": _num(
            summary.get("mean_normalized_utr_edit_distance_vs_native")
            if summary
            else None
        ),
        "mean_full_transcript_edit_distance_vs_native": _num(
            summary.get("mean_utr_edit_distance_vs_native") if summary else None
        ),
        "mean_utr_length": _num(
            summary.get("mean_designed_utr_length") if summary else None
        ),
        "mean_utr_length_delta": _num(
            summary.get("mean_utr_length_delta") if summary else None
        ),
        "exact_native_utr_match_fraction": _num(
            summary.get("exact_native_utr_match_fraction") if summary else None
        ),
        "unique_designed_utr_fraction": _num(
            summary.get("unique_designed_utr_fraction") if summary else None
        ),
        "cds_unchanged_fraction": _num(
            summary.get("cds_unchanged_fraction") if summary else None
        ),
        "three_utr_unchanged_fraction": _num(
            summary.get("three_utr_unchanged_fraction") if summary else None
        ),
        "protein_identity_exact_1_fraction": _num(
            summary.get("protein_identity_exact_1_fraction") if summary else None
        ),
        "within_edit_budget_fraction": None,
        "mean_wall_clock_s": _num(
            summary.get("mean_wall_clock_s") if summary else None
        ),
        "utr5_only_protocol": True,
        "model_only": True,
        "source_conditioned": False,
        "protocol_fidelity": (
            summary.get("protocol_fidelity") if summary else None
        ),
        "protocol_fidelity_sufficient_for_sota_reproduction": bool(
            summary
            and summary.get(
                "protocol_fidelity_sufficient_for_sota_reproduction"
            )
            is True
        ),
    }
    return row, scored, sorted(set(failures))


def _utailor_row(
    summary: Optional[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
    input_by_id: Mapping[str, Mapping[str, object]],
    measured: bool,
) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    transcript_ids = [str(row.get("transcript_id")) for row in output_rows]
    expected_ids = {
        transcript_id
        for transcript_id, row in input_by_id.items()
        if 25 <= len(_normalise_rna(row.get("native_five_utr"))) <= 100
    }
    if len(transcript_ids) != len(set(transcript_ids)):
        failures.append("duplicate_transcript_id")
    if set(transcript_ids) != expected_ids:
        failures.append("strict_eligible_transcript_coverage_mismatch")
    eligibility = _mapping(summary.get("eligibility") if summary else {})
    complete = bool(
        measured
        and summary
        and summary.get("n_inputs") == len(input_by_id)
        and summary.get("n_eligible_inputs") == len(expected_ids)
        and summary.get("n_ineligible_inputs")
        == len(input_by_id) - len(expected_ids)
        and summary.get("n_outputs") == len(expected_ids)
        and summary.get("n_failures") == 0
        and eligibility.get("policy")
        == "official_input_length_25_100_strict"
        and not failures
    )
    return {
        "method": "UTailoR_official_strict_25_100nt",
        "status": (
            "measured_external_protocol_subset"
            if complete
            else "missing_or_invalid"
        ),
        "evidence_type": "official_external_source_conditioned_generator",
        "n": len(output_rows),
        "n_total_input_pack": len(input_by_id),
        "eligibility_policy": eligibility.get("policy"),
        "mean_te_proxy": _num(
            summary.get("mean_te_proxy") if summary else None
        ),
        "mean_te_proxy_delta_vs_native": _num(
            summary.get("mean_te_proxy_delta_vs_native")
            if summary
            else None
        ),
        "mean_uaug_count": _num(
            summary.get("mean_uaug_count") if summary else None
        ),
        "mean_kozak_score": _num(
            summary.get("mean_kozak_score") if summary else None
        ),
        "mean_start_accessibility_proxy": _num(
            summary.get("mean_start_accessibility_proxy")
            if summary
            else None
        ),
        "mean_utr_edit_distance_vs_native": _num(
            summary.get("mean_utr_edit_distance_vs_native")
            if summary
            else None
        ),
        "mean_normalized_utr_edit_distance_vs_native": _num(
            summary.get("mean_normalized_utr_edit_distance_vs_native")
            if summary
            else None
        ),
        "mean_full_transcript_edit_distance_vs_native": _num(
            summary.get("mean_utr_edit_distance_vs_native")
            if summary
            else None
        ),
        "mean_utr_length": _num(
            summary.get("mean_designed_utr_length") if summary else None
        ),
        "mean_utr_length_delta": _num(
            summary.get("mean_utr_length_delta") if summary else None
        ),
        "exact_native_utr_match_fraction": _num(
            summary.get("exact_native_utr_match_fraction")
            if summary
            else None
        ),
        "unique_designed_utr_fraction": _num(
            summary.get("unique_designed_utr_fraction")
            if summary
            else None
        ),
        "cds_unchanged_fraction": _num(
            summary.get("cds_unchanged_fraction") if summary else None
        ),
        "three_utr_unchanged_fraction": _num(
            summary.get("three_utr_unchanged_fraction") if summary else None
        ),
        "protein_identity_exact_1_fraction": _num(
            summary.get("protein_identity_exact_1_fraction")
            if summary
            else None
        ),
        "within_edit_budget_fraction": None,
        "mean_wall_clock_s": _num(
            summary.get("mean_wall_clock_s") if summary else None
        ),
        "external_utailor_mean_original_rl": _num(
            summary.get("mean_external_utailor_original_rl")
            if summary
            else None
        ),
        "external_utailor_mean_optimized_rl": _num(
            summary.get("mean_external_utailor_optimized_rl")
            if summary
            else None
        ),
        "external_utailor_mean_increased_rl": _num(
            summary.get("mean_external_utailor_increased_rl")
            if summary
            else None
        ),
        "utr5_only_protocol": True,
        "model_only": True,
        "source_conditioned": True,
        "protocol_subset": True,
        "protocol_fidelity": (
            summary.get("protocol_fidelity") if summary else None
        ),
        "protocol_fidelity_sufficient_for_sota_reproduction": bool(
            summary
            and summary.get(
                "protocol_fidelity_sufficient_for_sota_reproduction"
            )
            is True
        ),
        "license_status": _mapping(
            summary.get("license") if summary else {}
        ).get("status"),
    }, sorted(set(failures))


def _distributional_comparison(
    local: Mapping[str, Mapping[str, float]],
    utrgan: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    common = sorted(set(local) & set(utrgan))
    metrics = (
        "te_proxy",
        "te_proxy_delta_vs_native",
        "uaug_count",
        "kozak_score",
        "start_accessibility_proxy",
        "utr_edit_distance_vs_native",
        "normalized_utr_edit_distance_vs_native",
        "utr_length_delta",
    )
    rows = []
    for metric in metrics:
        local_mean = _mean([local[key].get(metric) for key in common])
        utrgan_mean = _mean([utrgan[key].get(metric) for key in common])
        rows.append(
            {
                "metric": metric,
                "n_common_rows": len(common),
                "mef_local_search_mean": local_mean,
                "utrgan_mean": utrgan_mean,
                "mef_minus_utrgan": (
                    local_mean - utrgan_mean
                    if local_mean is not None and utrgan_mean is not None
                    else None
                ),
                "paired_p": None,
                "paired_inference_valid": False,
            }
        )
    return {
        "n_common_rows": len(common),
        "rows": rows,
        "paired_inference_valid": False,
        "paired_inference_block_reason": (
            "UTRGAN generates an unconditional batch and the adapter assigns "
            "candidates to ordered source rows; this is not source-conditioned "
            "pairing. Per-row deltas are descriptive only."
        ),
    }


def _load_utr5_model_artifacts(
    absolute: Mapping[str, str],
    prefix: str,
) -> tuple[
    Optional[Mapping[str, object]],
    list[Mapping[str, object]],
    list[dict[str, object]],
    list[str],
]:
    summary = _load_json(absolute[f"{prefix}_summary"])
    seed_payloads = [
        payload
        for path in sorted(
            glob(absolute[f"{prefix}_seed_glob"], recursive=True)
        )
        if (payload := _load_json(path)) is not None
    ]
    candidate_paths = sorted(
        glob(absolute[f"{prefix}_candidate_glob"], recursive=True)
    )
    source_rows = _load_jsonl(absolute[f"{prefix}_sources"])
    if summary and not seed_payloads:
        per_seed = summary.get("per_seed", [])
        if isinstance(per_seed, list):
            seed_payloads = [
                payload
                for row in per_seed
                if isinstance(row, Mapping)
                and isinstance(row.get("eval_json_path"), str)
                and (
                    payload := _load_json(str(row.get("eval_json_path")))
                )
                is not None
            ]
    if summary and not candidate_paths:
        per_seed = summary.get("per_seed", [])
        if isinstance(per_seed, list):
            candidate_paths = [
                str(row.get("candidate_path"))
                for row in per_seed
                if isinstance(row, Mapping)
                and isinstance(row.get("candidate_path"), str)
                and os.path.exists(str(row.get("candidate_path")))
            ]
    if summary and not source_rows:
        summary_source = summary.get("source_path")
        if isinstance(summary_source, str) and os.path.exists(summary_source):
            source_rows = _load_jsonl(summary_source)
    if summary and not source_rows:
        source_rows = _load_jsonl(absolute["mef_shared_sources"])
    return summary, seed_payloads, source_rows, candidate_paths


def _seed_delta_map(
    summary: Optional[Mapping[str, object]],
) -> dict[int, float]:
    rows = summary.get("per_seed", []) if summary else []
    rows = rows if isinstance(rows, list) else []
    result: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("metrics"), Mapping
        ):
            continue
        seed = row.get("seed")
        value = _num(
            _mapping(row.get("metrics")).get("delta_oracle_te_vs_source")
        )
        if isinstance(seed, int) and value is not None:
            result[int(seed)] = value
    return result


def _compare_utr5_models(
    *,
    comparison_id: str,
    run_name: str,
    run_summary: Optional[Mapping[str, object]],
    baseline_name: str,
    baseline_summary: Optional[Mapping[str, object]],
) -> dict[str, object]:
    run = _seed_delta_map(run_summary)
    baseline = _seed_delta_map(baseline_summary)
    seeds = sorted(set(run) & set(baseline))
    run_values = [run[seed] for seed in seeds]
    baseline_values = [baseline[seed] for seed in seeds]
    differences = [
        run_value - baseline_value
        for run_value, baseline_value in zip(run_values, baseline_values)
    ]
    ci = (
        bootstrap_ci(
            differences,
            seeds=seeds,
            n_bootstrap=1000,
        )
        if len(seeds) >= 5
        else {"mean": None, "low": None, "high": None, "n": len(seeds)}
    )
    p_value = (
        paired_permutation_pvalue(
            run_values,
            baseline_values,
            seed=0,
            n_permutations=2000,
        )
        if seeds
        else None
    )
    delta = _num(ci.get("mean"))
    if delta is None or p_value is None:
        signal = "missing"
    elif p_value < 0.05 and delta > 0:
        signal = "significant_positive"
    elif p_value < 0.05 and delta < 0:
        signal = "significant_negative"
    elif delta > 0:
        signal = "positive_not_significant"
    elif delta < 0:
        signal = "negative_not_significant"
    else:
        signal = "neutral"
    return {
        "comparison_id": comparison_id,
        "metric": "delta_oracle_te_vs_source",
        "run": run_name,
        "baseline": baseline_name,
        "n_paired_seeds": len(seeds),
        "seeds": seeds,
        "run_mean": _mean(run_values),
        "baseline_mean": _mean(baseline_values),
        "delta": delta,
        "ci_low": _num(ci.get("low")),
        "ci_high": _num(ci.get("high")),
        "paired_p": p_value,
        "signal": signal,
    }


def _score_model_subset(
    *,
    candidate_paths: Sequence[str],
    source_rows: Sequence[Mapping[str, object]],
    eligible_ids: set[str],
) -> tuple[list[dict[str, object]], list[str]]:
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for path in candidate_paths:
        candidates = _load_jsonl(path)
        if len(candidates) != len(source_rows):
            failures.append(f"coverage_mismatch:{path}")
            continue
        deltas: list[float] = []
        edits: list[float] = []
        uaug: list[float] = []
        kozak: list[float] = []
        access: list[float] = []
        for source, candidate in zip(source_rows, candidates):
            transcript_id = str(source.get("transcript_id"))
            if transcript_id not in eligible_ids:
                continue
            candidate_id = str(candidate.get("transcript_id", ""))
            if not candidate_id.startswith(transcript_id):
                failures.append(f"order_mismatch:{path}:{transcript_id}")
                continue
            source_utr = _normalise_rna(source.get("five_utr"))
            candidate_utr = _normalise_rna(candidate.get("five_utr"))
            cds = _normalise_rna(source.get("cds"))
            source_score = score_utr(source_utr, cds[:12])
            candidate_score = score_utr(candidate_utr, cds[:12])
            features = _mapping(candidate_score.get("features"))
            deltas.append(
                float(
                    candidate_score["ensemble_te"]
                    - source_score["ensemble_te"]
                )
            )
            edits.append(float(edit_distance(candidate_utr, source_utr)))
            uaug.append(float(features.get("uaug_count", 0.0)))
            kozak.append(float(features.get("kozak", 0.0)))
            access.append(
                float(features.get("start_accessibility", 0.0))
            )
        seed_dir = os.path.basename(os.path.dirname(path))
        try:
            seed = int(seed_dir.split("_")[-1])
        except (TypeError, ValueError):
            failures.append(f"seed_parse_failed:{path}")
            continue
        rows.append(
            {
                "seed": seed,
                "n": len(deltas),
                "mean_te_proxy_delta_vs_native": _mean(deltas),
                "mean_utr_edit_distance_vs_native": _mean(edits),
                "mean_uaug_count": _mean(uaug),
                "mean_kozak_score": _mean(kozak),
                "mean_start_accessibility_proxy": _mean(access),
            }
        )
    return sorted(rows, key=lambda row: int(row["seed"])), sorted(
        set(failures)
    )


def _compare_model_subset_to_utailor(
    *,
    comparison_id: str,
    model_name: str,
    model_rows: Sequence[Mapping[str, object]],
    utailor_delta: Optional[float],
    utailor_mean_edit_distance: Optional[float],
    model_hard_edit_budget: Optional[int] = None,
) -> dict[str, object]:
    model_values = [
        float(value)
        for row in model_rows
        if (
            value := _num(row.get("mean_te_proxy_delta_vs_native"))
        )
        is not None
    ]
    baseline_values = (
        [float(utailor_delta)] * len(model_values)
        if utailor_delta is not None
        else []
    )
    model_edit_values = [
        float(value)
        for row in model_rows
        if (
            value := _num(row.get("mean_utr_edit_distance_vs_native"))
        )
        is not None
    ]
    model_mean = _mean(model_values)
    model_mean_edit = _mean(model_edit_values)
    differences = [
        run - baseline
        for run, baseline in zip(model_values, baseline_values)
    ]
    seeds = [
        int(row["seed"])
        for row in model_rows
        if isinstance(row.get("seed"), int)
    ]
    ci = (
        bootstrap_ci(differences, seeds=seeds, n_bootstrap=1000)
        if len(differences) >= 5
        else {"mean": None, "low": None, "high": None}
    )
    p_value = (
        paired_permutation_pvalue(
            model_values,
            baseline_values,
            seed=0,
            n_permutations=2000,
        )
        if model_values and baseline_values
        else None
    )
    delta = _num(ci.get("mean"))
    if delta is None or p_value is None:
        signal = "missing"
    elif p_value < 0.05 and delta > 0:
        signal = "significant_positive"
    elif p_value < 0.05 and delta < 0:
        signal = "significant_negative"
    elif delta > 0:
        signal = "positive_not_significant"
    elif delta < 0:
        signal = "negative_not_significant"
    else:
        signal = "neutral"
    return {
        "comparison_id": comparison_id,
        "metric": "mean_te_proxy_delta_vs_native",
        "run": model_name,
        "baseline": "UTailoR_official_strict_25_100nt",
        "n_eligible_records": (
            int(model_rows[0].get("n", 0)) if model_rows else 0
        ),
        "n_paired_model_seeds": len(model_values),
        "model_mean": model_mean,
        "utailor_deterministic_mean": utailor_delta,
        "model_minus_utailor": delta,
        "model_mean_edit_distance": model_mean_edit,
        "utailor_mean_edit_distance": utailor_mean_edit_distance,
        "mean_edit_distance_gap": (
            model_mean_edit - utailor_mean_edit_distance
            if model_mean_edit is not None
            and utailor_mean_edit_distance is not None
            else None
        ),
        "model_hard_edit_budget": model_hard_edit_budget,
        "model_te_delta_per_edit": (
            model_mean / model_mean_edit
            if model_mean is not None
            and model_mean_edit is not None
            and model_mean_edit > 0
            else None
        ),
        "utailor_te_delta_per_edit": (
            utailor_delta / utailor_mean_edit_distance
            if utailor_delta is not None
            and utailor_mean_edit_distance is not None
            and utailor_mean_edit_distance > 0
            else None
        ),
        "ci_low": _num(ci.get("low")),
        "ci_high": _num(ci.get("high")),
        "paired_p": p_value,
        "signal": signal,
        "edit_budget_matched": False,
        "edit_budget_alignment": (
            "closer_hard_budget_5_vs_unbounded_external_observed_mean"
            if model_hard_edit_budget == 5
            else "unmatched"
        ),
        "paired_inference_scope": (
            "MEF seed variability against one deterministic UTailoR output "
            "on the same eligible transcripts"
        ),
        "ready_for_superiority_claim": False,
    }


def build_t5_external_utr_comparison(project_root: str) -> dict[str, object]:
    paths = dict(DEFAULT_PATHS)
    absolute = {
        key: (
            os.path.join(project_root, value)
            if key != "mef_model_seed_glob"
            else os.path.join(project_root, value)
        )
        for key, value in paths.items()
    }
    input_pack = _load_json(absolute["input_pack"])
    input_rows = _load_jsonl(absolute["utr_inputs"])
    input_by_id = {str(row.get("transcript_id")): row for row in input_rows}
    audit = _load_json(absolute["real_run_audit"])
    audit_rows = audit.get("rows", []) if audit else []
    audit_rows = audit_rows if isinstance(audit_rows, list) else []
    utrgan_measured = any(
        isinstance(row, Mapping)
        and row.get("model_name") == "UTRGAN"
        and row.get("status") == "measured"
        and row.get("real_metric_ready") is True
        for row in audit_rows
    )
    utrgan_summary = _load_json(absolute["utrgan_summary"])
    utrgan_outputs = _load_jsonl(absolute["utrgan_outputs"])
    utrgan_paper10000_summary = _load_json(
        absolute["utrgan_paper10000_summary"]
    )
    utrgan_paper10000_outputs = _load_jsonl(
        absolute["utrgan_paper10000_outputs"]
    )
    utailor_summary = _load_json(absolute["utailor_summary"])
    utailor_outputs = _load_jsonl(absolute["utailor_outputs"])
    utailor_measured = any(
        isinstance(row, Mapping)
        and row.get("model_name") == "UTailoR"
        and row.get("status") == "measured"
        and row.get("real_metric_ready") is True
        for row in audit_rows
    )
    local_payload = _load_json(absolute["mef_utr_local_search"])
    model_summary = _load_json(absolute["mef_model_summary"])
    model_seed_payloads = [
        payload
        for path in sorted(glob(absolute["mef_model_seed_glob"]))
        if (payload := _load_json(path)) is not None
    ]
    (
        utr5_model_summary,
        utr5_model_seed_payloads,
        utr5_model_sources,
        utr5_model_candidate_paths,
    ) = _load_utr5_model_artifacts(absolute, "mef_utr5_model")
    (
        utr_teacher_summary,
        utr_teacher_seed_payloads,
        utr_teacher_sources,
        utr_teacher_candidate_paths,
    ) = _load_utr5_model_artifacts(absolute, "mef_utr_teacher")
    (
        sequential_summary,
        sequential_seed_payloads,
        sequential_sources,
        sequential_candidate_paths,
    ) = _load_utr5_model_artifacts(
        absolute,
        "mef_sequential_utr_teacher",
    )
    (
        utailor_budget5_summary,
        utailor_budget5_seed_payloads,
        utailor_budget5_sources,
        utailor_budget5_candidate_paths,
    ) = _load_utr5_model_artifacts(
        absolute,
        "mef_utailor_budget5",
    )
    utailor_strict_subset_summary = _load_json(
        absolute["utailor_strict_subset_summary"]
    )

    native = _native_row(input_rows)
    model = _mef_model_row(model_summary, model_seed_payloads)
    utr5_model, utr5_model_failures = _mef_utr5_model_row(
        utr5_model_summary,
        utr5_model_seed_payloads,
        utr5_model_sources,
        utr5_model_candidate_paths,
    )
    utr_teacher, utr_teacher_failures = _mef_utr5_model_row(
        utr_teacher_summary,
        utr_teacher_seed_payloads,
        utr_teacher_sources,
        utr_teacher_candidate_paths,
        method="MEF_pure_utr_teacher_utr5only_top64",
        evidence_type="ten_seed_pure_utr_teacher_utr5_only_model",
    )
    sequential_utr_teacher, sequential_failures = _mef_utr5_model_row(
        sequential_summary,
        sequential_seed_payloads,
        sequential_sources,
        sequential_candidate_paths,
        method="MEF_full_then_utr_teacher_utr5only_top64",
        evidence_type="ten_seed_sequential_utr_teacher_utr5_only_model",
    )
    utailor_budget5_teacher, utailor_budget5_failures = _mef_utr5_model_row(
        utailor_budget5_summary,
        utailor_budget5_seed_payloads,
        utailor_budget5_sources,
        utailor_budget5_candidate_paths,
        method=(
            "MEF_pure_utr_teacher_budget5_"
            "utailor_strict_25_100nt"
        ),
        evidence_type=(
            "ten_seed_pure_utr_teacher_utr5_only_"
            "utailor_protocol_subset_budget5"
        ),
        expected_n=315,
        measured_status=(
            "measured_internal_model_10seed_"
            "utailor_protocol_subset_budget5"
        ),
    )
    local, local_scored, local_failures = _local_search_row(
        local_payload, input_by_id
    )
    utrgan, utrgan_scored, utrgan_failures = _utrgan_row(
        utrgan_summary,
        utrgan_outputs,
        input_by_id,
        utrgan_measured,
    )
    paper_input_pack = _mapping(
        utrgan_paper10000_summary.get("input_pack")
        if utrgan_paper10000_summary
        else {}
    )
    utrgan_paper10000_measured = bool(
        utrgan_paper10000_summary
        and utrgan_paper10000_summary.get("protocol_fidelity")
        == "official_code_paper_default_10000_steps"
        and utrgan_paper10000_summary.get(
            "protocol_fidelity_sufficient_for_sota_reproduction"
        )
        is True
        and utrgan_paper10000_summary.get("n_inputs") == len(input_by_id)
        and utrgan_paper10000_summary.get("n_outputs") == len(input_by_id)
        and utrgan_paper10000_summary.get("n_failures") == 0
        and utrgan_paper10000_summary.get("cds_unchanged_fraction") == 1.0
        and utrgan_paper10000_summary.get("three_utr_unchanged_fraction")
        == 1.0
        and utrgan_paper10000_summary.get(
            "protein_identity_exact_1_fraction"
        )
        == 1.0
        and paper_input_pack.get("summary_sha256")
        == _sha256_file(absolute["input_pack"])
    )
    (
        utrgan_paper10000,
        utrgan_paper10000_scored,
        utrgan_paper10000_failures,
    ) = _utrgan_row(
        utrgan_paper10000_summary,
        utrgan_paper10000_outputs,
        input_by_id,
        utrgan_paper10000_measured,
        method="UTRGAN_official_paper_default_10000_steps",
        measured_status="measured_external_paper_default",
    )
    utailor, utailor_failures = _utailor_row(
        utailor_summary,
        utailor_outputs,
        input_by_id,
        utailor_measured,
    )
    distributional = _distributional_comparison(local_scored, utrgan_scored)
    utrgan_protocol_distributional = _distributional_comparison(
        utrgan_paper10000_scored,
        utrgan_scored,
    )
    for protocol_row in utrgan_protocol_distributional["rows"]:
        if not isinstance(protocol_row, dict):
            continue
        protocol_row["paper_default_mean"] = protocol_row.pop(
            "mef_local_search_mean"
        )
        protocol_row["budgeted_10_step_mean"] = protocol_row.pop(
            "utrgan_mean"
        )
        protocol_row["paper_default_minus_budgeted"] = protocol_row.pop(
            "mef_minus_utrgan"
        )
    utrgan_protocol_distributional["comparison_scope"] = (
        "descriptive ordered-batch protocol comparison; both generations are "
        "stochastic and not source-conditioned"
    )
    utrgan_protocol_distributional["paper_default_method"] = (
        "UTRGAN_official_paper_default_10000_steps"
    )
    utrgan_protocol_distributional["budgeted_method"] = (
        "UTRGAN_official_budgeted_10_steps"
    )
    model_ablation_comparisons = [
        _compare_utr5_models(
            comparison_id="pure_utr_teacher_vs_region_adapter",
            run_name="MEF_pure_utr_teacher_utr5only_top64",
            run_summary=utr_teacher_summary,
            baseline_name="MEF_region_adapter_utr5only_top64",
            baseline_summary=utr5_model_summary,
        ),
        _compare_utr5_models(
            comparison_id="sequential_utr_teacher_vs_region_adapter",
            run_name="MEF_full_then_utr_teacher_utr5only_top64",
            run_summary=sequential_summary,
            baseline_name="MEF_region_adapter_utr5only_top64",
            baseline_summary=utr5_model_summary,
        ),
        _compare_utr5_models(
            comparison_id="pure_vs_sequential_utr_teacher",
            run_name="MEF_pure_utr_teacher_utr5only_top64",
            run_summary=utr_teacher_summary,
            baseline_name="MEF_full_then_utr_teacher_utr5only_top64",
            baseline_summary=sequential_summary,
        ),
    ]
    utailor_eligible_ids = {
        transcript_id
        for transcript_id, row in input_by_id.items()
        if 25 <= len(_normalise_rna(row.get("native_five_utr"))) <= 100
    }
    region_subset_rows, region_subset_failures = _score_model_subset(
        candidate_paths=utr5_model_candidate_paths,
        source_rows=utr5_model_sources,
        eligible_ids=utailor_eligible_ids,
    )
    pure_subset_rows, pure_subset_failures = _score_model_subset(
        candidate_paths=utr_teacher_candidate_paths,
        source_rows=utr_teacher_sources,
        eligible_ids=utailor_eligible_ids,
    )
    sequential_subset_rows, sequential_subset_failures = _score_model_subset(
        candidate_paths=sequential_candidate_paths,
        source_rows=sequential_sources,
        eligible_ids=utailor_eligible_ids,
    )
    budget5_subset_rows, budget5_subset_failures = _score_model_subset(
        candidate_paths=utailor_budget5_candidate_paths,
        source_rows=utailor_budget5_sources,
        eligible_ids=utailor_eligible_ids,
    )
    utailor_delta = _num(
        utailor.get("mean_te_proxy_delta_vs_native")
    )
    utailor_mean_edit = _num(
        utailor.get("mean_utr_edit_distance_vs_native")
    )
    model_vs_utailor_subset_comparisons = [
        _compare_model_subset_to_utailor(
            comparison_id="region_adapter_vs_utailor_strict_subset",
            model_name="MEF_region_adapter_utr5only_top64",
            model_rows=region_subset_rows,
            utailor_delta=utailor_delta,
            utailor_mean_edit_distance=utailor_mean_edit,
        ),
        _compare_model_subset_to_utailor(
            comparison_id="pure_utr_teacher_vs_utailor_strict_subset",
            model_name="MEF_pure_utr_teacher_utr5only_top64",
            model_rows=pure_subset_rows,
            utailor_delta=utailor_delta,
            utailor_mean_edit_distance=utailor_mean_edit,
        ),
        _compare_model_subset_to_utailor(
            comparison_id=(
                "sequential_utr_teacher_vs_utailor_strict_subset"
            ),
            model_name="MEF_full_then_utr_teacher_utr5only_top64",
            model_rows=sequential_subset_rows,
            utailor_delta=utailor_delta,
            utailor_mean_edit_distance=utailor_mean_edit,
        ),
        _compare_model_subset_to_utailor(
            comparison_id=(
                "pure_utr_teacher_budget5_vs_utailor_strict_subset"
            ),
            model_name=(
                "MEF_pure_utr_teacher_budget5_"
                "utailor_strict_25_100nt"
            ),
            model_rows=budget5_subset_rows,
            utailor_delta=utailor_delta,
            utailor_mean_edit_distance=utailor_mean_edit,
            model_hard_edit_budget=5,
        ),
    ]

    input_ready = bool(
        input_pack
        and input_pack.get("artifact_kind") == "external_sota_input_pack"
        and input_pack.get("ready_for_external_real_run") is True
        and len(input_rows) == 1024
        and len(input_by_id) == len(input_rows)
    )
    local_ready = local.get("status") == "measured_internal_oracle_guided_ceiling"
    utrgan_ready = utrgan.get("status") == "measured_external_budgeted"
    utrgan_paper10000_ready = (
        utrgan_paper10000.get("status")
        == "measured_external_paper_default"
    )
    utailor_ready = (
        utailor.get("status") == "measured_external_protocol_subset"
    )
    utr5_model_ready = (
        utr5_model.get("status")
        == "measured_internal_model_10seed_utr5only"
    )
    utr_teacher_ready = (
        utr_teacher.get("status")
        == "measured_internal_model_10seed_utr5only"
    )
    sequential_ready = (
        sequential_utr_teacher.get("status")
        == "measured_internal_model_10seed_utr5only"
    )
    utailor_subset_contract = _mapping(utailor_strict_subset_summary)
    utailor_subset_contract_ok = bool(
        utailor_subset_contract.get("artifact_kind")
        == "utailor_strict_input_domain_subset"
        and utailor_subset_contract.get("eligibility_policy")
        == "official_input_length_25_100_strict"
        and utailor_subset_contract.get("source_n") == 1024
        and utailor_subset_contract.get("subset_n") == 315
        and utailor_subset_contract.get("subset_sha256")
        == _sha256_file(absolute["mef_utailor_budget5_sources"])
    )
    utailor_budget5_ready = (
        utailor_budget5_teacher.get("status")
        == (
            "measured_internal_model_10seed_"
            "utailor_protocol_subset_budget5"
        )
        and utailor_subset_contract_ok
    )
    measured_model_rows = [
        row
        for row in (utr5_model, utr_teacher, sequential_utr_teacher)
        if row.get("status") == "measured_internal_model_10seed_utr5only"
    ]
    best_utr5_model = max(
        measured_model_rows,
        key=lambda row: float(
            row.get("mean_te_proxy_delta_vs_native") or -math.inf
        ),
        default=None,
    )
    constraints_exact = bool(
        local.get("cds_unchanged_fraction") == 1.0
        and local.get("three_utr_unchanged_fraction") == 1.0
        and local.get("protein_identity_exact_1_fraction") == 1.0
        and local.get("within_edit_budget_fraction") == 1.0
        and utrgan.get("cds_unchanged_fraction") == 1.0
        and utrgan.get("three_utr_unchanged_fraction") == 1.0
        and utrgan.get("protein_identity_exact_1_fraction") == 1.0
        and (
            not utrgan_paper10000_ready
            or (
                utrgan_paper10000.get("cds_unchanged_fraction") == 1.0
                and utrgan_paper10000.get(
                    "three_utr_unchanged_fraction"
                )
                == 1.0
                and utrgan_paper10000.get(
                    "protein_identity_exact_1_fraction"
                )
                == 1.0
            )
        )
        and (
            not utailor_ready
            or (
                utailor.get("cds_unchanged_fraction") == 1.0
                and utailor.get("three_utr_unchanged_fraction") == 1.0
                and utailor.get("protein_identity_exact_1_fraction") == 1.0
            )
        )
    )
    utrgan_descriptive_ready = bool(
        input_ready and local_ready and utrgan_ready and constraints_exact
    )
    descriptive_ready = bool(
        utrgan_descriptive_ready and utailor_ready
    )
    return {
        "artifact_kind": "t5_external_utr_baseline_comparison",
        "claim_policy": CLAIM_POLICY,
        "sources": {
            **paths,
            "input_pack_sha256": _sha256_file(absolute["input_pack"]),
            "utr_inputs_sha256": _sha256_file(absolute["utr_inputs"]),
            "utrgan_summary_sha256": _sha256_file(absolute["utrgan_summary"]),
            "utrgan_outputs_sha256": _sha256_file(absolute["utrgan_outputs"]),
            "utrgan_paper10000_summary_sha256": _sha256_file(
                absolute["utrgan_paper10000_summary"]
            ),
            "utrgan_paper10000_outputs_sha256": _sha256_file(
                absolute["utrgan_paper10000_outputs"]
            ),
            "utailor_summary_sha256": _sha256_file(
                absolute["utailor_summary"]
            ),
            "utailor_outputs_sha256": _sha256_file(
                absolute["utailor_outputs"]
            ),
            "mef_utr_local_search_sha256": _sha256_file(
                absolute["mef_utr_local_search"]
            ),
            "mef_model_summary_sha256": _sha256_file(
                absolute["mef_model_summary"]
            ),
            "mef_utr5_model_summary_sha256": _sha256_file(
                absolute["mef_utr5_model_summary"]
            ),
            "mef_utr5_model_sources_sha256": _sha256_file(
                absolute["mef_utr5_model_sources"]
            ),
            "mef_utailor_budget5_summary_sha256": _sha256_file(
                absolute["mef_utailor_budget5_summary"]
            ),
            "mef_utailor_budget5_sources_sha256": _sha256_file(
                absolute["mef_utailor_budget5_sources"]
            ),
            "utailor_strict_subset_summary_sha256": _sha256_file(
                absolute["utailor_strict_subset_summary"]
            ),
        },
        "summary": {
            "input_pack_ready_head1024": input_ready,
            "utrgan_measured_complete_head1024": utrgan_ready,
            "utrgan_paper10000_measured_complete_head1024": (
                utrgan_paper10000_ready
            ),
            "utailor_measured_strict_25_100_subset": utailor_ready,
            "utailor_eligible_rows": utailor.get("n"),
            "ready_for_t5_utrgan_descriptive_table": (
                utrgan_descriptive_ready
            ),
            "mef_utr_local_search_complete_head1024": local_ready,
            "hard_constraints_exact_1": constraints_exact,
            "shared_proxy_oracle": True,
            "fixed_region_protocol_matched_for_local_search_and_utrgan": True,
            "optimization_budget_matched": False,
            "utailor_strict_subset_budget5_mef_available": (
                utailor_budget5_ready
            ),
            "utailor_strict_subset_manifest_valid": (
                utailor_subset_contract_ok
            ),
            "utailor_strict_subset_budget_alignment": (
                "closer_hard_budget_5_vs_unbounded_external_observed_mean"
                if utailor_budget5_ready
                else "budget5_run_missing_or_incomplete"
            ),
            "source_conditioning_matched": False,
            "mef_model_utr5_only_run_available": utr5_model_ready,
            "mef_utr5_model_te_signal": utr5_model.get(
                "te_proxy_delta_seed_signal"
            ),
            "mef_pure_utr_teacher_run_available": utr_teacher_ready,
            "mef_pure_utr_teacher_te_signal": utr_teacher.get(
                "te_proxy_delta_seed_signal"
            ),
            "mef_sequential_utr_teacher_run_available": sequential_ready,
            "mef_sequential_utr_teacher_te_signal": (
                sequential_utr_teacher.get("te_proxy_delta_seed_signal")
            ),
            "n_mef_utr5only_model_runs_measured": len(measured_model_rows),
            "best_mef_utr5only_model": (
                best_utr5_model.get("method") if best_utr5_model else None
            ),
            "best_mef_utr5only_model_delta_te": (
                best_utr5_model.get("mean_te_proxy_delta_vs_native")
                if best_utr5_model
                else None
            ),
            "ready_for_t5_utr_descriptive_table": descriptive_ready,
            "ready_for_paired_per_record_inference": False,
            "ready_for_model_only_descriptive_head_to_head": bool(
                utr5_model_ready and utrgan_ready
            ),
            "ready_for_model_only_head_to_head": False,
            "ready_for_mef_superiority_claim": False,
            "ready_for_external_sota_claim": False,
            "local_search_validation_failures": local_failures,
            "mef_utr5_model_validation_failures": utr5_model_failures,
            "mef_pure_utr_teacher_validation_failures": (
                utr_teacher_failures
            ),
            "mef_sequential_utr_teacher_validation_failures": (
                sequential_failures
            ),
            "mef_utailor_budget5_validation_failures": (
                utailor_budget5_failures
            ),
            "utrgan_validation_failures": utrgan_failures,
            "utrgan_paper10000_validation_failures": (
                utrgan_paper10000_failures
            ),
            "utailor_validation_failures": utailor_failures,
            "mef_utailor_subset_scoring_failures": sorted(
                set(
                    region_subset_failures
                    + pure_subset_failures
                    + sequential_subset_failures
                    + budget5_subset_failures
                )
            ),
            "remaining_model_fairness_action": (
                (
                    "The MEF hard-budget-5 UTailoR strict-subset run narrows "
                    "the edit-effort mismatch but is not an exact budget "
                    "match because official UTailoR is not hard-capped per "
                    "record. UTRGAN paper-default is complete; paired "
                    "inference remains invalid because UTRGAN is not "
                    "source-conditioned."
                )
                if utailor_budget5_ready and utrgan_paper10000_ready
                else (
                    "Complete the UTRGAN paper-default 10000-step run. The "
                    "MEF hard-budget-5 UTailoR strict-subset run narrows the "
                    "edit-effort mismatch but is not an exact budget match."
                )
                if utailor_budget5_ready
                else (
                    "Complete the MEF pure-UTR-teacher hard-budget-5 run on "
                    "the exact 315-record UTailoR subset. Then rerun UTRGAN "
                    "with its paper-faithful 10000-step protocol or predeclare "
                    "a matched compute budget."
                )
                if utr5_model_ready
                else (
                    "Run the MEF decoder with 5UTR-only editable positions on "
                    "the same head1024 input pack and ten seeds; rerun UTRGAN "
                    "with its paper-faithful 10000-step protocol or predeclare "
                    "a matched compute budget."
                )
            ),
        },
        "rows": [
            native,
            model,
            utr5_model,
            utr_teacher,
            sequential_utr_teacher,
            utailor_budget5_teacher,
            local,
            utailor,
            utrgan,
            utrgan_paper10000,
        ],
        "distributional_comparison": distributional,
        "utrgan_paper10000_vs_budgeted10": (
            utrgan_protocol_distributional
        ),
        "model_ablation_comparisons": model_ablation_comparisons,
        "model_vs_utailor_subset_comparisons": (
            model_vs_utailor_subset_comparisons
        ),
        "model_utailor_subset_per_seed": {
            "MEF_region_adapter_utr5only_top64": region_subset_rows,
            "MEF_pure_utr_teacher_utr5only_top64": pure_subset_rows,
            "MEF_full_then_utr_teacher_utr5only_top64": (
                sequential_subset_rows
            ),
            (
                "MEF_pure_utr_teacher_budget5_"
                "utailor_strict_25_100nt"
            ): budget5_subset_rows,
        },
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    def canonicalise(value: object) -> object:
        if isinstance(value, float):
            return float(format(value, ".14g")) if math.isfinite(value) else value
        if isinstance(value, Mapping):
            return {
                str(key): canonicalise(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [canonicalise(item) for item in value]
        return value

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(canonicalise(report), fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object) -> str:
    number = _num(value)
    return "NA" if number is None else f"{number:.6f}"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = _mapping(report.get("summary"))
    rows = report.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    comparison = _mapping(report.get("distributional_comparison"))
    comparison_rows = comparison.get("rows", [])
    comparison_rows = comparison_rows if isinstance(comparison_rows, list) else []
    utrgan_protocol = _mapping(
        report.get("utrgan_paper10000_vs_budgeted10")
    )
    utrgan_protocol_rows = utrgan_protocol.get("rows", [])
    utrgan_protocol_rows = (
        utrgan_protocol_rows
        if isinstance(utrgan_protocol_rows, list)
        else []
    )
    model_comparisons = report.get("model_ablation_comparisons", [])
    model_comparisons = (
        model_comparisons if isinstance(model_comparisons, list) else []
    )
    utailor_comparisons = report.get(
        "model_vs_utailor_subset_comparisons", []
    )
    utailor_comparisons = (
        utailor_comparisons
        if isinstance(utailor_comparisons, list)
        else []
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# T5 External 5'UTR Baseline Comparison\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            "- Descriptive table ready: "
            f"`{summary.get('ready_for_t5_utr_descriptive_table')}`; "
            "model-only head-to-head ready: "
            f"`{summary.get('ready_for_model_only_head_to_head')}`; "
            "MEF superiority claim ready: "
            f"`{summary.get('ready_for_mef_superiority_claim')}`\n"
        )
        fh.write(
            "- Paired per-record inference ready: "
            f"`{summary.get('ready_for_paired_per_record_inference')}`. "
            f"{comparison.get('paired_inference_block_reason', '')}\n\n"
        )
        fh.write(
            "| Method | Status | n | TE | delta TE | uAUG | Kozak | Access | "
            "UTR edit | UTR length delta | exact native | unique | CDS fixed | "
            "3UTR fixed | protein exact-1 | seed p | signal | sec/seq |\n"
        )
        fh.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---|---:|\n"
        )
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('method')} | `{row.get('status')}` | "
                f"{row.get('n') if row.get('n') is not None else 'NA'} | "
                f"{_fmt(row.get('mean_te_proxy'))} | "
                f"{_fmt(row.get('mean_te_proxy_delta_vs_native'))} | "
                f"{_fmt(row.get('mean_uaug_count'))} | "
                f"{_fmt(row.get('mean_kozak_score'))} | "
                f"{_fmt(row.get('mean_start_accessibility_proxy'))} | "
                f"{_fmt(row.get('mean_utr_edit_distance_vs_native'))} | "
                f"{_fmt(row.get('mean_utr_length_delta'))} | "
                f"{_fmt(row.get('exact_native_utr_match_fraction'))} | "
                f"{_fmt(row.get('unique_designed_utr_fraction'))} | "
                f"{_fmt(row.get('cds_unchanged_fraction'))} | "
                f"{_fmt(row.get('three_utr_unchanged_fraction'))} | "
                f"{_fmt(row.get('protein_identity_exact_1_fraction'))} | "
                f"{_fmt(row.get('te_proxy_delta_seed_paired_p_vs_source'))} | "
                f"{row.get('te_proxy_delta_seed_signal', 'NA')} | "
                f"{_fmt(row.get('mean_wall_clock_s'))} |\n"
            )
        fh.write("\n## Descriptive Distribution Differences\n\n")
        fh.write("| Metric | n | MEF local search | UTRGAN | MEF - UTRGAN | paired p |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for row in comparison_rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| `{row.get('metric')}` | {row.get('n_common_rows')} | "
                f"{_fmt(row.get('mef_local_search_mean'))} | "
                f"{_fmt(row.get('utrgan_mean'))} | "
                f"{_fmt(row.get('mef_minus_utrgan'))} | NA |\n"
            )
        fh.write("\n## UTRGAN Paper-Default vs Budgeted Protocol\n\n")
        fh.write(
            "| Metric | n | paper 10000 | budgeted 10 | paper - budgeted | "
            "paired p |\n"
        )
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for row in utrgan_protocol_rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| `{row.get('metric')}` | {row.get('n_common_rows')} | "
                f"{_fmt(row.get('paper_default_mean'))} | "
                f"{_fmt(row.get('budgeted_10_step_mean'))} | "
                f"{_fmt(row.get('paper_default_minus_budgeted'))} | NA |\n"
            )
        fh.write(
            "\nBoth UTRGAN protocols are stochastic de novo batches rather "
            "than source-conditioned outputs; this section is descriptive and "
            "does not report paired inference.\n"
        )
        fh.write("\n## MEF 5'UTR-Only Model Ablations\n\n")
        fh.write(
            "| Comparison | Run | Baseline | n seeds | run | baseline | delta | "
            "95% CI | paired p | signal |\n"
        )
        fh.write("|---|---|---|---:|---:|---:|---:|---|---:|---|\n")
        for row in model_comparisons:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| `{row.get('comparison_id')}` | {row.get('run')} | "
                f"{row.get('baseline')} | {row.get('n_paired_seeds')} | "
                f"{_fmt(row.get('run_mean'))} | "
                f"{_fmt(row.get('baseline_mean'))} | "
                f"{_fmt(row.get('delta'))} | "
                f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] | "
                f"{_fmt(row.get('paired_p'))} | "
                f"`{row.get('signal')}` |\n"
            )
        fh.write("\n## MEF vs UTailoR Strict 25-100 nt Subset\n\n")
        fh.write(
            "| Comparison | n records | n model seeds | MEF delta TE | "
            "UTailoR delta TE | MEF - UTailoR | 95% CI | paired p | signal | "
            "MEF edits | UTailoR edits | TE/edit MEF/UTailoR | hard budget | "
            "budget matched | alignment |\n"
        )
        fh.write(
            "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---|"
            "---:|---:|---|\n"
        )
        for row in utailor_comparisons:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| `{row.get('comparison_id')}` | "
                f"{row.get('n_eligible_records')} | "
                f"{row.get('n_paired_model_seeds')} | "
                f"{_fmt(row.get('model_mean'))} | "
                f"{_fmt(row.get('utailor_deterministic_mean'))} | "
                f"{_fmt(row.get('model_minus_utailor'))} | "
                f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] | "
                f"{_fmt(row.get('paired_p'))} | "
                f"`{row.get('signal')}` | "
                f"{_fmt(row.get('model_mean_edit_distance'))} | "
                f"{_fmt(row.get('utailor_mean_edit_distance'))} | "
                f"{_fmt(row.get('model_te_delta_per_edit'))}/"
                f"{_fmt(row.get('utailor_te_delta_per_edit'))} | "
                f"{row.get('model_hard_edit_budget') or 'NA'} | "
                f"`{row.get('edit_budget_matched')}` | "
                f"`{row.get('edit_budget_alignment')}` |\n"
            )
        fh.write("\n## Remaining Gate\n\n")
        fh.write(f"{summary.get('remaining_model_fairness_action')}\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument(
        "--out-json",
        default="docs/t5_external_utr_baseline_comparison.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/t5_external_utr_baseline_comparison.md",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = os.path.abspath(args.project_root)
    out_json = (
        args.out_json
        if os.path.isabs(args.out_json)
        else os.path.join(root, args.out_json)
    )
    out_md = (
        args.out_md
        if os.path.isabs(args.out_md)
        else os.path.join(root, args.out_md)
    )
    report = build_t5_external_utr_comparison(root)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(
        json.dumps(
            {
                "json_path": out_json,
                "markdown_path": out_md,
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_t5_external_utr_comparison",
    "write_report_json",
    "write_report_markdown",
    "main",
]
