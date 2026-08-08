"""Fail-closed R3X ordinary-only runner and synthetic engineering fixture."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, NoReturn, Sequence

import numpy as np
import torch

from .artifact_io import atomic_publish_bytes
from .artifact_graph import (
    build_prediction_manifest,
    build_seed_commit,
    make_artifact_ref,
    replay_ensemble_manifest,
    replay_seed_commit,
    write_ensemble_from_seed_commits_exclusive,
    write_prediction_jsonl_exclusive,
    write_prediction_manifest_exclusive,
    write_seed_commit_exclusive,
)
from .control_summary import (
    ACTION_SUPPORT_NOT_QUALIFIED,
    build_not_run_negative_control_summary,
    load_negative_control_summary,
    write_negative_control_summary_exclusive,
)
from .baselines import (
    absolute_candidate_plus_anchor,
    anchor_only,
    candidate_kmer_ridge,
    edit_only_ridge,
    fit_optional_hist_gradient_boosting,
    foundation_baseline_status,
    group_balanced_prior,
    nearest_neighbor,
    paired_kmer_ridge,
    sign_logistic_diagnostic,
    zero_predictor,
)
from .baseline_selection import (
    BaselineSelectionError,
    DEVELOPMENT_PARTITION,
    QUALIFIED_STATUS as DEVELOPMENT_BASELINE_QUALIFIED_STATUS,
    build_development_baseline_evidence,
    build_development_baseline_selection,
    load_development_baseline_selection_exact,
    write_development_baseline_evidence_exclusive,
    write_development_baseline_selection_exclusive,
    write_development_evaluation_jsonl_exclusive,
)
from .checkpoint import (
    build_checkpoint,
    load_checkpoint_exact,
    save_checkpoint_exclusive,
    validate_checkpoint,
)
from .features import NUC_TO_INDEX, normalize_sequence, validated_substitutions
from .dataset import (
    _synthetic_engineering_materialization_scope,
    materialize_method_rows,
)
from .estimands import (
    PrimaryEstimandSlice,
    build_primary_estimand_slices,
    merge_estimand_seed_runs,
    row_order_sha256,
    stable_row_id,
)
from .information_sets import InformationSet, canonical_feature_row
from .global_prediction import (
    replay_global_prediction_commit,
    write_global_method_predictions_exclusive,
)
from .metrics import (
    component_balanced_cell_metrics,
    macro_metrics,
    within_pool_rank_metrics,
)
from .models import (
    ExactPotentialCritic,
    FullPairUnrestrictedSiamese,
    ModelConfig,
    ParameterMatchedUnrestrictedSiamese,
    R2StyleNoExplicitStudyCrossAttentionCritic,
    SparseEditSetCritic,
    count_parameters,
)
from .paired_artifact import (
    build_paired_comparison_artifact,
    build_verified_paired_reference,
    load_paired_comparison_artifact_exact,
    write_paired_comparison_artifact_exclusive,
)
from .protocol import (
    ProtocolHoldError,
    REQUIRED_ACTIVATION_AUTHORITIES,
    assert_ordinary_path,
    assert_ordinary_record,
    _audit_legacy_outcome_bearing_activation_manifest,
    canonical_sha256,
    load_route_authorization,
    read_exact_bytes,
    require_staged_disclosure_before_outcome_access,
    sha256_file,
)
from .report import (
    ReportProtocolError,
    evaluate_effect_gate,
    validate_run_report,
    write_json_exclusive,
)
from .selection_artifact import (
    build_checkpoint_selection_artifact,
    load_checkpoint_selection_artifact_exact,
    write_checkpoint_selection_artifact_exclusive,
)
from .run_complete import (
    ORDINARY_GO_CANDIDATE,
    SCIENTIFIC_HOLD,
    build_run_complete,
    replay_run_complete,
    write_run_complete_exclusive,
)
from .splits import (
    audit_component_disjointness,
    build_split_manifest,
    sequence_identity,
    verify_split_manifest_against_rows,
)
from .statistics import summarize_seeds
from .train import (
    apply_condition_unknown_dropout,
    balanced_epoch_indices,
    effect_loss,
    select_checkpoint,
)


EXECUTION_ORDER = [
    "PUBLICATION_GRADE_STRONG_BASELINES",
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION",
    "A_EXACT_POTENTIAL_DIFFERENCE",
    "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
    "B_SPARSE_EDIT_SET_ADDITIVE",
    "B_SPARSE_EDIT_SET_INTERACTION_ONLY_IF_ACTION_SUPPORT_PUBLICATION_QUALIFIED",
]
FROZEN_SEEDS = (11, 23, 42, 67, 101)

DEVELOPMENT_BASELINE_ELIGIBLE_ORDER = (
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
)

IMPLEMENTED_BASELINE_METHODS = {
    "ZERO_DELTA",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
    "EDIT_ONLY_RIDGE",
    "EDIT_ONLY_SIGN_LOGISTIC_DIAGNOSTIC",
    "CANDIDATE_KMER_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "HISTOGRAM_GRADIENT_BOOSTING",
    "MEASURED_ANCHOR_ONLY",
    "CANDIDATE_ABSOLUTE_MINUS_MEASURED_ANCHOR",
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _authorization() -> tuple[Path, str]:
    path = _repo_root() / "configs/r3x/r3_route_authorization_v1.json"
    return path, sha256_file(path)


def _digits(value: int, width: int) -> str:
    alphabet = "ACGU"
    output = []
    for _ in range(width):
        output.append(alphabet[value % 4])
        value //= 4
    return "".join(reversed(output))


def synthetic_rows(n_rows: int) -> list[dict]:
    if n_rows < 18:
        raise ValueError("synthetic fixture requires at least 18 independent rows")
    rows = []
    for index in range(n_rows):
        source = "A" + _digits(index, 7)
        candidate = "C" + source[1:]
        source_value = 1.0 + index / (10.0 * n_rows)
        gc = (source.count("G") + source.count("C")) / len(source)
        motif = 0.15 if "GG" in source else -0.10 if "UU" in source else 0.0
        delta = 2.0 * (gc - 0.40) + motif
        row_id = hashlib.sha256(f"synthetic-r3x-{index}".encode()).hexdigest()
        evidence = hashlib.sha256(f"synthetic-alias-{index}".encode()).hexdigest()
        rows.append({
            "r3_input_row_sha256": row_id,
            "record_id": f"synthetic_{index:03d}",
            "study": "synthetic_study",
            "endpoint": "synthetic_endpoint",
            "endpoint_id": "synthetic_endpoint",
            "publication_context_id": "synthetic_context",
            "context_id_v2": "synthetic_context",
            "region": "5UTR",
            "normalized_source_sequence": source,
            "normalized_candidate_sequence": candidate,
            "source_sequence": source,
            "candidate_sequence": candidate,
            "source_sequence_sha256": sequence_identity(source),
            "candidate_sequence_sha256": sequence_identity(candidate),
            "edit_list": [{"op": "SUB", "pos": 0, "ref": "A", "alt": "C"}],
            "edit_count": 1,
            "actual_edit_count": 1,
            "source_group_id_v2": f"synthetic_pool_{index:03d}",
            "frozen_relation_atoms": [{
                "atom_type": "SOURCE_ALIAS",
                "atom_id": f"synthetic_alias_{index:03d}",
                "member": "SOURCE",
                "status": "FROZEN",
                "evidence_sha256": evidence,
            }],
            "source_value": source_value,
            "source_anchor_status": "MEASURED_SINGLE_VALUE",
            "candidate_value": source_value + delta,
            "delta": delta,
        })
    return rows


def _attach_split(rows: Sequence[Mapping], split: Mapping) -> list[dict]:
    output = []
    assignments = split["row_assignments"]
    for row in rows:
        row_id = row.get("r3_input_row_sha256") or row.get("record_id")
        if row_id not in assignments:
            raise ValueError(f"split manifest lacks row {row_id}")
        changed = copy.deepcopy(dict(row))
        changed["source_component_id"] = assignments[row_id]["source_component_id"]
        changed["partition"] = assignments[row_id]["partition"]
        output.append(changed)
    return output


def _cell(row: Mapping) -> str:
    return "|".join((
        str(row.get("study") or ""),
        str(row.get("estimand_id") or row.get("endpoint_id") or row.get("endpoint") or ""),
        str(row.get("context_id_v2") or row.get("publication_context_id") or ""),
        str(row.get("actual_edit_count") or row.get("edit_count") or ""),
    ))


def _metric_bundle(rows: Sequence[Mapping], predictions: np.ndarray, logvar: np.ndarray | None = None) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("predictions do not align with evaluation rows")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_cell(row)].append(index)
    cells = {}
    for key, indices in sorted(groups.items()):
        y = np.asarray([float(rows[index]["delta"]) for index in indices])
        pred = np.asarray(predictions)[indices]
        lv = None if logvar is None else np.asarray(logvar)[indices]
        components = [str(rows[index].get("source_component_id") or "") for index in indices]
        pools = [str(rows[index].get("source_group_id_v2") or "") for index in indices]
        cells[key] = component_balanced_cell_metrics(
            y, pred, components, logvar=lv
        )
        pool_metrics = within_pool_rank_metrics(y, pred, pools, components)
        cells[key]["delta_spearman"] = pool_metrics["spearman"]
        cells[key]["ranking_pool_metrics"] = pool_metrics
        cells[key]["selection_gain"] = {
            "metric": "POOL_MACRO_ORACLE_NORMALIZED_SELECTION_GAIN",
            "value": pool_metrics["selection_gain"],
            "status": pool_metrics["status"],
            "tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        }
    return {"cells": cells, "macro": macro_metrics(cells)}


def _row_set_sha256(rows: Sequence[Mapping]) -> str:
    ids = [str(row.get("r3_input_row_sha256") or row.get("record_id") or "") for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise ReportProtocolError("method rows require unique stable IDs")
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _project_pair(
    train_rows: Sequence[Mapping], test_rows: Sequence[Mapping], information_set: InformationSet,
    *, target_fields: Sequence[str],
) -> tuple[list[dict], list[dict], str]:
    train, train_fingerprint = materialize_method_rows(
        train_rows, information_set, target_fields=target_fields
    )
    test, test_fingerprint = materialize_method_rows(
        test_rows, information_set, target_fields=target_fields
    )
    if train_fingerprint != test_fingerprint:
        raise ReportProtocolError("train/test information-set fingerprints differ")
    return train, test, train_fingerprint


def _baseline_methods(
    train_rows: list[dict], test_rows: list[dict], *, optional_hist_authorized: bool = False
) -> dict:
    methods = {}
    runners = [
        ("ZERO_DELTA", zero_predictor, InformationSet.NO_ANCHOR_ZERO_PRIOR, ("delta",)),
        ("SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR", group_balanced_prior, InformationSet.NO_ANCHOR_ZERO_PRIOR, ("delta",)),
        ("EDIT_ONLY_RIDGE", edit_only_ridge, InformationSet.EDIT_ONLY, ("delta",)),
        ("CANDIDATE_KMER_RIDGE", candidate_kmer_ridge, InformationSet.CANDIDATE_ONLY, ("delta",)),
        ("PAIRED_LOCAL_GLOBAL_KMER_RIDGE", paired_kmer_ridge, InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR, ("delta",)),
        ("PAIRED_FEATURE_NEAREST_NEIGHBOR", nearest_neighbor, InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR, ("delta",)),
        ("MEASURED_ANCHOR_ONLY", anchor_only, InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED, ("delta", "candidate_value")),
        ("CANDIDATE_ABSOLUTE_MINUS_MEASURED_ANCHOR", absolute_candidate_plus_anchor, InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED, ("delta", "candidate_value")),
    ]
    row_set_sha = _row_set_sha256(test_rows)
    for method_id, runner, information_set, targets in runners:
        projected_train, projected_test, fingerprint = _project_pair(
            train_rows, test_rows, information_set, target_fields=targets
        )
        result = runner(projected_train, projected_test)
        methods[method_id] = {
            "status": result.status,
            "information_set": result.information_set,
            "leaderboard": result.leaderboard,
            "role": result.role,
            "metrics": _metric_bundle(test_rows, result.predictions),
            "metadata": {
                **dict(result.metadata),
                "information_set_fingerprint": fingerprint,
                "prediction_row_set_sha256": row_set_sha,
            },
            "predictions": [float(value) for value in result.predictions],
        }
    sign_train, sign_test, sign_fingerprint = _project_pair(
        train_rows, test_rows, InformationSet.EDIT_ONLY, target_fields=("delta",)
    )
    sign = sign_logistic_diagnostic(sign_train, sign_test)
    methods["EDIT_ONLY_SIGN_LOGISTIC_DIAGNOSTIC"] = {
        "status": sign.status,
        "information_set": sign.information_set,
        "leaderboard": sign.leaderboard,
        "role": sign.role,
        "metrics": "NOT_EFFECT_GATE_SUBSTITUTE",
        "metadata": {
            **dict(sign.metadata),
            "information_set_fingerprint": sign_fingerprint,
            "prediction_row_set_sha256": row_set_sha,
        },
    }
    hist_train, hist_test, hist_fingerprint = _project_pair(
        train_rows, test_rows, InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        target_fields=("delta",),
    )
    optional = fit_optional_hist_gradient_boosting(
        hist_train, hist_test, execution_authorized=optional_hist_authorized
    )
    methods["HISTOGRAM_GRADIENT_BOOSTING"] = {
        "status": optional.status,
        "information_set": optional.information_set,
        "leaderboard": optional.leaderboard,
        "reason": optional.reason,
        "metadata": {
            **dict(optional.metadata),
            "information_set_fingerprint": hist_fingerprint,
            "prediction_row_set_sha256": row_set_sha,
        },
    }
    if optional.status == "RUN":
        methods["HISTOGRAM_GRADIENT_BOOSTING"]["metrics"] = _metric_bundle(
            test_rows, optional.predictions
        )
        methods["HISTOGRAM_GRADIENT_BOOSTING"]["predictions"] = [
            float(value) for value in optional.predictions
        ]
    foundation = foundation_baseline_status(exposure_qualified=False, feature_path=None)
    methods["FOUNDATION_EMBEDDING_LINEAR_HEAD"] = {
        "status": foundation.status,
        "information_set": foundation.information_set,
        "leaderboard": foundation.leaderboard,
        "reason": foundation.reason,
    }
    return methods


def _canonical_partition_rows(
    slices: Sequence[PrimaryEstimandSlice], partition: str
) -> list[dict]:
    rows = [dict(row) for item in slices for row in item.rows_for(partition)]
    rows.sort(key=stable_row_id)
    ids = [stable_row_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ReportProtocolError("primary-estimand partitions contain overlapping row identities")
    return rows


def _validate_estimand_model_capacity(
    slices: Sequence[PrimaryEstimandSlice], cfg: ModelConfig
) -> dict:
    if not slices:
        raise ReportProtocolError("model capacity preflight requires primary-estimand slices")
    vocabs = {}
    max_endpoint_levels = 0
    max_context_levels = 0
    for item in slices:
        vocab = _vocab(item.train)
        max_endpoint_levels = max(max_endpoint_levels, len(vocab["endpoint"]))
        max_context_levels = max(max_context_levels, len(vocab["context"]))
        for row in (*item.train, *item.development, *item.internal_test):
            source = normalize_sequence(
                row.get("source_sequence") or row.get("normalized_source_sequence")
            )
            candidate = normalize_sequence(
                row.get("candidate_sequence") or row.get("normalized_candidate_sequence")
            )
            if len(source) != len(candidate) or len(source) > cfg.max_len:
                raise ReportProtocolError(
                    f"estimand {item.estimand_id} has unequal or over-max sequence input"
                )
            if len(validated_substitutions(row)) > cfg.max_edits:
                raise ReportProtocolError(
                    f"estimand {item.estimand_id} exceeds frozen edit capacity"
                )
        vocabs[item.estimand_id] = {
            "vocab": vocab,
            "vocab_sha256": canonical_sha256(vocab),
            "partition_binding_sha256": item.partition_binding_sha256,
        }
    expected_endpoints = max_endpoint_levels
    expected_contexts = max_context_levels
    if cfg.n_endpoints != expected_endpoints or cfg.n_contexts != expected_contexts:
        raise ReportProtocolError(
            "frozen model condition capacity must equal one UNK row plus the maximum "
            "train-only per-estimand vocabulary size"
        )
    return {
        "status": "PASS_PER_ESTIMAND_TRAIN_ONLY_VOCAB_CAPACITY",
        "n_endpoints": expected_endpoints,
        "n_contexts": expected_contexts,
        "per_estimand": vocabs,
    }


def _baseline_methods_per_estimand(
    slices: Sequence[PrimaryEstimandSlice],
    global_test_rows: Sequence[Mapping],
    *,
    optional_hist_authorized: bool = False,
) -> dict:
    per_estimand = {
        item.estimand_id: _baseline_methods(
            list(item.train), list(item.internal_test),
            optional_hist_authorized=optional_hist_authorized,
        )
        for item in slices
    }
    method_sets = {tuple(sorted(methods)) for methods in per_estimand.values()}
    if len(method_sets) != 1:
        raise ReportProtocolError("baseline method set changes across estimands")
    expected_ids = [stable_row_id(row) for row in global_test_rows]
    output = {}
    slice_index = {item.estimand_id: item for item in slices}
    for method_id in next(iter(method_sets)):
        pieces = {
            estimand_id: methods[method_id]
            for estimand_id, methods in per_estimand.items()
        }
        statuses = {str(piece.get("status") or "") for piece in pieces.values()}
        prediction_capable = all(
            piece.get("status") == "RUN" and isinstance(piece.get("predictions"), list)
            for piece in pieces.values()
        )
        if prediction_capable:
            information_sets = {piece.get("information_set") for piece in pieces.values()}
            fingerprints = {
                piece.get("metadata", {}).get("information_set_fingerprint")
                for piece in pieces.values()
            }
            if len(information_sets) != 1 or len(fingerprints) != 1 or None in fingerprints:
                raise ReportProtocolError(
                    f"baseline {method_id} changes information-set contract across estimands"
                )
            predictions_by_id = {}
            for estimand_id, piece in pieces.items():
                row_ids = slice_index[estimand_id].row_ids_for("internal_test")
                values = piece["predictions"]
                if len(row_ids) != len(values):
                    raise ReportProtocolError(
                        f"baseline {method_id} prediction count differs in estimand {estimand_id}"
                    )
                for row_id, value in zip(row_ids, values):
                    if row_id in predictions_by_id or not math.isfinite(float(value)):
                        raise ReportProtocolError(
                            f"baseline {method_id} has duplicate or non-finite predictions"
                        )
                    predictions_by_id[row_id] = float(value)
            if set(predictions_by_id) != set(expected_ids):
                raise ReportProtocolError(
                    f"baseline {method_id} does not cover the complete global test row set"
                )
            predictions = [predictions_by_id[row_id] for row_id in expected_ids]
            first = pieces[sorted(pieces)[0]]
            output[method_id] = {
                "status": "RUN",
                "information_set": next(iter(information_sets)),
                "leaderboard": first.get("leaderboard"),
                "role": first.get("role"),
                "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
                "metrics": _metric_bundle(global_test_rows, np.asarray(predictions)),
                "predictions": predictions,
                "metadata": {
                    "information_set_fingerprint": next(iter(fingerprints)),
                    "prediction_row_set_sha256": _row_set_sha256(global_test_rows),
                    "partition_binding_sha256_by_estimand": {
                        item.estimand_id: item.partition_binding_sha256 for item in slices
                    },
                },
                "per_estimand": {
                    estimand_id: {
                        key: copy.deepcopy(value)
                        for key, value in piece.items() if key != "predictions"
                    }
                    for estimand_id, piece in pieces.items()
                },
            }
        elif statuses == {"RUN"}:
            # The sign-only diagnostic is intentionally not converted into a
            # regression prediction or an effect-gate substitute.
            output[method_id] = {
                "status": "RUN_DIAGNOSTIC_PER_ESTIMAND",
                "role": "DIAGNOSTIC_ONLY_NOT_EFFECT_GATE",
                "per_estimand": copy.deepcopy(pieces),
            }
        else:
            output[method_id] = {
                "status": "NOT_RUN_INCOMPLETE_ESTIMAND_COVERAGE",
                "per_estimand": copy.deepcopy(pieces),
            }
    return output


def _vocab(train_rows: Sequence[Mapping]) -> dict:
    endpoints = sorted({str(row.get("endpoint")) for row in train_rows})
    contexts = sorted({str(row.get("publication_context_id")) for row in train_rows})
    if "__UNK__" in endpoints or "__UNK__" in contexts:
        raise ReportProtocolError("ordinary condition labels cannot impersonate explicit __UNK__")
    return {
        "endpoint": {
            "__UNK__": 0,
            **{value: index + 1 for index, value in enumerate(endpoints)},
        },
        "context": {
            "__UNK__": 0,
            **{value: index + 1 for index, value in enumerate(contexts)},
        },
    }


def _tensorize(
    rows: Sequence[Mapping],
    vocab: Mapping,
    cfg: ModelConfig,
    *,
    include_target: bool = True,
) -> dict:
    n = len(rows)
    source = torch.zeros((n, cfg.max_len), dtype=torch.long)
    candidate = torch.zeros_like(source)
    sequence_mask = torch.zeros_like(source, dtype=torch.bool)
    positions = torch.zeros((n, cfg.max_edits), dtype=torch.long)
    ref = torch.zeros_like(positions)
    alt = torch.zeros_like(positions)
    edit_mask = torch.zeros_like(positions, dtype=torch.bool)
    endpoint = torch.zeros(n, dtype=torch.long)
    context = torch.zeros(n, dtype=torch.long)
    target = torch.zeros(n, dtype=torch.float32)
    base_to_token = {base: index + 1 for index, base in enumerate("ACGU")}
    for row_index, row in enumerate(rows):
        src = normalize_sequence(row.get("source_sequence") or row.get("normalized_source_sequence"))
        cand = normalize_sequence(row.get("candidate_sequence") or row.get("normalized_candidate_sequence"))
        if len(src) != len(cand) or len(src) > cfg.max_len:
            raise ValueError("R3X tensorizer refuses unequal or over-max sequences; no truncation")
        edits = validated_substitutions(row)
        if len(edits) not in {1, 3, 5}:
            raise ValueError("R3X primary tensorizer accepts edit counts {1,3,5} only")
        source[row_index, : len(src)] = torch.tensor([base_to_token[base] for base in src])
        candidate[row_index, : len(cand)] = torch.tensor([base_to_token[base] for base in cand])
        sequence_mask[row_index, : len(src)] = True
        for edit_index, (pos, source_nt, target_nt) in enumerate(edits):
            positions[row_index, edit_index] = pos
            ref[row_index, edit_index] = base_to_token[source_nt]
            alt[row_index, edit_index] = base_to_token[target_nt]
            edit_mask[row_index, edit_index] = True
        endpoint_name = str(row.get("endpoint"))
        context_name = str(row.get("publication_context_id"))
        endpoint[row_index] = int(vocab["endpoint"].get(endpoint_name, 0))
        context[row_index] = int(vocab["context"].get(context_name, 0))
        if include_target:
            target[row_index] = float(row["delta"])
    tensors = {
        "source": source,
        "candidate": candidate,
        "sequence_mask": sequence_mask,
        "positions": positions,
        "ref": ref,
        "alt": alt,
        "edit_mask": edit_mask,
        "endpoint": endpoint,
        "context": context,
    }
    if include_target:
        tensors["target"] = target
    return tensors


def _forward(model, batch: Mapping[str, torch.Tensor]) -> dict:
    if isinstance(model, SparseEditSetCritic):
        return model(
            batch["source"], batch["sequence_mask"], batch["positions"],
            batch["ref"], batch["alt"], batch["edit_mask"],
            batch["endpoint"], batch["context"],
            candidate_tokens=batch["candidate"],
        )
    return model(
        batch["source"], batch["candidate"], batch["sequence_mask"],
        batch["sequence_mask"], batch["endpoint"], batch["context"],
    )


def _index(batch: Mapping[str, torch.Tensor], indices: Sequence[int]) -> dict:
    index = torch.tensor(indices, dtype=torch.long)
    return {key: value[index] for key, value in batch.items()}


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _predict_batched(
    model,
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    means, logvars = [], []
    with torch.no_grad():
        n_rows = int(tensors["source"].shape[0])
        for start in range(0, n_rows, int(batch_size)):
            indices = range(start, min(start + int(batch_size), n_rows))
            output = _forward(model, _to_device(_index(tensors, indices), device))
            means.append(output["mean"].detach().cpu().numpy())
            logvars.append(output["logvar"].detach().cpu().numpy())
    return {"mean": np.concatenate(means), "logvar": np.concatenate(logvars)}


def _train_neural(
    model,
    train_rows: list[dict],
    dev_rows: list[dict],
    test_rows: list[dict] | None,
    *,
    cfg: ModelConfig,
    seed: int,
    epochs: int,
    batch_size: int = 128,
    device_name: str = "cpu",
    condition_unknown_dropout: float = 0.15,
    optimizer_config: Mapping | None = None,
    loss_config: Mapping | None = None,
    score_internal_test: bool = True,
    return_selected_model: bool = False,
) -> dict:
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("frozen run requests CUDA but CUDA is unavailable")
    device = torch.device(device_name)
    train_rows, train_fingerprint = materialize_method_rows(
        train_rows,
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        target_fields=("delta",),
    )
    dev_rows, dev_fingerprint = materialize_method_rows(
        dev_rows,
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        target_fields=("delta",),
    )
    if train_fingerprint != dev_fingerprint:
        raise ReportProtocolError("neural train/development information sets differ")
    materialized_test: list[dict] | None = None
    if score_internal_test:
        if test_rows is None:
            raise ReportProtocolError("internal-test scoring requires explicit test rows")
        materialized_test, test_fingerprint = materialize_method_rows(
            test_rows,
            InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
            target_fields=("delta",),
        )
        if test_fingerprint != train_fingerprint:
            raise ReportProtocolError("neural partitions use unequal information-set contracts")
    elif test_rows is not None:
        raise ReportProtocolError(
            "development-only fit must not receive internal-test rows"
        )
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    vocab = _vocab(train_rows)
    train = _tensorize(train_rows, vocab, cfg)
    dev = _tensorize(dev_rows, vocab, cfg)
    test = (
        _tensorize(materialized_test, vocab, cfg)
        if materialized_test is not None else None
    )
    model = model.to(device)
    optimizer_config = dict(optimizer_config or {
        "name": "Adam", "learning_rate": 2e-3, "weight_decay": 1e-4,
    })
    loss_config = dict(loss_config or {
        "huber_weight": 1.0, "nll_weight": 0.25,
        "sign_weight": 0.25, "sign_temperature": 1.0,
    })
    if set(optimizer_config) != {"name", "learning_rate", "weight_decay"} or optimizer_config["name"] != "Adam":
        raise ReportProtocolError("R3X runner currently supports the exact frozen Adam optimizer schema")
    lr = optimizer_config["learning_rate"]
    weight_decay = optimizer_config["weight_decay"]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0 for value in (lr, weight_decay)) or float(lr) <= 0.0:
        raise ReportProtocolError("frozen optimizer values are invalid")
    if set(loss_config) != {"huber_weight", "nll_weight", "sign_weight", "sign_temperature"}:
        raise ReportProtocolError("frozen loss config fields are incomplete or unexpected")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    epoch_metrics = []
    unknown_dropout_manifests = []
    states = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        indices = balanced_epoch_indices(train_rows, seed=int(seed) + epoch)
        presented = _index(train, indices)
        masked_endpoint, masked_context, dropout_manifest = apply_condition_unknown_dropout(
            presented["endpoint"], presented["context"],
            rate=condition_unknown_dropout, seed=int(seed) * 100000 + epoch,
        )
        presented["endpoint"] = masked_endpoint
        presented["context"] = masked_context
        unknown_dropout_manifests.append({"epoch": epoch, **dropout_manifest})
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = range(start, min(start + int(batch_size), len(indices)))
            batch = _to_device(_index(presented, batch_indices), device)
            optimizer.zero_grad()
            outputs = _forward(model, batch)
            losses = effect_loss(outputs, batch["target"], **loss_config)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        dev_out = _predict_batched(
            model, dev, batch_size=batch_size, device=device
        )
        dev_bundle = _metric_bundle(
            dev_rows,
            dev_out["mean"],
            dev_out["logvar"],
        )
        macro = dev_bundle["macro"]
        if macro["macro_delta_spearman"] is None:
            # Selector requires a finite value; undefined rank evidence is
            # retained as the worst possible development value.
            macro["macro_delta_spearman"] = -1.0
        epoch_metrics.append({"epoch": epoch, **macro})
        states[epoch] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    selection = select_checkpoint(epoch_metrics)
    model.load_state_dict(states[selection["selected_epoch"]])
    test_out = None
    test_bundle = None
    if score_internal_test:
        assert test is not None and materialized_test is not None
        test_out = _predict_batched(
            model, test, batch_size=batch_size, device=device
        )
        test_bundle = _metric_bundle(
            materialized_test,
            test_out["mean"],
            test_out["logvar"],
        )
    wall_time = time.perf_counter() - started
    train_presentations = len(balanced_epoch_indices(train_rows, seed=int(seed) + 1)) * int(epochs)
    encoder_multiplier = 1 if isinstance(model, SparseEditSetCritic) else 2
    result = {
        "status": "RUN",
        "architecture_id": model.architecture_id,
        "information_set": model.information_set,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": str(device),
        "condition_unknown_dropout": {
            "rate": float(condition_unknown_dropout),
            "epoch_manifests": unknown_dropout_manifests,
        },
        "optimizer_config": optimizer_config,
        "loss_config": loss_config,
        "parameter_count": count_parameters(model),
        "information_set_fingerprint": train_fingerprint,
        "checkpoint_selection": selection,
        "forward_equivalent_accounting": {
            "train_rows_presented": train_presentations,
            "development_rows_scored": len(dev_rows) * int(epochs),
            "internal_test_rows_scored": len(materialized_test or ()),
            "sequence_encoder_forward_equivalents": encoder_multiplier * (
                train_presentations + len(dev_rows) * int(epochs)
                + len(materialized_test or ())
            ),
            "encoder_sequences_per_row_forward": encoder_multiplier,
            "wall_time_seconds": float(wall_time),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "status": "MEASURED_WALL_TIME_AND_SEQUENCE_ENCODER_EQUIVALENTS",
        },
    }
    if score_internal_test:
        assert test_out is not None and test_bundle is not None
        result["test_metrics"] = test_bundle
        result["test_predictions"] = {
            "row_ids": [
                str(row.get("r3_input_row_sha256") or row.get("record_id"))
                for row in materialized_test
            ],
            "mean": [float(value) for value in test_out["mean"]],
            "logvar": [float(value) for value in test_out["logvar"]],
            "row_set_sha256": _row_set_sha256(materialized_test),
        }
    if return_selected_model:
        # Private runner-only values are removed before any JSON report is
        # assembled.  They allow the caller to publish the selected state and
        # then score internal-test rows only from a fresh exact checkpoint load.
        result["_selected_model"] = model
        result["_condition_vocab"] = vocab
        result["_development_epoch_metrics"] = epoch_metrics
    return result


@_synthetic_engineering_materialization_scope()
def run_synthetic_engineering_fixture(
    *,
    output_dir: Path | None,
    seed: int = 11,
    epochs: int = 1,
    n_rows: int = 30,
) -> dict:
    authorization_path, authorization_sha = _authorization()
    load_route_authorization(authorization_path, authorization_sha, require_training_ready=False)
    if output_dir is not None:
        output_dir = Path(output_dir)
        assert_ordinary_path(output_dir, purpose="synthetic R3X output")
        if output_dir.exists():
            raise ReportProtocolError(f"synthetic output already exists: {output_dir}")
        if not output_dir.parent.is_dir():
            raise ReportProtocolError("synthetic output parent must already exist")
    rows = synthetic_rows(n_rows)
    split = build_split_manifest(rows, seed=seed)
    rows = _attach_split(rows, split)
    train_rows = [row for row in rows if row["partition"] == "train"]
    dev_rows = [row for row in rows if row["partition"] == "development"]
    test_rows = [row for row in rows if row["partition"] == "internal_test"]
    if min(map(len, (train_rows, dev_rows, test_rows))) < 1:
        raise ValueError("synthetic split is degenerate")
    methods = _baseline_methods(train_rows, test_rows)
    vocab = _vocab(train_rows)
    cfg = ModelConfig(
        d_model=8, n_heads=2, n_layers=1, ff_dim=16, max_len=8,
        n_endpoints=len(vocab["endpoint"]) + 1,
        n_contexts=len(vocab["context"]) + 1,
        pair_rank=2,
    )
    # The unrestricted full-pair and de-studied R2 comparators are trained
    # before A, on the same projected rows and paired-sequence information set.
    # The older low-rank A control remains a transparent algebraic diagnostic;
    # it is not eligible to stand in for either publication-grade comparator.
    torch.manual_seed(seed)
    full_pair = FullPairUnrestrictedSiamese(cfg)
    methods["FULL_PAIR_UNRESTRICTED_SIAMESE"] = _train_neural(
        full_pair, train_rows, dev_rows, test_rows, cfg=cfg, seed=seed, epochs=epochs,
    )
    methods["FULL_PAIR_UNRESTRICTED_SIAMESE"]["role"] = (
        "PREREGISTERED_STRONG_SAME_INFORMATION_SET_COMPARATOR_"
        "PENDING_DEVELOPMENT_SELECTOR"
    )
    torch.manual_seed(seed)
    r2_no_study = R2StyleNoExplicitStudyCrossAttentionCritic(cfg)
    methods["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"] = _train_neural(
        r2_no_study, train_rows, dev_rows, test_rows, cfg=cfg, seed=seed, epochs=epochs,
    )
    methods["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"]["role"] = (
        "DEVELOPMENT_BASELINE_PENDING_ARCHITECTURE_CROSSWALK_AND_STUDY_PROXY_AUDIT"
    )
    methods["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"]["strongest_baseline_eligible"] = False
    torch.manual_seed(seed)
    exact = ExactPotentialCritic(cfg)
    methods["A_EXACT_POTENTIAL_DIFFERENCE"] = _train_neural(
        exact, train_rows, dev_rows, test_rows, cfg=cfg, seed=seed, epochs=epochs,
    )
    methods["A_EXACT_POTENTIAL_DIFFERENCE"]["role"] = "PRIMARY_CRITIC_CANDIDATE"
    torch.manual_seed(seed)
    control = ParameterMatchedUnrestrictedSiamese(cfg)
    methods["A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"] = _train_neural(
        control, train_rows, dev_rows, test_rows, cfg=cfg, seed=seed, epochs=epochs,
    )
    methods["A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"]["role"] = (
        "LOW_RANK_ALGEBRAIC_DIAGNOSTIC_NOT_STRONGEST_BASELINE_ELIGIBLE"
    )
    full_pair_gap = abs(count_parameters(full_pair) - count_parameters(exact)) / count_parameters(exact)
    low_rank_gap = abs(count_parameters(control) - count_parameters(exact)) / count_parameters(exact)
    methods["FULL_PAIR_UNRESTRICTED_SIAMESE"]["parameter_gap_fraction_vs_A"] = full_pair_gap
    methods["A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"]["parameter_gap_fraction_vs_A"] = low_rank_gap
    torch.manual_seed(seed)
    additive = SparseEditSetCritic(cfg, interaction=False)
    methods["B_SPARSE_EDIT_SET_ADDITIVE"] = _train_neural(
        additive, train_rows, dev_rows, test_rows, cfg=cfg, seed=seed, epochs=epochs,
    )
    methods["B_SPARSE_EDIT_SET_INTERACTION"] = {
        "status": "NOT_RUN_ACTION_SUPPORT",
        "reason": "synthetic fixture has no frozen publication-qualified action-support policy",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
    }
    report = {
        "schema_version": "r3x_ordinary_run_report_v1",
        "status": "SYNTHETIC_ENGINEERING_ONLY",
        "route_authorization_sha256": authorization_sha,
        "execution_order": EXECUTION_ORDER,
        "seed": int(seed),
        "epochs": int(epochs),
        "split_manifest_sha256": split["manifest_sha256"],
        "scientific_scope": "SYNTHETIC_ENGINEERING_ONLY",
        "n_rows": {"train": len(train_rows), "development": len(dev_rows), "internal_test": len(test_rows)},
        "methods": methods,
        "GO_EFFECT": False,
        "downstream_unlock": False,
        "sealed_access": False,
        "active_contract_mutation": False,
        "sign_gate": 0.60,
        "stable_selection_metric_gate": "HOLD_PENDING_FROZEN_REAL_ORDINARY_POLICY",
        "limitations": [
            "SYNTHETIC_FIXTURE_IS_NOT_SCIENTIFIC_EFFECT_EVIDENCE",
            "NO_MODEL_FAMILY_WINNER_SELECTED",
            "NO_O0_G1_E0_UNLOCK",
        ],
    }
    validate_run_report(report)
    if output_dir is not None:
        output_dir.mkdir()
        write_json_exclusive(output_dir / "R3X_RUN_REPORT.json", report)
    return report


def preflight_real_ordinary(
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    *,
    prospective_output_dir: Path,
) -> NoReturn:
    require_staged_disclosure_before_outcome_access(
        activation_manifest_path,
        activation_manifest_sha256,
        prospective_output_dir=prospective_output_dir,
    )
    raise ProtocolHoldError(
        "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED",
        "staged disclosure guard returned without a stage-specific validator",
    )


def _read_json_exact(path: Path, expected_sha256: str, *, purpose: str) -> dict:
    raw = read_exact_bytes(path, expected_sha256, purpose=purpose)
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReportProtocolError(f"{purpose} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportProtocolError(f"{purpose} must be a JSON object")
    return payload


def _read_ready_jsonl_exact(path: Path, expected_sha256: str) -> list[dict]:
    raw = read_exact_bytes(path, expected_sha256, purpose="R3X ready dataset")
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output
    rows = []
    try:
        text = raw.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(row, dict):
                raise ReportProtocolError(f"ready dataset line {line_number} is not an object")
            assert_ordinary_record(row, purpose=f"ready dataset line {line_number}")
            rows.append(row)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReportProtocolError(f"ready dataset is not valid JSONL: {exc}") from exc
    if not rows:
        raise ReportProtocolError("ready dataset contains no rows")
    return rows


def _validate_ready_scientific_authorities(
    rows: Sequence[Mapping],
    inclusion: Mapping,
    crosswalk: Mapping,
    estimand_registry: Mapping,
) -> dict:
    inclusion_rows = inclusion.get("rows")
    if not isinstance(inclusion_rows, list):
        raise ReportProtocolError("inclusion manifest lacks row-level registry")
    inclusion_index = {str(item.get("row_id")): item for item in inclusion_rows if isinstance(item, Mapping)}
    if len(inclusion_index) != len(inclusion_rows) or len(inclusion_index) != len(rows):
        raise ReportProtocolError("inclusion row registry is not one-to-one")

    crosswalk_entries = crosswalk.get("entries")
    estimand_entries = estimand_registry.get("entries")
    if not isinstance(crosswalk_entries, list) or not isinstance(estimand_entries, list):
        raise ReportProtocolError("crosswalk/estimand authorities lack entries")
    crosswalk_fields = (
        "study", "endpoint", "assay_arm", "library", "source_asset",
        "region", "source_anchor_rule",
    )
    crosswalk_index = {}
    for entry in crosswalk_entries:
        key = tuple(str(entry.get(field) or "") for field in crosswalk_fields)
        if "" in key or key in crosswalk_index:
            raise ReportProtocolError("crosswalk keys must be complete and unique")
        crosswalk_index[key] = entry
    estimand_index = {}
    for entry in estimand_entries:
        key = (str(entry.get("study") or ""), str(entry.get("endpoint") or ""))
        if "" in key or key in estimand_index:
            raise ReportProtocolError("estimand keys must be complete and unique")
        estimand_index[key] = entry

    for raw in rows:
        canonical = canonical_feature_row(raw)
        row_id = str(raw.get("r3_input_row_sha256") or "")
        registered = inclusion_index.get(row_id)
        if registered is None or registered.get("role") != "ORDINARY_PRIMARY_READY":
            raise ReportProtocolError("ready row is absent from ordinary inclusion registry")
        if registered.get("row_content_sha256") != canonical_sha256(raw):
            raise ReportProtocolError("ready row content does not match inclusion registry")
        key = tuple(str(raw.get(field) or "") for field in crosswalk_fields)
        binding = crosswalk_index.get(key)
        if binding is None:
            raise ReportProtocolError("ready row is not covered by the frozen context/source crosswalk")
        if canonical.get("publication_context_id") != binding.get("publication_context_id"):
            raise ReportProtocolError("ready row publication context disagrees with crosswalk")
        endpoint_key = (str(raw.get("study") or ""), str(canonical.get("endpoint") or ""))
        estimand = estimand_index.get(endpoint_key)
        if estimand is None or str(raw.get("estimand_id") or "") != str(estimand.get("estimand_id") or ""):
            raise ReportProtocolError("ready row is not bound to the frozen endpoint estimand")
        source_value = raw.get(estimand["raw_source_field"])
        candidate_value = raw.get(estimand["raw_candidate_field"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (source_value, candidate_value)):
            raise ReportProtocolError("estimand raw values must be finite")
        source_value = float(source_value)
        candidate_value = float(candidate_value)
        transform = estimand.get("transform")
        if transform == "IDENTITY_DIFFERENCE":
            expected_delta = candidate_value - source_value
        elif transform == "NATURAL_LOG_RATIO":
            if source_value <= 0.0 or candidate_value <= 0.0:
                raise ReportProtocolError("log-ratio estimand requires positive values")
            expected_delta = math.log(candidate_value / source_value)
        elif transform == "DECAY_RATE_DIFFERENCE":
            if source_value <= 0.0 or candidate_value <= 0.0:
                raise ReportProtocolError("decay-rate estimand requires positive half-lives")
            expected_delta = math.log(2.0) / source_value - math.log(2.0) / candidate_value
        else:
            raise ReportProtocolError("unsupported frozen estimand transform")
        if estimand.get("direction") == "LOWER_IS_MORE_STABLE":
            expected_delta = -expected_delta
        observed_delta = raw.get("delta")
        if isinstance(observed_delta, bool) or not isinstance(observed_delta, (int, float)) or not math.isclose(float(observed_delta), expected_delta, rel_tol=1e-10, abs_tol=1e-12):
            raise ReportProtocolError("ready row delta does not replay the frozen estimand")
    return {
        "status": "PASS_READY_CROSSWALK_ESTIMAND_INCLUSION_BINDING",
        "n_rows": len(rows),
    }


def _aggregate_seed_runs(
    runs: Sequence[Mapping],
    test_rows: Sequence[Mapping],
    *,
    expected_seeds: Sequence[int] = FROZEN_SEEDS,
) -> dict:
    if not runs:
        raise ReportProtocolError("neural method has no seed runs")
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    actual_seeds = [run.get("seed") for run in runs]
    if (
        any(isinstance(seed, bool) or not isinstance(seed, int) for seed in actual_seeds)
        or len(actual_seeds) != len(set(actual_seeds))
        or set(actual_seeds) != set(expected_seeds)
        or len(actual_seeds) != len(expected_seeds)
    ):
        raise ReportProtocolError(
            "neural aggregation requires every frozen seed exactly once"
        )
    by_seed = {int(run["seed"]): run for run in runs}
    runs = [by_seed[seed] for seed in expected_seeds]
    architectures = {str(run.get("architecture_id") or "") for run in runs}
    information_sets = {str(run.get("information_set") or "") for run in runs}
    if len(architectures) != 1 or "" in architectures:
        raise ReportProtocolError("architecture changes across optimizer seeds")
    if len(information_sets) != 1 or "" in information_sets:
        raise ReportProtocolError("information set changes across optimizer seeds")
    metric_keys = (
        "macro_sign_accuracy", "macro_delta_spearman", "macro_rmse",
        "macro_mae", "macro_gaussian_nll",
    )
    summaries = {}
    for key in metric_keys:
        raw_values = {
            int(run["seed"]): run["test_metrics"]["macro"].get(key)
            for run in runs
        }
        if any(value is None for value in raw_values.values()):
            summaries[key] = {
                "status": "HOLD_AT_LEAST_ONE_SEED_METRIC_UNDEFINED",
                "values": {str(seed): value for seed, value in sorted(raw_values.items())},
                "inferential_unit": "NOT_BIOLOGICAL_REPLICATES",
            }
        else:
            summaries[key] = summarize_seeds(
                {seed: float(value) for seed, value in raw_values.items()}
            )
    parameter_counts = {int(run["parameter_count"]) for run in runs}
    if len(parameter_counts) != 1:
        raise ReportProtocolError("parameter count changes across seeds")
    fingerprints = {str(run.get("information_set_fingerprint") or "") for run in runs}
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ReportProtocolError("information-set fingerprint changes across seeds")
    prediction_rows = [run["test_predictions"] for run in runs]
    expected_ids = prediction_rows[0]["row_ids"]
    true_expected_ids = [stable_row_id(row) for row in test_rows]
    if expected_ids != true_expected_ids:
        raise ReportProtocolError(
            "seed prediction row order does not equal the supplied test-row order"
        )
    if any(prediction["row_ids"] != expected_ids for prediction in prediction_rows[1:]):
        raise ReportProtocolError("seed prediction row order/identity differs")
    row_set_hashes = {str(prediction.get("row_set_sha256") or "") for prediction in prediction_rows}
    if len(row_set_hashes) != 1 or "" in row_set_hashes:
        raise ReportProtocolError("prediction row-set hash changes across seeds")
    expected_row_set_hash = _row_set_sha256(test_rows)
    if row_set_hashes != {expected_row_set_hash}:
        raise ReportProtocolError("seed prediction row-set hash disagrees with supplied test rows")
    means = np.asarray([prediction["mean"] for prediction in prediction_rows], dtype=np.float64)
    logvars = np.asarray([prediction["logvar"] for prediction in prediction_rows], dtype=np.float64)
    expected_shape = (len(runs), len(test_rows))
    if means.shape != expected_shape or logvars.shape != expected_shape:
        raise ReportProtocolError("seed prediction arrays do not align with the test rows")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(logvars)):
        raise ReportProtocolError("seed prediction arrays contain non-finite values")
    ensemble_mean = np.mean(means, axis=0)
    ensemble_variance = np.mean(np.exp(logvars) + means**2, axis=0) - ensemble_mean**2
    ensemble_variance = np.maximum(ensemble_variance, 1e-12)
    ensemble_logvar = np.log(ensemble_variance)
    ensemble_metrics = _metric_bundle(
        test_rows, ensemble_mean, ensemble_logvar
    )
    return {
        "status": "RUN",
        "architecture_id": architectures.pop(),
        "information_set": information_sets.pop(),
        "information_set_fingerprint": fingerprints.pop(),
        "prediction_row_set_sha256": row_set_hashes.pop(),
        "parameter_count": parameter_counts.pop(),
        "seed_metric_summaries": summaries,
        "seed_runs": {str(run["seed"]): dict(run) for run in runs},
        "seed_inference_status": "DESCRIPTIVE_OPTIMIZATION_REPEATS_NOT_BIOLOGICAL_REPLICATES",
        "primary_prediction_aggregation": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
        "ensemble_prediction": {
            "row_ids": expected_ids,
            "row_set_sha256": prediction_rows[0]["row_set_sha256"],
            "mean": [float(value) for value in ensemble_mean],
            "logvar_total_variance": [float(value) for value in ensemble_logvar],
            "n_models": len(runs),
            "inference_cost_multiplier": len(runs),
        },
        "ensemble_test_metrics": ensemble_metrics,
    }


def _selection_epoch_projection(records: Sequence[Mapping]) -> list[dict]:
    return [
        {
            "epoch": record["epoch"],
            "macro_sign_accuracy": record["macro_sign_accuracy"],
            "macro_delta_spearman": record["macro_delta_spearman"],
            "macro_gaussian_nll": record["macro_gaussian_nll"],
        }
        for record in records
    ]


def _artifact_unit_directory(
    run_root: Path,
    *,
    method_id: str,
    item: PrimaryEstimandSlice,
    seed: int,
) -> Path:
    if not method_id or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in method_id):
        raise ReportProtocolError("neural artifact method ID is not path-safe")
    unit = (
        Path(run_root) / "neural_artifacts" / method_id
        / item.partition_binding_sha256[:24] / f"seed_{int(seed)}"
    )
    unit.mkdir(parents=True)
    return unit


def _development_cell_id(row: Mapping) -> str:
    """Return a path/schema-safe immutable ID for one registered metric cell."""

    raw = _cell(row)
    if not raw or raw == "|||":
        raise ReportProtocolError(
            "development selector cannot derive a registered metric cell"
        )
    return "CELL_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24].upper()


def _development_neural_unit_directory(
    run_root: Path,
    *,
    method_id: str,
    item: PrimaryEstimandSlice,
    seed: int,
) -> Path:
    """Keep selector checkpoints in a literal development-only namespace."""

    if not method_id or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for character in method_id
    ):
        raise ReportProtocolError(
            "development neural predictor method ID is not path-safe"
        )
    unit = (
        Path(run_root)
        / "development"
        / "predictors"
        / method_id
        / item.partition_binding_sha256[:24]
        / f"seed_{int(seed)}"
    )
    unit.mkdir(parents=True)
    return unit


def _commit_and_predict_development_neural_seed(
    *,
    run: dict,
    factory,
    item: PrimaryEstimandSlice,
    cfg: ModelConfig,
    method_id: str,
    seed: int,
    batch_size: int,
    device_name: str,
    artifact_context: Mapping,
) -> dict:
    """Commit an exact selected checkpoint and score development only.

    The function never receives an internal-test inference view.  It does bind
    the frozen partition row *identities* in the checkpoint because the shared
    checkpoint schema requires all three partition hashes; no internal-test
    outcomes or features enter selection or prediction.
    """

    model = run.pop("_selected_model", None)
    condition_vocab = run.pop("_condition_vocab", None)
    epoch_metrics = run.pop("_development_epoch_metrics", None)
    if (
        model is None
        or not isinstance(condition_vocab, Mapping)
        or not isinstance(epoch_metrics, list)
    ):
        raise ReportProtocolError(
            "development selector training did not return selected state"
        )
    run_root = Path(artifact_context["run_root"])
    unit = _development_neural_unit_directory(
        run_root, method_id=method_id, item=item, seed=seed
    )
    partition_hashes = {
        partition: _row_set_sha256(item.rows_for(partition))
        for partition in ("train", "development", "internal_test")
    }
    selection_payload = build_checkpoint_selection_artifact(
        run_start_sha256=artifact_context["run_start_sha256"],
        activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        authority_hashes=artifact_context["authority_hashes"],
        estimand_binding_sha256=item.partition_binding_sha256,
        method_id=method_id,
        training_variant_id="PRIMARY",
        seed=int(seed),
        partition_row_set_sha256={
            "train": partition_hashes["train"],
            "development": partition_hashes["development"],
        },
        training_config_sha256=artifact_context["training_config_sha256"],
        epoch_candidates=_selection_epoch_projection(epoch_metrics),
    )
    if (
        selection_payload["selected_epoch"]
        != run["checkpoint_selection"]["selected_epoch"]
    ):
        raise ReportProtocolError(
            "development checkpoint selection disagrees with in-process state"
        )
    selection_path = unit / "checkpoint_selection.json"
    selection_sha = write_checkpoint_selection_artifact_exclusive(
        selection_path, selection_payload
    )
    loaded_selection = load_checkpoint_selection_artifact_exact(
        selection_path,
        selection_sha,
        expected_run_start_sha256=artifact_context["run_start_sha256"],
        expected_activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        expected_authority_hashes=artifact_context["authority_hashes"],
        expected_estimand_binding_sha256=item.partition_binding_sha256,
        expected_method_id=method_id,
        expected_training_variant_id="PRIMARY",
        expected_seed=int(seed),
        expected_partition_row_set_sha256={
            "train": partition_hashes["train"],
            "development": partition_hashes["development"],
        },
        expected_training_config_sha256=artifact_context[
            "training_config_sha256"
        ],
    )
    selection_ref = make_artifact_ref(run_root, selection_path, selection_sha)

    checkpoint_payload = build_checkpoint(
        model,
        cfg,
        authority_hashes=artifact_context["authority_hashes"],
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        seed=int(seed),
        scientific_scope=artifact_context["scientific_scope"],
        run_start_sha256=artifact_context["run_start_sha256"],
        run_plan_sha256=artifact_context["run_plan_sha256"],
        activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        estimand_binding_sha256=item.partition_binding_sha256,
        method_id=method_id,
        training_variant_id="PRIMARY",
        training_config_sha256=artifact_context["training_config_sha256"],
        condition_vocab=condition_vocab,
        information_set_fingerprint=run["information_set_fingerprint"],
        selected_epoch=loaded_selection["selected_epoch"],
        checkpoint_selection_sha256=selection_sha,
        partition_row_set_sha256=partition_hashes,
    )
    checkpoint_path = unit / "checkpoint.pt"
    checkpoint_sha = save_checkpoint_exclusive(
        checkpoint_path, checkpoint_payload
    )
    validated_checkpoint = validate_checkpoint(
        load_checkpoint_exact(checkpoint_path, checkpoint_sha),
        expected_architecture_id=run["architecture_id"],
        expected_authority_hashes=artifact_context["authority_hashes"],
        expected_model_config=cfg,
        expected_condition_vocab=condition_vocab,
        expected_run_start_sha256=artifact_context["run_start_sha256"],
        expected_run_plan_sha256=artifact_context["run_plan_sha256"],
        expected_activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        expected_training_config_sha256=artifact_context[
            "training_config_sha256"
        ],
        expected_checkpoint_selection_sha256=selection_sha,
        expected_seed=int(seed),
        expected_scientific_scope=artifact_context["scientific_scope"],
        expected_method_id=method_id,
        expected_training_variant_id="PRIMARY",
        expected_estimand_binding_sha256=item.partition_binding_sha256,
        expected_information_set_fingerprint=run[
            "information_set_fingerprint"
        ],
        expected_partition_row_set_sha256=partition_hashes,
        expected_state_shapes={
            key: tuple(value.shape) for key, value in model.state_dict().items()
        },
    )

    scored_model = factory()
    if scored_model.architecture_id != run["architecture_id"]:
        raise ReportProtocolError(
            "development checkpoint scorer factory changed architecture"
        )
    scored_model.load_state_dict(validated_checkpoint["state_dict"], strict=True)
    device = torch.device(device_name)
    scored_model = scored_model.to(device)
    projected_development, development_fingerprint = materialize_method_rows(
        item.development,
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        target_fields=(),
    )
    if development_fingerprint != run["information_set_fingerprint"]:
        raise ReportProtocolError(
            "development checkpoint scorer information-set fingerprint drifted"
        )
    if any("delta" in row for row in projected_development):
        raise ReportProtocolError(
            "development inference view unexpectedly contains target labels"
        )
    development_tensors = _tensorize(
        projected_development,
        validated_checkpoint["condition_vocab"],
        cfg,
        include_target=False,
    )
    prediction = _predict_batched(
        scored_model,
        development_tensors,
        batch_size=batch_size,
        device=device,
    )
    rows = [
        {
            "schema_version": "r3x_prediction_row_v1",
            "row_id": stable_row_id(row),
            "predicted_delta_mean": float(mean),
            "predicted_delta_logvar": float(logvar),
        }
        for row, mean, logvar in zip(
            item.development, prediction["mean"], prediction["logvar"]
        )
    ]
    prediction_path = unit / "development_predictions.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(prediction_path, rows)
    return {
        "information_set_fingerprint": run["information_set_fingerprint"],
        "prediction_rows": rows,
        "checkpoint_selection_ref": selection_ref,
        "checkpoint_ref": make_artifact_ref(
            run_root, checkpoint_path, checkpoint_sha
        ),
        "development_prediction_ref": make_artifact_ref(
            run_root, prediction_path, prediction_sha
        ),
        "selected_epoch": loaded_selection["selected_epoch"],
    }


def _build_and_replay_development_baseline_selector(
    *,
    slices: Sequence[PrimaryEstimandSlice],
    cfg: ModelConfig,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    device_name: str,
    condition_unknown_dropout: float,
    optimizer_config: Mapping,
    loss_config: Mapping,
    artifact_context: Mapping,
    expected_paired_baseline_id: str,
) -> tuple[dict, dict, dict, dict]:
    """Build and exact-replay the mandatory pre-test development selector.

    Returns the selector payload/reference, report verification context, and a
    runner-only provenance commit.  The latter is deliberately marked as a
    schema gap: the current selector schema recursively replays prediction and
    evaluation bytes, but cannot itself bind predictor checkpoints or scalar
    fitted-state provenance.
    """

    run_root = Path(artifact_context["run_root"])
    development_root = run_root / "development"
    development_root.mkdir()
    evaluation_root = development_root / "evaluation"
    evaluation_root.mkdir()

    grouped_rows: dict[str, dict[str, list[dict]]] = {}
    evaluation_refs: dict[tuple[str, str], dict] = {}
    required_estimand_cells: list[dict] = []
    for item in sorted(slices, key=lambda value: value.estimand_id):
        cells: dict[str, list[dict]] = defaultdict(list)
        for row in item.development:
            cells[_development_cell_id(row)].append(dict(row))
        if not cells:
            raise ReportProtocolError(
                f"development selector estimand {item.estimand_id} has no rows"
            )
        grouped_rows[item.estimand_id] = {}
        layout_cells = []
        estimand_dir = evaluation_root / hashlib.sha256(
            item.estimand_id.encode("utf-8")
        ).hexdigest()[:24]
        estimand_dir.mkdir()
        for cell_id, rows in sorted(cells.items()):
            rows.sort(key=stable_row_id)
            grouped_rows[item.estimand_id][cell_id] = rows
            evaluation_rows = []
            for row in rows:
                component = str(row.get("source_component_id") or "")
                pool = str(row.get("source_group_id_v2") or "")
                if not component or not pool:
                    raise ReportProtocolError(
                        "development selector requires frozen component and pool IDs"
                    )
                evaluation_rows.append({
                    "schema_version": "r3x_development_evaluation_row_v1",
                    "row_id": stable_row_id(row),
                    "partition": DEVELOPMENT_PARTITION,
                    "estimand_id": item.estimand_id,
                    "cell_id": cell_id,
                    "true_delta": float(row["delta"]),
                    "source_component_id": component,
                    "source_group_id_v2": pool,
                })
            evaluation_path = estimand_dir / f"{cell_id}.jsonl"
            evaluation_sha = write_development_evaluation_jsonl_exclusive(
                evaluation_path, evaluation_rows
            )
            evaluation_refs[(item.estimand_id, cell_id)] = make_artifact_ref(
                run_root, evaluation_path, evaluation_sha
            )
            layout_cells.append({
                "cell_id": cell_id,
                "registered_source_group_ids_v2": sorted({
                    str(row["source_group_id_v2"]) for row in rows
                }),
            })
        required_estimand_cells.append({
            "estimand_id": item.estimand_id,
            "cells": layout_cells,
        })

    predictions_by_method: dict[str, dict[str, dict[str, float]]] = {
        method_id: {} for method_id in DEVELOPMENT_BASELINE_ELIGIBLE_ORDER
    }
    fingerprints: set[str] = set()
    scalar_fit_metadata: dict[str, dict] = {
        method_id: {}
        for method_id in DEVELOPMENT_BASELINE_ELIGIBLE_ORDER
        if method_id != "FULL_PAIR_UNRESTRICTED_SIAMESE"
    }
    scalar_runners = {
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": paired_kmer_ridge,
        "PAIRED_FEATURE_NEAREST_NEIGHBOR": nearest_neighbor,
    }
    for item in slices:
        for method_id, runner in scalar_runners.items():
            projected_train, projected_development, fingerprint = _project_pair(
                list(item.train),
                list(item.development),
                InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
                target_fields=("delta",),
            )
            result = runner(projected_train, projected_development)
            if result.status != "RUN" or len(result.predictions) != len(
                item.development
            ):
                raise ReportProtocolError(
                    f"development selector predictor {method_id} is incomplete"
                )
            fingerprints.add(fingerprint)
            predictions_by_method[method_id][item.estimand_id] = {
                stable_row_id(row): float(value)
                for row, value in zip(item.development, result.predictions)
            }
            scalar_fit_metadata[method_id][item.estimand_id] = {
                "train_row_set_sha256": _row_set_sha256(item.train),
                "development_row_set_sha256": _row_set_sha256(
                    item.development
                ),
                "fit_metadata": copy.deepcopy(dict(result.metadata)),
            }

    neural_seed_provenance: dict[str, dict[str, dict]] = {}
    neural_prediction_rows: dict[str, list[dict]] = {
        item.estimand_id: [] for item in slices
    }
    for item in slices:
        seed_rows = []
        neural_seed_provenance[item.estimand_id] = {}
        for seed in seeds:
            torch.manual_seed(int(seed))
            run = _train_neural(
                FullPairUnrestrictedSiamese(cfg),
                list(item.train),
                list(item.development),
                None,
                cfg=cfg,
                seed=int(seed),
                epochs=epochs,
                batch_size=batch_size,
                device_name=device_name,
                condition_unknown_dropout=condition_unknown_dropout,
                optimizer_config=optimizer_config,
                loss_config=loss_config,
                score_internal_test=False,
                return_selected_model=True,
            )
            committed = _commit_and_predict_development_neural_seed(
                run=run,
                factory=lambda: FullPairUnrestrictedSiamese(cfg),
                item=item,
                cfg=cfg,
                method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
                seed=int(seed),
                batch_size=batch_size,
                device_name=device_name,
                artifact_context=artifact_context,
            )
            fingerprints.add(committed["information_set_fingerprint"])
            seed_rows.append(committed["prediction_rows"])
            neural_seed_provenance[item.estimand_id][str(seed)] = {
                "checkpoint_selection_ref": committed[
                    "checkpoint_selection_ref"
                ],
                "checkpoint_ref": committed["checkpoint_ref"],
                "development_prediction_ref": committed[
                    "development_prediction_ref"
                ],
                "selected_epoch": committed["selected_epoch"],
            }
        expected_ids = [stable_row_id(row) for row in item.development]
        by_seed = []
        for rows in seed_rows:
            mapping = {
                row["row_id"]: row for row in rows
            }
            if set(mapping) != set(expected_ids):
                raise ReportProtocolError(
                    "development neural seed prediction coverage drifted"
                )
            by_seed.append(mapping)
        means = np.asarray([
            [mapping[row_id]["predicted_delta_mean"] for row_id in expected_ids]
            for mapping in by_seed
        ], dtype=np.float64)
        logvars = np.asarray([
            [mapping[row_id]["predicted_delta_logvar"] for row_id in expected_ids]
            for mapping in by_seed
        ], dtype=np.float64)
        ensemble_mean = np.mean(means, axis=0)
        ensemble_variance = np.mean(
            np.exp(logvars) + means**2, axis=0
        ) - ensemble_mean**2
        ensemble_logvar = np.log(np.maximum(ensemble_variance, 1e-12))
        neural_prediction_rows[item.estimand_id] = [
            {
                "schema_version": "r3x_prediction_row_v1",
                "row_id": row_id,
                "predicted_delta_mean": float(mean),
                "predicted_delta_logvar": float(logvar),
            }
            for row_id, mean, logvar in zip(
                expected_ids, ensemble_mean, ensemble_logvar
            )
        ]
        predictions_by_method["FULL_PAIR_UNRESTRICTED_SIAMESE"][
            item.estimand_id
        ] = {
            row["row_id"]: float(row["predicted_delta_mean"])
            for row in neural_prediction_rows[item.estimand_id]
        }

    if len(fingerprints) != 1:
        raise ReportProtocolError(
            "development selector predictors differ in information-set fingerprint"
        )
    information_set_fingerprint = fingerprints.pop()
    evidence_refs = {}
    prediction_refs: dict[str, dict[str, dict[str, dict]]] = {}
    for method_id in DEVELOPMENT_BASELINE_ELIGIBLE_ORDER:
        method_root = development_root / "evidence" / method_id
        method_root.mkdir(parents=True)
        estimand_entries = []
        prediction_refs[method_id] = {}
        for item in sorted(slices, key=lambda value: value.estimand_id):
            prediction_refs[method_id][item.estimand_id] = {}
            cells = []
            for cell_id, rows in sorted(
                grouped_rows[item.estimand_id].items()
            ):
                row_ids = [stable_row_id(row) for row in rows]
                prediction_map = predictions_by_method[method_id][
                    item.estimand_id
                ]
                if not set(row_ids).issubset(prediction_map):
                    raise ReportProtocolError(
                        f"development predictor {method_id} misses cell rows"
                    )
                prediction_rows = [
                    {
                        "schema_version": "r3x_prediction_row_v1",
                        "row_id": row_id,
                        "predicted_delta_mean": float(prediction_map[row_id]),
                        "predicted_delta_logvar": None,
                    }
                    for row_id in row_ids
                ]
                estimand_dir = method_root / hashlib.sha256(
                    item.estimand_id.encode("utf-8")
                ).hexdigest()[:24]
                estimand_dir.mkdir(exist_ok=True)
                prediction_path = estimand_dir / f"{cell_id}.jsonl"
                prediction_sha = write_prediction_jsonl_exclusive(
                    prediction_path, prediction_rows
                )
                prediction_ref = make_artifact_ref(
                    run_root, prediction_path, prediction_sha
                )
                prediction_refs[method_id][item.estimand_id][
                    cell_id
                ] = prediction_ref
                cells.append({
                    "cell_id": cell_id,
                    "prediction_jsonl_ref": prediction_ref,
                    "development_evaluation_jsonl_ref": evaluation_refs[
                        (item.estimand_id, cell_id)
                    ],
                })
            estimand_entries.append({
                "estimand_id": item.estimand_id,
                "cells": cells,
            })
        evidence = build_development_baseline_evidence(
            run_root,
            run_start_sha256=artifact_context["run_start_sha256"],
            activation_manifest_sha256=artifact_context[
                "activation_manifest_sha256"
            ],
            authority_hashes=artifact_context["authority_hashes"],
            method_id=method_id,
            information_set_fingerprint=information_set_fingerprint,
            estimands=estimand_entries,
        )
        evidence_path = method_root / "DEVELOPMENT_EVIDENCE.json"
        evidence_sha = write_development_baseline_evidence_exclusive(
            run_root, evidence_path, evidence
        )
        evidence_refs[method_id] = make_artifact_ref(
            run_root, evidence_path, evidence_sha
        )

    selection = build_development_baseline_selection(
        run_root,
        run_start_sha256=artifact_context["run_start_sha256"],
        activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        authority_hashes=artifact_context["authority_hashes"],
        eligible_method_order=DEVELOPMENT_BASELINE_ELIGIBLE_ORDER,
        development_evidence_refs=evidence_refs,
        required_estimand_cells=required_estimand_cells,
    )
    selection_path = development_root / "R3X_DEVELOPMENT_BASELINE_SELECTION.json"
    selection_sha = write_development_baseline_selection_exclusive(
        run_root, selection_path, selection
    )
    selection_ref = make_artifact_ref(run_root, selection_path, selection_sha)
    replayed = load_development_baseline_selection_exact(
        run_root,
        selection_path,
        selection_sha,
        expected_run_start_sha256=artifact_context["run_start_sha256"],
        expected_activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        expected_authority_hashes=artifact_context["authority_hashes"],
        expected_eligible_method_order=DEVELOPMENT_BASELINE_ELIGIBLE_ORDER,
        expected_required_estimand_cells=required_estimand_cells,
    )

    provenance = {
        "schema_version": "r3x_runner_development_selector_provenance_v1",
        "status": "DEVELOPMENT_PROVISIONAL_NOT_CONFIRMATORY_SCHEMA_EXTENSION_REQUIRED",
        "run_start_sha256": artifact_context["run_start_sha256"],
        "activation_manifest_sha256": artifact_context[
            "activation_manifest_sha256"
        ],
        "authority_hashes": dict(artifact_context["authority_hashes"]),
        "eligible_method_order": list(DEVELOPMENT_BASELINE_ELIGIBLE_ORDER),
        "selector_ref": selection_ref,
        "development_evidence_refs": evidence_refs,
        "predictor_provenance": {
            "FULL_PAIR_UNRESTRICTED_SIAMESE": {
                "kind": "EXACT_SELECTED_CHECKPOINT_FIXED_SEED_ENSEMBLE",
                "seeds": [int(seed) for seed in seeds],
                "by_estimand_seed": neural_seed_provenance,
                "cell_prediction_refs": prediction_refs[
                    "FULL_PAIR_UNRESTRICTED_SIAMESE"
                ],
            },
            "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": {
                "kind": "DETERMINISTIC_TRAIN_ONLY_FIT_REPLAY_REQUIRED",
                "fit_by_estimand": scalar_fit_metadata[
                    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
                ],
                "cell_prediction_refs": prediction_refs[
                    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
                ],
            },
            "PAIRED_FEATURE_NEAREST_NEIGHBOR": {
                "kind": "DETERMINISTIC_TRAIN_ONLY_FIT_REPLAY_REQUIRED",
                "fit_by_estimand": scalar_fit_metadata[
                    "PAIRED_FEATURE_NEAREST_NEIGHBOR"
                ],
                "cell_prediction_refs": prediction_refs[
                    "PAIRED_FEATURE_NEAREST_NEIGHBOR"
                ],
            },
        },
        "selector_schema_gap": (
            "r3x_development_baseline_evidence_v1 recursively binds prediction "
            "and evaluation bytes but has no predictor_provenance_ref field; "
            "this runner commit is not a substitute for that schema extension"
        ),
        "outcome_loader_boundary": (
            "ready/preflight authority validation parses outcome-bearing rows, "
            "but this selector consumes only ordinary development outcomes; "
            "internal-test blind-loader isolation is not asserted"
        ),
        "expected_paired_baseline_id": expected_paired_baseline_id,
        "ordinary_only": True,
        "internal_test_predictions_scored": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    provenance_path = development_root / "R3X_DEVELOPMENT_SELECTOR_PROVENANCE.json"
    provenance_sha = write_json_exclusive(provenance_path, provenance)
    provenance_ref = make_artifact_ref(
        run_root, provenance_path, provenance_sha
    )
    verification_context = {
        "expected_run_start_sha256": artifact_context["run_start_sha256"],
        "expected_activation_manifest_sha256": artifact_context[
            "activation_manifest_sha256"
        ],
        "expected_authority_hashes": dict(artifact_context["authority_hashes"]),
        "expected_eligible_method_order": list(
            DEVELOPMENT_BASELINE_ELIGIBLE_ORDER
        ),
        "expected_required_estimand_cells": required_estimand_cells,
    }
    return replayed, selection_ref, verification_context, provenance_ref


def _commit_and_score_neural_seed(
    *,
    run: dict,
    factory,
    item: PrimaryEstimandSlice,
    cfg: ModelConfig,
    method_id: str,
    seed: int,
    batch_size: int,
    device_name: str,
    artifact_context: Mapping,
) -> dict:
    """Publish selection/checkpoint, then score test only from exact bytes."""

    model = run.pop("_selected_model", None)
    condition_vocab = run.pop("_condition_vocab", None)
    epoch_metrics = run.pop("_development_epoch_metrics", None)
    if model is None or not isinstance(condition_vocab, Mapping) or not isinstance(epoch_metrics, list):
        raise ReportProtocolError("artifact-mode training did not return selected development state")
    run_root = Path(artifact_context["run_root"])
    unit = _artifact_unit_directory(
        run_root, method_id=method_id, item=item, seed=seed
    )
    partition_hashes = {
        partition: _row_set_sha256(item.rows_for(partition))
        for partition in ("train", "development", "internal_test")
    }
    selection_payload = build_checkpoint_selection_artifact(
        run_start_sha256=artifact_context["run_start_sha256"],
        activation_manifest_sha256=artifact_context["activation_manifest_sha256"],
        authority_hashes=artifact_context["authority_hashes"],
        estimand_binding_sha256=item.partition_binding_sha256,
        method_id=method_id,
        training_variant_id="PRIMARY",
        seed=int(seed),
        partition_row_set_sha256={
            "train": partition_hashes["train"],
            "development": partition_hashes["development"],
        },
        training_config_sha256=artifact_context["training_config_sha256"],
        epoch_candidates=_selection_epoch_projection(epoch_metrics),
    )
    if selection_payload["selected_epoch"] != run["checkpoint_selection"]["selected_epoch"]:
        raise ReportProtocolError(
            "development selection artifact disagrees with in-process selector"
        )
    selection_path = unit / "checkpoint_selection.json"
    selection_sha = write_checkpoint_selection_artifact_exclusive(
        selection_path, selection_payload
    )
    loaded_selection = load_checkpoint_selection_artifact_exact(
        selection_path,
        selection_sha,
        expected_run_start_sha256=artifact_context["run_start_sha256"],
        expected_activation_manifest_sha256=artifact_context["activation_manifest_sha256"],
        expected_authority_hashes=artifact_context["authority_hashes"],
        expected_estimand_binding_sha256=item.partition_binding_sha256,
        expected_method_id=method_id,
        expected_training_variant_id="PRIMARY",
        expected_seed=int(seed),
        expected_partition_row_set_sha256={
            "train": partition_hashes["train"],
            "development": partition_hashes["development"],
        },
        expected_training_config_sha256=artifact_context["training_config_sha256"],
    )
    selection_ref = make_artifact_ref(run_root, selection_path, selection_sha)

    checkpoint_payload = build_checkpoint(
        model,
        cfg,
        authority_hashes=artifact_context["authority_hashes"],
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        seed=int(seed),
        scientific_scope=artifact_context["scientific_scope"],
        run_start_sha256=artifact_context["run_start_sha256"],
        run_plan_sha256=artifact_context["run_plan_sha256"],
        activation_manifest_sha256=artifact_context["activation_manifest_sha256"],
        estimand_binding_sha256=item.partition_binding_sha256,
        method_id=method_id,
        training_variant_id="PRIMARY",
        training_config_sha256=artifact_context["training_config_sha256"],
        condition_vocab=condition_vocab,
        information_set_fingerprint=run["information_set_fingerprint"],
        selected_epoch=loaded_selection["selected_epoch"],
        checkpoint_selection_sha256=selection_sha,
        partition_row_set_sha256=partition_hashes,
    )
    checkpoint_path = unit / "checkpoint.pt"
    checkpoint_sha = save_checkpoint_exclusive(checkpoint_path, checkpoint_payload)
    loaded_checkpoint = load_checkpoint_exact(checkpoint_path, checkpoint_sha)
    validated_checkpoint = validate_checkpoint(
        loaded_checkpoint,
        expected_architecture_id=run["architecture_id"],
        expected_authority_hashes=artifact_context["authority_hashes"],
        expected_model_config=cfg,
        expected_condition_vocab=condition_vocab,
        expected_run_start_sha256=artifact_context["run_start_sha256"],
        expected_run_plan_sha256=artifact_context["run_plan_sha256"],
        expected_activation_manifest_sha256=artifact_context[
            "activation_manifest_sha256"
        ],
        expected_training_config_sha256=artifact_context[
            "training_config_sha256"
        ],
        expected_checkpoint_selection_sha256=selection_sha,
        expected_seed=int(seed),
        expected_scientific_scope=artifact_context["scientific_scope"],
        expected_method_id=method_id,
        expected_training_variant_id="PRIMARY",
        expected_estimand_binding_sha256=item.partition_binding_sha256,
        expected_information_set_fingerprint=run["information_set_fingerprint"],
        expected_partition_row_set_sha256=partition_hashes,
        expected_state_shapes={
            key: tuple(value.shape) for key, value in model.state_dict().items()
        },
    )
    if validated_checkpoint["checkpoint_selection_sha256"] != selection_ref["sha256"]:
        raise ReportProtocolError("checkpoint does not bind the committed selection bytes")
    checkpoint_ref = make_artifact_ref(run_root, checkpoint_path, checkpoint_sha)

    scored_model = factory()
    if scored_model.architecture_id != run["architecture_id"]:
        raise ReportProtocolError("checkpoint scorer factory changed architecture")
    scored_model.load_state_dict(validated_checkpoint["state_dict"], strict=True)
    device = torch.device(device_name)
    scored_model = scored_model.to(device)
    projected_test, test_fingerprint = materialize_method_rows(
        item.internal_test,
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        target_fields=(),
    )
    if test_fingerprint != run["information_set_fingerprint"]:
        raise ReportProtocolError("checkpoint scorer information-set fingerprint drifted")
    if any("delta" in row for row in projected_test):
        raise ReportProtocolError("checkpoint inference view unexpectedly contains target labels")
    test_tensors = _tensorize(
        projected_test,
        validated_checkpoint["condition_vocab"],
        cfg,
        include_target=False,
    )
    test_out = _predict_batched(
        scored_model, test_tensors, batch_size=batch_size, device=device
    )
    prediction_rows = [
        {
            "schema_version": "r3x_prediction_row_v1",
            "row_id": stable_row_id(row),
            "predicted_delta_mean": float(mean),
            "predicted_delta_logvar": float(logvar),
        }
        for row, mean, logvar in zip(
            item.internal_test, test_out["mean"], test_out["logvar"]
        )
    ]
    prediction_path = unit / "internal_test_predictions.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(
        prediction_path, prediction_rows
    )
    prediction_ref = make_artifact_ref(run_root, prediction_path, prediction_sha)
    prediction_manifest = build_prediction_manifest(
        run_start_sha256=artifact_context["run_start_sha256"],
        estimand_binding_sha256=item.partition_binding_sha256,
        method_id=method_id,
        training_variant_id="PRIMARY",
        seed=int(seed),
        checkpoint_sha256=checkpoint_sha,
        prediction_jsonl_ref=prediction_ref,
        prediction_rows=prediction_rows,
        information_set_fingerprint=run["information_set_fingerprint"],
    )
    manifest_path = unit / "prediction_manifest.json"
    manifest_sha = write_prediction_manifest_exclusive(
        manifest_path, prediction_manifest
    )
    manifest_ref = make_artifact_ref(run_root, manifest_path, manifest_sha)
    seed_commit = build_seed_commit(
        checkpoint_selection_ref=selection_ref,
        checkpoint_ref=checkpoint_ref,
        prediction_manifest_ref=manifest_ref,
        prediction_jsonl_ref=prediction_ref,
        prediction_manifest=prediction_manifest,
    )
    commit_path = unit / "SEED_COMMIT.json"
    commit_sha = write_seed_commit_exclusive(commit_path, seed_commit)
    commit_ref = make_artifact_ref(run_root, commit_path, commit_sha)
    replay = replay_seed_commit(run_root, commit_ref, expected_seed=int(seed))
    replay_rows = replay["rows"]
    expected_ids = [stable_row_id(row) for row in item.internal_test]
    if [row["row_id"] for row in replay_rows] != expected_ids:
        raise ReportProtocolError("committed seed prediction order differs from estimand rows")
    means = np.asarray(
        [row["predicted_delta_mean"] for row in replay_rows], dtype=np.float64
    )
    logvars = np.asarray(
        [row["predicted_delta_logvar"] for row in replay_rows], dtype=np.float64
    )
    run["test_metrics"] = _metric_bundle(item.internal_test, means, logvars)
    run["test_predictions"] = {
        "row_ids": expected_ids,
        "mean": [float(value) for value in means],
        "logvar": [float(value) for value in logvars],
        "row_set_sha256": partition_hashes["internal_test"],
        "row_order_sha256": item.row_order_sha256_for("internal_test"),
    }
    accounting = run["forward_equivalent_accounting"]
    encoder_multiplier = int(accounting["encoder_sequences_per_row_forward"])
    accounting["internal_test_rows_scored"] = len(item.internal_test)
    accounting["sequence_encoder_forward_equivalents"] += (
        encoder_multiplier * len(item.internal_test)
    )
    run["artifact_refs"] = {
        "checkpoint_selection": selection_ref,
        "checkpoint": checkpoint_ref,
        "prediction_manifest": manifest_ref,
        "prediction_jsonl": prediction_ref,
        "seed_commit": commit_ref,
    }
    return run


def _train_neural_family_per_estimand(
    *,
    factory,
    slices: Sequence[PrimaryEstimandSlice],
    global_test_rows: Sequence[Mapping],
    cfg: ModelConfig,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    device_name: str,
    condition_unknown_dropout: float,
    optimizer_config: Mapping,
    loss_config: Mapping,
    method_id: str | None = None,
    artifact_context: Mapping | None = None,
) -> dict:
    if (artifact_context is None) != (method_id is None):
        raise ReportProtocolError(
            "artifact-mode neural training requires both method_id and artifact_context"
        )
    merged_seed_runs = []
    seed_commit_refs_by_estimand: dict[str, dict[str, dict]] = {
        item.estimand_id: {} for item in slices
    }
    for seed in seeds:
        estimand_runs = []
        for item in slices:
            torch.manual_seed(int(seed))
            run = _train_neural(
                factory(), list(item.train), list(item.development),
                (None if artifact_context is not None else list(item.internal_test)),
                cfg=cfg, seed=int(seed), epochs=epochs,
                batch_size=batch_size, device_name=device_name,
                condition_unknown_dropout=condition_unknown_dropout,
                optimizer_config=optimizer_config, loss_config=loss_config,
                score_internal_test=artifact_context is None,
                return_selected_model=artifact_context is not None,
            )
            if artifact_context is not None:
                run = _commit_and_score_neural_seed(
                    run=run,
                    factory=factory,
                    item=item,
                    cfg=cfg,
                    method_id=str(method_id),
                    seed=int(seed),
                    batch_size=batch_size,
                    device_name=device_name,
                    artifact_context=artifact_context,
                )
                seed_commit_refs_by_estimand[item.estimand_id][str(seed)] = dict(
                    run["artifact_refs"]["seed_commit"]
                )
            run.update({
                "estimand_id": item.estimand_id,
                "registry_sha256": item.registry_sha256,
                "registry_entry_set_sha256": item.registry_entry_set_sha256,
                "partition_binding_sha256": item.partition_binding_sha256,
            })
            # The existing prediction key is order-sensitive despite its
            # historical name.  Record the explicit ordered digest as well.
            run["test_predictions"]["row_order_sha256"] = (
                item.row_order_sha256_for("internal_test")
            )
            estimand_runs.append(run)
        merged = merge_estimand_seed_runs(slices, estimand_runs)
        predictions = merged["test_predictions"]
        merged["test_metrics"] = _metric_bundle(
            global_test_rows,
            np.asarray(predictions["mean"], dtype=np.float64),
            np.asarray(predictions["logvar"], dtype=np.float64),
        )
        merged["test_predictions"] = {
            "row_ids": predictions["row_ids"],
            "mean": predictions["mean"],
            "logvar": predictions["logvar"],
            "row_set_sha256": predictions["row_set_sha256"],
            "row_order_sha256": predictions["row_order_sha256"],
            "estimand_id_by_row": predictions["estimand_id_by_row"],
        }
        merged["estimand_runs"] = {
            item.estimand_id: estimand_runs[index]
            for index, item in enumerate(slices)
        }
        merged_seed_runs.append(merged)
    aggregated = _aggregate_seed_runs(
        merged_seed_runs, global_test_rows, expected_seeds=seeds
    )
    if artifact_context is not None:
        exact_by_id: dict[str, dict] = {}
        ensemble_refs: dict[str, dict] = {}
        run_root = Path(artifact_context["run_root"])
        for item in slices:
            ensemble_dir = (
                run_root / "neural_artifacts" / str(method_id)
                / item.partition_binding_sha256[:24] / "ensemble"
            )
            ensemble_dir.mkdir(parents=True)
            committed = write_ensemble_from_seed_commits_exclusive(
                run_root=run_root,
                seed_commit_refs=seed_commit_refs_by_estimand[item.estimand_id],
                prediction_path=ensemble_dir / "internal_test_predictions.jsonl",
                manifest_path=ensemble_dir / "ENSEMBLE_COMMIT.json",
            )
            replayed = replay_ensemble_manifest(
                run_root, committed["manifest_ref"]
            )
            if [row["row_id"] for row in replayed["rows"]] != [
                stable_row_id(row) for row in item.internal_test
            ]:
                raise ReportProtocolError(
                    "committed estimand ensemble order differs from frozen rows"
                )
            for row in replayed["rows"]:
                if row["row_id"] in exact_by_id:
                    raise ReportProtocolError(
                        "estimand ensemble artifacts contain duplicate row identities"
                    )
                exact_by_id[row["row_id"]] = dict(row)
            ensemble_refs[item.estimand_id] = {
                "manifest": committed["manifest_ref"],
                "prediction_jsonl": committed["prediction_ref"],
            }
        expected_global_ids = [stable_row_id(row) for row in global_test_rows]
        if set(exact_by_id) != set(expected_global_ids):
            raise ReportProtocolError(
                "estimand ensemble artifacts do not exactly cover global test rows"
            )
        exact_rows = [exact_by_id[row_id] for row_id in expected_global_ids]
        exact_mean = np.asarray(
            [row["predicted_delta_mean"] for row in exact_rows], dtype=np.float64
        )
        exact_logvar = np.asarray(
            [row["predicted_delta_logvar"] for row in exact_rows], dtype=np.float64
        )
        aggregated["ensemble_prediction"] = {
            "row_ids": expected_global_ids,
            "row_set_sha256": _row_set_sha256(global_test_rows),
            "mean": [float(value) for value in exact_mean],
            "logvar_total_variance": [float(value) for value in exact_logvar],
            "n_models": len(seeds),
            "inference_cost_multiplier": len(seeds),
            "source": "EXACT_PER_ESTIMAND_ENSEMBLE_COMMIT_REPLAY",
        }
        aggregated["ensemble_test_metrics"] = _metric_bundle(
            global_test_rows, exact_mean, exact_logvar
        )
        aggregated["ensemble_artifact_refs_by_estimand"] = ensemble_refs
    first = merged_seed_runs[0]
    aggregated.update({
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "n_primary_estimands": len(slices),
        "models_per_row": len(seeds),
        "total_fitted_models": len(seeds) * len(slices),
        "parameter_count_scope": "SUM_OF_INDEPENDENT_ESTIMAND_MODELS",
        "parameter_count_by_estimand": dict(first["parameter_count_by_estimand"]),
        "partition_binding_sha256_by_estimand": {
            item.estimand_id: item.partition_binding_sha256 for item in slices
        },
    })
    return aggregated


def _selection_policy_for_method(metric_registry: Mapping, method: Mapping) -> dict:
    policy = metric_registry.get("selection_metric_policy")
    if not isinstance(policy, Mapping) or policy.get("status") != "FROZEN":
        return {
            "status": "HOLD_PENDING_SEPARATE_PREREGISTRATION",
            "pass": False,
            "reason": "stable selection metric policy is not frozen",
        }
    if policy.get("metric_id") != "ORACLE_NORMALIZED_SELECTION_GAIN":
        return {
            "status": "HOLD_UNSUPPORTED_FROZEN_SELECTION_METRIC",
            "pass": False,
            "reason": "runner has no registered evaluator for this metric ID",
        }
    threshold = policy.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return {"status": "HOLD_INVALID_FROZEN_SELECTION_POLICY", "pass": False}
    values = [
        cell["selection_gain"]["value"]
        for cell in method["ensemble_test_metrics"]["cells"].values()
    ]
    if not values or any(value is None for value in values):
        return {
            "schema_version": "r3x_selection_metric_result_v1",
            "status": "HOLD_PRIMARY_CELL_UNDEFINED",
            "pass": False,
            "reason": "at least one ensemble held-out cell lacks a complete qualified pool metric",
        }
    value = float(np.mean(values))
    return {
        "schema_version": "r3x_selection_metric_result_v1",
        "status": "FROZEN_EVALUATED",
        "metric_id": policy["metric_id"],
        "threshold": float(threshold),
        "value": value,
        "tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        "aggregation": "FIXED_ENSEMBLE_THEN_WITHIN_POOL_THEN_CELL_MACRO",
    }


def _method_prediction_rows(
    method: Mapping, test_rows: Sequence[Mapping]
) -> tuple[list[dict], list[float]]:
    prediction = method.get("ensemble_prediction")
    expected_ids = [stable_row_id(row) for row in test_rows]
    if isinstance(prediction, Mapping):
        if prediction.get("row_ids") != expected_ids:
            raise ReportProtocolError("ensemble prediction order differs from frozen test rows")
        means = prediction.get("mean")
        logvars = prediction.get("logvar_total_variance")
    else:
        means = method.get("predictions")
        logvars = [None] * len(expected_ids)
    if (
        not isinstance(means, list) or not isinstance(logvars, list)
        or len(means) != len(expected_ids) or len(logvars) != len(expected_ids)
    ):
        raise ReportProtocolError("ensemble prediction arrays do not align with frozen test rows")
    rows = []
    normalized_means = []
    for row_id, mean, logvar in zip(expected_ids, means, logvars):
        if (
            isinstance(mean, bool) or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
        ) or (
            logvar is not None and (
                isinstance(logvar, bool) or not isinstance(logvar, (int, float))
                or not math.isfinite(float(logvar))
            )
        ):
            raise ReportProtocolError("ensemble prediction contains non-finite values")
        normalized_means.append(float(mean))
        rows.append({
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row_id,
            "predicted_delta_mean": float(mean),
            "predicted_delta_logvar": None if logvar is None else float(logvar),
        })
    return rows, normalized_means


def _report_without_prediction_arrays(value):
    """Keep metrics and immutable refs, but never embed prediction vectors."""

    if isinstance(value, Mapping):
        return {
            key: _report_without_prediction_arrays(child)
            for key, child in value.items()
            if key not in {"predictions", "test_predictions", "ensemble_prediction"}
        }
    if isinstance(value, list):
        return [_report_without_prediction_arrays(child) for child in value]
    if isinstance(value, tuple):
        return [_report_without_prediction_arrays(child) for child in value]
    return value


def _preload_exact_run_authorities(
    *,
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    authorities: Mapping[str, Mapping],
) -> tuple[bytes, dict[str, bytes]]:
    """Read every activation byte stream once before creating the run root.

    Validation already checks the semantic content.  This second exact read
    closes the remaining publication-time TOCTOU window and gives the run a
    byte-for-byte authority snapshot that can later be referenced by the final
    commit.  A changed authority therefore fails before ``output_dir`` exists.
    """

    activation_raw = read_exact_bytes(
        Path(activation_manifest_path),
        activation_manifest_sha256,
        purpose="R3X activation snapshot",
    )
    if set(authorities) != set(REQUIRED_ACTIVATION_AUTHORITIES):
        raise ReportProtocolError(
            "verified activation does not contain exactly eleven authorities"
        )
    authority_raw = {
        name: read_exact_bytes(
            Path(authorities[name]["path"]),
            authorities[name]["sha256"],
            purpose=f"R3X {name} authority snapshot",
        )
        for name in REQUIRED_ACTIVATION_AUTHORITIES
    }
    return activation_raw, authority_raw


def _publish_run_authority_snapshots(
    *,
    run_root: Path,
    activation_raw: bytes,
    activation_manifest_sha256: str,
    authority_raw: Mapping[str, bytes],
    authorities: Mapping[str, Mapping],
) -> tuple[dict, dict[str, dict]]:
    """Atomically publish immutable, run-local copies of all input bytes."""

    snapshot_dir = Path(run_root) / "authority_snapshots"
    snapshot_dir.mkdir()
    activation_path = snapshot_dir / "ACTIVATION_MANIFEST.json"
    activation_sha = atomic_publish_bytes(activation_path, activation_raw)
    if activation_sha != activation_manifest_sha256:
        raise ReportProtocolError("published activation snapshot hash drifted")
    activation_ref = make_artifact_ref(run_root, activation_path, activation_sha)

    authority_refs: dict[str, dict] = {}
    for name in REQUIRED_ACTIVATION_AUTHORITIES:
        path = snapshot_dir / f"{name}.authority"
        digest = atomic_publish_bytes(path, authority_raw[name])
        expected = str(authorities[name]["sha256"])
        if digest != expected:
            raise ReportProtocolError(
                f"published authority snapshot hash drifted for {name}"
            )
        authority_refs[name] = make_artifact_ref(run_root, path, digest)
    return activation_ref, authority_refs


def _commit_method_metrics_artifact(
    *,
    run_root: Path,
    method_id: str,
    method: Mapping,
    global_prediction_commit_ref: Mapping,
    run_start_sha256: str,
) -> dict:
    """Publish the exact metric bundle used by the paired/report layers."""

    metrics = method.get("ensemble_test_metrics")
    if not isinstance(metrics, Mapping):
        raise ReportProtocolError(
            f"method {method_id} lacks an ensemble test metric bundle"
        )
    information_set = method.get("information_set")
    fingerprint = method.get("information_set_fingerprint")
    if not isinstance(information_set, str) or not isinstance(fingerprint, str):
        raise ReportProtocolError(
            f"method {method_id} lacks information-set metric bindings"
        )
    payload = {
        "schema_version": "r3x_method_internal_test_metrics_v1",
        "artifact_role": "ORDINARY_INTERNAL_TEST_METHOD_METRICS",
        "method_id": method_id,
        "run_start_sha256": run_start_sha256,
        "global_prediction_commit_sha256": global_prediction_commit_ref[
            "sha256"
        ],
        "information_set": information_set,
        "information_set_fingerprint": fingerprint,
        "metrics": copy.deepcopy(dict(metrics)),
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    directory = Path(run_root) / "method_metrics" / method_id
    directory.mkdir(parents=True)
    path = directory / "INTERNAL_TEST_METRICS.json"
    digest = write_json_exclusive(path, payload)
    return make_artifact_ref(run_root, path, digest)


def _run_complete_method_entry(
    method_id: str,
    method: Mapping,
    *,
    global_prediction_ref: Mapping,
    metrics_ref: Mapping,
) -> dict:
    ensembles = method.get("ensemble_artifact_refs_by_estimand")
    if not isinstance(ensembles, Mapping) or not ensembles:
        raise ReportProtocolError(
            f"method {method_id} lacks per-estimand ensemble commits"
        )
    return {
        "method_id": method_id,
        "global_prediction_ref": dict(global_prediction_ref),
        "metrics_ref": dict(metrics_ref),
        "checkpoint_ensemble_refs": {
            str(estimand_id): dict(refs["manifest"])
            for estimand_id, refs in sorted(ensembles.items())
        },
    }


def run_real_ordinary(
    *,
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    output_dir: Path,
) -> NoReturn:
    """Fail closed until staged disclosure authorities are implemented."""

    output_dir = Path(output_dir)
    require_staged_disclosure_before_outcome_access(
        activation_manifest_path,
        activation_manifest_sha256,
        prospective_output_dir=output_dir,
    )
    raise ProtocolHoldError(
        "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED",
        "staged disclosure guard returned without a stage-specific runner",
    )


def _run_legacy_outcome_bearing_development_provisional(
    *,
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    output_dir: Path,
) -> dict:
    """Disabled reference scaffold; never a confirmatory execution path."""

    raise ProtocolHoldError(
        "HOLD_LEGACY_OUTCOME_BEARING_RUNNER_DISABLED",
        "legacy monolithic ready-data runner is retained only as reference code",
    )

    output_dir = Path(output_dir)
    activation = _audit_legacy_outcome_bearing_activation_manifest(
        activation_manifest_path,
        activation_manifest_sha256,
        prospective_output_dir=output_dir,
    )
    authorities = activation["verified_authorities"]
    rows = _read_ready_jsonl_exact(
        Path(authorities["ready_dataset"]["path"]),
        authorities["ready_dataset"]["sha256"],
    )
    row_ids_in_order = [
        str(row.get("r3_input_row_sha256") or "") for row in rows
    ]
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in row_ids_in_order):
        raise ReportProtocolError("real ready rows require lowercase SHA-256 r3_input_row_sha256 IDs")
    if len(row_ids_in_order) != len(set(row_ids_in_order)):
        raise ReportProtocolError("real ready dataset contains duplicate row identities")
    split = _read_json_exact(
        Path(authorities["source_component_split_manifest"]["path"]),
        authorities["source_component_split_manifest"]["sha256"],
        purpose="source-component split manifest",
    )
    if split.get("status") != "FROZEN":
        raise ReportProtocolError("real split manifest must declare status=FROZEN")
    verify_split_manifest_against_rows(rows, split)
    assignment_ids = set(split.get("row_assignments", {}))
    row_ids = set(row_ids_in_order)
    if row_ids != assignment_ids:
        raise ReportProtocolError(
            "ready dataset and split manifest row identities are not exactly equal"
        )
    inclusion = _read_json_exact(
        Path(authorities["inclusion_manifest"]["path"]),
        authorities["inclusion_manifest"]["sha256"],
        purpose="inclusion manifest",
    )
    ready_row_ids_sha = hashlib.sha256(
        json.dumps(sorted(row_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if inclusion.get("n_ready_rows") != len(rows) or inclusion.get("ready_row_ids_sha256") != ready_row_ids_sha:
        raise ReportProtocolError("inclusion manifest row count/set does not match ready bytes")
    crosswalk = _read_json_exact(
        Path(authorities["context_source_crosswalk"]["path"]),
        authorities["context_source_crosswalk"]["sha256"],
        purpose="context/source crosswalk",
    )
    estimand_registry = _read_json_exact(
        Path(authorities["endpoint_estimand_registry"]["path"]),
        authorities["endpoint_estimand_registry"]["sha256"],
        purpose="endpoint estimand registry",
    )
    _validate_ready_scientific_authorities(
        rows, inclusion, crosswalk, estimand_registry
    )
    split_scope = split.get("scientific_scope")
    scientific_scope = activation.get("scientific_scope")
    allowed_authority_upgrade = (
        split_scope == "DEVELOPMENT_ONLY_SAME_ENDPOINT_ESTIMAND_AUTHORITY_REQUIRED"
        and scientific_scope == "PUBLICATION_ELIGIBLE_SAME_ESTIMAND"
    )
    if split_scope != scientific_scope and not allowed_authority_upgrade:
        raise ReportProtocolError(
            "activation and replayed split scientific scopes disagree outside the "
            "single allowed registry-verified same-estimand upgrade"
        )
    rows = _attach_split(rows, split)
    estimand_slices = build_primary_estimand_slices(
        rows,
        estimand_registry,
        estimand_registry_sha256=authorities["endpoint_estimand_registry"]["sha256"],
    )
    train_rows = _canonical_partition_rows(estimand_slices, "train")
    dev_rows = _canonical_partition_rows(estimand_slices, "development")
    test_rows = _canonical_partition_rows(estimand_slices, "internal_test")
    if min(map(len, (train_rows, dev_rows, test_rows))) < 1:
        raise ReportProtocolError("frozen real split has an empty partition")

    run_config = _read_json_exact(
        Path(authorities["model_optimizer_seed_config"]["path"]),
        authorities["model_optimizer_seed_config"]["sha256"],
        purpose="model/optimizer/seed config",
    )
    if run_config.get("status") != "FROZEN":
        raise ReportProtocolError("real run config must declare status=FROZEN")
    seeds = run_config.get("seeds")
    if seeds != [11, 23, 42, 67, 101]:
        raise ReportProtocolError("real ordinary seeds must be exactly [11,23,42,67,101]")
    epochs = run_config.get("epochs")
    batch_size = run_config.get("batch_size")
    device = run_config.get("device")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ReportProtocolError("frozen epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ReportProtocolError("frozen batch_size must be a positive integer")
    if not isinstance(device, str) or not device:
        raise ReportProtocolError("frozen device is required")
    model_fields = run_config.get("model")
    if not isinstance(model_fields, Mapping):
        raise ReportProtocolError("frozen run config lacks model settings")
    cfg = ModelConfig(**dict(model_fields))
    estimand_capacity = _validate_estimand_model_capacity(estimand_slices, cfg)

    metric_registry = _read_json_exact(
        Path(authorities["metric_zero_policy_registry"]["path"]),
        authorities["metric_zero_policy_registry"]["sha256"],
        purpose="metric/zero-policy registry",
    )
    if metric_registry.get("status") != "FROZEN" or metric_registry.get("zero_policy") != "EXCLUDE_TRUE_ZERO":
        raise ReportProtocolError("metric registry is not frozen to EXCLUDE_TRUE_ZERO")

    paired_policy = _read_json_exact(
        Path(authorities["paired_strongest_baseline_comparison_policy"]["path"]),
        authorities["paired_strongest_baseline_comparison_policy"]["sha256"],
        purpose="paired strongest-baseline comparison policy",
    )
    required_baselines = set(paired_policy.get("required_baseline_method_ids") or ())
    missing_implementations = sorted(required_baselines - IMPLEMENTED_BASELINE_METHODS)
    if missing_implementations:
        raise ReportProtocolError(
            "BASELINE_SUITE_IMPLEMENTATION_INCOMPLETE: "
            + ", ".join(missing_implementations)
        )
    family_members = paired_policy.get("comparison_family_member_ids")
    primary_family_member = paired_policy.get("primary_comparison_member_id")
    if family_members != [primary_family_member]:
        raise ReportProtocolError(
            "HOLD_COMPARISON_FAMILY_EVALUATOR_INCOMPLETE: the current validator "
            "implements only the preregistered primary sign member; additional "
            "Holm members require their own hash-bound evaluators before training"
        )
    if paired_policy.get("primary_candidate_model_id") != "A_EXACT_POTENTIAL_DIFFERENCE":
        raise ReportProtocolError(
            "HOLD_PRIMARY_CANDIDATE_NOT_AUTHORIZED_FOR_A_FIRST_ROUTE"
        )
    if paired_policy.get("strongest_baseline_id") != "FULL_PAIR_UNRESTRICTED_SIAMESE":
        raise ReportProtocolError(
            "HOLD_GLOBAL_BASELINE_ARTIFACT_UNSUPPORTED: the frozen A-first route "
            "currently requires the full-pair unrestricted comparator"
        )
    minimum_sign_components = int(
        paired_policy["primary_sign_inference"]["bootstrap"][
            "min_components_per_cell"
        ]
    )
    sign_support: dict[str, set[str]] = defaultdict(set)
    for row in test_rows:
        if float(row["delta"]) != 0.0:
            sign_support[_cell(row)].add(str(row.get("source_component_id") or ""))
    underpowered_sign_cells = sorted(
        cell for cell in {_cell(row) for row in test_rows}
        if "" in sign_support.get(cell, set())
        or len(sign_support.get(cell, set())) < minimum_sign_components
    )
    if underpowered_sign_cells:
        raise ReportProtocolError(
            "HOLD_INSUFFICIENT_PRIMARY_SIGN_COMPONENTS_BEFORE_TRAINING: "
            + ", ".join(underpowered_sign_cells)
        )
    # Fail before output/training if the alleged matched control is not matched
    # for this frozen model configuration.
    torch.manual_seed(0)
    preflight_exact = ExactPotentialCritic(cfg)
    torch.manual_seed(0)
    preflight_control = FullPairUnrestrictedSiamese(cfg)
    preflight_gap = abs(
        count_parameters(preflight_control) - count_parameters(preflight_exact)
    ) / count_parameters(preflight_exact)
    if preflight_gap > 0.05:
        raise ReportProtocolError(
            "A unrestricted control exceeds the frozen 5% parameter tolerance before training"
        )

    activation_snapshot_raw, authority_snapshot_raw = (
        _preload_exact_run_authorities(
            activation_manifest_path=activation_manifest_path,
            activation_manifest_sha256=activation_manifest_sha256,
            authorities=authorities,
        )
    )

    authority_hashes = {
        f"{name}_sha256": str(value["sha256"])
        for name, value in authorities.items()
    }
    run_plan = {
        "schema_version": "r3x_run_plan_v1",
        "activation_manifest_sha256": activation_manifest_sha256,
        "authority_hashes": authority_hashes,
        "execution_order": EXECUTION_ORDER,
        "primary_estimand_partition_binding_sha256": {
            item.estimand_id: item.partition_binding_sha256
            for item in estimand_slices
        },
        "seeds": seeds,
        "training_variant_id": "PRIMARY",
        "scientific_scope": scientific_scope,
        "sign_gate": 0.60,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    # The output becomes visible only after every authority and row/split/config
    # invariant above has passed.  RUN_PLAN is published before RUN_START, and
    # every later seed checkpoint binds the exact hashes of both artifacts.
    output_dir.mkdir()
    activation_snapshot_ref, authority_snapshot_refs = (
        _publish_run_authority_snapshots(
            run_root=output_dir,
            activation_raw=activation_snapshot_raw,
            activation_manifest_sha256=activation_manifest_sha256,
            authority_raw=authority_snapshot_raw,
            authorities=authorities,
        )
    )
    run_plan_sha = write_json_exclusive(
        output_dir / "R3X_RUN_PLAN.json", run_plan
    )
    run_start_sha = write_json_exclusive(output_dir / "R3X_RUN_START.json", {
        "schema_version": "r3x_run_start_v1",
        "activation_manifest_sha256": activation_manifest_sha256,
        "run_plan_sha256": run_plan_sha,
        "authority_hashes": authority_hashes,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "sealed_access": False,
        "downstream_unlock": False,
    })
    artifact_context = {
        "run_root": output_dir,
        "run_start_sha256": run_start_sha,
        "run_plan_sha256": run_plan_sha,
        "activation_manifest_sha256": activation_manifest_sha256,
        "authority_hashes": authority_hashes,
        "training_config_sha256": authorities[
            "model_optimizer_seed_config"
        ]["sha256"],
        "scientific_scope": scientific_scope,
    }

    # Publication-comparator selection is development-only and must finish
    # before any baseline or neural method receives an internal-test inference
    # view.  The ready-dataset/preflight layer above has already parsed frozen
    # outcome-bearing rows for authority and support validation; this ordering
    # guarantee is therefore about prediction/scoring, not a claim of a fully
    # blinded internal-test loader process.
    try:
        (
            development_baseline_selection,
            development_baseline_selection_ref,
            development_baseline_selection_context,
            development_selector_provenance_ref,
        ) = _build_and_replay_development_baseline_selector(
            slices=estimand_slices,
            cfg=cfg,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            device_name=device,
            condition_unknown_dropout=float(
                run_config.get("condition_unknown_dropout")
            ),
            optimizer_config=run_config["optimizer"],
            loss_config=run_config["loss"],
            artifact_context=artifact_context,
            expected_paired_baseline_id=paired_policy[
                "strongest_baseline_id"
            ],
        )
    except BaselineSelectionError as exc:
        write_json_exclusive(
            output_dir / "R3X_DEVELOPMENT_SELECTOR_STOP.json",
            {
                "schema_version": "r3x_development_selector_stop_v1",
                "status": "SAFE_STOP_BEFORE_INTERNAL_TEST_PREDICTION",
                "reason": str(exc),
                "run_start_sha256": run_start_sha,
                "ordinary_only": True,
                "internal_test_predictions_scored": False,
                "sealed_access": False,
                "downstream_unlock": False,
            },
        )
        raise ReportProtocolError(
            "HOLD_DEVELOPMENT_BASELINE_SELECTOR_REPLAY_FAILED_BEFORE_TEST: "
            + str(exc)
        ) from exc
    selector_qualified = (
        development_baseline_selection["status"]
        == DEVELOPMENT_BASELINE_QUALIFIED_STATUS
    )
    selector_matches_paired = (
        development_baseline_selection["selected_method_id"]
        == paired_policy["strongest_baseline_id"]
    )
    if not selector_qualified or not selector_matches_paired:
        write_json_exclusive(
            output_dir / "R3X_DEVELOPMENT_SELECTOR_STOP.json",
            {
                "schema_version": "r3x_development_selector_stop_v1",
                "status": "SAFE_STOP_BEFORE_INTERNAL_TEST_PREDICTION",
                "selector_status": development_baseline_selection["status"],
                "selected_method_id": development_baseline_selection[
                    "selected_method_id"
                ],
                "expected_paired_baseline_id": paired_policy[
                    "strongest_baseline_id"
                ],
                "development_baseline_selection_ref": (
                    development_baseline_selection_ref
                ),
                "development_selector_provenance_ref": (
                    development_selector_provenance_ref
                ),
                "run_start_sha256": run_start_sha,
                "ordinary_only": True,
                "internal_test_predictions_scored": False,
                "sealed_access": False,
                "downstream_unlock": False,
            },
        )
        raise ReportProtocolError(
            "HOLD_DEVELOPMENT_BASELINE_SELECTOR_BEFORE_TEST: status="
            f"{development_baseline_selection['status']} selected="
            f"{development_baseline_selection['selected_method_id']} expected="
            f"{paired_policy['strongest_baseline_id']}"
        )

    methods = _baseline_methods_per_estimand(
        estimand_slices,
        test_rows,
        optional_hist_authorized=bool(
            paired_policy.get("hist_gradient_boosting_execution_authorized", False)
        ),
    )
    neural_strong_baselines = {"FULL_PAIR_UNRESTRICTED_SIAMESE"}
    incomplete_pre_neural = sorted(
        method_id for method_id in required_baselines - neural_strong_baselines
        if method_id not in methods or methods[method_id].get("status") != "RUN"
    )
    if incomplete_pre_neural:
        # This happens after RUN_START only when a frozen optional backend fails
        # at execution time.  No critic is trained after an incomplete suite.
        raise ReportProtocolError(
            "BASELINE_SUITE_RUNTIME_INCOMPLETE: " + ", ".join(incomplete_pre_neural)
        )
    families = (
        ("FULL_PAIR_UNRESTRICTED_SIAMESE", lambda: FullPairUnrestrictedSiamese(cfg)),
        ("R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION", lambda: R2StyleNoExplicitStudyCrossAttentionCritic(cfg)),
        ("A_EXACT_POTENTIAL_DIFFERENCE", lambda: ExactPotentialCritic(cfg)),
        ("A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL", lambda: ParameterMatchedUnrestrictedSiamese(cfg)),
        ("B_SPARSE_EDIT_SET_ADDITIVE", lambda: SparseEditSetCritic(cfg, interaction=False)),
    )
    for method_id, factory in families:
        methods[method_id] = _train_neural_family_per_estimand(
            factory=factory,
            slices=estimand_slices,
            global_test_rows=test_rows,
            cfg=cfg,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            device_name=device,
            condition_unknown_dropout=float(
                run_config.get("condition_unknown_dropout")
            ),
            optimizer_config=run_config["optimizer"],
            loss_config=run_config["loss"],
            method_id=method_id,
            artifact_context=artifact_context,
        )
    methods["FULL_PAIR_UNRESTRICTED_SIAMESE"]["role"] = (
        "DEVELOPMENT_SELECTED_PREREGISTERED_STRONG_SAME_INFORMATION_SET_"
        "COMPARATOR"
    )
    methods["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"]["role"] = (
        "DEVELOPMENT_BASELINE_PENDING_ARCHITECTURE_CROSSWALK_AND_STUDY_PROXY_AUDIT"
    )
    methods["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"]["strongest_baseline_eligible"] = False
    methods["A_EXACT_POTENTIAL_DIFFERENCE"]["role"] = "PRIMARY_CRITIC_CANDIDATE"
    methods["A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"]["role"] = (
        "LOW_RANK_ALGEBRAIC_DIAGNOSTIC_NOT_STRONGEST_BASELINE_ELIGIBLE"
    )
    incomplete_required = sorted(
        method_id for method_id in required_baselines
        if method_id not in methods or methods[method_id].get("status") != "RUN"
    )
    if incomplete_required:
        raise ReportProtocolError(
            "BASELINE_SUITE_RUNTIME_INCOMPLETE: " + ", ".join(incomplete_required)
        )
    exact_params = methods["A_EXACT_POTENTIAL_DIFFERENCE"]["parameter_count_by_estimand"]
    full_pair_params = methods["FULL_PAIR_UNRESTRICTED_SIAMESE"]["parameter_count_by_estimand"]
    low_rank_params = methods["A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"]["parameter_count_by_estimand"]
    if set(exact_params) != set(full_pair_params) or set(exact_params) != set(low_rank_params):
        raise ReportProtocolError("parameter matching estimand sets differ")
    full_pair_gaps = {
        estimand_id: abs(full_pair_params[estimand_id] - exact_params[estimand_id])
        / exact_params[estimand_id]
        for estimand_id in exact_params
    }
    low_rank_gaps = {
        estimand_id: abs(low_rank_params[estimand_id] - exact_params[estimand_id])
        / exact_params[estimand_id]
        for estimand_id in exact_params
    }
    if any(value > 0.05 for value in (*full_pair_gaps.values(), *low_rank_gaps.values())):
        raise ReportProtocolError("A unrestricted control exceeds the frozen 5% parameter tolerance")
    methods["FULL_PAIR_UNRESTRICTED_SIAMESE"]["parameter_gap_fraction_vs_A_by_estimand"] = full_pair_gaps
    methods["A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"]["parameter_gap_fraction_vs_A_by_estimand"] = low_rank_gaps
    methods["B_SPARSE_EDIT_SET_INTERACTION"] = {
        "status": "NOT_RUN_ACTION_SUPPORT",
        "reason": "activation schema contains no separately frozen publication-qualified action-support policy",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
    }

    primary_method_id = paired_policy["primary_candidate_model_id"]
    if primary_method_id not in methods:
        raise ReportProtocolError("frozen primary candidate is not an executed method")
    primary_method = methods[primary_method_id]
    baseline_method_id = paired_policy["strongest_baseline_id"]
    if baseline_method_id not in methods or methods[baseline_method_id].get("status") != "RUN":
        raise ReportProtocolError(
            "preregistered paired comparator is absent or incomplete after "
            "per-estimand execution"
        )
    baseline_method = methods[baseline_method_id]
    global_dir = output_dir / "global_method_predictions"
    global_dir.mkdir()

    def commit_global_method(method_id: str, method: Mapping) -> tuple[dict, list[float]]:
        source_refs = method.get("ensemble_artifact_refs_by_estimand")
        if not isinstance(source_refs, Mapping):
            raise ReportProtocolError(
                "paired neural method lacks per-estimand ensemble artifact refs"
            )
        method_dir = global_dir / method_id
        method_dir.mkdir()
        committed = write_global_method_predictions_exclusive(
            run_root=output_dir,
            ensemble_manifest_refs_by_estimand={
                estimand_id: refs["manifest"]
                for estimand_id, refs in source_refs.items()
            },
            estimand_slices=estimand_slices,
            global_test_rows=test_rows,
            prediction_path=method_dir / "internal_test_predictions.jsonl",
            commit_path=method_dir / "GLOBAL_PREDICTION_COMMIT.json",
        )
        replayed = replay_global_prediction_commit(
            output_dir,
            committed["commit_ref"],
            estimand_slices=estimand_slices,
            global_test_rows=test_rows,
        )
        means = [float(row["predicted_delta_mean"]) for row in replayed["rows"]]
        logvars = np.asarray(
            [row["predicted_delta_logvar"] for row in replayed["rows"]],
            dtype=np.float64,
        )
        recomputed_metrics = _metric_bundle(
            test_rows, np.asarray(means, dtype=np.float64), logvars
        )
        if recomputed_metrics != method["ensemble_test_metrics"]:
            raise ReportProtocolError(
                "global prediction artifact metrics differ from method ensemble metrics"
            )
        return committed, means

    candidate_global, candidate_prediction = commit_global_method(
        primary_method_id, primary_method
    )
    baseline_global, baseline_prediction = commit_global_method(
        baseline_method_id, baseline_method
    )
    candidate_prediction_sha = candidate_global["prediction_sha256"]
    baseline_prediction_sha = baseline_global["prediction_sha256"]
    candidate_prediction_ref = candidate_global["prediction_ref"]
    baseline_prediction_ref = baseline_global["prediction_ref"]
    primary_method["global_prediction_artifact_refs"] = {
        "commit": candidate_global["commit_ref"],
        "prediction_jsonl": candidate_prediction_ref,
    }
    baseline_method["global_prediction_artifact_refs"] = {
        "commit": baseline_global["commit_ref"],
        "prediction_jsonl": baseline_prediction_ref,
    }
    candidate_metrics_ref = _commit_method_metrics_artifact(
        run_root=output_dir,
        method_id=primary_method_id,
        method=primary_method,
        global_prediction_commit_ref=candidate_global["commit_ref"],
        run_start_sha256=run_start_sha,
    )
    baseline_metrics_ref = _commit_method_metrics_artifact(
        run_root=output_dir,
        method_id=baseline_method_id,
        method=baseline_method,
        global_prediction_commit_ref=baseline_global["commit_ref"],
        run_start_sha256=run_start_sha,
    )
    primary_method["metrics_artifact_ref"] = candidate_metrics_ref
    baseline_method["metrics_artifact_ref"] = baseline_metrics_ref

    def information_fingerprint(method: Mapping) -> str:
        value = method.get("information_set_fingerprint")
        if value is None and isinstance(method.get("metadata"), Mapping):
            value = method["metadata"].get("information_set_fingerprint")
        if not isinstance(value, str):
            raise ReportProtocolError("paired method lacks information-set fingerprint")
        return value

    paired_build_kwargs = {
        "policy": paired_policy,
        "policy_sha256": authorities[
            "paired_strongest_baseline_comparison_policy"
        ]["sha256"],
        "authority_sha256": authority_hashes,
        "candidate_model_id": primary_method_id,
        "baseline_model_id": baseline_method_id,
        "candidate_information_set_fingerprint": information_fingerprint(primary_method),
        "baseline_information_set_fingerprint": information_fingerprint(baseline_method),
        "candidate_prediction_artifact_sha256": candidate_prediction_sha,
        "baseline_prediction_artifact_sha256": baseline_prediction_sha,
        "additional_family_raw_p_values": {},
        "additional_family_evidence_sha256": {},
    }
    paired_payload = build_paired_comparison_artifact(
        test_rows,
        candidate_prediction,
        baseline_prediction,
        **paired_build_kwargs,
    )
    paired_path = output_dir / "R3X_PAIRED_COMPARISON_V2.json"
    paired_sha = write_paired_comparison_artifact_exclusive(
        paired_path, paired_payload
    )
    paired_payload = load_paired_comparison_artifact_exact(
        paired_path,
        paired_sha,
        test_rows,
        candidate_prediction,
        baseline_prediction,
        **paired_build_kwargs,
    )
    paired_result = build_verified_paired_reference(
        artifact_relative_path=paired_path.relative_to(output_dir).as_posix(),
        artifact_sha256=paired_sha,
        artifact_payload=paired_payload,
    )
    paired_verification_context = {
        "rows": test_rows,
        "candidate_prediction": candidate_prediction,
        "baseline_prediction": baseline_prediction,
        "build_kwargs": paired_build_kwargs,
    }
    negative_control_payload = build_not_run_negative_control_summary(
        method_protocol_ref=authority_snapshot_refs["method_protocol"],
        ordinary_internal_test_row_binding_sha256=_row_set_sha256(test_rows),
        reason_code="CONTROLS_NOT_EXECUTED",
        action_support_status=ACTION_SUPPORT_NOT_QUALIFIED,
    )
    negative_control_path = output_dir / "R3X_NEGATIVE_CONTROL_SUMMARY_HOLD.json"
    negative_control_sha = write_negative_control_summary_exclusive(
        negative_control_path,
        negative_control_payload,
        run_root=output_dir,
        evidence_verifier=None,
        verifier_context=None,
    )
    negative_control_ref = make_artifact_ref(
        output_dir, negative_control_path, negative_control_sha
    )
    loaded_negative_control = load_negative_control_summary(
        negative_control_path,
        negative_control_sha,
        run_root=output_dir,
        evidence_verifier=None,
        verifier_context=None,
    )
    if loaded_negative_control["overall_status"] != "HOLD":
        raise ReportProtocolError(
            "unexecuted preregistered controls must force scientific HOLD"
        )

    def reject_unexpected_completed_control_evidence(_raw, _metadata):
        raise ReportProtocolError(
            "completed negative-control evidence has no integrated scientific verifier"
        )

    negative_control_verifier_context = {
        "expected_internal_test_row_binding_sha256": _row_set_sha256(test_rows),
        "mode": "HOLD_UNEXECUTED_CONTROLS_ONLY",
    }
    candidate_metrics = dict(primary_method["ensemble_test_metrics"]["macro"])
    selection_policy = _selection_policy_for_method(metric_registry, primary_method)
    gates = {
        primary_method_id: evaluate_effect_gate(
            candidate_model_id=primary_method_id,
            candidate_metrics=candidate_metrics,
            information_set=primary_method["information_set"],
            strongest_baseline_comparison=paired_result,
            selection_metric_policy=selection_policy,
            scientific_scope=str(scientific_scope),
            ordinary_run_root=output_dir,
            paired_verification_context=paired_verification_context,
            development_baseline_selection_reference=(
                development_baseline_selection_ref
            ),
            development_baseline_selection_verification_context=(
                development_baseline_selection_context
            ),
            negative_control_summary_reference=negative_control_ref,
            negative_control_evidence_verifier=(
                reject_unexpected_completed_control_evidence
            ),
            negative_control_verifier_context=(
                negative_control_verifier_context
            ),
        )
    }
    gate_inputs = {
        primary_method_id: {
            "candidate_metrics": candidate_metrics,
            "information_set": primary_method["information_set"],
            "strongest_baseline_comparison": paired_result,
            "selection_metric_policy": selection_policy,
            "development_baseline_selection_reference": (
                development_baseline_selection_ref
            ),
            "negative_control_summary_reference": negative_control_ref,
        }
    }
    any_go = gates[primary_method_id]["GO_EFFECT"]
    report_methods = _report_without_prediction_arrays(methods)
    report = {
        "schema_version": "r3x_ordinary_run_report_v1",
        "status": (
            "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS"
            if any_go else "ORDINARY_RUN_COMPLETE_SCIENTIFIC_HOLD"
        ),
        "activation_manifest_sha256": activation_manifest_sha256,
        "run_plan_sha256": run_plan_sha,
        "run_start_sha256": run_start_sha,
        "execution_order": EXECUTION_ORDER,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "scientific_scope": scientific_scope,
        "n_rows": {"train": len(train_rows), "development": len(dev_rows), "internal_test": len(test_rows)},
        "primary_estimands": {
            item.estimand_id: {
                "studies": list(item.studies),
                "endpoints": list(item.endpoints),
                "partition_binding_sha256": item.partition_binding_sha256,
                "n_rows": {
                    "train": len(item.train),
                    "development": len(item.development),
                    "internal_test": len(item.internal_test),
                },
            }
            for item in estimand_slices
        },
        "estimand_model_capacity": estimand_capacity,
        "development_baseline_selection_ref": (
            development_baseline_selection_ref
        ),
        "development_selector_provenance_ref": (
            development_selector_provenance_ref
        ),
        "methods": report_methods,
        "effect_gates": gates,
        "effect_gate_inputs": gate_inputs,
        "GO_EFFECT": any_go,
        "downstream_unlock": False,
        "sealed_access": False,
        "active_contract_mutation": False,
        "sign_gate": 0.60,
        "limitations": [
            "PRIMARY_MODEL_FAMILY_FROZEN_BEFORE_INTERNAL_TEST",
            "PAIRED_PREDICTION_ARTIFACT_V2_COMMITTED_AND_RECOMPUTED",
            "FULL_PAIR_SELECTED_ON_ORDINARY_DEVELOPMENT_BEFORE_TEST_SCORING",
            "SELECTOR_SCHEMA_LACKS_RECURSIVE_PREDICTOR_PROVENANCE_FIELD",
            "READY_PREFLIGHT_PARSES_OUTCOME_ROWS_NO_BLIND_LOADER_CLAIM",
            "PREREGISTERED_NEGATIVE_CONTROLS_NOT_RUN_SCIENTIFIC_HOLD",
            "SELECTION_METRIC_REMAINS_HOLD_UNLESS_SEPARATELY_FROZEN",
            "O0_G1_E0_REMAIN_LOCKED",
        ],
    }
    validate_run_report(
        report,
        ordinary_run_root=output_dir,
        paired_verification_context_by_method={
            primary_method_id: paired_verification_context
        },
        development_baseline_selection_verification_context_by_method={
            primary_method_id: development_baseline_selection_context
        },
        negative_control_evidence_verifier_by_method={
            primary_method_id: reject_unexpected_completed_control_evidence
        },
        negative_control_verifier_context_by_method={
            primary_method_id: negative_control_verifier_context
        },
    )
    report_path = output_dir / "R3X_RUN_REPORT.json"
    report_sha = write_json_exclusive(report_path, report)
    run_plan_ref = make_artifact_ref(
        output_dir, output_dir / "R3X_RUN_PLAN.json", run_plan_sha
    )
    run_start_ref = make_artifact_ref(
        output_dir, output_dir / "R3X_RUN_START.json", run_start_sha
    )
    report_ref = make_artifact_ref(output_dir, report_path, report_sha)
    paired_ref = make_artifact_ref(output_dir, paired_path, paired_sha)
    required_complete_methods = sorted(
        [primary_method_id, baseline_method_id]
    )
    run_complete_payload = build_run_complete(
        status=(
            ORDINARY_GO_CANDIDATE if any_go else SCIENTIFIC_HOLD
        ),
        run_plan_ref=run_plan_ref,
        run_start_ref=run_start_ref,
        activation_manifest_ref=activation_snapshot_ref,
        authority_refs=authority_snapshot_refs,
        required_method_ids=required_complete_methods,
        method_artifacts={
            primary_method_id: _run_complete_method_entry(
                primary_method_id,
                primary_method,
                global_prediction_ref=candidate_global["commit_ref"],
                metrics_ref=candidate_metrics_ref,
            ),
            baseline_method_id: _run_complete_method_entry(
                baseline_method_id,
                baseline_method,
                global_prediction_ref=baseline_global["commit_ref"],
                metrics_ref=baseline_metrics_ref,
            ),
        },
        paired_artifact_ref=paired_ref,
        report_ref=report_ref,
        negative_control_summary_ref=negative_control_ref,
    )
    run_complete_path = output_dir / "RUN_COMPLETE.json"
    run_complete_sha = write_run_complete_exclusive(
        run_complete_path,
        run_complete_payload,
        run_root=output_dir,
    )
    replayed_complete = replay_run_complete(
        output_dir,
        make_artifact_ref(
            output_dir, run_complete_path, run_complete_sha
        ),
    )
    if (
        replayed_complete["declared_status"]
        != (ORDINARY_GO_CANDIDATE if any_go else SCIENTIFIC_HOLD)
        or replayed_complete["sealed_access"] is not False
        or replayed_complete["downstream_unlock"] is not False
    ):
        raise ReportProtocolError("RUN_COMPLETE exact replay changed run disposition")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--activation-manifest", type=Path)
    parser.add_argument("--activation-manifest-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-rows", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if args.synthetic_fixture:
            if args.output_dir is None:
                raise ReportProtocolError("--synthetic-fixture requires --output-dir")
            result = run_synthetic_engineering_fixture(
                output_dir=args.output_dir, seed=args.seed, epochs=args.epochs, n_rows=args.n_rows,
            )
        elif args.preflight_only:
            if args.activation_manifest is None or args.activation_manifest_sha256 is None or args.output_dir is None:
                raise ReportProtocolError(
                    "--preflight-only requires activation manifest, exact SHA-256 and output dir"
                )
            result = preflight_real_ordinary(
                args.activation_manifest,
                args.activation_manifest_sha256,
                prospective_output_dir=args.output_dir,
            )
        else:
            if args.activation_manifest is None or args.activation_manifest_sha256 is None or args.output_dir is None:
                raise ReportProtocolError(
                    "real ordinary execution requires activation manifest, exact SHA-256 and output dir"
                )
            result = run_real_ordinary(
                activation_manifest_path=args.activation_manifest,
                activation_manifest_sha256=args.activation_manifest_sha256,
                output_dir=args.output_dir,
            )
    except (ProtocolHoldError, ReportProtocolError, ValueError) as exc:
        error = {
            "status": getattr(exc, "code", "HOLD_R3X_RUNNER"),
            "message": str(exc),
            "sealed_access": False,
            "downstream_unlock": False,
        }
        print(json.dumps(error, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
