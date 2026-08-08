"""End-to-end ordinary-only execution through the strict real R3X runner.

This fixture deliberately uses the production authority schemas rather than
the synthetic engineering shortcut.  All data are generated locally, every
row is an independent source-relation component, and the deterministic split
has at least ten non-zero components in every publication cell/partition.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.r3x.artifact_graph import (
    make_artifact_ref,
    read_artifact_ref,
)
from scripts.r3x.artifact_io import canonical_json_bytes
from scripts.r3x.control_summary import (
    ACTION_SUPPORT_NOT_QUALIFIED,
    CONTROL_IDS,
    HOLD,
    NOT_RUN,
    load_negative_control_summary,
)
from scripts.r3x.protocol import (
    ProtocolHoldError,
    REQUIRED_ACTIVATION_AUTHORITIES,
    STAGED_DISCLOSURE_HOLD_CODE,
    canonical_sha256,
)
from scripts.r3x.run_complete import (
    COMMIT_SEMANTICS,
    SCIENTIFIC_HOLD,
    replay_run_complete,
)
from scripts.r3x.run_ordinary import (
    main,
    preflight_real_ordinary,
    run_real_ordinary,
    synthetic_rows,
)
from scripts.r3x.splits import build_split_manifest


FROZEN_SEEDS = [11, 23, 42, 67, 101]
REQUIRED_BASELINES = [
    "ZERO_DELTA",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
    "EDIT_ONLY_RIDGE",
    "CANDIDATE_KMER_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
]
NEURAL_METHODS = [
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION",
    "A_EXACT_POTENTIAL_DIFFERENCE",
    "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
    "B_SPARSE_EDIT_SET_ADDITIVE",
]
REQUIRED_COMPLETE_METHODS = [
    "A_EXACT_POTENTIAL_DIFFERENCE",
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
]


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: dict) -> str:
    raw = _json_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _authority_ref(path: Path, digest: str) -> dict:
    return {"path": str(path), "sha256": digest, "status": "FROZEN"}


def _route_authorization() -> dict:
    return {
        "schema_version": "r3_route_authorization_v1",
        "authorization_id": "local-e2e-ordinary-fixture",
        "authorized_at_utc": "2026-08-08T00:00:00Z",
        "authorization_source": "USER_MESSAGE_THIS_THREAD",
        "status": "USER_AUTHORIZED_INPUTS_FROZEN",
        "selected_route": "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY",
        "authorized_sequence": [
            "PUBLICATION_GRADE_STRONG_BASELINES",
            "A_EXACT_POTENTIAL_DIFFERENCE",
            "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
            "B_SPARSE_EDIT_SET_ADDITIVE",
            "B_SPARSE_EDIT_SET_INTERACTION_ONLY_IF_SEPARATELY_QUALIFIED",
        ],
        "code_implementation_authorized": True,
        "ordinary_training_authorized_when_all_inputs_frozen": True,
        "currently_ready_for_real_training": True,
        "required_unclosed_inputs": [],
        "c3_redline": {
            "draft_authorized": True,
            "active_contract_mutation_authorized": False,
            "proposed_positioning": "DGM_DOOB_DERIVED_LEGAL_EDIT_ADAPTATION",
            "original_general_exact_guidance_claim": "REJECT_AS_ORIGINAL_GENERAL_THEORY",
        },
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "formal_sealed_evaluator": False,
            "sign_accuracy_gate": 0.60,
            "sign_gate_change_authorized": False,
            "post_hoc_metric_threshold_seed_or_endpoint_selection": False,
            "legacy_signed_ratio_enrichment": "PROHIBITED",
            "top_selection_gate": "HOLD_PENDING_SEPARATE_PREREGISTRATION",
            "o0_g1_unlock": False,
            "authority_contract_overwrite": False,
        },
    }


def _ordinary_rows() -> list[dict]:
    rows = synthetic_rows(134)
    for index, row in enumerate(rows):
        # This stable study assignment is independent of every target value and
        # is fixed before the source-component split is constructed.
        study_byte = hashlib.sha256(f"study-9-{index}".encode("ascii")).digest()[0]
        study = "study_a" if study_byte < 128 else "study_b"
        context = f"publication_context_{study[-1]}"
        delta = 0.75 if index % 2 else -0.75
        row.update(
            {
                "study": study,
                "endpoint": "shared_endpoint",
                "endpoint_id": "shared_endpoint",
                "estimand_id": "shared_estimand",
                "publication_context_id": context,
                "context_id_v2": context,
                "assay_arm": "ordinary_assay_arm",
                "library": f"ordinary_library_{study[-1]}",
                "source_asset": f"ordinary_source_asset_{study[-1]}",
                "source_anchor_rule": "ROW_EXPLICIT_SOURCE_SEQUENCE",
                # A ranking pool may never cross source components.  These
                # one-row pools are intentionally ineligible for ranking and
                # therefore keep that secondary metric in explicit HOLD.
                "source_group_id_v2": f"ordinary_ranking_pool_{index:03d}",
                "candidate_value": float(row["source_value"]) + delta,
                "delta": delta,
            }
        )
    return rows


def _build_activation_bundle(tmp_path: Path) -> tuple[Path, str, dict, dict]:
    refs: dict[str, dict] = {}

    authorization_path = tmp_path / "route_authorization.json"
    authorization_sha = _write_json(authorization_path, _route_authorization())
    refs["route_authorization"] = _authority_ref(
        authorization_path, authorization_sha
    )

    rows = _ordinary_rows()
    dataset_path = tmp_path / "ordinary_ready.jsonl"
    dataset_raw = b"".join(_json_bytes(row) + b"\n" for row in rows)
    dataset_path.write_bytes(dataset_raw)
    dataset_sha = hashlib.sha256(dataset_raw).hexdigest()
    refs["ready_dataset"] = _authority_ref(dataset_path, dataset_sha)

    crosswalk = {
        "schema_version": "r3_ordinary_context_crosswalk_v1",
        "status": "FROZEN",
        "entries": [
            {
                "study": study,
                "endpoint": "shared_endpoint",
                "assay_arm": "ordinary_assay_arm",
                "library": f"ordinary_library_{suffix}",
                "source_asset": f"ordinary_source_asset_{suffix}",
                "region": "5UTR",
                "source_anchor_rule": "ROW_EXPLICIT_SOURCE_SEQUENCE",
                "publication_context_id": f"publication_context_{suffix}",
                "context_evidence_id": f"context_evidence_{suffix}",
                "source_relation_evidence_id": f"source_relation_evidence_{suffix}",
                "allowed_row_relation_evidence": [
                    {"evidence_id": f"source_relation_evidence_{suffix}"}
                ],
            }
            for study, suffix in (("study_a", "a"), ("study_b", "b"))
        ],
    }
    crosswalk_path = tmp_path / "context_source_crosswalk.json"
    crosswalk_sha = _write_json(crosswalk_path, crosswalk)
    refs["context_source_crosswalk"] = _authority_ref(
        crosswalk_path, crosswalk_sha
    )

    estimand_registry = {
        "schema_version": "r3_endpoint_estimand_registry_v1",
        "status": "FROZEN",
        "entries": [
            {
                "study": study,
                "endpoint": "shared_endpoint",
                "estimand_id": "shared_estimand",
                "raw_source_field": "source_value",
                "raw_candidate_field": "candidate_value",
                "raw_unit": "fixture_effect_unit",
                "direction": "HIGHER_IS_MORE_STABLE",
                "transform": "IDENTITY_DIFFERENCE",
                "valid_domain": "FINITE",
                "role": "PRIMARY",
                "replicate_policy": "SINGLE_OBSERVATION_ONLY",
                "source_evidence_sha256": hashlib.sha256(
                    f"estimand-evidence-{study}".encode("ascii")
                ).hexdigest(),
            }
            for study in ("study_a", "study_b")
        ],
    }
    estimand_path = tmp_path / "endpoint_estimand_registry.json"
    estimand_sha = _write_json(estimand_path, estimand_registry)
    refs["endpoint_estimand_registry"] = _authority_ref(
        estimand_path, estimand_sha
    )

    row_ids = sorted(row["r3_input_row_sha256"] for row in rows)
    inclusion = {
        "schema_version": "r3x_inclusion_manifest_v1",
        "status": "FROZEN",
        "n_ready_rows": len(rows),
        "ready_dataset_sha256": dataset_sha,
        "ready_row_ids_sha256": hashlib.sha256(
            json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "context_source_crosswalk_sha256": crosswalk_sha,
        "endpoint_estimand_registry_sha256": estimand_sha,
        "ordinary_membership_registry_sha256": hashlib.sha256(
            b"local-ordinary-membership-registry"
        ).hexdigest(),
        "protected_member_count": 0,
        "ordinary_only": True,
        "rows": [
            {
                "row_id": row["r3_input_row_sha256"],
                "row_content_sha256": canonical_sha256(row),
                "role": "ORDINARY_PRIMARY_READY",
            }
            for row in rows
        ],
    }
    inclusion_path = tmp_path / "inclusion_manifest.json"
    inclusion_sha = _write_json(inclusion_path, inclusion)
    refs["inclusion_manifest"] = _authority_ref(inclusion_path, inclusion_sha)

    split = build_split_manifest(
        rows,
        seed=42,
        publication_claim=True,
        support_policy={
            "status": "FROZEN",
            "minimum_components_per_cell_partition": 10,
        },
        authority_bindings={
            "ready_dataset_sha256": dataset_sha,
            "inclusion_manifest_sha256": inclusion_sha,
        },
    )
    assert split["support_check"]["status"] == (
        "PASS_FROZEN_MINIMUM_COMPONENT_SUPPORT"
    )
    assert all(
        count >= 10
        for partitions in split["cell_component_counts"].values()
        for count in partitions.values()
    )
    split_path = tmp_path / "source_component_split_manifest.json"
    split_sha = _write_json(split_path, split)
    refs["source_component_split_manifest"] = _authority_ref(split_path, split_sha)

    metric_registry = {
        "schema_version": "r3x_metric_zero_policy_registry_v1",
        "status": "FROZEN",
        "zero_policy": "EXCLUDE_TRUE_ZERO",
        "primary_cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
        "component_weighting": "EQUAL_COMPONENT_THEN_EQUAL_ROW_WITHIN_COMPONENT",
        "ranking_aggregation": "WITHIN_SOURCE_POOL_THEN_CELL_MACRO",
        "undefined_primary_cell_policy": "HOLD_PRIMARY_CELL_UNDEFINED",
        "seed_policy": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
        "selection_tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        "selection_metric_policy": {
            "status": "HOLD_PENDING_SEPARATE_PREREGISTRATION"
        },
    }
    metric_path = tmp_path / "metric_zero_policy_registry.json"
    metric_sha = _write_json(metric_path, metric_registry)
    refs["metric_zero_policy_registry"] = _authority_ref(metric_path, metric_sha)

    code_path = tmp_path / "analysis_source.py"
    code_path.write_bytes(b"ANALYSIS_FIXTURE_VERSION = 1\n")
    code_entry = {
        "path": code_path.name,
        "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
    }
    code_manifest = {
        "schema_version": "r3x_analysis_code_tree_manifest_v1",
        "status": "FROZEN",
        "root_path": ".",
        "files": [code_entry],
        "tree_sha256": canonical_sha256([code_entry]),
    }
    code_manifest_path = tmp_path / "analysis_code_tree_manifest.json"
    code_manifest_sha = _write_json(code_manifest_path, code_manifest)
    refs["analysis_code_tree_manifest"] = _authority_ref(
        code_manifest_path, code_manifest_sha
    )

    method_protocol = {
        "schema_version": "r3_ordinary_experiment_protocol_v1",
        "status": "FROZEN",
        "selected_route": "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY",
        "real_ordinary_training_ready": True,
        "negative_controls": list(CONTROL_IDS),
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "O0_G1_E0_unlock": False,
        },
    }
    method_path = tmp_path / "method_protocol.json"
    method_sha = _write_json(method_path, method_protocol)
    refs["method_protocol"] = _authority_ref(method_path, method_sha)

    paired_policy = {
        "schema_version": "r3x_paired_strongest_baseline_comparison_policy_v1",
        "status": "FROZEN",
        "policy_id": "local_e2e_a_vs_full_pair",
        "baseline_suite_status": "PUBLICATION_QUALIFIED",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "strongest_baseline_id": "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "comparison_family_id": "local_e2e_primary_sign_family",
        "comparison_family_member_ids": ["a_vs_full_pair_primary_sign"],
        "primary_comparison_member_id": "a_vs_full_pair_primary_sign",
        "primary_sign_inference": {
            "metric_id": "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY",
            "alpha": 0.05,
            "confidence": 0.95,
            "bootstrap": {
                "n_resamples": 2000,
                "seed": 23,
                "min_components_per_cell": 10,
                "max_attempt_multiplier": 100,
            },
            "permutation": {
                "n_permutations": 10000,
                "seed": 42,
                "exact_component_limit": 20,
                "min_components_per_cell": 10,
            },
        },
        "required_baseline_method_ids": REQUIRED_BASELINES,
        "ready_dataset_sha256": dataset_sha,
        "inclusion_manifest_sha256": inclusion_sha,
        "source_component_split_manifest_sha256": split_sha,
        "metric_zero_policy_registry_sha256": metric_sha,
    }
    paired_path = tmp_path / "paired_policy.json"
    paired_sha = _write_json(paired_path, paired_policy)
    refs["paired_strongest_baseline_comparison_policy"] = _authority_ref(
        paired_path, paired_sha
    )

    # This config is created last because its authority map must bind every
    # other activated authority exactly and may not bind itself recursively.
    run_config = {
        "schema_version": "r3x_model_optimizer_seed_config_v1",
        "status": "FROZEN",
        "seeds": FROZEN_SEEDS,
        "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "epochs": 1,
        "batch_size": 256,
        "device": "cpu",
        "condition_unknown_dropout": 0.15,
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
        },
        "loss": {
            "huber_weight": 1.0,
            "nll_weight": 0.25,
            "sign_weight": 0.25,
            "sign_temperature": 1.0,
        },
        "checkpoint_selection": {
            "rule": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH"
        },
        "model": {
            "vocab_size": 5,
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "ff_dim": 16,
            "max_len": 8,
            "n_endpoints": 2,
            "n_contexts": 3,
            "max_edits": 5,
            "pair_rank": 2,
        },
        "authority_sha256": {
            name: ref["sha256"] for name, ref in refs.items()
        },
    }
    run_config_path = tmp_path / "model_optimizer_seed_config.json"
    run_config_sha = _write_json(run_config_path, run_config)
    refs["model_optimizer_seed_config"] = _authority_ref(
        run_config_path, run_config_sha
    )

    activation = {
        "schema_version": "r3_activation_manifest_v1",
        "status": "FROZEN",
        "mode": "ORDINARY_ONLY",
        "claim_scope": "PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        "authorities": refs,
        "study_endpoint_map": {
            "study_a": ["shared_endpoint"],
            "study_b": ["shared_endpoint"],
        },
        "study_estimand_map": {
            "study_a": ["shared_estimand"],
            "study_b": ["shared_estimand"],
        },
    }
    activation_path = tmp_path / "activation.json"
    activation_sha = _write_json(activation_path, activation)
    return activation_path, activation_sha, split, refs


def test_real_ordinary_runner_stops_before_monolithic_ready_outcome_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    activation_path, activation_sha, split, refs = _build_activation_bundle(tmp_path)
    output = tmp_path / "ordinary_e2e_output"
    assert not output.exists()
    assert all(
        partitions["internal_test"] >= 10
        for partitions in split["cell_component_counts"].values()
    )

    calls = {"legacy_runner": 0, "ready_reader": 0}

    def forbidden_legacy_runner(*_args, **_kwargs):
        calls["legacy_runner"] += 1
        raise AssertionError(
            "legacy monolithic runner must not open ready outcomes"
        )

    def forbidden_ready_reader(*_args, **_kwargs):
        calls["ready_reader"] += 1
        raise AssertionError("ready outcome reader must remain unopened")

    monkeypatch.setattr(
        "scripts.r3x.run_ordinary._run_legacy_outcome_bearing_development_provisional",
        forbidden_legacy_runner,
    )
    monkeypatch.setattr(
        "scripts.r3x.run_ordinary._read_ready_jsonl_exact",
        forbidden_ready_reader,
    )
    with pytest.raises(ProtocolHoldError) as run_exc:
        run_real_ordinary(
            activation_manifest_path=activation_path,
            activation_manifest_sha256=activation_sha,
            output_dir=output,
        )
    assert run_exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    preflight_output = tmp_path / "ordinary_preflight_output"
    with pytest.raises(ProtocolHoldError) as preflight_exc:
        preflight_real_ordinary(
            activation_path,
            activation_sha,
            prospective_output_dir=preflight_output,
        )
    assert preflight_exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    cli_preflight_output = tmp_path / "ordinary_cli_preflight_output"
    assert main([
        "--preflight-only",
        "--activation-manifest",
        str(activation_path),
        "--activation-manifest-sha256",
        activation_sha,
        "--output-dir",
        str(cli_preflight_output),
    ]) == 2
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["status"] == STAGED_DISCLOSURE_HOLD_CODE
    assert STAGED_DISCLOSURE_HOLD_CODE in cli_result["message"]
    assert cli_result["sealed_access"] is False
    assert cli_result["downstream_unlock"] is False

    # A future edit that makes the current guard return is not authorization to
    # fall through into the retained legacy outcome-bearing scaffold.
    monkeypatch.setattr(
        "scripts.r3x.run_ordinary.require_staged_disclosure_before_outcome_access",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ProtocolHoldError) as guard_return_run_exc:
        run_real_ordinary(
            activation_manifest_path=activation_path,
            activation_manifest_sha256=activation_sha,
            output_dir=tmp_path / "guard-return-run",
        )
    assert guard_return_run_exc.value.code == (
        "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED"
    )
    with pytest.raises(ProtocolHoldError) as guard_return_preflight_exc:
        preflight_real_ordinary(
            activation_path,
            activation_sha,
            prospective_output_dir=tmp_path / "guard-return-preflight",
        )
    assert guard_return_preflight_exc.value.code == (
        "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED"
    )
    assert calls == {"legacy_runner": 0, "ready_reader": 0}
    assert not output.exists()
    assert not preflight_output.exists()
    assert not cli_preflight_output.exists()
    assert not list(tmp_path.glob("**/internal_test_predictions.jsonl"))
    assert not list(tmp_path.glob("**/RUN_COMPLETE.json"))
    # The source fixture remains ordinary and the sealed boundary is unchanged;
    # execution stops because physical outcome disclosure has not been staged.
    assert refs["ready_dataset"]["status"] == "FROZEN"
