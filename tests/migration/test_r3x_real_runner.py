from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.protocol import ProtocolHoldError, STAGED_DISCLOSURE_HOLD_CODE
from scripts.r3x.run_ordinary import (
    _run_legacy_outcome_bearing_development_provisional,
    run_real_ordinary,
    synthetic_rows,
)
from scripts.r3x.splits import build_split_manifest


def write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready_authorization() -> dict:
    return {
        "schema_version": "r3_route_authorization_v1",
        "authorization_id": "fixture-ready-auth",
        "authorized_at_utc": "2026-08-07T00:00:00Z",
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
        },
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "formal_sealed_evaluator": False,
            "sign_accuracy_gate": 0.60,
            "sign_gate_change_authorized": False,
            "post_hoc_metric_threshold_seed_or_endpoint_selection": False,
            "legacy_signed_ratio_enrichment": "PROHIBITED",
            "o0_g1_unlock": False,
            "authority_contract_overwrite": False,
        },
    }


def test_status_only_fixture_authorities_cannot_start_real_training(tmp_path: Path) -> None:
    rows = synthetic_rows(30)
    dataset_path = tmp_path / "ordinary_ready.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    split = build_split_manifest(rows, seed=11)
    split["status"] = "FROZEN"
    split_path = tmp_path / "source_component_split.json"
    split_sha = write_json(split_path, split)

    auth_path = tmp_path / "route_authorization.json"
    auth_sha = write_json(auth_path, ready_authorization())

    authorities = {
        "route_authorization": (auth_path, auth_sha),
        "ready_dataset": (dataset_path, dataset_sha),
        "source_component_split_manifest": (split_path, split_sha),
    }
    frozen_payloads = {
        "inclusion_manifest": {"schema_version": "fixture_inclusion_v1", "status": "FROZEN"},
        "context_source_crosswalk": {"schema_version": "fixture_crosswalk_v1", "status": "FROZEN"},
        "endpoint_estimand_registry": {"schema_version": "fixture_estimand_v1", "status": "FROZEN"},
        "metric_zero_policy_registry": {
            "schema_version": "fixture_metric_v1",
            "status": "FROZEN",
            "zero_policy": "EXCLUDE_TRUE_ZERO",
            "selection_metric_policy": {
                "status": "HOLD_PENDING_SEPARATE_PREREGISTRATION"
            },
        },
        "model_optimizer_seed_config": {
            "schema_version": "fixture_run_config_v1",
            "status": "FROZEN",
            "seeds": [11, 23, 42, 67, 101],
            "epochs": 1,
            "batch_size": 64,
            "device": "cpu",
            "strongest_paired_baseline_id": "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
            "model": {
                "vocab_size": 5,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "ff_dim": 16,
                "max_len": 8,
                "n_endpoints": 2,
                "n_contexts": 2,
                "max_edits": 5,
                "pair_rank": 2
            },
        },
        "analysis_code_tree_manifest": {"schema_version": "fixture_code_v1", "status": "FROZEN"},
    }
    for name, payload in frozen_payloads.items():
        path = tmp_path / f"{name}.json"
        authorities[name] = (path, write_json(path, payload))

    activation = {
        "schema_version": "r3_activation_manifest_v1",
        "status": "FROZEN",
        "mode": "ORDINARY_ONLY",
        "claim_scope": "DEVELOPMENT_ONLY",
        "study_endpoint_map": {"synthetic_study": ["synthetic_endpoint"]},
        "authorities": {
            name: {"path": str(path), "sha256": digest, "status": "FROZEN"}
            for name, (path, digest) in authorities.items()
        },
    }
    activation_path = tmp_path / "activation.json"
    activation_sha = write_json(activation_path, activation)
    output = tmp_path / "real-ordinary-output"
    with pytest.raises(ProtocolHoldError) as exc:
        run_real_ordinary(
            activation_manifest_path=activation_path,
            activation_manifest_sha256=activation_sha,
            output_dir=output,
        )
    assert exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    assert not output.exists()


def test_private_legacy_runner_is_disabled_before_any_outcome_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("legacy outcome reader must remain unreachable")

    monkeypatch.setattr(
        "scripts.r3x.run_ordinary._audit_legacy_outcome_bearing_activation_manifest",
        forbidden,
    )
    monkeypatch.setattr(
        "scripts.r3x.run_ordinary._read_ready_jsonl_exact",
        forbidden,
    )
    with pytest.raises(ProtocolHoldError) as exc:
        _run_legacy_outcome_bearing_development_provisional(
            activation_manifest_path=tmp_path / "unused-activation.json",
            activation_manifest_sha256="0" * 64,
            output_dir=tmp_path / "unused-output",
        )
    assert exc.value.code == "HOLD_LEGACY_OUTCOME_BEARING_RUNNER_DISABLED"
    assert calls == []
    assert not (tmp_path / "unused-output").exists()
