"""Fail-closed protocol tests for the R3 ordinary-only route."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.r3x.protocol as protocol_module  # noqa: E402

from scripts.r3x.protocol import (  # noqa: E402
    ProtocolHoldError,
    _LEGACY_DEVELOPMENT_AUDIT_ACKNOWLEDGEMENT,
    REQUIRED_STAGED_DISCLOSURE_CAPABILITIES,
    STAGED_DISCLOSURE_HOLD_CODE,
    assert_ordinary_path,
    assert_ordinary_record,
    _audit_legacy_outcome_bearing_activation_manifest,
    canonical_sha256,
    load_route_authorization,
    require_staged_disclosure_before_outcome_access,
    sha256_file,
    validate_activation_manifest,
)
from scripts.r3x.splits import build_split_manifest, sequence_identity  # noqa: E402


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_development_audit(path, digest, **kwargs):
    return _audit_legacy_outcome_bearing_activation_manifest(
        path,
        digest,
        _development_audit_acknowledgement=(
            _LEGACY_DEVELOPMENT_AUDIT_ACKNOWLEDGEMENT
        ),
        **kwargs,
    )


def _authorization(*, ready: bool = True, **boundary_overrides) -> dict:
    hard_boundaries = {
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
    }
    hard_boundaries.update(boundary_overrides)
    return {
        "schema_version": "r3_route_authorization_v1",
        "authorization_id": "fixture-auth",
        "authorized_at_utc": "2026-08-07T00:00:00Z",
        "authorization_source": "USER_MESSAGE_THIS_THREAD",
        "status": (
            "USER_AUTHORIZED_INPUTS_FROZEN"
            if ready
            else "USER_AUTHORIZED_INPUTS_NOT_YET_QUALIFIED"
        ),
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
        "currently_ready_for_real_training": ready,
        "required_unclosed_inputs": [] if ready else ["FROZEN_CONTEXT_SOURCE_CROSSWALK"],
        "c3_redline": {
            "draft_authorized": True,
            "active_contract_mutation_authorized": False,
            "proposed_positioning": "DGM_DOOB_DERIVED_LEGAL_EDIT_ADAPTATION",
            "original_general_exact_guidance_claim": "REJECT_AS_ORIGINAL_GENERAL_THEORY",
        },
        "hard_boundaries": hard_boundaries,
    }


def _activation_fixture(
    tmp_path: Path,
    *,
    claim_scope: str = "DEVELOPMENT_ONLY",
    single_study: bool = False,
    shared_estimand: bool = False,
):
    auth_path = tmp_path / "route_authorization.json"
    auth_sha = _write_json(auth_path, _authorization())
    refs = {"route_authorization": {
        "path": str(auth_path), "sha256": auth_sha, "status": "FROZEN",
    }}

    if single_study:
        study_specs = [("study_a", "endpoint_a", "context_a", "estimand_a")]
    elif shared_estimand:
        study_specs = [
            ("study_a", "endpoint_a", "context_a", "estimand_a"),
            ("study_b", "endpoint_a", "context_b", "estimand_a"),
        ]
    else:
        study_specs = [
            ("study_a", "endpoint_a", "context_a", "estimand_a"),
            ("study_b", "endpoint_b", "context_b", "estimand_b"),
        ]
    rows = []
    for index in range(6):
        study, endpoint, context, estimand_id = study_specs[
            0 if single_study or index < 3 else 1
        ]
        source = "A" * (index + 3) + "C"
        candidate = "A" * (index + 3) + "G"
        rows.append({
            "r3_input_row_sha256": hashlib.sha256(f"row-{index}".encode()).hexdigest(),
            "normalized_source_sequence": source,
            "normalized_candidate_sequence": candidate,
            "source_sequence_sha256": sequence_identity(source),
            "candidate_sequence_sha256": sequence_identity(candidate),
            "source_group_id_v2": f"pool_{index}",
            "frozen_relation_atoms": [{
                "atom_type": "SOURCE_ALIAS",
                "atom_id": f"alias_{index}",
                "member": "SOURCE",
                "status": "FROZEN",
                "evidence_sha256": "d" * 64,
            }],
            "study": study,
            "endpoint": endpoint,
            "publication_context_id": context,
            "estimand_id": estimand_id,
        })
    dataset_path = tmp_path / "ordinary_ready.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    refs["ready_dataset"] = {
        "path": str(dataset_path), "sha256": dataset_sha, "status": "FROZEN",
    }

    crosswalk_path = tmp_path / "context_source_crosswalk.json"
    crosswalk_sha = _write_json(crosswalk_path, {
        "schema_version": "r3_ordinary_context_crosswalk_v1", "status": "FROZEN",
        "entries": [
            {
                "study": study, "endpoint": endpoint, "assay_arm": "arm",
                "library": "library", "source_asset": f"ordinary_asset_{study}",
                "region": "5UTR", "source_anchor_rule": "single",
                "publication_context_id": context,
                "context_evidence_id": f"context_evidence_{study}",
                "source_relation_evidence_id": f"relation_evidence_{study}",
                "allowed_row_relation_evidence": [{"evidence_id": f"relation_evidence_{study}"}],
            }
            for study, endpoint, context, _ in study_specs
        ],
    })
    refs["context_source_crosswalk"] = {
        "path": str(crosswalk_path), "sha256": crosswalk_sha, "status": "FROZEN",
    }

    estimand_path = tmp_path / "endpoint_estimand_registry.json"
    estimand_sha = _write_json(estimand_path, {
        "schema_version": "r3_endpoint_estimand_registry_v1", "status": "FROZEN",
        "entries": [
            {
                "study": study, "endpoint": endpoint, "estimand_id": estimand_id,
                "raw_source_field": "source_value", "raw_candidate_field": "candidate_value",
                "raw_unit": "unit", "direction": "HIGHER_IS_MORE_STABLE",
                "transform": "IDENTITY_DIFFERENCE", "valid_domain": "FINITE", "role": "PRIMARY",
                "replicate_policy": "SINGLE_OBSERVATION_ONLY", "source_evidence_sha256": "e" * 64,
            }
            for study, endpoint, _, estimand_id in study_specs
        ],
    })
    refs["endpoint_estimand_registry"] = {
        "path": str(estimand_path), "sha256": estimand_sha, "status": "FROZEN",
    }

    row_ids = sorted(row["r3_input_row_sha256"] for row in rows)
    row_ids_sha = hashlib.sha256(json.dumps(row_ids, separators=(",", ":")).encode()).hexdigest()
    inclusion_path = tmp_path / "inclusion_manifest.json"
    inclusion_sha = _write_json(inclusion_path, {
        "schema_version": "r3x_inclusion_manifest_v1", "status": "FROZEN",
        "n_ready_rows": len(rows), "ready_dataset_sha256": dataset_sha,
        "ready_row_ids_sha256": row_ids_sha,
        "context_source_crosswalk_sha256": crosswalk_sha,
        "endpoint_estimand_registry_sha256": estimand_sha,
        "ordinary_membership_registry_sha256": "a" * 64,
        "protected_member_count": 0, "ordinary_only": True,
        "rows": [
            {
                "row_id": row["r3_input_row_sha256"],
                "row_content_sha256": canonical_sha256(row),
                "role": "ORDINARY_PRIMARY_READY",
            }
            for row in rows
        ],
    })
    refs["inclusion_manifest"] = {
        "path": str(inclusion_path), "sha256": inclusion_sha, "status": "FROZEN",
    }

    split_path = tmp_path / "source_component_split_manifest.json"
    split_payload = build_split_manifest(
        rows,
        seed=42,
        authority_bindings={
            "ready_dataset_sha256": dataset_sha,
            "inclusion_manifest_sha256": inclusion_sha,
        },
    )
    split_payload["status"] = "FROZEN"
    split_payload_without_hash = dict(split_payload)
    split_payload_without_hash.pop("manifest_sha256")
    split_payload["manifest_sha256"] = canonical_sha256(split_payload_without_hash)
    split_sha = _write_json(split_path, split_payload)
    refs["source_component_split_manifest"] = {
        "path": str(split_path), "sha256": split_sha, "status": "FROZEN",
    }

    metric_path = tmp_path / "metric_zero_policy_registry.json"
    metric_sha = _write_json(metric_path, {
        "schema_version": "r3x_metric_zero_policy_registry_v1", "status": "FROZEN",
        "zero_policy": "EXCLUDE_TRUE_ZERO",
        "primary_cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
        "component_weighting": "EQUAL_COMPONENT_THEN_EQUAL_ROW_WITHIN_COMPONENT",
        "ranking_aggregation": "WITHIN_SOURCE_POOL_THEN_CELL_MACRO",
        "undefined_primary_cell_policy": "HOLD_PRIMARY_CELL_UNDEFINED",
        "seed_policy": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
        "selection_tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
    })
    refs["metric_zero_policy_registry"] = {
        "path": str(metric_path), "sha256": metric_sha, "status": "FROZEN",
    }

    code_path = tmp_path / "code_fixture.py"
    code_path.write_text("VALUE = 1\n", encoding="utf-8")
    code_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
    code_entries = [{"path": "code_fixture.py", "sha256": code_sha}]
    code_manifest_path = tmp_path / "analysis_code_tree_manifest.json"
    code_manifest_sha = _write_json(code_manifest_path, {
        "schema_version": "r3x_analysis_code_tree_manifest_v1", "status": "FROZEN",
        "root_path": ".", "files": code_entries,
        "tree_sha256": canonical_sha256(code_entries),
    })
    refs["analysis_code_tree_manifest"] = {
        "path": str(code_manifest_path), "sha256": code_manifest_sha, "status": "FROZEN",
    }

    method_path = tmp_path / "method_protocol.json"
    method_sha = _write_json(method_path, {
        "schema_version": "r3_ordinary_experiment_protocol_v1", "status": "FROZEN",
        "selected_route": "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY",
        "real_ordinary_training_ready": True,
        "negative_controls": [
            "SOURCE_CANDIDATE_PAIRING_SHUFFLE",
            "CONTEXT_LABEL_PERMUTATION",
            "A_IDENTITY_AND_SWAP_INVARIANTS",
            "SOURCE_CONTEXT_SHUFFLE_WITHIN_ACTION",
            "EDIT_POSITION_SHUFFLE_WITHIN_SOURCE",
            "TARGET_TOKEN_SHUFFLE_WITHIN_CELL",
            "B_ADDITIVE_VERSUS_INTERACTION",
        ],
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "O0_G1_E0_unlock": False,
        },
    })
    refs["method_protocol"] = {
        "path": str(method_path), "sha256": method_sha, "status": "FROZEN",
    }

    paired_path = tmp_path / "paired_policy.json"
    paired_sha = _write_json(paired_path, {
        "schema_version": "r3x_paired_strongest_baseline_comparison_policy_v1",
        "status": "FROZEN", "policy_id": "policy", "baseline_suite_status": "PUBLICATION_QUALIFIED",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "strongest_baseline_id": "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", "comparison_family_id": "family",
        "comparison_family_member_ids": ["a_vs_fullpair_sign"],
        "primary_comparison_member_id": "a_vs_fullpair_sign",
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
        "required_baseline_method_ids": [
            "ZERO_DELTA", "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR", "EDIT_ONLY_RIDGE",
            "CANDIDATE_KMER_RIDGE", "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
            "PAIRED_FEATURE_NEAREST_NEIGHBOR", "FULL_PAIR_UNRESTRICTED_SIAMESE",
        ],
        "ready_dataset_sha256": dataset_sha, "inclusion_manifest_sha256": inclusion_sha,
        "source_component_split_manifest_sha256": split_sha,
        "metric_zero_policy_registry_sha256": metric_sha,
    })
    refs["paired_strongest_baseline_comparison_policy"] = {
        "path": str(paired_path), "sha256": paired_sha, "status": "FROZEN",
    }

    digest_map = {name: value["sha256"] for name, value in refs.items()}
    run_path = tmp_path / "model_optimizer_seed_config.json"
    run_sha = _write_json(run_path, {
        "schema_version": "r3x_model_optimizer_seed_config_v1", "status": "FROZEN",
        "seeds": [11, 23, 42, 67, 101], "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "condition_unknown_dropout": 0.15,
        "optimizer": {"name": "Adam", "learning_rate": 0.002, "weight_decay": 0.0001},
        "loss": {"huber_weight": 1.0, "nll_weight": 0.25, "sign_weight": 0.25, "sign_temperature": 1.0},
        "checkpoint_selection": {"rule": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH"},
        "model": {"architecture": "fixture"}, "authority_sha256": digest_map,
    })
    refs["model_optimizer_seed_config"] = {
        "path": str(run_path), "sha256": run_sha, "status": "FROZEN",
    }
    manifest = {
        "schema_version": "r3_activation_manifest_v1",
        "status": "FROZEN",
        "mode": "ORDINARY_ONLY",
        "claim_scope": claim_scope,
        "authorities": refs,
        "study_endpoint_map": {
            study: [endpoint] for study, endpoint, _, _ in study_specs
        },
        "study_estimand_map": {
            study: [estimand_id] for study, _, _, estimand_id in study_specs
        },
    }
    path = tmp_path / "activation.json"
    digest = _write_json(path, manifest)
    return path, digest, refs


def test_canonical_sha_is_order_invariant_and_file_hash_is_exact(tmp_path: Path):
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    path = tmp_path / "ordinary.json"
    path.write_bytes(b"{\"a\":1}\n")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_authorization_validates_but_real_training_is_hold():
    path = REPO_ROOT / "configs/r3x/r3_route_authorization_v1.json"
    parsed = load_route_authorization(path, sha256_file(path), require_training_ready=False)
    assert parsed["selected_route"] == "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY"
    with pytest.raises(ProtocolHoldError) as exc:
        load_route_authorization(path, sha256_file(path), require_training_ready=True)
    assert exc.value.code == "HOLD_INPUT_AUTHORITY"
    assert "FROZEN_CONTEXT_SOURCE_CROSSWALK" in exc.value.details["missing"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("schema_version", "wrong"), "schema"),
        (("selected_route", "A_ONLY"), "route"),
        (("code_implementation_authorized", False), "implementation"),
    ],
)
def test_authorization_rejects_wrong_schema_route_or_permission(
    tmp_path: Path, mutation: tuple[str, object], match: str
):
    payload = _authorization()
    payload[mutation[0]] = mutation[1]
    path = tmp_path / "authorization.json"
    digest = _write_json(path, payload)
    with pytest.raises(ProtocolHoldError, match=match):
        load_route_authorization(path, digest)


@pytest.mark.parametrize(
    ("boundary", "value"),
    [
        ("ordinary_only", False),
        ("sealed_member_access", True),
        ("formal_sealed_evaluator", True),
        ("sign_accuracy_gate", 0.59),
        ("sign_gate_change_authorized", True),
        ("post_hoc_metric_threshold_seed_or_endpoint_selection", True),
        ("legacy_signed_ratio_enrichment", "ALLOWED"),
        ("o0_g1_unlock", True),
        ("authority_contract_overwrite", True),
    ],
)
def test_authorization_rejects_any_boundary_relaxation(
    tmp_path: Path, boundary: str, value: object
):
    path = tmp_path / "authorization.json"
    digest = _write_json(path, _authorization(**{boundary: value}))
    with pytest.raises(ProtocolHoldError) as exc:
        load_route_authorization(path, digest)
    assert exc.value.code == "HOLD_ROUTE_AUTHORIZATION"


@pytest.mark.parametrize("token", ["sealed", "restricted", "GSE246381"])
def test_protected_path_is_rejected_before_open(tmp_path: Path, token: str):
    path = tmp_path / f"{token}_members.jsonl"
    with pytest.raises(ProtocolHoldError) as exc:
        assert_ordinary_path(path, purpose="fixture")
    assert exc.value.code == "HOLD_PROTECTED_INPUT"
    assert not path.exists()


def test_symlink_to_protected_location_is_rejected(tmp_path: Path):
    protected_dir = tmp_path / "sealed_members"
    protected_dir.mkdir()
    target = protected_dir / "rows.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "ordinary_link.jsonl"
    link.symlink_to(target)
    with pytest.raises(ProtocolHoldError, match="protected"):
        assert_ordinary_path(link)


def test_nested_protected_record_is_rejected_but_false_flag_is_allowed():
    assert_ordinary_record({"is_sealed": False, "metadata": {"role": "ordinary"}})
    with pytest.raises(ProtocolHoldError) as exc:
        assert_ordinary_record({"metadata": {"access": {"role": "sealed_final"}}})
    assert exc.value.code == "HOLD_PROTECTED_INPUT"


def test_staged_disclosure_guard_holds_before_monolithic_authority_access(
    tmp_path: Path,
):
    path = tmp_path / "activation.json"
    digest = _write_json(
        path,
        {
            "schema_version": "r3_activation_manifest_v1",
            "status": "FROZEN",
            "authorities": {
                "ready_dataset": {
                    "path": str(tmp_path / "must-not-be-opened.jsonl"),
                    "sha256": "a" * 64,
                    "status": "FROZEN",
                }
            },
        },
    )
    output = tmp_path / "future-output"
    with pytest.raises(ProtocolHoldError) as exc:
        require_staged_disclosure_before_outcome_access(
            path,
            digest,
            prospective_output_dir=output,
        )
    assert exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    assert exc.value.details["ordinary_internal_test_feature_access"] is False
    assert exc.value.details["ordinary_internal_test_outcome_access"] is False
    assert exc.value.details["ordinary_internal_test_predictions_generated"] is False
    assert sorted(exc.value.details["missing_capabilities"]) == sorted(
        REQUIRED_STAGED_DISCLOSURE_CAPABILITIES
    )
    assert not output.exists()


def test_staged_disclosure_metadata_cannot_bypass_unimplemented_v2_validator(
    tmp_path: Path,
):
    path = tmp_path / "activation.json"
    digest = _write_json(
        path,
        {
            "schema_version": "r3_activation_manifest_v2_staged_disclosure",
            "staged_disclosure": {
                key: True for key in REQUIRED_STAGED_DISCLOSURE_CAPABILITIES
            },
        },
    )
    with pytest.raises(ProtocolHoldError) as exc:
        require_staged_disclosure_before_outcome_access(path, digest)
    assert exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    assert exc.value.details["missing_capabilities"] == []
    assert sorted(exc.value.details["invalid_capabilities"]) == sorted(
        REQUIRED_STAGED_DISCLOSURE_CAPABILITIES
    )
    assert exc.value.details["unverified_capabilities"] == []


def test_public_activation_validator_cannot_bypass_staged_disclosure(
    tmp_path: Path,
):
    path, digest, _ = _activation_fixture(tmp_path)
    with pytest.raises(ProtocolHoldError) as exc:
        validate_activation_manifest(path, digest)
    assert exc.value.code == STAGED_DISCLOSURE_HOLD_CODE


def test_public_activation_guard_return_is_not_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "activation.json"
    digest = _write_json(path, {"schema_version": "future_fixture"})
    monkeypatch.setattr(
        "scripts.r3x.protocol.require_staged_disclosure_before_outcome_access",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ProtocolHoldError) as exc:
        validate_activation_manifest(path, digest)
    assert exc.value.code == "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED"


def test_staged_guard_reads_only_exact_activation_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path, digest, refs = _activation_fixture(tmp_path)
    reads = []
    original = protocol_module.read_exact_bytes

    def recording_reader(candidate, expected, *, purpose):
        reads.append(Path(candidate).resolve())
        return original(candidate, expected, purpose=purpose)

    monkeypatch.setattr(protocol_module, "read_exact_bytes", recording_reader)
    with pytest.raises(ProtocolHoldError) as exc:
        require_staged_disclosure_before_outcome_access(path, digest)
    assert exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    assert reads == [path.resolve()]
    assert Path(refs["ready_dataset"]["path"]).resolve() not in reads


def test_staged_disclosure_guard_checks_activation_hash_and_output_first(
    tmp_path: Path,
):
    path = tmp_path / "activation.json"
    digest = _write_json(path, {"schema_version": "r3_activation_manifest_v1"})
    with pytest.raises(ProtocolHoldError) as hash_exc:
        require_staged_disclosure_before_outcome_access(path, "0" * 64)
    assert hash_exc.value.code == "HOLD_HASH_MISMATCH"

    output = tmp_path / "existing-output"
    output.mkdir()
    with pytest.raises(ProtocolHoldError) as output_exc:
        require_staged_disclosure_before_outcome_access(
            path,
            digest,
            prospective_output_dir=output,
        )
    assert output_exc.value.code == (
        "HOLD_OUTPUT_EXISTS_BEFORE_STAGED_DISCLOSURE_PREFLIGHT"
    )


def test_activation_manifest_verifies_every_exact_hash(tmp_path: Path):
    path, digest, _ = _activation_fixture(tmp_path)
    result = _legacy_development_audit(path, digest)
    assert result["status"] == "LEGACY_OUTCOME_BEARING_DEVELOPMENT_AUDIT_COMPLETE"
    assert len(result["verified_authorities"]) == 11
    assert result["scientific_scope"] == (
        "DEVELOPMENT_ONLY_LEGACY_OUTCOME_BEARING_AUDIT"
    )


def test_private_legacy_audit_requires_acknowledgement_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path, digest, _ = _activation_fixture(tmp_path)
    reads = []

    def forbidden_reader(*_args, **_kwargs):
        reads.append(True)
        raise AssertionError("legacy authority bytes must remain unread")

    monkeypatch.setattr(protocol_module, "read_exact_bytes", forbidden_reader)
    with pytest.raises(ProtocolHoldError) as exc:
        _audit_legacy_outcome_bearing_activation_manifest(path, digest)
    assert exc.value.code == (
        "HOLD_LEGACY_DEVELOPMENT_AUDIT_ACKNOWLEDGEMENT_REQUIRED"
    )
    assert reads == []


def test_activation_hash_mismatch_holds_before_any_output(tmp_path: Path):
    path, digest, refs = _activation_fixture(tmp_path)
    output = tmp_path / "run-output"
    Path(refs["endpoint_estimand_registry"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(
            path, digest, prospective_output_dir=output
        )
    assert exc.value.code == "HOLD_HASH_MISMATCH"
    assert not output.exists()


def test_proposed_crosswalk_or_estimand_is_not_activation_authority(tmp_path: Path):
    path, digest, refs = _activation_fixture(tmp_path)
    ref_path = Path(refs["context_source_crosswalk"]["path"])
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    payload["status"] = "PROPOSED"
    new_sha = _write_json(ref_path, payload)
    activation = json.loads(path.read_text(encoding="utf-8"))
    activation["authorities"]["context_source_crosswalk"]["sha256"] = new_sha
    digest = _write_json(path, activation)
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_INPUT_AUTHORITY"


def test_two_study_endpoint_nested_design_cannot_claim_publication_transfer(tmp_path: Path):
    path, digest, _ = _activation_fixture(
        tmp_path, claim_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND"
    )
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_TWO_STUDY_NESTED_ENDPOINTS"


def test_single_verified_study_cannot_claim_publication_scope(tmp_path: Path):
    path, digest, _ = _activation_fixture(
        tmp_path,
        claim_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        single_study=True,
    )
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_SINGLE_STUDY_PUBLICATION_SCOPE"


def test_activation_phantom_scope_map_cannot_override_ready_rows(tmp_path: Path):
    path, _, _ = _activation_fixture(tmp_path)
    activation = json.loads(path.read_text(encoding="utf-8"))
    activation["study_endpoint_map"] = {
        "study_a": ["endpoint_a"],
        "study_b": ["endpoint_a"],
    }
    activation["study_estimand_map"] = {
        "study_a": ["estimand_a"],
        "study_b": ["estimand_a"],
    }
    activation["claim_scope"] = "PUBLICATION_ELIGIBLE_SAME_ESTIMAND"
    digest = _write_json(path, activation)
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_ACTIVATION_SCOPE_MISMATCH"


def test_publication_scope_requires_and_accepts_verified_shared_endpoint_estimand(tmp_path: Path):
    path, digest, _ = _activation_fixture(
        tmp_path,
        claim_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        shared_estimand=True,
    )
    result = _legacy_development_audit(path, digest)
    assert result["scientific_scope"] == (
        "DEVELOPMENT_ONLY_LEGACY_OUTCOME_BEARING_AUDIT"
    )
    assert result["legacy_claim_scope_diagnostic"] == (
        "PUBLICATION_ELIGIBLE_SAME_ESTIMAND"
    )
    assert result["verified_study_estimand_map"] == {
        "study_a": ["estimand_a"],
        "study_b": ["estimand_a"],
    }


def test_activation_manifest_expected_hash_is_mandatory(tmp_path: Path):
    path, _, _ = _activation_fixture(tmp_path)
    with pytest.raises(ProtocolHoldError, match="expected SHA-256"):
        _legacy_development_audit(path, None)


def test_activation_rejects_unverified_extra_authority(tmp_path: Path):
    path, _, _ = _activation_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = tmp_path / "extra.json"
    extra_sha = _write_json(extra, {"status": "FROZEN"})
    payload["authorities"]["unregistered_extra"] = {
        "path": str(extra), "sha256": extra_sha, "status": "FROZEN"
    }
    digest = _write_json(path, payload)
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_INPUT_AUTHORITY"
    assert exc.value.details["unexpected"] == ["unregistered_extra"]


def test_status_only_wrong_schema_authority_cannot_activate(tmp_path: Path):
    path, _, refs = _activation_fixture(tmp_path)
    crosswalk = Path(refs["context_source_crosswalk"]["path"])
    wrong_sha = _write_json(crosswalk, {"schema_version": "fixture_crosswalk_v1", "status": "FROZEN"})
    activation = json.loads(path.read_text(encoding="utf-8"))
    activation["authorities"]["context_source_crosswalk"]["sha256"] = wrong_sha
    digest = _write_json(path, activation)
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_INPUT_AUTHORITY"
    assert "unsupported authority schema" in str(exc.value)


def test_duplicate_json_keys_are_rejected_from_exact_bytes(tmp_path: Path):
    path = tmp_path / "activation.json"
    path.write_text(
        '{"schema_version":"r3_activation_manifest_v1",'
        '"schema_version":"r3_activation_manifest_v1","status":"FROZEN"}',
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert "duplicate JSON key" in exc.value.details["error"]


def test_code_tree_manifest_is_recomputed_against_runtime_bytes(tmp_path: Path):
    path, digest, refs = _activation_fixture(tmp_path)
    code_manifest = json.loads(
        Path(refs["analysis_code_tree_manifest"]["path"]).read_text(encoding="utf-8")
    )
    code_path = Path(refs["analysis_code_tree_manifest"]["path"]).parent / code_manifest["files"][0]["path"]
    code_path.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ProtocolHoldError) as exc:
        _legacy_development_audit(path, digest)
    assert exc.value.code == "HOLD_HASH_MISMATCH"
