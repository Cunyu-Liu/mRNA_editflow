"""Negative tests for decision, manifest, supersession and sealed AST drift."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy

import pytest
import yaml


def _codes(issues):
    return {issue.code for issue in issues}


def _dec019_successor_initial_i(config):
    """Recover the exact synthetic I state from either an on-disk I or B config."""

    initial = deepcopy(config)
    binding = initial["implementation_binding"]
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        binding[key] = "UNKNOWN_NOT_ASSERTED"
    return initial


def _copy_runner_and_guard(validator, repo_root, tmp_path):
    runner_target = tmp_path / validator.SEALED_RUNNER_PATH
    guard_target = tmp_path / validator.SEALED_GUARD_PATH
    runner_target.parent.mkdir(parents=True)
    guard_target.parent.mkdir(parents=True)
    shutil.copy2(repo_root / validator.SEALED_RUNNER_PATH, runner_target)
    shutil.copy2(repo_root / validator.SEALED_GUARD_PATH, guard_target)
    return runner_target, guard_target


def _copy_manifest_bundle(validator, repo_root, tmp_path):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    paths = set(validator.required_bundle_paths())
    paths.update(row["path"] for row in manifest["files"])
    for relative in sorted(paths):
        source = repo_root / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return manifest


def _validate_rehashed_interim_bypass(
    validator,
    repo_root,
    case_root,
    monkeypatch,
    mutate,
):
    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    interim_path = case_root / validator.A1_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    mutate(interim)
    interim_path.write_text(yaml.safe_dump(interim, sort_keys=False), encoding="utf-8")
    interim_hash = validator.sha256_file(interim_path)
    monkeypatch.setattr(validator, "EXPECTED_A1_INTERIM_SHA256", interim_hash)

    entry = next(
        row for row in manifest["files"] if row["path"] == validator.A1_INTERIM_PATH
    )
    entry["sha256"] = interim_hash
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(case_root))
    assert "A1_INTERIM_CANONICAL_HASH" not in codes
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    return codes


def _validate_rehashed_gse114002_public_gap_audit_bypass(
    validator,
    repo_root,
    case_root,
    monkeypatch,
    mutate,
):
    """Rehash audit, interim, and manifest while leaving validator semantics fixed."""

    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    audit_path = case_root / validator.GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    mutate(audit)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    audit_hash = validator.sha256_file(audit_path)

    interim_path = case_root / validator.A1_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    lineage = interim["artifact_lineage"][
        validator.GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_LINEAGE_ID
    ]
    lineage["bytes"] = audit_path.stat().st_size
    lineage["sha256"] = audit_hash
    interim_path.write_text(yaml.safe_dump(interim, sort_keys=False), encoding="utf-8")
    interim_hash = validator.sha256_file(interim_path)
    monkeypatch.setattr(validator, "EXPECTED_A1_INTERIM_SHA256", interim_hash)

    next(
        row
        for row in manifest["files"]
        if row["path"] == validator.GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH
    )["sha256"] = audit_hash
    next(
        row
        for row in manifest["files"]
        if row["path"] == validator.A1_INTERIM_PATH
    )["sha256"] = interim_hash
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(case_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A1_INTERIM_CANONICAL_HASH" not in codes
    return codes


def _validate_manifest_mutation(
    validator,
    repo_root,
    case_root,
    mutate,
):
    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    mutate(manifest)
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return _codes(validator.validate_registry_manifest(case_root))


def _validate_rehashed_dec019_leaf_bypass(
    validator,
    repo_root,
    case_root,
    monkeypatch,
    relative,
    mutate,
):
    """Rehash a DEC-019 leaf, interim reference, and manifest as one bypass."""

    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    leaf_path = case_root / relative
    if leaf_path.suffix == ".json":
        leaf = json.loads(leaf_path.read_text(encoding="utf-8"))
        mutate(leaf)
        leaf_path.write_text(json.dumps(leaf, indent=2) + "\n", encoding="utf-8")
    else:
        leaf = yaml.safe_load(leaf_path.read_text(encoding="utf-8"))
        mutate(leaf)
        leaf_path.write_text(yaml.safe_dump(leaf, sort_keys=False), encoding="utf-8")
    leaf_sha256 = validator.sha256_file(leaf_path)
    next(row for row in manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = leaf_sha256

    interim_path = case_root / validator.A1_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    interim_hash_keys = {
        validator.DEC019_AMENDMENT_PATH: "dec019_amendment_sha256",
        validator.DEC020_AMENDMENT_PATH: "dec020_amendment_sha256",
        validator.DECISION_LOG_PATH: "decision_log_sha256",
        validator.REGISTRY_PATHS["data"]: "data_role_registry_sha256",
        validator.REGISTRY_PATHS["claim"]: "claim_evidence_matrix_sha256",
    }
    hash_key = interim_hash_keys.get(relative)
    if hash_key is not None:
        interim["authority"][hash_key] = leaf_sha256
    if relative in interim["authority"]["active_authority_leaf_sha256"]:
        interim["authority"]["active_authority_leaf_sha256"][relative] = leaf_sha256
    interim_path.write_text(yaml.safe_dump(interim, sort_keys=False), encoding="utf-8")
    interim_sha256 = validator.sha256_file(interim_path)
    monkeypatch.setattr(validator, "EXPECTED_A1_INTERIM_SHA256", interim_sha256)
    next(
        row for row in manifest["files"] if row["path"] == validator.A1_INTERIM_PATH
    )["sha256"] = interim_sha256

    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    codes = _codes(validator.validate_bundle(case_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A1_INTERIM_CANONICAL_HASH" not in codes
    return codes


def test_dec020_authority_commit_registration_is_closed(validator, repo_root):
    config, _, registries = validator.load_bundle_documents(repo_root)
    assert validator.validate_dec020_authority(repo_root, config, registries) == []

    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    exact14 = set(validator.DEC020_AUTHORITY_COMMIT_EXACT_CHANGED_PATHS)
    assert len(exact14) == 14
    assert exact14 - {validator.REGISTRY_MANIFEST_PATH} <= manifest_paths
    assert validator.REGISTRY_MANIFEST_PATH not in manifest_paths
    assert validator.DEC020_AMENDMENT_PATH in manifest_paths
    assert set(validator.GSE200304_DEC020_V4_STATIC_LEAF_SHA256).issubset(manifest_paths)

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    current = interim["dec020_current_disposition"]
    assert current["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert current["qualified"] is True
    assert current["canonical_materialization_execution_authorized"] is False
    assert current["training_allowed"] is False
    assert current["model_selection_allowed"] is False
    assert current["next_phase_authorized"] is False
    assert current["latest_settled_runtime_event_id"] == "A1-EVT-051"
    sync = current["authority_runtime_sync"]
    assert sync == {
        "predecessor_event_id": "A1-EVT-050",
        "next_event_id": "A1-EVT-051",
        "next_event_id_preallocated": False,
        "status": "SYNCED_EVT_051",
    }
    assert current["future_v4_successor_registration"]["registered_in_static_manifest"] is True
    assert current["future_v4_successor_registration"]["may_execute"] is False
    assert current["future_v4_successor_registration"]["lifecycle_status"] == "ADJUDICATED_POST_IMPLEMENTATION_COMMIT_I_BOUND_PRODUCTION"
    assert current["future_v4_successor_registration"]["may_adjudicate"] is False


def test_dec020_authority_synchronized_rehash_cannot_fake_route_pass(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(amendment):
        scratch = amendment["route_conditional_gate"]["scratch_route"]
        scratch["current_status"] = "PASS"
        scratch["checkpoint_specific_exposure_pass_claimed"] = True
        amendment["authorization_projection"]["training_allowed"] = True

    codes = _validate_rehashed_dec019_leaf_bypass(
        validator,
        repo_root,
        tmp_path,
        monkeypatch,
        validator.DEC020_AMENDMENT_PATH,
        mutate,
    )
    assert "DEC020_ACTIVE_AUTHORITY_LEAF_DRIFT" in codes
    assert "DEC020_SCRATCH_ROUTE" in codes
    assert "DEC020_AMENDMENT_SEMANTICS" in codes


def test_dec020_interim_cannot_preallocate_event_or_unlock_v4(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(interim):
        current = interim["dec020_current_disposition"]
        current["authority_runtime_sync"]["next_event_id"] = "A1-EVT-050"
        current["authority_runtime_sync"]["next_event_id_preallocated"] = True
        current["current_qualified_counts"]["ordinary"] = 1
        current["future_v4_successor_registration"]["may_execute"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path,
        monkeypatch,
        mutate,
    )
    assert "A1_INTERIM_DEC020" in codes


def test_dec020_manifest_rejects_unregistered_v4_path(
    validator,
    repo_root,
    tmp_path,
):
    def mutate(manifest):
        manifest["files"].append(
            {
                "path": "configs/route_a_v3_gse200304_dec020_scratch_v4.json",
                "role": "GSE200304_DEC020_SCRATCH_V4_CONFIG",
                "sha256": "0" * 64,
            }
        )

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        tmp_path,
        mutate,
    )
    assert "REGISTRY_MANIFEST_CLOSURE" in codes


def test_dec021_authority_is_public_aggregate_geometry_only_and_preserves_history(
    validator,
    repo_root,
):
    config, _, registries = validator.load_bundle_documents(repo_root)
    assert validator.validate_dec021_authority(repo_root, config, registries) == []

    amendment = validator._load_yaml(repo_root, validator.DEC021_AMENDMENT_PATH)
    assert amendment["scope"] == {
        "dataset_id": "GSE256185",
        "role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
        "authority_surface": "ORDINARY_PUBLIC_ONLY",
        "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
        "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
        "row_output_allowed": False,
        "sequence_output_allowed": False,
        "effect_output_allowed": False,
        "private_or_restricted_input_allowed": False,
        "sealed_contact_allowed": False,
    }
    for field in (
        "sequence_evaluation",
        "edit_budget_evaluation",
        "effect_evaluation",
        "true_a2_status_evaluation",
        "qualification_evaluation",
    ):
        assert amendment["preflight_semantics"][field] == "OUT_OF_SCOPE_NOT_EVALUATED"
    assert amendment["authorization_projection"] == {
        "changes_current_qualified_counts": False,
        "current_qualified_independent_ordinary_studies": 1,
        "current_qualified_a1_studies": 1,
        "current_qualified_true_a2_dense_studies": 0,
        "current_canonical_record_count": 6547,
        "gse256185_ordinary_study_contribution": 0,
        "gse256185_a1_study_contribution": 0,
        "gse256185_true_a2_dense_study_contribution": 0,
        "gse256185_canonical_record_count": 0,
        "phase_complete": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "qualifier_execution_allowed": False,
        "canonical_materialization_allowed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    data = registries["data"]
    assert "GSE256185" not in data["ordinary_candidate_dataset_ids"]
    assert "GSE256185" not in data["true_a2_recovery_candidate_dataset_ids"]
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    current = interim["dec021_current_disposition"]
    assert amendment["historical_preservation"]["latest_settled_runtime_event_id"] == (
        "A1-EVT-051"
    )
    assert amendment["historical_preservation"]["evt051_settled_state_changed"] is False
    assert current["latest_settled_runtime_event_id"] == "A1-EVT-053"
    assert current["settled_runtime_event_changed"] is True
    assert current["runtime_event_emitted"] is True
    assert current["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert current["gse256185_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }


def test_dec021_cannot_promote_preflight_to_qualification_or_training(
    validator,
    repo_root,
    tmp_path,
):
    _copy_manifest_bundle(validator, repo_root, tmp_path)

    qualification_path = tmp_path / validator.A1_QUALIFICATION_CONFIG_PATH
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["scope"]["included_dataset_ids"].append("GSE256185")
    qualification["dec021_public_identifier_and_pool_geometry_preflight_authority"]["qualification_allowed"] = True
    qualification["dec021_public_identifier_and_pool_geometry_preflight_authority"]["training_allowed"] = True
    qualification_path.write_text(json.dumps(qualification, indent=2) + "\n", encoding="utf-8")

    config_path = tmp_path / validator.CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["a1_qualification_authority"]["gse256185_public_identifier_and_pool_geometry_preflight"]["sequence_evaluation"] = "PASS"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    data_path = tmp_path / validator.REGISTRY_PATHS["data"]
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["ordinary_candidate_dataset_ids"].append("GSE256185")
    row = next(item for item in data["datasets"] if item["dataset_id"] == "GSE256185")
    row["qualified"] = True
    row["ordinary_gate_contribution"] = 1
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    mutated_config, _, mutated_registries = validator.load_bundle_documents(tmp_path)
    codes = _codes(
        validator.validate_dec021_authority(
            tmp_path,
            mutated_config,
            mutated_registries,
        )
    )
    assert "DEC021_A1_SCOPE" in codes
    assert "DEC021_A1_POLICY" in codes
    assert "DEC021_ROOT_POLICY" in codes
    assert "DEC021_DATA_ROLE" in codes


def test_gse256185_public_geometry_registration_is_closed(
    validator,
    repo_root,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(validator.GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256)
    assert len(static_paths) == 3
    assert static_paths.issubset(manifest_paths)
    assert validator.GSE256185_PUBLIC_GEOMETRY_RUNTIME_CONFIG_PATH not in manifest_paths
    assert validator.validate_gse256185_public_geometry_registration(repo_root) == []

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    lineage = interim["artifact_lineage"][validator.GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID]
    assert lineage["path"] == validator.GSE256185_PUBLIC_GEOMETRY_REPORT_PATH
    assert lineage["bytes"] == validator.GSE256185_PUBLIC_GEOMETRY_REPORT_BYTES
    assert lineage["sha256"] == validator.GSE256185_PUBLIC_GEOMETRY_REPORT_SHA256
    assert lineage["status"] == (
        "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_COMPLETE_NOT_QUALIFIED"
    )
    assert lineage["aggregate_pool_geometry"] == {
        "total_body_row_count": 11404,
        "group_count": 652,
        "single_parent_group_count": 637,
        "dual_parent_group_count": 15,
        "single_parent_groups_with_at_least_3_candidate_rows": 634,
        "strict_candidate_rows_in_at_least_3_candidate_groups": 7292,
        "reasoned_family_closure_candidate_rows_in_at_least_3_candidate_groups": 7294,
        "identifier_grammar_anomaly_counts": {
            "MISSING_GROUP_ROLE_DELIMITER": 1,
            "UNSIGNED_CCC_ROLE": 1,
        },
        "strict_axis_is_frozen_observed_identifier_grammar": True,
        "reasoned_family_closure_axis_status": (
            "REASONED_FAMILY_CLOSURE_NOT_PUBLISHER_EXPLICIT"
        ),
        "reasoned_family_closure_axis_is_publisher_explicit": False,
    }
    assert lineage["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert lineage["gse256185_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert lineage["qualified"] is False
    assert lineage["training_allowed"] is False
    assert lineage["gpu_work_allowed"] is False
    assert lineage["model_selection_allowed"] is False
    assert lineage["next_phase_authorized"] is False
    assert lineage["predecessor_runtime_event_id"] == "A1-EVT-052"
    assert lineage["expected_next_runtime_event_id"] == (
        validator.GSE256185_PUBLIC_GEOMETRY_RUNTIME_EVENT_ID
    )
    assert lineage["next_runtime_event_id_preallocated"] is False
    assert lineage["runtime_sync_status"] == "SYNCED_EVT_053"
    assert lineage["scope_attestation"]["raw_asset_registered"] is False
    assert lineage["scope_attestation"]["row_record_output_count"] == 0
    assert lineage["scope_attestation"]["member_identifier_output_count"] == 0
    assert lineage["scope_attestation"]["sequence_value_output_count"] == 0
    assert lineage["scope_attestation"]["effect_value_output_count"] == 0


def test_gse256185_rehashed_interim_cannot_promote_geometry_to_qualification(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(interim):
        lineage = interim["artifact_lineage"][
            validator.GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID
        ]
        lineage["qualified"] = True
        lineage["gse256185_contribution"]["ordinary"] = 1
        lineage["training_allowed"] = True
        boundary = interim["dataset_boundary_summary"]["GSE256185"]
        boundary["qualified"] = True
        boundary["ordinary_study_contribution"] = 1
        boundary["training_allowed"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path,
        monkeypatch,
        mutate,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE256185" in codes


def test_gse256185_synchronized_static_leaf_rehash_cannot_bypass_binding(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    config_path = tmp_path / validator.GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["frozen_outer_truth"]["training_allowed"] = True
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config_sha256 = validator.sha256_file(config_path)
    next(
        row
        for row in manifest["files"]
        if row["path"] == validator.GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH
    )["sha256"] = config_sha256

    interim_path = tmp_path / validator.A1_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    interim["artifact_lineage"][validator.GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID][
        "producer_lineage"
    ]["config_sha256"] = config_sha256
    interim_path.write_text(
        yaml.safe_dump(interim, sort_keys=False),
        encoding="utf-8",
    )
    interim_sha256 = validator.sha256_file(interim_path)
    monkeypatch.setattr(validator, "EXPECTED_A1_INTERIM_SHA256", interim_sha256)
    next(
        row
        for row in manifest["files"]
        if row["path"] == validator.A1_INTERIM_PATH
    )["sha256"] = interim_sha256
    (tmp_path / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A1_INTERIM_CANONICAL_HASH" not in codes
    assert "GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF" in codes
    assert "A1_INTERIM_LINEAGE" in codes


def test_decision_log_requires_all_ids_and_historical_m0_decision(validator, repo_root):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    bypass["decisions"] = [row for row in bypass["decisions"] if row["decision_id"] != "V3-DEC-014"]
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_ID_CLOSURE" in codes
    assert "DECISION_LOG_KEY_DECISION" in codes


def test_decision_015_historical_security_design_is_preserved(validator, repo_root):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-015")
    decision["status"] = "MUTABLE"
    decision["effective_phase"] = "A10"
    decision["requires_user_authorization"] = True
    decision["sealed_contact"] = True
    decision["evidence_refs"].remove(validator.SEALED_GUARD_PATH)
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_SECURITY_DESIGN" in codes
    assert "DECISION_LOG_SECURITY_EVIDENCE" in codes


def test_decision_016_superseding_a0_boundary_is_frozen(validator, repo_root):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-016")
    decision["status"] = "MUTABLE"
    decision["supersedes_decision_id"] = "V3-DEC-014"
    decision["effective_phase"] = "A9"
    decision["requires_user_authorization"] = True
    decision["sealed_contact"] = True
    decision["resolution"] = decision["resolution"].replace(
        "unconditional A0-A9 hard disable",
        "conditional toggle",
    )
    decision["evidence_refs"].remove(validator.SEALED_RUNNER_PATH)
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_A0_PHASE_BOUNDARY" in codes
    assert "DECISION_LOG_A0_PHASE_BOUNDARY_EVIDENCE" in codes
    assert "DECISION_LOG_KEY_DECISION" in codes


def test_decision_017_user_authorized_data_role_amendment_is_frozen(
    validator,
    repo_root,
):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-017")
    decision["status"] = "MUTABLE"
    decision["user_authorization_status"] = "NOT_GRANTED"
    decision["preserves_decision_ids"] = []
    decision["sealed_contact"] = True
    decision["resolution"] = decision["resolution"].replace(
        "zero contribution",
        "counts as a qualified true A2",
    )
    decision["evidence_refs"].remove(validator.REGISTRY_PATHS["data"])
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_A1_ROLE_AMENDMENT" in codes
    assert "DECISION_LOG_A1_ROLE_AMENDMENT_EVIDENCE" in codes
    assert "DECISION_LOG_KEY_DECISION" in codes


def test_decision_018_official_role_authority_and_raw_replay_boundary_is_frozen(
    validator,
    repo_root,
):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-018")
    decision["dimension"] = "gse200302_qualification"
    decision["status"] = "QUALIFIED"
    decision["role_authority_status"] = "RAW_REPLAY_CLOSED"
    decision["prior_blocker_status"] = "OPEN"
    decision["replacement_blocker"] = "NONE"
    decision["replacement_blocker_status"] = "CLOSED"
    decision["role_grid_status"] = "MATCH"
    decision["pdna_may_substitute_for_80s_rna"] = True
    decision["runtime_sync_status"] = "EVT-035"
    decision["evidence_refs"].remove(validator.GSE200302_ROLE_CONFIG_PATH)
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_ENTRY_DRIFT" in codes
    assert "DECISION_LOG_DIMENSION" in codes
    assert "DECISION_LOG_GSE200302_ROLE_AUTHORITY" in codes
    assert "DECISION_LOG_GSE200302_ROLE_AUTHORITY_EVIDENCE" in codes


def test_decision_019_user_authorized_measurement_and_split_authority_is_frozen(
    validator,
    repo_root,
):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-019")
    decision["status"] = "MUTABLE"
    decision["user_authorization_status"] = "NOT_GRANTED"
    decision["current_qualified_counts"]["ordinary"] = 1
    decision["training_allowed"] = True
    decision["sealed_contact"] = True
    decision["resolution"] += " K2 is dropped and K5 now receives qualification credit."
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_ENTRY_DRIFT" in codes
    assert "DECISION_LOG_DEC019" in codes


def test_decision_prefix_digest_rejects_conflicting_semantics_and_old_entry_drift(
    validator,
    repo_root,
):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    contradictory = deepcopy(decision_log)
    decision = next(row for row in contradictory["decisions"] if row["decision_id"] == "V3-DEC-017")
    decision["resolution"] += (
        " Notwithstanding the foregoing, GSE145046 is now QUALIFIED_TRUE_A2 "
        "and contributes one true-A2 gate credit."
    )
    codes = _codes(validator.validate_decision_log(contradictory))
    assert "DECISION_LOG_ENTRY_DRIFT" in codes

    historical = deepcopy(decision_log)
    historical["decisions"][0]["status"] = "REWRITTEN_AFTER_ACCEPTANCE"
    codes = _codes(validator.validate_decision_log(historical))
    assert "DECISION_LOG_ENTRY_DRIFT" in codes


def test_rehashed_manifest_cannot_bypass_decision_prefix_digest(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    decision_path = tmp_path / validator.DECISION_LOG_PATH
    decision_log = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    decision = next(row for row in decision_log["decisions"] if row["decision_id"] == "V3-DEC-017")
    decision["resolution"] += (
        " Notwithstanding the foregoing, GSE145046 is now QUALIFIED_TRUE_A2 "
        "and contributes one true-A2 gate credit."
    )
    decision_path.write_text(yaml.safe_dump(decision_log, sort_keys=False), encoding="utf-8")
    entry = next(row for row in manifest["files"] if row["path"] == validator.DECISION_LOG_PATH)
    entry["sha256"] = validator.sha256_file(decision_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "DECISION_LOG_ENTRY_DRIFT" in codes


def test_rehashed_manifest_cannot_bypass_decision_018_role_boundary(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    decision_path = tmp_path / validator.DECISION_LOG_PATH
    decision_log = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    decision = next(row for row in decision_log["decisions"] if row["decision_id"] == "V3-DEC-018")
    decision["replacement_blocker"] = "NONE"
    decision["replacement_blocker_status"] = "CLOSED"
    decision["pdna_may_substitute_for_80s_rna"] = True
    decision["resolution"] += " pDNA is treated as 80S_RNA and raw replay is now qualified."
    decision_path.write_text(yaml.safe_dump(decision_log, sort_keys=False), encoding="utf-8")
    entry = next(row for row in manifest["files"] if row["path"] == validator.DECISION_LOG_PATH)
    entry["sha256"] = validator.sha256_file(decision_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "DECISION_LOG_ENTRY_DRIFT" in codes
    assert "DECISION_LOG_GSE200302_ROLE_AUTHORITY" in codes


def test_a1_interim_semantics_and_rehashed_manifest_cannot_grant_a1(
    validator,
    repo_root,
    tmp_path,
):
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    assert validator.validate_a1_interim_lineage(repo_root, interim) == []

    bypass = deepcopy(interim)
    bypass["scope"]["training_allowed"] = True
    bypass["gate_snapshot"]["qualified_a2_dense_studies"] = 1
    bypass["gate_snapshot"]["phase_complete"] = True
    bypass["gate_snapshot"]["next_phase_authorized"] = True
    bypass["dataset_boundary_summary"]["GSE145046"]["true_a2_gate_contribution"] = 1
    bypass["claim_boundaries"]["a1_phase_complete"] = True
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, bypass))
    assert "A1_INTERIM_SCOPE" in codes
    assert "A1_INTERIM_GATE" in codes
    assert "A1_INTERIM_GSE145046" in codes
    assert "A1_INTERIM_CLAIMS" in codes

    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    interim_path = tmp_path / validator.A1_INTERIM_PATH
    interim_path.write_text(yaml.safe_dump(bypass, sort_keys=False), encoding="utf-8")
    entry = next(row for row in manifest["files"] if row["path"] == validator.A1_INTERIM_PATH)
    entry["sha256"] = validator.sha256_file(interim_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A1_INTERIM_CANONICAL_HASH" in codes
    assert "A1_INTERIM_GATE" in codes


def test_a1_interim_gse200304_blocked_counts_and_lineage_are_fail_closed(
    validator,
    repo_root,
):
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    assert validator.validate_a1_interim_lineage(repo_root, interim) == []

    summary_mutations = {
        "qualified": True,
        "training_allowed": True,
        "canonical_intervention_record_count": 1,
        "ordinary_gate_contribution": 1,
        "acquisition_changes_qualification_gate": True,
        "raw_replay_preflight_changes_qualification_gate": True,
    }
    for key, value in summary_mutations.items():
        bypass = deepcopy(interim)
        bypass["dataset_boundary_summary"]["GSE200304"][key] = value
        codes = _codes(validator.validate_a1_interim_lineage(repo_root, bypass))
        assert "A1_INTERIM_GSE200304" in codes, key

    lineage_count_bypass = deepcopy(interim)
    final_bundle = lineage_count_bypass["artifact_lineage"][
        "gse200304_gap_qualification_v1"
    ]
    final_bundle["a1_study_contribution"] = 1
    final_bundle["canonical_record_count"] = 1
    codes = _codes(
        validator.validate_a1_interim_lineage(repo_root, lineage_count_bypass)
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    lineage_hash_bypass = deepcopy(interim)
    lineage_hash_bypass["artifact_lineage"]["gse200304_gap_qualification_v1"][
        "terminal_marker_sha256"
    ] = "0" * 64
    codes = _codes(
        validator.validate_a1_interim_lineage(repo_root, lineage_hash_bypass)
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    ena_body_bypass = deepcopy(interim)
    ena_body_bypass["artifact_lineage"]["gse200304_ena_fastq_manifest_bundle"][
        "fastq_body_download_count"
    ] = 48
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, ena_body_bypass))
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    raw_preflight_bypass = deepcopy(interim)
    raw_preflight = raw_preflight_bypass["artifact_lineage"][
        "gse200304_raw_replay_preflight_v1"
    ]
    raw_preflight["qualified"] = True
    raw_preflight["hard_unknown_blockers"].pop()
    raw_preflight["reference_aggregate_truth"]["u_to_t_normalization_applied"] = True
    codes = _codes(
        validator.validate_a1_interim_lineage(repo_root, raw_preflight_bypass)
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    omitted_failure = deepcopy(interim)
    omitted_failure["artifact_lineage"].pop(
        "gse200304_raw_replay_preflight_attempt_001_failure"
    )
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, omitted_failure))
    assert "A1_INTERIM_LINEAGE_ID_SET" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE_ID_SET" in codes


def test_gse149487_stop_before_data_preflight_ledger_is_exact_and_rehash_safe(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    assert validator.validate_a1_interim_lineage(repo_root, interim) == []

    lineage = interim["artifact_lineage"]
    preflight = lineage["gse149487_full_a1_stop_before_data_preflight_v1"]
    assert preflight["artifact_id"] == validator.GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_ID
    assert preflight["path"] == validator.GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_PATH
    assert preflight["bytes"] == 7218
    assert preflight["sha256"] == validator.GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_SHA256
    assert preflight["blockers"] == validator.GSE149487_PLUMAGE_PREFLIGHT_BLOCKERS
    assert preflight["historical_r4_closure"]["reference_only_not_reopened"] is True
    assert preflight["gate_truth"]["qualified"] is False
    assert preflight["counters"]["payload_open_count"] == 0
    assert "gse149487_plumage_reconstruction_v4" in lineage
    assert "gse149487_reconstruction_attempt_003_failure" in lineage

    summary = interim["dataset_boundary_summary"]["GSE149487"]
    assert summary["stop_before_data_preflight"] == {
        "artifact_lineage_id": "gse149487_full_a1_stop_before_data_preflight_v1",
        "authority_status": "PASS_CONFIG_ONLY_BINDING_VERIFIED",
        "inventory_status": "PASS_METADATA_ONLY_STOP_BEFORE_DATA",
        "outcome": "NOT_READY_FOR_STUDY_QUALIFICATION",
        "blocker_count": 11,
        "manifest_open_count": 0,
        "payload_hash_count": 0,
        "payload_open_count": 0,
        "scientific_processing_count": 0,
        "qualifier_execution_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
        "canonical_record_count": 0,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "qualified": False,
        "ready_for_study_qualification": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "changes_qualification_gate": False,
        "historical_r4_reopened": False,
    }

    def upgrade_metadata_pass(interim):
        record = interim["artifact_lineage"][
            "gse149487_full_a1_stop_before_data_preflight_v1"
        ]
        record["inventory_audit"]["status"] = "PASS_PAYLOAD_INTEGRITY"
        record["counters"]["payload_open_count"] = 1
        record["gate_truth"]["qualified"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "metadata_pass_upgrade",
        monkeypatch,
        upgrade_metadata_pass,
    )
    assert "A1_INTERIM_GSE149487_PREFLIGHT" in codes

    def rewrite_blockers_and_reopen_r4(interim):
        record = interim["artifact_lineage"][
            "gse149487_full_a1_stop_before_data_preflight_v1"
        ]
        record["blockers"].pop()
        record["historical_r4_closure"]["exact_blockers"].pop()
        record["historical_r4_closure"]["reference_only_not_reopened"] = False
        record["historical_r4_closure"]["rerun_is_qualification_path"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "blocker_and_r4_rewrite",
        monkeypatch,
        rewrite_blockers_and_reopen_r4,
    )
    assert "A1_INTERIM_GSE149487_PREFLIGHT" in codes

    def upgrade_summary_and_claim(interim):
        summary = interim["dataset_boundary_summary"]["GSE149487"][
            "stop_before_data_preflight"
        ]
        summary["changes_qualification_gate"] = True
        summary["ordinary_study_contribution"] = 1
        summary["qualified"] = True
        interim["claim_boundaries"][
            "gse149487_stop_before_data_preflight_is_study_qualification"
        ] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "summary_and_claim_upgrade",
        monkeypatch,
        upgrade_summary_and_claim,
    )
    assert "A1_INTERIM_GSE149487" in codes
    assert "A1_INTERIM_CLAIMS" in codes


def test_rehashed_gse200304_interim_cannot_bypass_semantic_closure(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    summary_mutations = {
        "qualified": True,
        "training_allowed": True,
        "canonical_intervention_record_count": 1,
        "ordinary_gate_contribution": 1,
        "acquisition_changes_qualification_gate": True,
        "raw_replay_preflight_changes_qualification_gate": True,
    }
    for key, value in summary_mutations.items():
        def mutate_summary(interim, key=key, value=value):
            interim["dataset_boundary_summary"]["GSE200304"][key] = value

        codes = _validate_rehashed_interim_bypass(
            validator,
            repo_root,
            tmp_path / f"summary_{key}",
            monkeypatch,
            mutate_summary,
        )
        assert "A1_INTERIM_GSE200304" in codes, key

    for key in (
        "raw_sequence_or_label_payload_embedded",
        "record_contains_row_or_member_payload",
        "record_contains_sequence_values",
        "record_contains_raw_label_values",
    ):
        def mutate_scope(interim, key=key):
            interim["scope"][key] = True

        codes = _validate_rehashed_interim_bypass(
            validator,
            repo_root,
            tmp_path / f"scope_{key}",
            monkeypatch,
            mutate_scope,
        )
        assert "A1_INTERIM_SCOPE" in codes, key

    def mutate_metadata_gate(interim):
        interim["gate_snapshot"]["metadata_only_qualification_count"] = 1

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "metadata_gate",
        monkeypatch,
        mutate_metadata_gate,
    )
    assert "A1_INTERIM_GATE" in codes

    def inject_unknown_fastq_gate_fields(interim):
        acquisition = interim["artifact_lineage"]["gse200304_fastq_acquisition_v1"]
        acquisition["qualification_status"] = "QUALIFIED"
        acquisition["gate_credit_override"] = 1

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "fastq_unknown_gate_fields",
        monkeypatch,
        inject_unknown_fastq_gate_fields,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE_KEYS" in codes

    def mutate_unvalidated_study_summaries(interim):
        interim["dataset_boundary_summary"]["GSE149487"]["qualified"] = True
        interim["dataset_boundary_summary"]["GSE149487"]["training_allowed"] = True
        interim["dataset_boundary_summary"]["three_utr_candidates"][
            "qualified_studies"
        ] = 3
        interim["dataset_boundary_summary"]["three_utr_candidates"][
            "transfer_claim_status"
        ] = "ESTABLISHED"

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "other_study_summary_upgrade",
        monkeypatch,
        mutate_unvalidated_study_summaries,
    )
    assert "A1_INTERIM_GSE149487" in codes
    assert "A1_INTERIM_THREE_UTR" in codes

    def inject_excluded_dataset_into_scope(interim):
        interim["scope"]["included_dataset_ids"].append("GSE246381")

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "excluded_dataset_scope_injection",
        monkeypatch,
        inject_excluded_dataset_into_scope,
    )
    assert "A1_INTERIM_SCOPE" in codes

    def mutate_bool_to_int(interim):
        interim["gate_snapshot"]["next_phase_authorized"] = 0
        interim["dataset_boundary_summary"]["GSE200304"]["qualified"] = 0

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "bool_int_type_confusion",
        monkeypatch,
        mutate_bool_to_int,
    )
    assert "A1_INTERIM_GATE" in codes
    assert "A1_INTERIM_GSE200304" in codes

    closed_lineage_ids = (
        "gse200304_public_asset_bundle",
        "gse200304_ena_fastq_manifest_bundle",
        "gse200304_fastq_acquisition_v1",
        "gse200304_fastq_independent_consumer_verification_v1",
        "gse200304_gap_qualification_attempt_001_failure",
        "gse200304_gap_qualification_attempt_002_failure",
        "gse200304_gap_qualification_attempt_003_failure",
        "gse200304_gap_qualification_v1",
        "gse200304_raw_replay_preflight_attempt_001_failure",
        "gse200304_raw_replay_preflight_v1",
        "gse200302_srr_role_authority_v1",
    )
    for lineage_id in closed_lineage_ids:
        def mutate_member(interim, lineage_id=lineage_id):
            interim["artifact_lineage"][lineage_id]["files"][0]["bytes"] += 1

        codes = _validate_rehashed_interim_bypass(
            validator,
            repo_root,
            tmp_path / f"member_{lineage_id}",
            monkeypatch,
            mutate_member,
        )
        assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes, lineage_id

    def mutate_failure_semantics(interim):
        failure = interim["artifact_lineage"][
            "gse200304_gap_qualification_attempt_001_failure"
        ]
        failure["bundled_status"] = "IN_PROGRESS"
        failure["failure_report_bytes"] += 1
        failure["sha256sums_bytes"] += 1
        failure["terminal_marker_bytes"] += 1

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "failure_semantics",
        monkeypatch,
        mutate_failure_semantics,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_fastq_acquisition_semantics(interim):
        acquisition = interim["artifact_lineage"]["gse200304_fastq_acquisition_v1"]
        acquisition["paper_native_count_reconstruction_status"] = "PASS"
        acquisition["ordinary_study_contribution"] = 1
        acquisition["qualified"] = True
        acquisition["training_allowed"] = True
        acquisition["next_phase_authorized"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "fastq_acquisition_semantics",
        monkeypatch,
        mutate_fastq_acquisition_semantics,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_fastq_consumer_semantics(interim):
        consumer = interim["artifact_lineage"][
            "gse200304_fastq_independent_consumer_verification_v1"
        ]
        consumer["first_descendant_head_attempt_status"] = "PASS"
        consumer["acceptance_scope"] = "A1_DATA_QUALIFICATION"
        consumer["a1_study_contribution"] = 1
        consumer["qualified"] = True
        consumer["model_selection_allowed"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "fastq_consumer_semantics",
        monkeypatch,
        mutate_fastq_consumer_semantics,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_ena_semantics(interim):
        ena = interim["artifact_lineage"]["gse200304_ena_fastq_manifest_bundle"]
        ena["status"] = "IN_PROGRESS"
        ena["official_metadata_and_object_lengths_status"] = "NOT_VERIFIED"
        ena["metadata_only"] = False
        ena["contains_fastq_body_payload"] = True
        ena["fastq_body_download_count"] = 48
        ena["used_by_current_qualifier"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "ena_semantics",
        monkeypatch,
        mutate_ena_semantics,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_raw_preflight_upgrade(interim):
        preflight = interim["artifact_lineage"]["gse200304_raw_replay_preflight_v1"]
        preflight["status"] = "QUALIFIED"
        preflight["ordinary_study_contribution"] = 1
        preflight["qualified"] = True
        preflight["phase_complete"] = True
        preflight["training_started"] = True
        preflight["model_selection_started"] = True
        preflight["training_allowed"] = True
        preflight["model_selection_allowed"] = True
        preflight["next_phase_authorized"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "raw_preflight_upgrade",
        monkeypatch,
        mutate_raw_preflight_upgrade,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_raw_preflight_blockers(interim):
        preflight = interim["artifact_lineage"]["gse200304_raw_replay_preflight_v1"]
        preflight["hard_unknown_blockers"].pop()
        preflight["hard_unknown_blocker_count"] = 16

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "raw_preflight_blocker_reduction",
        monkeypatch,
        mutate_raw_preflight_blockers,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_raw_preflight_control_u_truth(interim):
        aggregate = interim["artifact_lineage"]["gse200304_raw_replay_preflight_v1"][
            "reference_aggregate_truth"
        ]
        aggregate["u_to_t_normalization_applied"] = True
        aggregate["control_row_exclusion_applied"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "raw_preflight_control_u_drift",
        monkeypatch,
        mutate_raw_preflight_control_u_truth,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_raw_preflight_type_confusion(interim):
        preflight = interim["artifact_lineage"]["gse200304_raw_replay_preflight_v1"]
        preflight["hard_unknown_blocker_count"] = True
        preflight["fastq_body_read_count_by_preflight"] = False
        preflight["phase_complete"] = 0

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "raw_preflight_type_confusion",
        monkeypatch,
        mutate_raw_preflight_type_confusion,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_raw_preflight_claim(interim):
        interim["claim_boundaries"][
            "gse200304_raw_replay_preflight_is_study_qualification"
        ] = True
        interim["claim_boundaries"][
            "gse200304_raw_replay_preflight_resolves_control_u_policy"
        ] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "raw_preflight_claim_upgrade",
        monkeypatch,
        mutate_raw_preflight_claim,
    )
    assert "A1_INTERIM_CLAIMS" in codes

    def omit_raw_preflight_failure(interim):
        interim["artifact_lineage"].pop(
            "gse200304_raw_replay_preflight_attempt_001_failure"
        )

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "raw_preflight_failure_omission",
        monkeypatch,
        omit_raw_preflight_failure,
    )
    assert "A1_INTERIM_LINEAGE_ID_SET" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE_ID_SET" in codes

    def add_nonterminal_fastq_lineage(interim):
        interim["artifact_lineage"]["gse200304_fastq_acquisition_in_progress"] = {
            "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_FASTQ_ACQUISITION_IN_PROGRESS",
            "status": "IN_PROGRESS",
        }

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "extra_nonterminal_lineage",
        monkeypatch,
        add_nonterminal_fastq_lineage,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE_ID_SET" in codes

    def mutate_boundary_payload_contact(interim):
        boundary = interim["boundary_deviation"]
        boundary["restricted_or_sealed_member_content_read"] = True
        boundary["raw_label_read"] = True
        boundary["used_in_a1_reasoning"] = True
        boundary["payload"] = {"prohibited_member": "injected"}

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "boundary_payload_contact",
        monkeypatch,
        mutate_boundary_payload_contact,
    )
    assert "A1_INTERIM_BOUNDARY_DEVIATION" in codes

    def mutate_post_hoc_power(interim):
        power = interim["power_prefreeze"]
        power["status"] = "POST_HOC_AFTER_MODEL_RESULTS"
        power["model_results_may_change_this_rule"] = True
        power["selected_under_contract_underspecification"] = False

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "post_hoc_power",
        monkeypatch,
        mutate_post_hoc_power,
    )
    assert "A1_INTERIM_POWER_PREFREEZE" in codes

    def fake_full_repository_pass(interim):
        interim["verification"]["full_repository_tests"] = {
            "status": "PASS",
            "passed": 999,
            "failed": 0,
        }

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "fake_full_repository_pass",
        monkeypatch,
        fake_full_repository_pass,
    )
    assert "A1_INTERIM_VERIFICATION" in codes

    def mutate_non_gse_lineage_and_inject_payload(interim):
        reconstruction = interim["artifact_lineage"][
            "gse149487_plumage_reconstruction_v4"
        ]
        reconstruction["status"] = "IN_PROGRESS"
        reconstruction["raw_payload"] = {"sequence": "ACGT"}

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "non_gse_lineage_payload",
        monkeypatch,
        mutate_non_gse_lineage_and_inject_payload,
    )
    assert "A1_INTERIM_LINEAGE" in codes

    def mutate_outer_git_binding(interim):
        binding = interim["artifact_lineage"][
            "gse200304_raw_replay_preflight_v1"
        ]["outer_git_binding"]
        binding["binding_commit"] = "0" * 40
        binding["protocol_config_sha256"] = "0" * 64
        binding["worktree_and_index_clean"] = 1

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "outer_git_binding_drift",
        monkeypatch,
        mutate_outer_git_binding,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def mutate_targeted_passed_count(interim):
        interim["artifact_lineage"]["gse200304_raw_replay_preflight_v1"][
            "targeted_test_passed"
        ] = 58
        interim["verification"][
            "targeted_gse200304_raw_replay_preflight_tests"
        ]["passed"] = 58

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "targeted_passed_drift",
        monkeypatch,
        mutate_targeted_passed_count,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes
    assert "A1_INTERIM_VERIFICATION" in codes


def test_gse200302_role_authority_lineage_is_exact_and_fail_closed(
    validator,
    repo_root,
):
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    assert validator.validate_a1_interim_lineage(repo_root, interim) == []

    semantic_bypass = deepcopy(interim)
    role = semantic_bypass["artifact_lineage"]["gse200302_srr_role_authority_v1"]
    role["status"] = "RAW_REPLAY_CLOSED"
    role["mapping_row_count"] = 25
    role["measurement_families"] = ["High_Poly", "Low_Poly", "80S_RNA", "Total_RNA"]
    role["replacement_blocker_status"] = "CLOSED"
    role["pdna_may_substitute_for_80s_rna"] = True
    role["artifact_intrinsic_model_selection_field_present"] = True
    role["artifact_intrinsic_model_selection_status"] = "ENCODED_ALLOWED"
    role["qualified"] = True
    role["training_authorized"] = True
    role["next_phase_authorized"] = True
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, semantic_bypass))
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    member_bypass = deepcopy(interim)
    member_bypass["artifact_lineage"]["gse200302_srr_role_authority_v1"]["files"][0]["bytes"] += 1
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, member_bypass))
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes

    extra_member = deepcopy(interim)
    extra_member["artifact_lineage"]["gse200302_srr_role_authority_v1"]["files"].append(
        {
            "path": f"{validator.GSE200302_ROLE_ARTIFACT_ROOT}/IN_PROGRESS.json",
            "bytes": 2,
            "sha256": "0" * 64,
        }
    )
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, extra_member))
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes

    omitted = deepcopy(interim)
    omitted["artifact_lineage"].pop("gse200302_srr_role_authority_v1")
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, omitted))
    assert "A1_INTERIM_LINEAGE_ID_SET" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE_ID_SET" in codes

    stale_summary = deepcopy(interim)
    current = stale_summary["dataset_boundary_summary"]["GSE200304"][
        "primary_subseries_role_authority"
    ]
    current["prior_blocker_status"] = "OPEN"
    current["replacement_blocker_status"] = "CLOSED"
    current["pdna_may_substitute_for_80s_rna"] = True
    stale_summary["dataset_boundary_summary"]["GSE200304"]["next_phase_authorized"] = True
    codes = _codes(validator.validate_a1_interim_lineage(repo_root, stale_summary))
    assert "A1_INTERIM_GSE200304" in codes


def test_rehashed_gse200302_role_authority_cannot_bypass_exact_semantics(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate_role_and_rehash(interim):
        role = interim["artifact_lineage"]["gse200302_srr_role_authority_v1"]
        role["bundle_digest"] = "0" * 64
        role["files"][0]["sha256"] = "1" * 64
        role["replacement_blocker"] = "NONE"
        role["replacement_blocker_status"] = "CLOSED"
        role["pdna_may_substitute_for_80s_rna"] = True
        role["qualified"] = True
        role["training_authorized"] = True
        interim["dataset_boundary_summary"]["GSE200304"][
            "primary_subseries_role_authority"
        ]["bundle_digest"] = "0" * 64

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "synchronized_role_rehash",
        monkeypatch,
        mutate_role_and_rehash,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes
    assert "A1_INTERIM_GSE200304" in codes

    def add_nonterminal_role_lineage(interim):
        interim["artifact_lineage"]["gse200302_srr_role_authority_in_progress"] = {
            "path": f"{validator.GSE200302_ROLE_ARTIFACT_ROOT}_IN_PROGRESS",
            "dataset_id": "GSE200304",
            "status": "IN_PROGRESS",
        }

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "extra_role_lineage",
        monkeypatch,
        add_nonterminal_role_lineage,
    )
    assert "A1_INTERIM_LINEAGE_ID_SET" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE_ID_SET" in codes


def test_gse114002_endpoint_geometry_lineage_is_exact_and_rehash_resistant(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    assert validator.validate_a1_interim_lineage(repo_root, interim) == []
    attempt_001 = interim["artifact_lineage"][
        validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID
    ]
    attempt_002 = interim["artifact_lineage"][
        validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID
    ]
    assert attempt_001 == {
        **validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_EXPECTED_RECORD,
        "files": validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_EXPECTED_FILES,
    }
    assert attempt_002 == {
        **validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_EXPECTED_RECORD,
        "files": validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_EXPECTED_FILES,
    }
    assert set(attempt_001["unresolved_blockers"]) - set(
        attempt_002["unresolved_blockers"]
    ) == set(validator.GSE114002_ENDPOINT_GEOMETRY_CLOSED_BLOCKERS)

    def rewrite_historical_failure(interim):
        failure = interim["artifact_lineage"][
            validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID
        ]
        failure["status"] = "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED"
        failure["failure_preserved"] = False
        failure["unresolved_blockers"] = list(
            validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS
        )

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "gse114002_rewritten_failure",
        monkeypatch,
        rewrite_historical_failure,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE114002_ENDPOINT_GEOMETRY_HISTORY" in codes

    def upgrade_current_gate_and_counts(interim):
        current = interim["artifact_lineage"][
            validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID
        ]
        current["gate_snapshot"]["ordinary_study_contribution"] = 1
        current["gate_snapshot"]["a1_intervention_study_contribution"] = 1
        current["gate_snapshot"]["true_a2_dense_study_contribution"] = 1
        current["gate_snapshot"]["canonical_record_count"] = 1
        current["gate_snapshot"]["qualified"] = True
        current["gate_snapshot"]["training_allowed"] = True
        current["gate_snapshot"]["model_selection_allowed"] = True
        current["mechanical_diagnostics"][
            "eligible_provisional_pool_count"
        ] = True
        current["mechanical_diagnostics"][
            "diagnostic_only_not_effective_n"
        ] = 1

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "gse114002_gate_and_type_upgrade",
        monkeypatch,
        upgrade_current_gate_and_counts,
    )
    assert "A1_INTERIM_LINEAGE" in codes

    def tamper_attempt_member_triples(interim):
        for lineage_id in (
            validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID,
            validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID,
        ):
            member = interim["artifact_lineage"][lineage_id]["files"][0]
            member["bytes"] += 1
            member["sha256"] = "0" * 64

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "gse114002_member_triples",
        monkeypatch,
        tamper_attempt_member_triples,
    )
    assert "A1_INTERIM_LINEAGE" in codes

    def tamper_current_summary_and_runtime(interim):
        current = interim["dataset_boundary_summary"]["GSE114002"][
            "endpoint_geometry_reconciliation"
        ]
        current["current_artifact_lineage_id"] = (
            validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID
        )
        current["blocker_count"] = 0
        current["qualified"] = True
        current["runtime_sync_status"] = "EVT-039"

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "gse114002_summary_and_runtime",
        monkeypatch,
        tamper_current_summary_and_runtime,
    )
    assert "A1_INTERIM_GSE114002" in codes

    def hide_public_boundary_deviation(interim):
        boundary = interim["boundary_deviation"]
        boundary["count"] = 4
        boundary["classifications"].pop()
        boundary["descriptions"].pop()
        boundary["ordinary_public_excluded_dataset_policy_lines_displayed"] = 0

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "gse114002_boundary_deviation_hidden",
        monkeypatch,
        hide_public_boundary_deviation,
    )
    assert "A1_INTERIM_BOUNDARY_DEVIATION" in codes

    def add_unregistered_gse114002_lineage(interim):
        interim["artifact_lineage"][
            "gse114002_endpoint_geometry_reconciliation_v2_attempt_003"
        ] = {
            "dataset_id": "GSE114002",
            "status": "QUALIFIED",
        }

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "gse114002_extra_lineage",
        monkeypatch,
        add_unregistered_gse114002_lineage,
    )
    assert "A1_INTERIM_LINEAGE_ID_SET" in codes


def test_gse114002_endpoint_geometry_producer_rehash_cannot_bypass_binding(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    producer_paths = {
        validator.GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH,
        validator.GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH,
        validator.GSE114002_ENDPOINT_GEOMETRY_TEST_PATH,
    }
    assert producer_paths.issubset(
        {path for path, _role in validator.EXPECTED_REGISTRY_MANIFEST_PATH_ROLES}
    )

    config_path = tmp_path / validator.GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    next(
        row
        for row in manifest["files"]
        if row["path"] == validator.GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH
    )["sha256"] = validator.sha256_file(config_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A1_INTERIM_GSE114002_ENDPOINT_GEOMETRY_BINDING" in codes


def test_gse114002_public_authority_gap_audit_is_exact_aggregate_only_and_blocked(
    validator,
    repo_root,
):
    assert validator.validate_gse114002_public_authority_gap_audit(repo_root) == []
    audit = validator._load_json(
        repo_root,
        validator.GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH,
    )
    assert audit["status"] == "PUBLIC_AUTHORITY_GAPS_AUDITED_NOT_QUALIFIED"
    assert audit["scope_attestation"] == {
        "ordinary_public_sources_only": True,
        "aggregate_only": True,
        "ordinary_locator_metadata_only": True,
        "sequence_values_included": False,
        "row_identifier_values_included": False,
        "raw_label_values_included": False,
        "per_member_hashes_included": False,
        "model_weight_hashes_included": False,
        "real_row_level_payload_opened": False,
        "model_weight_payload_opened": False,
        "restricted_or_sealed_contact": False,
        "gse246381_contact": False,
        "qualifier_execution_count": 0,
        "training_run_count": 0,
        "gpu_work_count": 0,
        "model_selection_run_count": 0,
        "canonical_materialization_count": 0,
    }
    closure = audit["closure_and_remaining_evidence"]
    assert closure["science_blockers_closed_by_this_audit"] == []
    assert closure["remaining_science_blockers"] == (
        validator.GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS
    )
    assert audit["gate_snapshot"] == {
        "qualified_independent_ordinary_studies": 0,
        "qualified_a1_studies": 0,
        "qualified_true_a2_dense_studies": 0,
        "canonical_record_count": 0,
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "phase_complete": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }

    checkpoints = {
        row["checkpoint_family"]: row
        for row in audit["checkpoint_family_exposure"]
    }
    assert set(checkpoints) == {"OPTIMUS_5PRIME", "UTR_LM", "MRNABERT", "ORTHRUS"}
    for row in checkpoints.values():
        assert row["checkpoint_specific_exposure_status"] == "UNKNOWN_NOT_ASSERTED"
        assert row["near_duplicate_exposure_status"] == "NOT_RUN"
        assert row["overall_blocker_status"] == "OPEN"
    assert checkpoints["OPTIMUS_5PRIME"]["accession_exposure_status"] == (
        "EXPOSED_ACCESSION_LEVEL_NOT_CHECKPOINT_SPECIFIC"
    )
    assert checkpoints["UTR_LM"]["accession_exposure_status"] == (
        "EXPOSED_ACCESSION_LEVEL_NOT_CHECKPOINT_SPECIFIC"
    )
    for family in ("MRNABERT", "ORTHRUS"):
        assert checkpoints[family]["accession_exposure_status"] == (
            "NOT_DECLARED_DOES_NOT_ESTABLISH_ABSENCE"
        )


def test_gse114002_public_gap_synchronized_rehash_cannot_upgrade_exposure_or_gate(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(audit):
        audit["checkpoint_family_exposure"][0][
            "checkpoint_specific_exposure_status"
        ] = "UNTOUCHED"
        audit["checkpoint_family_exposure"][1][
            "near_duplicate_exposure_status"
        ] = "ZERO"
        audit["checkpoint_family_exposure"][2][
            "accession_exposure_status"
        ] = "ABSENT"
        audit["closure_and_remaining_evidence"][
            "science_blockers_closed_by_this_audit"
        ] = list(audit["closure_and_remaining_evidence"]["remaining_science_blockers"])
        audit["closure_and_remaining_evidence"]["remaining_science_blockers"] = []
        audit["gate_snapshot"]["qualified_true_a2_dense_studies"] = 1
        audit["gate_snapshot"]["canonical_record_count"] = 1
        audit["gate_snapshot"]["qualified"] = True
        audit["gate_snapshot"]["training_allowed"] = True

    codes = _validate_rehashed_gse114002_public_gap_audit_bypass(
        validator,
        repo_root,
        tmp_path / "public_gap_exposure_gate_rehash",
        monkeypatch,
        mutate,
    )
    assert "GSE114002_PUBLIC_GAP_AUDIT_CANONICAL_HASH" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_CHECKPOINTS" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_EXPOSURE_BYPASS" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_BLOCKERS" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_GATE" in codes
    assert "A1_INTERIM_LINEAGE" in codes


def test_gse114002_public_gap_synchronized_rehash_cannot_promote_inference_or_license(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(audit):
        mother = next(
            row
            for row in audit["field_and_source_claims"]
            if row["claim_id"] == "MOTHER_AND_MATCH_SCORE_SEMANTICS"
        )
        mother["evidence_status"] = "CONFIRMED"
        audit["merge_authority_claims"][
            "mother_and_match_score_may_be_used_as_join_authority"
        ] = True
        chemistry = next(
            row
            for row in audit["construct_and_chemistry_claims"]
            if row["claim_id"] == "DESIGNED_SAMPLE_RNA_CHEMISTRY"
        )
        chemistry["evidence_status"] = "CONFIRMED"
        chemistry["finding"] = "UNMODIFIED_U_CONFIRMED"
        data_license = next(
            row
            for row in audit["license_claims"]
            if row["claim_id"] == "GSE114002_DATA_REDISTRIBUTION_RIGHTS"
        )
        data_license["evidence_status"] = "CONFIRMED"
        data_license["finding"] = "GPL_3_0"

    codes = _validate_rehashed_gse114002_public_gap_audit_bypass(
        validator,
        repo_root,
        tmp_path / "public_gap_inference_license_rehash",
        monkeypatch,
        mutate,
    )
    assert "GSE114002_PUBLIC_GAP_AUDIT_FIELD_CLAIMS" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_MERGE" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_CONSTRUCT" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_LICENSE" in codes


def test_gse114002_public_gap_recursive_privacy_scan_rejects_payload_fields(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(audit):
        audit["source_registry"][0]["ordinary_locator_metadata"]["row_id"] = (
            "private-row"
        )
        audit["checkpoint_family_exposure"][0]["model_weight_sha256"] = "0" * 64
        audit["license_claims"][0]["sequence"] = "ACGT" * 8

    codes = _validate_rehashed_gse114002_public_gap_audit_bypass(
        validator,
        repo_root,
        tmp_path / "public_gap_privacy_rehash",
        monkeypatch,
        mutate,
    )
    assert "GSE114002_PUBLIC_GAP_AUDIT_PRIVACY" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_SOURCE_REGISTRY" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_CHECKPOINTS" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_LICENSE" in codes


def test_gse114002_public_gap_type_strict_and_runtime_event_boundary(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(audit):
        audit["scope_attestation"]["aggregate_only"] = 1
        audit["gate_snapshot"]["qualified"] = 0
        audit["lineage"]["runtime_sync_status"] = "EVT-040"
        audit["lineage"]["predecessor_runtime_event_id"] = "A1-EVT-040"

    codes = _validate_rehashed_gse114002_public_gap_audit_bypass(
        validator,
        repo_root,
        tmp_path / "public_gap_type_runtime_rehash",
        monkeypatch,
        mutate,
    )
    assert "GSE114002_PUBLIC_GAP_AUDIT_SCOPE" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_GATE" in codes
    assert "GSE114002_PUBLIC_GAP_AUDIT_LINEAGE" in codes


def test_gse114002_public_gap_interim_summary_cannot_upgrade_or_fake_evt040(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(interim):
        lineage = interim["artifact_lineage"][
            validator.GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_LINEAGE_ID
        ]
        lineage["science_blockers_closed_by_this_audit"] = list(
            lineage["unresolved_blockers"]
        )
        lineage["unresolved_blockers"] = []
        lineage["gate_snapshot"]["qualified"] = True
        lineage["runtime_sync_status"] = "EVT-040"
        summary = interim["dataset_boundary_summary"]["GSE114002"][
            "public_authority_gap_audit"
        ]
        summary["science_blockers_closed_count"] = 7
        summary["qualified"] = True
        summary["training_allowed"] = True
        summary["runtime_sync_status"] = "EVT-040"

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "public_gap_interim_upgrade",
        monkeypatch,
        mutate,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE114002" in codes


def test_published_endpoint_lineage_is_exact_and_rehash_resistant(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    assert validator.validate_a1_interim_lineage(repo_root, interim) == []
    node = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"]
    assert node == {
        **validator.GSE200304_PUBLISHED_ENDPOINT_EXPECTED_RECORD,
        "files": validator.GSE200304_PUBLISHED_ENDPOINT_EXPECTED_FILES,
    }

    def tamper_blocker_order(interim):
        interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"][
            "unresolved_blockers"
        ].reverse()

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_blocker_order",
        monkeypatch,
        tamper_blocker_order,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def clear_blockers_and_upgrade_gate(interim):
        node = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"]
        node["unresolved_blockers"] = []
        node["gate_snapshot"]["ordinary_study_contribution"] = 1
        node["gate_snapshot"]["a1_intervention_study_contribution"] = 1
        node["gate_snapshot"]["true_a2_dense_study_contribution"] = 1
        node["gate_snapshot"]["canonical_record_count"] = 1
        node["gate_snapshot"]["qualified"] = True
        node["gate_snapshot"]["training_allowed"] = True
        node["gate_snapshot"]["model_selection_allowed"] = True
        node["gate_snapshot"]["next_phase_authorized"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_cleared_blockers_and_gate_upgrade",
        monkeypatch,
        clear_blockers_and_upgrade_gate,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def tamper_counts_and_types(interim):
        node = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"]
        node["mechanical_aggregates"]["table_s2"]["raw_row_count"] = True
        node["mechanical_aggregates"]["table_s3"][
            "primary_pair_key_count"
        ] = 6773
        node["mechanical_aggregates"]["endpoint_boundary"][
            "standard_error"
        ] = 0

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_counts",
        monkeypatch,
        tamper_counts_and_types,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def tamper_implementation_binding(interim):
        binding = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"][
            "implementation_binding"
        ]
        binding["implementation_commit"] = "0" * 40
        binding["binding_commit"] = "1" * 40
        binding["protocol_config_sha256"] = "2" * 64
        binding["production_script_path"] = "scripts/route_a_v3/replacement.py"

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_binding",
        monkeypatch,
        tamper_implementation_binding,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def tamper_zero_false_gates(interim):
        node = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"]
        node["gate_snapshot"]["ordinary_study_contribution"] = False
        node["gate_snapshot"]["qualified"] = 0
        node["gate_snapshot"]["training_allowed"] = True
        node["access_and_materialization_boundary"]["canonical_write_count"] = False
        node["access_and_materialization_boundary"]["row_level_payload_included"] = 0

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_zero_false",
        monkeypatch,
        tamper_zero_false_gates,
    )
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes

    def tamper_member_triple(interim):
        member = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"][
            "files"
        ][0]
        member["path"] = member["path"].replace(
            "INPUT_INTEGRITY_AUDIT.json", "INPUT_INTEGRITY_AUDIT_REHASHED.json"
        )
        member["bytes"] += 1
        member["sha256"] = "3" * 64

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_member",
        monkeypatch,
        tamper_member_triple,
    )
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes

    def delete_terminal_member(interim):
        files = interim["artifact_lineage"]["gse200304_published_endpoint_a1_v1"][
            "files"
        ]
        files[:] = [
            member
            for member in files
            if not member["path"].endswith("/PUBLICATION_COMMIT.json")
        ]

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_terminal_member_deleted",
        monkeypatch,
        delete_terminal_member,
    )
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes

    def drift_terminal_member(interim):
        terminal = interim["artifact_lineage"][
            "gse200304_published_endpoint_a1_v1"
        ]["files"][-1]
        terminal["bytes"] = 974
        terminal["sha256"] = "4" * 64

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_terminal_member_drift",
        monkeypatch,
        drift_terminal_member,
    )
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes

    def tamper_summary(interim):
        current = interim["dataset_boundary_summary"]["GSE200304"][
            "published_endpoint_evidence"
        ]
        current["blocker_count"] = 7
        current["qualified"] = True
        current["runtime_sync_status"] = "EVT-037"

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_summary",
        monkeypatch,
        tamper_summary,
    )
    assert "A1_INTERIM_GSE200304" in codes

    def add_extra_lineage(interim):
        interim["artifact_lineage"]["gse200304_published_endpoint_in_progress"] = {
            "dataset_id": "GSE200304",
            "status": "IN_PROGRESS",
        }

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "published_endpoint_extra_lineage",
        monkeypatch,
        add_extra_lineage,
    )
    assert "A1_INTERIM_LINEAGE_ID_SET" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE_ID_SET" in codes


def test_published_endpoint_producer_rehash_cannot_bypass_frozen_binding(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    protocol_path = tmp_path / validator.GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH
    protocol_path.write_bytes(protocol_path.read_bytes() + b"\n")
    next(
        row
        for row in manifest["files"]
        if row["path"] == validator.GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH
    )["sha256"] = validator.sha256_file(protocol_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A1_INTERIM_GSE200304_PUBLISHED_ENDPOINT_BINDING" in codes


def test_gse200302_role_protocol_core_and_dynamic_binding_are_separate(
    validator,
    repo_root,
    tmp_path,
):
    manifest_paths = {
        path for path, _role in validator.EXPECTED_REGISTRY_MANIFEST_PATH_ROLES
    }
    assert validator.GSE200302_ROLE_CONFIG_PATH in validator.required_bundle_paths()
    assert validator.GSE200302_ROLE_CONFIG_PATH not in manifest_paths

    issues = validator.validate_gse200302_role_protocol(repo_root)
    non_binding_issues = [
        issue
        for issue in issues
        if issue.code != "GSE200302_ROLE_PROTOCOL_BINDING"
    ]
    assert non_binding_issues == []
    protocol = validator._load_json(repo_root, validator.GSE200302_ROLE_CONFIG_PATH)
    if protocol["implementation_binding"]["status"] == "UNKNOWN_NOT_ASSERTED":
        assert _codes(issues) == {"GSE200302_ROLE_PROTOCOL_BINDING"}
    else:
        assert issues == []

    bound_root = tmp_path / "synthetic_bound_protocol"
    for relative in (
        validator.GSE200302_ROLE_CONFIG_PATH,
        validator.GSE200302_ROLE_BUILDER_PATH,
        validator.GSE200302_ROLE_TEST_PATH,
    ):
        source = repo_root / relative
        target = bound_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    bound_path = bound_root / validator.GSE200302_ROLE_CONFIG_PATH
    bound = json.loads(bound_path.read_text(encoding="utf-8"))
    binding = bound["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = validator.sha256_file(
        bound_root / validator.GSE200302_ROLE_BUILDER_PATH
    )
    binding["implementation_test_sha256"] = validator.sha256_file(
        bound_root / validator.GSE200302_ROLE_TEST_PATH
    )
    bound_path.write_text(
        json.dumps(bound, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert validator.validate_gse200302_role_protocol(bound_root) == []

    binding["implementation_test_sha256"] = "0" * 64
    bound_path.write_text(
        json.dumps(bound, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert "GSE200302_ROLE_PROTOCOL_BINDING" in _codes(
        validator.validate_gse200302_role_protocol(bound_root)
    )


def test_synchronized_manifest_rehash_cannot_hide_role_protocol_core_mutation(
    validator,
    repo_root,
    tmp_path,
):
    case_root = tmp_path / "role_protocol_core_mutation"
    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    assert all(
        row["path"] != validator.GSE200302_ROLE_CONFIG_PATH
        for row in manifest["files"]
    )

    protocol_path = case_root / validator.GSE200302_ROLE_CONFIG_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["gate_contract"]["model_selection_allowed"] = True
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for row in manifest["files"]:
        row["sha256"] = validator.sha256_file(case_root / row["path"])
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    codes = _codes(validator.validate_bundle(case_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "GSE200302_ROLE_PROTOCOL_CORE" in codes
    assert "GSE200302_ROLE_PROTOCOL_GATES" in codes
    assert "GSE200302_ROLE_PROTOCOL_MODEL_SELECTION_FIELD" in codes


def test_gse149487_plumage_protocol_is_dynamic_but_semantically_closed(
    validator,
    repo_root,
):
    manifest_paths = {
        path for path, _role in validator.EXPECTED_REGISTRY_MANIFEST_PATH_ROLES
    }
    assert validator.GSE149487_PLUMAGE_PROTOCOL_PATH in validator.required_bundle_paths()
    assert validator.GSE149487_PLUMAGE_PROTOCOL_PATH not in manifest_paths
    for immutable_path in (
        validator.GSE149487_PLUMAGE_ASSET_MANIFEST_PATH,
        validator.GSE149487_PLUMAGE_HELPER_PATH,
        validator.GSE149487_PLUMAGE_QUALIFIER_PATH,
        validator.GSE149487_PLUMAGE_TEST_PATH,
        validator.GSE149487_PLUMAGE_PREFLIGHT_CONFIG_PATH,
        validator.GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_PATH,
        validator.GSE149487_PLUMAGE_PREFLIGHT_TEST_PATH,
    ):
        assert immutable_path in manifest_paths

    assert validator.validate_gse149487_plumage_protocol(repo_root) == []
    protocol = validator._load_json(
        repo_root,
        validator.GSE149487_PLUMAGE_PROTOCOL_PATH,
    )
    assert protocol["authority"]["active_authority_commit"] == (
        validator.GSE149487_PLUMAGE_ACTIVE_AUTHORITY_COMMIT
    )
    assert protocol["authority"]["active_amendment_decision_ids"] == [
        "V3-DEC-017",
        "V3-DEC-018",
    ]
    assert protocol["known_external_evidence_blockers"] == (
        validator.GSE149487_PLUMAGE_EXTERNAL_BLOCKERS
    )
    assert protocol["current_gate_contract"] == (
        validator.GSE149487_PLUMAGE_CURRENT_GATE_CONTRACT
    )
    assert validator._gse149487_plumage_nonbinding_core_sha256(protocol) == (
        validator.GSE149487_PLUMAGE_NONBINDING_CORE_SHA256
    )
    preflight_binding = protocol["stop_before_data_preflight_binding"]
    assert preflight_binding["binding_scheme"] == (
        validator.GSE149487_PLUMAGE_PREFLIGHT_BINDING_SCHEME
    )
    assert preflight_binding["status"] in {"UNKNOWN_NOT_ASSERTED", "BOUND"}
    if preflight_binding["status"] == "UNKNOWN_NOT_ASSERTED":
        assert preflight_binding["implementation_commit"] == "UNKNOWN_NOT_ASSERTED"
    else:
        assert preflight_binding["implementation_commit"] == (
            "d10a42a564ecac2af048b39c05cbc863ebdacd02"
        )
    assert preflight_binding["external_evidence_config_sha256"] == (
        validator.GSE149487_PLUMAGE_PREFLIGHT_CONFIG_SHA256
    )
    assert preflight_binding["preflight_script_sha256"] == (
        validator.GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_SHA256
    )
    assert preflight_binding["preflight_test_sha256"] == (
        validator.GSE149487_PLUMAGE_PREFLIGHT_TEST_SHA256
    )


def test_synchronized_manifest_rehash_cannot_hide_plumage_gate_or_authority_drift(
    validator,
    repo_root,
    tmp_path,
):
    case_root = tmp_path / "plumage_protocol_semantic_mutation"
    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    assert all(
        row["path"] != validator.GSE149487_PLUMAGE_PROTOCOL_PATH
        for row in manifest["files"]
    )

    protocol_path = case_root / validator.GSE149487_PLUMAGE_PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["authority"]["active_amendment_decision_ids"] = ["V3-DEC-017"]
    protocol["current_gate_contract"]["qualified"] = True
    protocol["current_gate_contract"]["ordinary_study_contribution"] = 1
    protocol["known_external_evidence_blockers"].pop()
    preflight_path = case_root / validator.GSE149487_PLUMAGE_PREFLIGHT_CONFIG_PATH
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["protocol_status"] = "SYNCHRONIZED_REHASH_DRIFT"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    protocol["stop_before_data_preflight_binding"][
        "external_evidence_config_sha256"
    ] = validator.sha256_file(preflight_path)
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for row in manifest["files"]:
        row["sha256"] = validator.sha256_file(case_root / row["path"])
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    codes = _codes(validator.validate_bundle(case_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "GSE149487_PLUMAGE_PROTOCOL_AUTHORITY" in codes
    assert "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE" in codes
    assert "GSE149487_PLUMAGE_PROTOCOL_GATES" in codes
    assert "GSE149487_PLUMAGE_PREFLIGHT_BINDING" in codes


def test_plumage_static_core_allows_only_the_three_b_binding_scalars(
    validator,
    repo_root,
    tmp_path,
):
    case_root = tmp_path / "plumage_exact_b_scalars"
    _copy_manifest_bundle(validator, repo_root, case_root)
    a1_contract_relative = "configs/route_a_v3_a1_qualification.json"
    (case_root / a1_contract_relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / a1_contract_relative, case_root / a1_contract_relative)
    protocol_path = case_root / validator.GSE149487_PLUMAGE_PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    implementation_commit = "1" * 40
    protocol["authority"]["implementation_commit"] = implementation_commit
    protocol["stop_before_data_preflight_binding"]["status"] = "BOUND"
    protocol["stop_before_data_preflight_binding"][
        "implementation_commit"
    ] = implementation_commit
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert validator.validate_gse149487_plumage_protocol(case_root) == []


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("canonical_materialization_bypass", "GSE149487_PLUMAGE_PROTOCOL_CANONICAL"),
        ("outcome_dependent_mapping", "GSE149487_PLUMAGE_PROTOCOL_MAPPING"),
        ("empty_qualification_gates", "GSE149487_PLUMAGE_PROTOCOL_QUALIFICATION_GATES"),
    ],
)
def test_synchronized_rehash_cannot_hide_plumage_core_semantic_bypass(
    validator,
    repo_root,
    tmp_path,
    mutation,
    expected_code,
):
    case_root = tmp_path / mutation
    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    protocol_path = case_root / validator.GSE149487_PLUMAGE_PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if mutation == "canonical_materialization_bypass":
        protocol["canonical_v3"][
            "materialize_only_when_every_qualification_gate_passes"
        ] = False
    elif mutation == "outcome_dependent_mapping":
        protocol["mapping"][
            "membership_may_depend_on_measured_effect_or_significance"
        ] = True
    elif mutation == "empty_qualification_gates":
        protocol["qualification_gates"] = []
    else:  # pragma: no cover - closed parametrization guard
        raise AssertionError(mutation)
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for row in manifest["files"]:
        row["sha256"] = validator.sha256_file(case_root / row["path"])
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    codes = _codes(validator.validate_bundle(case_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert expected_code in codes
    assert "GSE149487_PLUMAGE_PROTOCOL_NONBINDING_CORE" in codes


def test_dec019_successor_configs_use_stable_cores_without_manifest_cycle(
    validator,
    repo_root,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    assert validator.DEC019_SUCCESSOR_DYNAMIC_CONFIG_PATHS.isdisjoint(manifest_paths)
    for static_path in (
        validator.GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH,
        validator.GSE114002_DEC019_SUCCESSOR_TEST_PATH,
        validator.GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH,
        validator.GSE200304_DEC019_SUCCESSOR_TEST_PATH,
    ):
        assert static_path in manifest_paths

    assert validator.validate_dec019_successor_adjudicators(repo_root) == []
    expected = (
        (
            validator.GSE114002_DEC019_SUCCESSOR_CONFIG_PATH,
            validator.GSE114002_DEC019_SUCCESSOR_INITIAL_I_SHA256,
            validator.GSE114002_DEC019_SUCCESSOR_CORE_SHA256,
            validator.GSE114002_DEC019_SUCCESSOR_SCRIPT_SHA256,
            validator.GSE114002_DEC019_SUCCESSOR_TEST_SHA256,
        ),
        (
            validator.GSE200304_DEC019_SUCCESSOR_CONFIG_PATH,
            validator.GSE200304_DEC019_SUCCESSOR_INITIAL_I_SHA256,
            validator.GSE200304_DEC019_SUCCESSOR_CORE_SHA256,
            validator.GSE200304_DEC019_SUCCESSOR_SCRIPT_SHA256,
            validator.GSE200304_DEC019_SUCCESSOR_TEST_SHA256,
        ),
    )
    for relative, initial_sha256, core_sha256, script_sha256, test_sha256 in expected:
        path = repo_root / relative
        config = json.loads(path.read_text(encoding="utf-8"))
        binding = config["implementation_binding"]
        if binding["status"] == "UNKNOWN_NOT_ASSERTED":
            assert validator.sha256_file(path) == initial_sha256
        else:
            assert binding["status"] == "BOUND"
            assert validator._is_lower_hex(binding["implementation_commit"], 40)
            assert binding["implementation_script_sha256"] == script_sha256
            assert binding["implementation_test_sha256"] == test_sha256

        initial_config = _dec019_successor_initial_i(config)
        initial_payload = (json.dumps(initial_config, indent=2) + "\n").encode("utf-8")
        assert validator.sha256_bytes(initial_payload) == initial_sha256
        assert validator._dec019_successor_core_sha256(config) == core_sha256
        assert config["implementation_binding"]["config_core_sha256"] == core_sha256
        assert config["current_external_state"]["qualified"] is False
        assert config["current_external_state"]["canonical_record_count"] == 0


def test_dec019_successor_config_only_i_to_b_preserves_stable_core(
    validator,
    repo_root,
    tmp_path,
):
    _copy_manifest_bundle(validator, repo_root, tmp_path)
    config_path = tmp_path / validator.GSE114002_DEC019_SUCCESSOR_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    initial_core = validator._dec019_successor_core_sha256(config)
    binding = config["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "a" * 40
    binding["implementation_script_sha256"] = (
        validator.GSE114002_DEC019_SUCCESSOR_SCRIPT_SHA256
    )
    binding["implementation_test_sha256"] = (
        validator.GSE114002_DEC019_SUCCESSOR_TEST_SHA256
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    assert validator._dec019_successor_core_sha256(config) == initial_core
    assert validator.validate_dec019_successor_adjudicators(tmp_path) == []


@pytest.mark.parametrize(
    ("dataset_id", "mutation"),
    [
        ("GSE114002", "drop_k2"),
        ("GSE114002", "promote_k5"),
        ("GSE114002", "technical_uncertainty_supports_power"),
        ("GSE114002", "multiply_studies"),
        ("GSE114002", "power_bool"),
        ("GSE200304", "raw_replay_primary"),
        ("GSE200304", "multiply_studies"),
        ("GSE200304", "waive_rights"),
    ],
)
def test_dec019_successor_synchronized_core_rehash_cannot_change_policy(
    validator,
    repo_root,
    tmp_path,
    dataset_id,
    mutation,
):
    case_root = tmp_path / f"{dataset_id}_{mutation}"
    _copy_manifest_bundle(validator, repo_root, case_root)
    relative = (
        validator.GSE114002_DEC019_SUCCESSOR_CONFIG_PATH
        if dataset_id == "GSE114002"
        else validator.GSE200304_DEC019_SUCCESSOR_CONFIG_PATH
    )
    path = case_root / relative
    config = json.loads(path.read_text(encoding="utf-8"))
    policy = config["policy_boundary"]
    if mutation == "drop_k2":
        policy["eligible_edit_distances"] = [1, 3]
    elif mutation == "promote_k5":
        policy["k5_role"] = "QUALIFICATION_GATE"
    elif mutation == "technical_uncertainty_supports_power":
        policy["technical_uncertainty_prohibited_uses"].remove("POWER")
    elif mutation == "multiply_studies":
        policy["maximum_study_contribution_per_dataset"] = 2
        policy["gsm_pool_subseries_modality_endpoint_replicate_may_multiply_study_count"] = True
    elif mutation == "power_bool":
        policy["minimum_power"] = True
    elif mutation == "raw_replay_primary":
        policy["raw_replay_role"] = "PRIMARY_MEASUREMENT_ROUTE"
    elif mutation == "waive_rights":
        policy["rights_required"] = False
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(mutation)
    config["implementation_binding"]["config_core_sha256"] = (
        validator._dec019_successor_core_sha256(config)
    )
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_dec019_successor_adjudicators(case_root))
    assert "DEC019_SUCCESSOR_CORE_DRIFT" in codes


def test_dec019_successor_partial_binding_and_fake_gate_are_rejected(
    validator,
    repo_root,
    tmp_path,
):
    _copy_manifest_bundle(validator, repo_root, tmp_path)
    config_path = tmp_path / validator.GSE200304_DEC019_SUCCESSOR_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    initial_config = _dec019_successor_initial_i(config)
    config_path.write_text(
        json.dumps(initial_config, indent=2) + "\n",
        encoding="utf-8",
    )
    assert validator.validate_dec019_successor_adjudicators(tmp_path) == []

    partial_config = deepcopy(initial_config)
    partial_config["implementation_binding"]["status"] = "BOUND"
    current = partial_config["current_external_state"]
    current["qualified"] = True
    current["ordinary_study_contribution"] = 1
    current["a1_study_contribution"] = 1
    current["canonical_record_count"] = 1
    config_path.write_text(
        json.dumps(partial_config, indent=2) + "\n",
        encoding="utf-8",
    )

    codes = _codes(validator.validate_dec019_successor_adjudicators(tmp_path))
    assert "DEC019_SUCCESSOR_BINDING" in codes
    assert "DEC019_SUCCESSOR_CORE_DRIFT" in codes
    assert "DEC019_SUCCESSOR_CURRENT_STATE" in codes


def test_dec019_successor_config_cannot_be_added_to_static_manifest(
    validator,
    repo_root,
    tmp_path,
):
    case_root = tmp_path / "manifest_cycle"

    def add_dynamic_config(manifest):
        relative = validator.GSE114002_DEC019_SUCCESSOR_CONFIG_PATH
        manifest["files"].append(
            {
                "path": relative,
                "role": "DYNAMIC_CONFIG_EXACT_HASH",
                "sha256": validator.sha256_file(case_root / relative),
            }
        )

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        case_root,
        add_dynamic_config,
    )
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "DEC019_SUCCESSOR_MANIFEST_CYCLE" in codes


def test_dec019_successor_static_leaf_synchronized_manifest_rehash_is_rejected(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    relative = validator.GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH
    script_path = tmp_path / relative
    script_path.write_bytes(script_path.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(script_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "DEC019_SUCCESSOR_STATIC_LEAF_DRIFT" in codes


def test_gse200304_dec019_one_blocker_registration_is_closed(
    validator,
    repo_root,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(
        validator.GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256
    )

    assert len(static_paths) == 23
    assert static_paths.issubset(manifest_paths)
    assert validator.GSE200304_DEC019_V3_CONFIG_PATH not in manifest_paths
    assert (
        validator.validate_gse200304_dec019_post_adjudication_registration(
            repo_root
        )
        == []
    )

    config_path = repo_root / validator.GSE200304_DEC019_V3_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert validator.sha256_file(config_path) == validator.GSE200304_DEC019_V3_CONFIG_SHA256
    assert (
        validator._gse200304_dec019_v3_config_core_sha256(config)
        == validator.GSE200304_DEC019_V3_CONFIG_CORE_SHA256
    )
    assert (
        validator._gse200304_dec019_v3_descriptor_set_sha256(config)
        == validator.GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256
    )

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    lineage = interim["artifact_lineage"]
    historical = lineage[validator.GSE200304_DEC019_ADJUDICATION_LINEAGE_ID]
    upstream_pass = lineage[
        validator.GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID
    ]
    current = lineage[validator.GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID]
    assert historical["input_status_counts"] == {
        "PASS": 1,
        "BLOCKED": 3,
        "UNKNOWN_NOT_ASSERTED": 2,
        "NOT_RUN": 2,
    }
    assert upstream_pass["input_status_counts"] == (
        validator.GSE200304_DEC019_UPSTREAM_PASS_INPUT_STATUS_COUNTS
    )
    assert current["historical_predecessor_adjudication_lineage_id"] == (
        validator.GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID
    )
    assert current["pass_slot_ids"] == validator.GSE200304_DEC019_ONE_BLOCKER_PASS_SLOT_IDS
    assert current["input_status_counts"] == (
        validator.GSE200304_DEC019_ONE_BLOCKER_INPUT_STATUS_COUNTS
    )
    assert current["blockers"] == validator.GSE200304_DEC019_ONE_BLOCKER_BLOCKERS
    assert current["ordinary_study_contribution"] == 0
    assert current["a1_study_contribution"] == 0
    assert current["true_a2_study_contribution"] == 0
    assert current["canonical_record_count"] == 0
    assert current["qualified"] is False
    assert current["canonical_materialization_allowed"] is False
    assert current["training_allowed"] is False
    assert current["model_selection_allowed"] is False
    assert current["next_phase_authorized"] is False
    assert current["power_evidence_is_planning_only"] is True
    assert current["runtime_sync_status"] == "SYNCED_EVT_044"
    assert len(lineage[validator.GSE200304_UPSTREAM_AUTHORITY_LINEAGE_ID]["files"]) == 6
    assert len(lineage[validator.GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_LINEAGE_ID]["files"]) == 6
    assert len(lineage[validator.GSE200304_DEC019_GROUP_LINEAGE_ID]["files"]) == 4
    assert len(lineage[validator.GSE200304_DEC019_SPLIT_LINEAGE_ID]["files"]) == 4
    assert len(lineage[validator.GSE200304_DEC019_POWER_LINEAGE_ID]["files"]) == 2
    assert len(current["files"]) == 4


def test_gse200304_dec020_v4_post_adjudication_registration_is_closed(
    validator,
    repo_root,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(validator.GSE200304_DEC020_V4_STATIC_LEAF_SHA256)
    assert len(static_paths) == 3
    assert static_paths.issubset(manifest_paths)
    assert validator.validate_gse200304_dec020_v4_post_adjudication_registration(repo_root) == []

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    current = interim["dec020_current_disposition"]
    assert current["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert current["qualified"] is True
    assert current["canonical_materialization_qualification_eligible"] is True
    assert current["canonical_materialization_execution_authorized"] is False
    assert current["latest_settled_runtime_event_id"] == "A1-EVT-051"
    assert current["authority_runtime_sync"] == {
        "predecessor_event_id": "A1-EVT-050",
        "next_event_id": "A1-EVT-051",
        "next_event_id_preallocated": False,
        "status": "SYNCED_EVT_051",
    }
    assert current["selected_route_status"] == "PASS_DEC020_SCRATCH_ROUTE_SCOPED_REPORTED_ENDPOINT_A1_QUALIFIED"
    successor = current["future_v4_successor_registration"]
    assert successor["lifecycle_status"] == "ADJUDICATED_POST_IMPLEMENTATION_COMMIT_I_BOUND_PRODUCTION"
    assert successor["may_execute"] is False
    assert successor["may_adjudicate"] is False
    lineage = interim["artifact_lineage"][validator.GSE200304_DEC020_V4_LINEAGE_ID]
    assert lineage["input_status_counts"] == {"PASS": 7}
    assert lineage["qualified"] is True
    assert lineage["ordinary_study_contribution"] == 1
    assert lineage["a1_study_contribution"] == 1
    assert lineage["true_a2_study_contribution"] == 0
    assert lineage["canonical_record_count"] == 6547
    assert lineage["predecessor_runtime_event_id"] == "A1-EVT-050"
    assert lineage["expected_next_runtime_event_id"] == "A1-EVT-051"
    assert lineage["runtime_sync_status"] == "SYNCED_EVT_051"
    assert len(lineage["files"]) == 4


def test_post_fail_acquisition_registration_is_closed(
    validator,
    repo_root,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(validator.POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256)

    assert len(static_paths) == 6
    assert static_paths.issubset(manifest_paths)
    assert not (
        {
            validator.GSE200304_CHECKPOINT_EXPOSURE_FAIL_CONFIG_PATH,
            validator.GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG_PATH,
        }
        & validator.DEC019_SUCCESSOR_DYNAMIC_CONFIG_PATHS
    )
    assert validator.REGISTRY_MANIFEST_PATH not in manifest_paths
    assert validator.validate_post_fail_acquisition_registration(repo_root) == []

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    lineage = interim["artifact_lineage"]
    exposure = lineage[validator.GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID]
    acquisition = lineage[validator.GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID]

    assert exposure["status"] == "FAIL_CURRENT_PROTOCOL"
    assert exposure["current_exposure_gate_status"] == "UNKNOWN_NOT_ASSERTED"
    assert exposure["exact_blocker"] == "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"
    assert exposure["current_public_executable_foundation_checkpoint_count"] == 0
    assert exposure["audited_checkpoint_count"] == 0
    assert exposure["qualified"] is False
    assert exposure["training_allowed"] is False
    assert exposure["model_selection_allowed"] is False
    assert exposure["next_phase_authorized"] is False
    assert exposure["predecessor_runtime_event_id"] == "A1-EVT-044"
    assert exposure["expected_next_runtime_event_id"] == "A1-EVT-045"
    assert exposure["runtime_sync_status"] == "SYNCED_EVT_045"

    assert acquisition["status"] == "STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER"
    assert acquisition["acquisition_status"] == (
        "EXACT_21_ASSETS_ACQUIRED_AND_INTEGRITY_VERIFIED"
    )
    assert (
        acquisition["asset_count"],
        acquisition["geo_raw_count"],
        acquisition["supplement_count"],
        acquisition["total_verified_bytes"],
    ) == (21, 18, 3, 70032274)
    assert acquisition["ready_for_full_qualifier_input"] is False
    assert acquisition["ready_for_study_qualification"] is False
    assert acquisition["qualified"] is False
    assert acquisition["canonical_record_count"] == 0
    assert acquisition["training_allowed"] is False
    assert acquisition["model_selection_allowed"] is False
    assert acquisition["next_phase_authorized"] is False
    assert acquisition["predecessor_runtime_event_id"] == "A1-EVT-044"
    assert acquisition["expected_next_runtime_event_id"] == "A1-EVT-045"
    assert acquisition["runtime_sync_status"] == "SYNCED_EVT_045"

    current = interim["dec019_current_disposition"][
        "gse200304_published_processed_endpoint"
    ]
    assert current["input_status_counts"] == {
        "PASS": 7,
        "BLOCKED": 0,
        "UNKNOWN_NOT_ASSERTED": 1,
        "NOT_RUN": 0,
    }
    assert current["current_blockers"] == ["CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"]
    assert current["qualified"] is False
    assert current["training_allowed"] is False
    assert current["model_selection_allowed"] is False
    assert current["next_phase_authorized"] is False

    for lineage_id in (
        validator.GSE200304_DEC019_GROUP_LINEAGE_ID,
        validator.GSE200304_DEC019_SPLIT_LINEAGE_ID,
        validator.GSE200304_DEC019_POWER_LINEAGE_ID,
        validator.GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID,
    ):
        assert lineage[lineage_id]["runtime_sync_status"] == "SYNCED_EVT_044"


def test_gse217518_public_authority_preflight_registration_is_closed(
    validator,
    repo_root,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(
        validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256
    )

    assert len(static_paths) == 3
    assert static_paths.issubset(manifest_paths)
    assert (
        validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH
        not in manifest_paths
    )
    assert validator.REGISTRY_MANIFEST_PATH not in manifest_paths
    assert manifest["manifest_status"] == validator.A6_REGISTRATION_MANIFEST_STATUS
    assert (
        validator.validate_gse217518_public_authority_preflight_registration(
            repo_root
        )
        == []
    )

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    lineage = interim["artifact_lineage"]
    record = lineage[validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID]
    assert record["status"] == "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
    assert record["ready_for_ordinary_public_row_level_producer"] is False
    assert record["bytes"] == 6517
    assert record["sha256"] == (
        "4e43db6030ee0839edb011a35858ba52177a719be23b3cae1774b5aac58ac1c9"
    )
    assert record["qualified"] is False
    assert record["canonical_record_count"] == 0
    assert record["ordinary_study_contribution"] == 0
    assert record["a1_study_contribution"] == 0
    assert record["true_a2_study_contribution"] == 0
    assert record["training_allowed"] is False
    assert record["model_selection_allowed"] is False
    assert record["next_phase_authorized"] is False
    assert record["predecessor_runtime_event_id"] == "A1-EVT-045"
    assert record["expected_next_runtime_event_id"] == "A1-EVT-046"
    assert record["runtime_sync_status"] == "SYNCED_EVT_046"
    assert record["producer_lineage"] == {
        "implementation_commit": "a0e8bd7c751f94e116546d6164ec2de4faeae924",
        "binding_commit": "bcdbd5e0735e950be92cee557785d5f72d2013e9",
        "binding_diff_is_config_only": True,
        "remote_head_at_registration": "bcdbd5e0735e950be92cee557785d5f72d2013e9",
        "config_path": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH,
        "config_sha256": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
            validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH
        ],
        "script_path": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
        "script_sha256": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
            validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH
        ],
        "focused_test_path": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH,
        "focused_test_sha256": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
            validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH
        ],
    }

    summary = interim["dataset_boundary_summary"]["GSE217518"]
    assert summary["public_authority_preflight"]["artifact_lineage_id"] == (
        validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID
    )
    assert summary["public_authority_preflight"]["changes_qualification_gate"] is False
    assert summary["qualified"] is False
    assert summary["training_allowed"] is False
    assert summary["model_selection_allowed"] is False
    assert summary["next_phase_authorized"] is False

    gate = interim["gate_snapshot"]
    assert gate["qualified_independent_ordinary_studies"] == 1
    assert gate["qualified_a1_studies"] == 1
    assert gate["qualified_a2_dense_studies"] == 0
    assert gate["next_phase_authorized"] is False
    assert interim["dec019_current_disposition"]["runtime_sync_status"] == (
        "SYNCED_EVT_049"
    )
    assert lineage[validator.GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID][
        "runtime_sync_status"
    ] == "SYNCED_EVT_045"
    assert lineage[validator.GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID][
        "runtime_sync_status"
    ] == "SYNCED_EVT_045"


def test_gse217518_static_drift_dynamic_cycle_and_unlock_fail_closed(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    static_root = tmp_path / "static"
    manifest = _copy_manifest_bundle(validator, repo_root, static_root)
    relative = validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH
    leaf_path = static_root / relative
    leaf_path.write_bytes(leaf_path.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(leaf_path)
    manifest_path = static_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(static_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF" in codes

    dynamic_root = tmp_path / "dynamic_cycle"
    dynamic_manifest = _copy_manifest_bundle(validator, repo_root, dynamic_root)
    dynamic_manifest["files"].append(
        {
            "path": validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH,
            "role": "FORBIDDEN_DYNAMIC_EVT046_CONFIG",
            "sha256": "0" * 64,
        }
    )
    dynamic_manifest_path = dynamic_root / validator.REGISTRY_MANIFEST_PATH
    dynamic_manifest_path.write_text(
        json.dumps(dynamic_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse217518_public_authority_preflight_registration(
            dynamic_root
        )
    )
    assert "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_MANIFEST_DAG" in codes

    def forge_unlock(interim):
        record = interim["artifact_lineage"][
            validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID
        ]
        record["status"] = "READY_FOR_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
        record["ready_for_ordinary_public_row_level_producer"] = True
        record["qualified"] = True
        summary = interim["dataset_boundary_summary"]["GSE217518"]
        summary["public_authority_preflight"]["status"] = (
            "READY_FOR_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
        )
        summary["public_authority_preflight"][
            "ready_for_ordinary_public_row_level_producer"
        ] = True
        summary["qualified"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "interim_unlock",
        monkeypatch,
        forge_unlock,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE217518" in codes


def test_gse232572_public_recovery_audit_registration_is_closed(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(validator.GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256)

    assert len(static_paths) == 3
    assert static_paths.issubset(manifest_paths)
    assert validator.GSE232572_PUBLIC_RECOVERY_AUDIT_RUNTIME_CONFIG_PATH not in manifest_paths
    assert validator.REGISTRY_MANIFEST_PATH not in manifest_paths
    assert validator.validate_gse232572_public_recovery_audit_registration(repo_root) == []

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    record = interim["artifact_lineage"][
        validator.GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID
    ]
    assert record["artifact_type"] == "GSE232572_PUBLIC_RECOVERY_AUDIT_AGGREGATE_ONLY"
    assert record["status"] == "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED"
    assert record["registry_role"] == "AUDIT_ONLY"
    assert record["qualification_status"] == "AUDIT_PENDING"
    assert record["aggregate_only"] is True
    assert record["published_universe_row_count"] == 11929
    assert record["accepted_pair_count"] == 8068
    assert record["rejected_published_row_count"] == 3861
    assert record["rejection_reason_counts"] == {
        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
    }
    assert record["development_reconstruction_record_count"] == 8068
    assert record["canonical_materialization_allowed"] is False
    assert record["canonical_record_count"] == 0
    assert record["qualified"] is False
    assert record["ordinary_study_contribution"] == 0
    assert record["a1_study_contribution"] == 0
    assert record["true_a2_study_contribution"] == 0
    assert record["training_allowed"] is False
    assert record["model_selection_allowed"] is False
    assert record["next_phase_authorized"] is False
    assert record["predecessor_runtime_event_id"] == "A1-EVT-046"
    assert record["expected_next_runtime_event_id"] == "A1-EVT-047"
    assert record["runtime_sync_status"] == "SYNCED_EVT_047"
    assert "binding_commit" not in record["producer_lineage"]
    assert record["producer_lineage"]["config_inspected_predecessor_is_binding_commit"] is False

    summary = interim["dataset_boundary_summary"]["GSE232572"]
    audit = summary["public_recovery_audit"]
    assert audit["artifact_lineage_id"] == validator.GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID
    assert audit["changes_qualification_gate"] is False
    assert summary["qualified"] is False
    assert summary["training_allowed"] is False
    assert summary["model_selection_allowed"] is False
    assert summary["next_phase_authorized"] is False
    assert interim["artifact_lineage"][
        validator.GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID
    ]["runtime_sync_status"] == "SYNCED_EVT_046"
    assert interim["dec019_current_disposition"]["runtime_sync_status"] == "SYNCED_EVT_049"
    assert not any(
        row["path"].endswith(".private.jsonl") for row in manifest["files"]
    )

    static_root = tmp_path / "static_drift"
    static_manifest = _copy_manifest_bundle(validator, repo_root, static_root)
    relative = validator.GSE232572_PUBLIC_RECOVERY_AUDIT_SCRIPT_PATH
    static_leaf = static_root / relative
    static_leaf.write_bytes(static_leaf.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in static_manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(static_leaf)
    (static_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(static_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_public_recovery_audit_registration(
            static_root
        )
    )
    assert "GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF" in codes

    dynamic_root = tmp_path / "dynamic_cycle"
    dynamic_manifest = _copy_manifest_bundle(validator, repo_root, dynamic_root)
    dynamic_manifest["files"].append(
        {
            "path": validator.GSE232572_PUBLIC_RECOVERY_AUDIT_RUNTIME_CONFIG_PATH,
            "role": "FORBIDDEN_DYNAMIC_EVT047_CONFIG",
            "sha256": "0" * 64,
        }
    )
    (dynamic_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(dynamic_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_public_recovery_audit_registration(
            dynamic_root
        )
    )
    assert "GSE232572_PUBLIC_RECOVERY_AUDIT_MANIFEST_DAG" in codes

    def forge_qualification(document):
        node = document["artifact_lineage"][
            validator.GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID
        ]
        node["canonical_record_count"] = 8068
        node["qualified"] = True
        boundary = document["dataset_boundary_summary"]["GSE232572"]
        boundary["public_recovery_audit"]["canonical_record_count"] = 8068
        boundary["qualified"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "qualification_bypass",
        monkeypatch,
        forge_qualification,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE232572" in codes


def test_gse232572_development_v3_materialization_registration_is_closed(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256
    )

    assert len(static_paths) == 3
    assert static_paths.issubset(manifest_paths)
    assert (
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_RUNTIME_CONFIG_PATH
        not in manifest_paths
    )
    assert validator.REGISTRY_MANIFEST_PATH not in manifest_paths
    assert not any(path.endswith(".private.jsonl") for path in manifest_paths)
    assert (
        validator.validate_gse232572_development_v3_materialization_registration(
            repo_root
        )
        == []
    )

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    lineage = interim["artifact_lineage"]
    failure = lineage[
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID
    ]
    success = lineage[validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID]

    assert failure["artifact_type"] == (
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_"
        "ATTEMPT_001_FAIL_CLOSED_EVIDENCE"
    )
    assert failure["status"] == "STOP_BEFORE_DEVELOPMENT_V3_ROW_PRODUCTION"
    assert failure["failure_gate"] == "RECOVERY_AUTHORITY"
    assert failure["failure_code"] == (
        "MATERIALIZER_INPUTS_DIVERGE_FROM_RECOVERY_CONFIG"
    )
    assert failure["schema_valid_development_record_count"] == 0
    assert failure["canonical_record_count"] == 0
    assert failure["qualified"] is False
    assert failure["failed_attempt_preserved"] is True
    assert failure["historical_attempt_rewritten"] is False
    assert failure["superseded_for_current_execution_by_lineage_id"] == (
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID
    )
    assert failure["producer_lineage"]["implementation_i2_commit"] == (
        "5619dc39622de7f97f63811d51a0e04bdf668e48"
    )
    assert failure["producer_lineage"]["binding_b2_commit"] == (
        "89db6313c6331e767ac5074170e7ff5b3cab8e3e"
    )
    assert failure["producer_lineage"]["implementation_exact_changed_paths"] == [
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH,
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH,
    ]
    assert "script_sha256" not in failure["producer_lineage"]
    assert "config_sha256" not in failure["producer_lineage"]

    assert success["artifact_type"] == (
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT_AGGREGATE_ONLY"
    )
    assert success["status"] == "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED"
    assert success["scientific_disposition"] == (
        "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
    )
    assert success["published_universe_row_count"] == 11929
    assert success["schema_valid_development_record_count"] == 8068
    assert success["rejected_published_row_count"] == 3861
    assert success["rejection_reason_counts"] == {
        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
    }
    assert success["canonical_materialization_allowed"] is False
    assert success["canonical_record_count"] == 0
    assert success["qualified"] is False
    assert success["ordinary_study_contribution"] == 0
    assert success["a1_study_contribution"] == 0
    assert success["true_a2_study_contribution"] == 0
    assert success["public_redistribution_status"] == (
        "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
    )
    assert success["row_license_status"] == "UNKNOWN_BLOCKED"
    assert success["redistribution_allowed"] is False
    assert success["training_allowed"] is False
    assert success["model_selection_allowed"] is False
    assert success["next_phase_allowed"] is False
    assert success["failed_attempt_lineage_id"] == (
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID
    )
    assert success["producer_lineage"]["implementation_i3_commit"] == (
        "e923d7b992293ca7bb5889bf3c0b3bc6ce750e03"
    )
    assert success["producer_lineage"]["binding_b3_commit"] == (
        "b982275c25b7158a5a543a5e0c9fd23728fa0961"
    )
    assert success["producer_lineage"]["config_bytes"] == 9484
    assert success["producer_lineage"]["script_bytes"] == 55522
    assert success["producer_lineage"]["focused_test_bytes"] == 31585
    assert failure["runtime_sync_status"] == "SYNCED_EVT_048"
    assert success["runtime_sync_status"] == "SYNCED_EVT_048"
    assert failure["private_jsonl_read_count_for_ledger"] == 0
    assert failure["private_jsonl_registered_artifact_count"] == 0
    assert success["private_jsonl_read_count_for_ledger"] == 0
    assert success["private_jsonl_registered_artifact_count"] == 0

    summary = interim["dataset_boundary_summary"]["GSE232572"]
    materialization = summary["development_v3_materialization"]
    assert materialization["failed_attempt_artifact_lineage_id"] == (
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID
    )
    assert materialization["current_artifact_lineage_id"] == (
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID
    )
    assert materialization["schema_valid_development_record_count"] == 8068
    assert materialization["canonical_record_count"] == 0
    assert materialization["changes_qualification_gate"] is False
    assert materialization["runtime_sync_status"] == "SYNCED_EVT_048"
    assert summary["qualified"] is False
    assert summary["training_allowed"] is False
    assert summary["model_selection_allowed"] is False
    assert summary["next_phase_authorized"] is False
    assert lineage[validator.GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID][
        "runtime_sync_status"
    ] == "SYNCED_EVT_047"
    assert interim["dec019_current_disposition"]["runtime_sync_status"] == (
        "SYNCED_EVT_049"
    )

    static_root = tmp_path / "materializer_static_drift"
    static_manifest = _copy_manifest_bundle(validator, repo_root, static_root)
    relative = validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH
    static_leaf = static_root / relative
    static_leaf.write_bytes(static_leaf.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in static_manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(static_leaf)
    (static_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(static_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_development_v3_materialization_registration(
            static_root
        )
    )
    assert "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF" in codes

    dynamic_root = tmp_path / "materializer_dynamic_cycle"
    dynamic_manifest = _copy_manifest_bundle(validator, repo_root, dynamic_root)
    dynamic_manifest["files"].append(
        {
            "path": validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_RUNTIME_CONFIG_PATH,
            "role": "FORBIDDEN_DYNAMIC_EVT048_CONFIG",
            "sha256": "0" * 64,
        }
    )
    (dynamic_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(dynamic_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_development_v3_materialization_registration(
            dynamic_root
        )
    )
    assert "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_MANIFEST_DAG" in codes

    private_root = tmp_path / "private_jsonl_registration"
    private_manifest = _copy_manifest_bundle(validator, repo_root, private_root)
    private_manifest["files"].append(
        {
            "path": "reports/development_v3_records.private.jsonl",
            "role": "FORBIDDEN_PRIVATE_ROW_ARTIFACT",
            "sha256": "0" * 64,
        }
    )
    (private_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(private_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_development_v3_materialization_registration(
            private_root
        )
    )
    assert "GSE232572_DEVELOPMENT_V3_PRIVATE_JSONL_EXCLUDED" in codes

    def forge_canonical_unlock(document):
        node = document["artifact_lineage"][
            validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID
        ]
        node["canonical_record_count"] = 8068
        node["qualified"] = True
        node["redistribution_allowed"] = True
        node["training_allowed"] = True
        boundary = document["dataset_boundary_summary"]["GSE232572"]
        boundary["development_v3_materialization"]["canonical_record_count"] = 8068
        boundary["development_v3_materialization"]["qualified"] = True
        boundary["qualified"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "materialization_unlock",
        monkeypatch,
        forge_canonical_unlock,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE232572" in codes


def test_gse232572_qualification_authority_preflight_registration_is_closed(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    static_paths = set(
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256
    )

    assert len(static_paths) == 3
    assert static_paths.issubset(manifest_paths)
    assert (
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH
        not in manifest_paths
    )
    assert validator.REGISTRY_MANIFEST_PATH not in manifest_paths
    assert not any(path.endswith(".private.jsonl") for path in manifest_paths)
    assert (
        validator.validate_gse232572_qualification_authority_preflight_registration(
            repo_root
        )
        == []
    )

    interim = validator._load_yaml(repo_root, validator.A1_INTERIM_PATH)
    lineage = interim["artifact_lineage"]
    record = lineage[
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID
    ]
    assert record["path"] == (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_20260813T010116P0800/"
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT.json"
    )
    assert record["bytes"] == 9586
    assert record["sha256"] == (
        "00776c808cfa3e9ba2cfdb92b866c5f7c1bc92ea3818d17687cb9a8521b30d71"
    )
    assert record["artifact_type"] == (
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_AGGREGATE_ONLY"
    )
    assert record["overall_decision"] == "BLOCKED_MISSING_EXTERNAL_AUTHORITY"
    assert record["terminal_status"] == (
        "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION"
    )
    assert record["registered_aggregate_pass_count"] == 3
    assert record["registered_aggregate_passes"] == (
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_PASSES
    )
    assert record["open_qualification_blocker_count"] == 12
    assert record["qualification_blocker_statuses"] == (
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_BLOCKERS
    )
    assert record["schema_valid_development_record_count"] == 8068
    assert record["public_replicate_count"] == 3
    assert record["primary_label_standard_error"] is None
    assert record["checkpoint_specific_exposure"] == "UNKNOWN_NOT_ASSERTED"
    assert record["untouched_confirmatory"] is False
    assert record["recommended_future_scope"] == "PRIVATE_CANONICAL_ONLY"
    assert record["recommended_future_scope_approved"] is False
    assert record["future_contribution_authorization_status"] == "NOT_AUTHORIZED"
    assert record["canonical_record_count"] == 0
    assert record["ordinary_study_contribution"] == 0
    assert record["a1_study_contribution"] == 0
    assert record["true_a2_study_contribution"] == 0
    assert record["qualified"] is False
    assert record["training_allowed"] is False
    assert record["model_selection_allowed"] is False
    assert record["next_phase_allowed"] is False
    assert record["private_row_artifact_read_count_for_ledger"] == 0
    assert record["private_jsonl_registered_artifact_count"] == 0
    assert record["predecessor_runtime_event_id"] == "A1-EVT-048"
    assert record["expected_next_runtime_event_id"] == "A1-EVT-049"
    assert record["runtime_sync_status"] == "SYNCED_EVT_049"

    producer = record["producer_lineage"]
    assert producer["base_commit"] == "13baa39e87406b5bc81b7e236cee637f694bfd0f"
    assert producer["initial_implementation_commit"] == (
        "cb10350681a1f4fd7dbe5322d671d618d77aaebf"
    )
    assert producer["lifecycle_repair_implementation_commit"] == (
        "8ee914723b0d97d8ca07bab9ae7aaa1114e049dd"
    )
    assert producer["binding_commit"] == "d0778b92c1b90456a84bce60c7b7c3e039bc1ff5"
    assert producer["binding_diff_is_config_only"] is True
    assert producer["config_sha256"] == (
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
            validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH
        ]
    )

    assert lineage[
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID
    ]["runtime_sync_status"] == "SYNCED_EVT_048"
    assert lineage[
        validator.GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID
    ]["runtime_sync_status"] == "SYNCED_EVT_048"
    summary = interim["dataset_boundary_summary"]["GSE232572"]
    preflight = summary["qualification_authority_preflight"]
    assert preflight["artifact_lineage_id"] == (
        validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID
    )
    assert preflight["registered_aggregate_pass_count"] == 3
    assert preflight["open_qualification_blocker_count"] == 12
    assert preflight["canonical_record_count"] == 0
    assert preflight["changes_qualification_gate"] is False
    assert preflight["runtime_sync_status"] == "SYNCED_EVT_049"
    assert summary["qualified"] is False
    assert summary["training_allowed"] is False
    assert summary["model_selection_allowed"] is False
    assert summary["next_phase_authorized"] is False
    assert interim["dec019_current_disposition"]["runtime_sync_status"] == (
        "SYNCED_EVT_049"
    )

    static_root = tmp_path / "qualification_preflight_static_drift"
    static_manifest = _copy_manifest_bundle(validator, repo_root, static_root)
    relative = validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH
    static_leaf = static_root / relative
    static_leaf.write_bytes(static_leaf.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in static_manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(static_leaf)
    (static_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(static_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_qualification_authority_preflight_registration(
            static_root
        )
    )
    assert "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF" in codes

    dynamic_root = tmp_path / "qualification_preflight_dynamic_cycle"
    dynamic_manifest = _copy_manifest_bundle(validator, repo_root, dynamic_root)
    dynamic_manifest["files"].append(
        {
            "path": validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH,
            "role": "FORBIDDEN_DYNAMIC_EVT049_CONFIG",
            "sha256": "0" * 64,
        }
    )
    (dynamic_root / validator.REGISTRY_MANIFEST_PATH).write_text(
        json.dumps(dynamic_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    codes = _codes(
        validator.validate_gse232572_qualification_authority_preflight_registration(
            dynamic_root
        )
    )
    assert "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_MANIFEST_DAG" in codes

    def forge_qualification_unlock(document):
        node = document["artifact_lineage"][
            validator.GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID
        ]
        node["overall_decision"] = "QUALIFIED"
        node["open_qualification_blocker_count"] = 0
        node["canonical_record_count"] = 8068
        node["qualified"] = True
        node["training_allowed"] = True
        boundary = document["dataset_boundary_summary"]["GSE232572"]
        boundary["qualification_authority_preflight"]["qualified"] = True
        boundary["qualified"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "qualification_preflight_unlock",
        monkeypatch,
        forge_qualification_unlock,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE232572" in codes


def test_post_fail_acquisition_drift_and_interim_unlock_fail_closed(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    static_root = tmp_path / "static"
    manifest = _copy_manifest_bundle(validator, repo_root, static_root)
    relative = validator.GSE149487_PUBLIC_ASSET_ACQUISITION_SCRIPT_PATH
    leaf_path = static_root / relative
    leaf_path.write_bytes(leaf_path.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(leaf_path)
    manifest_path = static_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(static_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "POST_FAIL_ACQUISITION_STATIC_LEAF" in codes

    def forge_unlock(interim):
        exposure = interim["artifact_lineage"][
            validator.GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID
        ]
        exposure["current_exposure_gate_status"] = "PASS"
        exposure["qualified"] = True
        acquisition = interim["artifact_lineage"][
            validator.GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID
        ]
        acquisition["ready_for_study_qualification"] = True
        acquisition["qualified"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "interim_unlock",
        monkeypatch,
        forge_unlock,
    )
    assert "A1_INTERIM_LINEAGE" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes


def test_gse200304_dec019_post_adjudication_dynamic_and_static_drift_fail_closed(
    validator,
    repo_root,
    tmp_path,
):
    dynamic_root = tmp_path / "dynamic"
    _copy_manifest_bundle(validator, repo_root, dynamic_root)
    config_path = dynamic_root / validator.GSE200304_DEC019_V3_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evidence_descriptor_bindings"]["descriptor_set_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    codes = _codes(
        validator.validate_gse200304_dec019_post_adjudication_registration(
            dynamic_root
        )
    )
    assert "GSE200304_DEC019_V3_DYNAMIC_CONFIG" in codes
    assert "GSE200304_DEC019_V3_DESCRIPTOR_BINDING" in codes
    assert "GSE200304_DEC019_POST_ADJUDICATION_DAG" not in codes

    static_root = tmp_path / "static"
    manifest = _copy_manifest_bundle(validator, repo_root, static_root)
    relative = validator.GSE200304_DEC019_POWER_SCRIPT_PATH
    leaf_path = static_root / relative
    leaf_path.write_bytes(leaf_path.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(leaf_path)
    manifest_path = static_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    codes = _codes(validator.validate_bundle(static_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF" in codes


def test_gse200304_dec019_post_adjudication_interim_rehash_cannot_unlock(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def materialize_positive_input(interim):
        current = interim["dec019_current_disposition"][
            "gse200304_published_processed_endpoint"
        ]
        current["canonical_record_count"] = 6547
        current["canonical_materialization_allowed"] = True
        summary = interim["dataset_boundary_summary"]["GSE200304"][
            "dec019_post_adjudication"
        ]
        summary["canonical_record_count"] = 6547
        summary["canonical_materialization_allowed"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "materialize_positive_input",
        monkeypatch,
        materialize_positive_input,
    )
    assert "A1_INTERIM_DEC019_GSE200304" in codes
    assert "A1_INTERIM_GSE200304" in codes

    def change_settled_member_and_one_blocker_counts(interim):
        lineage = interim["artifact_lineage"]
        lineage[validator.GSE200304_DEC019_GROUP_LINEAGE_ID][
            "files"
        ][0]["sha256"] = "0" * 64
        lineage[
            validator.GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID
        ]["input_status_counts"]["PASS"] = 6

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path / "lineage_and_counts",
        monkeypatch,
        change_settled_member_and_one_blocker_counts,
    )
    assert "A1_INTERIM_GSE200304_CLOSED_FILES" in codes
    assert "A1_INTERIM_GSE200304_LINEAGE" in codes


def test_dec019_authority_synchronized_rehash_cannot_relax_routes(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(amendment):
        gse114002 = amendment["gse114002_designed_library_true_a2_route"]
        gse114002["technical_fraction_uncertainty"]["may_support_power"] = True
        gse114002["candidate_hamming_distance_eligibility_if_qualified"] = [1, 3]
        gse114002["maximum_independent_ordinary_study_contribution_if_qualified"] = 2
        gse114002["k5_role"] = "QUALIFICATION_GATE"
        gse200304 = amendment["gse200304_published_processed_endpoint_a1_route"]
        gse200304["raw_replay_role"] = "PRIMARY_MEASUREMENT_ROUTE"
        gse200304["maximum_independent_ordinary_study_contribution_if_qualified"] = 2
        amendment["uncertainty_and_power_authority"]["target_power_minimum"] = True
        amendment["split_freeze_boundary"]["a1_freeze_required"] = [
            "FINAL_BENCHMARK_MEMBERSHIP"
        ]
        amendment["split_freeze_boundary"]["a2_freeze_required"] = [
            "SOURCE_AUTHORITY"
        ]
        amendment["nonwaivable_authority"][
            "global_replicate_or_standard_error_relaxation_allowed"
        ] = True
        amendment["nonwaivable_authority"][
            "gse149487_three_biological_replicates_and_route_a_se_gate_changed"
        ] = True
        amendment["nonwaivable_authority"][
            "other_dataset_specific_stricter_replicate_or_standard_error_gates_changed"
        ] = True

    codes = _validate_rehashed_dec019_leaf_bypass(
        validator,
        repo_root,
        tmp_path,
        monkeypatch,
        validator.DEC019_AMENDMENT_PATH,
        mutate,
    )
    assert "DEC019_LEAF_AUTHORITY_DRIFT" in codes
    assert "DEC019_GSE114002_ROUTE" in codes
    assert "DEC019_GSE200304_ROUTE" in codes
    assert "DEC019_AMENDMENT_SEMANTICS" in codes


def test_dec019_data_role_synchronized_rehash_cannot_globalize_absence_route(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(registry):
        requirements = registry["common_audit_requirements"]
        requirements[requirements.index(
            "REPLICATE_AND_STANDARD_ERROR_OR_DATASET_SCOPED_ABSENCE_ADJUDICATION"
        )] = "REPLICATE_OR_STANDARD_ERROR"
        gse114002 = next(
            row for row in registry["datasets"] if row["dataset_id"] == "GSE114002"
        )
        gse114002["dec019_conditional_true_a2_route"][
            "replicate_and_standard_error_absence_adjudication_may_apply_to_other_datasets"
        ] = True

    codes = _validate_rehashed_dec019_leaf_bypass(
        validator,
        repo_root,
        tmp_path,
        monkeypatch,
        validator.REGISTRY_PATHS["data"],
        mutate,
    )
    assert "DEC020_ACTIVE_AUTHORITY_LEAF_DRIFT" in codes
    assert "DEC019_DATASET_SCOPED_UNCERTAINTY" in codes


def test_dec019_interim_successor_lineage_cannot_fake_qualification(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
):
    def mutate(interim):
        lineage = interim["artifact_lineage"][
            validator.GSE114002_DEC019_SUCCESSOR_LINEAGE_ID
        ]
        lineage["current_qualified"] = True
        lineage["current_ordinary_study_contribution"] = 2
        lineage["current_true_a2_dense_study_contribution"] = 2
        lineage["current_canonical_record_count"] = 1
        lineage["training_allowed"] = True

    codes = _validate_rehashed_interim_bypass(
        validator,
        repo_root,
        tmp_path,
        monkeypatch,
        mutate,
    )
    assert "A1_INTERIM_LINEAGE" in codes


def test_scheme_a_data_roles_cannot_restore_gse145046_as_true_a2(
    validator,
    repo_root,
):
    registry = validator._load_yaml(repo_root, validator.REGISTRY_PATHS["data"])
    assert validator.validate_scheme_a_data_roles(registry) == []

    bypass = deepcopy(registry)
    bypass["ordinary_candidate_dataset_ids"].append("GSE145046")
    gse145046 = next(row for row in bypass["datasets"] if row["dataset_id"] == "GSE145046")
    gse145046["true_a2_qualification_status"] = "QUALIFIED"
    gse145046["true_a2_gate_contribution"] = 1
    gse145046["source_relative_confirmatory_evidence_allowed"] = True
    gse145046["permanently_forbidden_gate_uses"] = []
    gse114002 = next(row for row in bypass["datasets"] if row["dataset_id"] == "GSE114002")
    gse114002["known_related_sequence_exposure_label"] = "UNEXPOSED"
    gse114002["fallback_if_designed_library_not_qualifiable"] = "LOWER_TRUE_A2_GATE"
    bypass["data_policy"]["ordinary_minimum_a2_dense_studies"] = 0
    codes = _codes(validator.validate_scheme_a_data_roles(bypass))
    assert "SCHEME_A_GATE_PRESERVATION" in codes
    assert "SCHEME_A_ORDINARY_CANDIDATES" in codes
    assert "SCHEME_A_GSE145046_ROLE" in codes
    assert "SCHEME_A_GSE145046_FORBIDDEN" in codes
    assert "SCHEME_A_GSE114002_BOUNDARY" in codes


def test_scheme_a_gse200304_official_role_authority_cannot_unlock_replay(
    validator,
    repo_root,
):
    registry = validator._load_yaml(repo_root, validator.REGISTRY_PATHS["data"])
    assert validator.validate_scheme_a_data_roles(registry) == []

    bypass = deepcopy(registry)
    gse200304 = next(row for row in bypass["datasets"] if row["dataset_id"] == "GSE200304")
    official = gse200304["official_srr_role_authority"]
    official["measurement_families"] = ["High_Poly", "Low_Poly", "80S_RNA", "Total_RNA"]
    official["mapping_row_count"] = 25
    official["bundle_digest"] = "0" * 64
    raw_role = gse200304["raw_replay_role_authority"]
    raw_role["prior_blocker_status"] = "OPEN"
    raw_role["replacement_blocker_status"] = "CLOSED"
    raw_role["role_grid_status"] = "MATCH"
    raw_role["pdna_may_substitute_for_80s_rna"] = True
    gse200304["qualified"] = True
    gse200304["training_allowed"] = True
    gse200304["model_selection_allowed"] = True
    gse200304["next_phase_authorized"] = True
    gse200304["ordinary_gate_contribution"] = 1
    codes = _codes(validator.validate_scheme_a_data_roles(bypass))
    assert "SCHEME_A_GSE200304_ROLE_AUTHORITY" in codes

    new_study = deepcopy(registry)
    new_study["datasets"].append({"dataset_id": "GSE200302"})
    codes = _codes(validator.validate_scheme_a_data_roles(new_study))
    assert "SCHEME_A_GSE200302_NOT_NEW_STUDY" in codes


def test_rehashed_registry_cannot_relabel_pdna_as_80s_or_close_successor_blocker(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    registry_path = tmp_path / validator.REGISTRY_PATHS["data"]
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    gse200304 = next(row for row in registry["datasets"] if row["dataset_id"] == "GSE200304")
    gse200304["official_srr_role_authority"]["measurement_families"] = [
        "High_Poly",
        "Low_Poly",
        "80S_RNA",
        "Total_RNA",
    ]
    gse200304["raw_replay_role_authority"]["replacement_blocker_status"] = "CLOSED"
    gse200304["raw_replay_role_authority"]["pdna_may_substitute_for_80s_rna"] = True
    gse200304["qualified"] = True
    gse200304["training_allowed"] = True
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    entry = next(row for row in manifest["files"] if row["path"] == validator.REGISTRY_PATHS["data"])
    entry["sha256"] = validator.sha256_file(registry_path)
    interim_path = tmp_path / validator.A1_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    interim["authority"]["data_role_registry_sha256"] = entry["sha256"]
    interim_path.write_text(yaml.safe_dump(interim, sort_keys=False), encoding="utf-8")
    interim_hash = validator.sha256_file(interim_path)
    next(row for row in manifest["files"] if row["path"] == validator.A1_INTERIM_PATH)[
        "sha256"
    ] = interim_hash
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(tmp_path))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "SCHEME_A_GSE200304_ROLE_AUTHORITY" in codes


def test_supersession_config_and_m0_scientific_failure_are_bound(validator, repo_root, bundle_documents):
    config, source_supersession, registries = bundle_documents
    supersession = deepcopy(source_supersession)
    supersession["new_authority"]["config_sha256"] = "0" * 64
    m0 = next(row for row in supersession["historical_gate_records"] if row["gate_id"] == "M0_SCIENTIFIC_ORIGINAL")
    m0["macro_sign_accuracy"] = 0.61
    m0["o0_valid"] = True
    codes = _codes(validator.validate_contract_authority(repo_root, config, supersession, registries))
    assert "SUPERSESSION_CONFIG_HASH" in codes
    assert "M0_SCIENTIFIC_BINDING" in codes
    assert "M0_SCIENTIFIC_THRESHOLD" in codes


def test_runner_and_guard_ast_are_fail_closed(validator, repo_root):
    assert validator.validate_runner_and_guard_ast(repo_root) == []


def test_runner_ast_rejects_guard_moved_after_runtime_import(
    validator,
    repo_root,
    tmp_path,
):
    runner_target, _ = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = runner_target.read_text(encoding="utf-8")
    guard_block = (
        '    if args.mode == "sealed-final":\n'
        "        assert_sealed_final_authorized(args)\n\n"
    )
    source = source.replace(guard_block, "", 1)
    source = source.replace(
        "    from scripts.e0x import prereg\n",
        "    from scripts.e0x import prereg\n" + guard_block,
        1,
    )
    runner_target.write_text(source, encoding="utf-8")
    codes = _codes(validator.validate_runner_and_guard_ast(tmp_path))
    assert "RUNNER_EARLY_GUARD_MISSING" in codes


def test_runner_ast_rejects_eager_project_import(
    validator,
    repo_root,
    tmp_path,
):
    runner_target, _ = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = runner_target.read_text(encoding="utf-8").replace(
        "from scripts.route_a_v3.sealed_guard import assert_sealed_final_authorized\n",
        "from scripts.route_a_v3.sealed_guard import assert_sealed_final_authorized\n"
        "from scripts.m4_sparse import config as EAGER_CONFIG\n",
        1,
    )
    runner_target.write_text(source, encoding="utf-8")
    assert "RUNNER_MODULE_PROJECT_IMPORT" in _codes(
        validator.validate_runner_and_guard_ast(tmp_path)
    )


def test_runner_ast_requires_defense_guard_expression_first(
    validator,
    repo_root,
    tmp_path,
):
    runner_target, _ = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = runner_target.read_text(encoding="utf-8").replace(
        "    assert_sealed_final_authorized(args)\n"
        "    from scripts.e0x import sealed\n",
        "    authorization = assert_sealed_final_authorized(args)\n"
        "    from scripts.e0x import sealed\n",
        1,
    )
    runner_target.write_text(source, encoding="utf-8")
    assert "RUN_SEALED_FIRST_GUARD" in _codes(
        validator.validate_runner_and_guard_ast(tmp_path)
    )


def test_guard_ast_rejects_toggle_and_reachable_return(
    validator,
    repo_root,
    tmp_path,
):
    _, guard_target = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = guard_target.read_text(encoding="utf-8").replace(
        "    raise RouteAV3SealedHardDisabled(HARD_DISABLED)\n",
        '    if getattr(call_args, "execution_authorized", False):\n'
        "        return None\n"
        "    raise RouteAV3SealedHardDisabled(HARD_DISABLED)\n",
        1,
    )
    guard_target.write_text(source, encoding="utf-8")
    codes = _codes(validator.validate_runner_and_guard_ast(tmp_path))
    assert "SEALED_GUARD_HARD_DISABLE_BODY" in codes
    assert "SEALED_GUARD_REACHABLE_SUCCESS" in codes


def test_guard_ast_rejects_any_path_or_manifest_import(
    validator,
    repo_root,
    tmp_path,
):
    _, guard_target = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = guard_target.read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\nfrom pathlib import Path\n",
        1,
    )
    guard_target.write_text(source, encoding="utf-8")
    codes = _codes(validator.validate_runner_and_guard_ast(tmp_path))
    assert "SEALED_GUARD_MODULE_SHAPE" in codes
    assert "SEALED_GUARD_IMPORT" in codes


def test_registry_manifest_detects_every_listed_hash_drift(validator, tmp_path, monkeypatch):
    goal_bytes = b"fixture Route A V3 authority\n"
    goal_hash = validator.sha256_bytes(goal_bytes)
    monkeypatch.setattr(validator, "SOURCE_CONTRACT_SHA256", goal_hash)
    goal_path = tmp_path / validator.GOAL_PATH
    goal_path.parent.mkdir(parents=True)
    goal_path.write_bytes(goal_bytes)

    entries = []
    for index, (relative, role) in enumerate(
        validator.EXPECTED_REGISTRY_MANIFEST_PATH_ROLES
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            goal_bytes
            if relative == validator.GOAL_PATH
            else f"fixture-{index}-{relative}\n".encode("utf-8")
        )
        path.write_bytes(data)
        entries.append(
            {
                "path": relative,
                "role": role,
                "sha256": validator.sha256_bytes(data),
            }
        )
    interim_path = tmp_path / validator.A1_INTERIM_PATH
    interim_path.write_text(
        yaml.safe_dump({"updated_at": validator.A6_REGISTRATION_LEDGER_AT}),
        encoding="utf-8",
    )
    next(row for row in entries if row["path"] == validator.A1_INTERIM_PATH)["sha256"] = validator.sha256_file(interim_path)
    manifest = {
        "contract_id": validator.CONTRACT_ID,
        "version": validator.VERSION,
        "schema_version": "1.0.0",
        "contract_path": validator.GOAL_PATH,
        "initial_contract_sha256": "d1c031aecdec710495f6861b380785cccd64663ac4bd97b4f479d6fdf372ea07",
        "contract_sha256": goal_hash,
        "active_amendment_decision_ids": validator.ACTIVE_AMENDMENT_DECISION_IDS,
        "base_commit": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
        "manifest_status": validator.A6_REGISTRATION_MANIFEST_STATUS,
        "initial_generated_at": "2026-08-10T10:10:05+08:00",
        "generated_at": validator.A6_REGISTRATION_MANIFEST_AT,
        "updated_at": validator.A6_REGISTRATION_MANIFEST_AT,
        "sealed_contact": False,
        "files": entries,
    }
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator.validate_registry_manifest(tmp_path) == []

    config_path = tmp_path / validator.CONFIG_PATH
    config_path.write_bytes(config_path.read_bytes() + b"drift")
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" in _codes(validator.validate_registry_manifest(tmp_path))


def test_registry_manifest_shape_and_types_are_closed(
    validator,
    repo_root,
    tmp_path,
):
    def mutate_role(manifest):
        manifest["files"][0]["role"] = "UNREGISTERED_ROLE"

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        tmp_path / "role",
        mutate_role,
    )
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "REGISTRY_MANIFEST_CLOSURE" in codes

    extra_root = tmp_path / "extra_entry"

    def append_extra_path(manifest):
        relative = "docs/execution/unregistered_authority.yaml"
        target = extra_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("unregistered: true\n", encoding="utf-8")
        manifest["files"].append(
            {
                "path": relative,
                "role": "ACTIVE_CONTRACT",
                "sha256": validator.sha256_file(target),
            }
        )

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        extra_root,
        append_extra_path,
    )
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "REGISTRY_MANIFEST_CLOSURE" in codes
    assert "REGISTRY_MANIFEST_COVERAGE" in codes

    def add_top_level_key(manifest):
        manifest["unregistered_metadata"] = "bypass"

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        tmp_path / "top_key",
        add_top_level_key,
    )
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "REGISTRY_MANIFEST_CLOSURE" in codes

    def add_entry_key(manifest):
        manifest["files"][0]["qualification_override"] = True

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        tmp_path / "entry_key",
        add_entry_key,
    )
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "REGISTRY_MANIFEST_CLOSURE" in codes

    def confuse_sealed_contact_type(manifest):
        manifest["sealed_contact"] = 0

    codes = _validate_manifest_mutation(
        validator,
        repo_root,
        tmp_path / "sealed_contact_int",
        confuse_sealed_contact_type,
    )
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "REGISTRY_MANIFEST_METADATA" in codes


def test_registry_manifest_cannot_predate_the_a1_interim_it_hashes(
    validator,
    repo_root,
    tmp_path,
):
    manifest = _copy_manifest_bundle(validator, repo_root, tmp_path)
    manifest["generated_at"] = "2026-08-10T10:10:05+08:00"
    manifest["updated_at"] = "2026-08-10T10:10:05+08:00"
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_registry_manifest(tmp_path))
    assert "REGISTRY_MANIFEST_TIME" in codes


def test_a6_cpu_partial_registration_is_closed(validator, repo_root):
    _, _, registries = validator.load_bundle_documents(repo_root)
    assert validator.validate_a6_cpu_exact_registration(repo_root, registries) == []

    interim = validator._load_yaml(repo_root, validator.A6_INTERIM_PATH)
    assert interim["record_status"] == "INTERIM_IN_PROGRESS_NOT_PHASE_COMPLETE"
    assert interim["run_state"]["run_status"] == "PASS"
    assert interim["phase_state"] == {
        "evidence_status": "IN_PROGRESS",
        "phase_complete": False,
    }
    assert interim["task_states"]["EXACT_GUIDANCE_TOY_GRAPH"] == {
        "evidence_status": "PASS",
        "result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "scope": "SYNTHETIC_TIME_HOMOGENEOUS_CPU_EXACT",
    }
    assert interim["task_states"]["FLOW_BASE_LEGAL_CTMC"] == {
        "evidence_status": "IN_PROGRESS",
        "result": "DEVELOPMENT_CPU_NONLEARNED_GILLESPIE_REPLAY_PARTIAL_PASS",
        "scope": "SYNTHETIC_NONLEARNED_CPU_GILLESPIE_BASE_RECOVERY",
        "formal_task_pass_asserted": False,
    }
    assert interim["claim_state"] == {
        "claim_id": "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW",
        "evidence_status": "IN_PROGRESS",
        "claim_status": "NOT_ESTABLISHED",
    }
    assert interim["boundaries"] == {
        "a6_pass_asserted": False,
        "formal_flow_base_task_pass_asserted": False,
        "l3_claim_established": False,
        "a7_evidence_status": "NOT_RUN",
        "a7_unlock": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "ordinary_data_read": False,
        "private_payload_access_allowed": False,
        "sealed_contact_allowed": False,
    }

    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    manifest_paths = {row["path"] for row in manifest["files"]}
    assert {
        validator.A6_INTERIM_PATH,
        *validator.A6_STATIC_PRODUCER_LEAF_SHA256,
        *validator.A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256,
    } <= manifest_paths
    assert validator.A6_REPORT_PATH not in manifest_paths
    assert validator.A6_GILLESPIE_REPORT_PATH not in manifest_paths
    assert not any(
        path.endswith("/RUN_MANIFEST.json") or path.endswith("/EVENT_LOG.jsonl")
        for path in manifest_paths
    )

    task_registry = registries["task"]
    a6_phase = next(
        row for row in task_registry["phase_tasks"] if row["phase_id"] == "A6"
    )
    assert a6_phase["evidence_status"] == "NOT_RUN"
    for task_id in ("EXACT_GUIDANCE_TOY_GRAPH", "FLOW_BASE_LEGAL_CTMC"):
        task = next(row for row in task_registry["tasks"] if row["task_id"] == task_id)
        assert task["evidence_status"] == "NOT_RUN"
        assert task["claim_status"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize(
    ("field_path", "promoted_value"),
    [
        (("record_status",), "PASS"),
        (("task_states", "FLOW_BASE_LEGAL_CTMC", "formal_task_pass_asserted"), True),
        (("claim_state", "claim_status"), "ESTABLISHED"),
        (("boundaries", "a7_unlock"), True),
        (("boundaries", "training_allowed"), True),
    ],
)
def test_a6_interim_rejects_partial_evidence_promotion(
    validator,
    repo_root,
    tmp_path,
    monkeypatch,
    field_path,
    promoted_value,
):
    case_root = tmp_path / "a6_partial_promotion"
    _copy_manifest_bundle(validator, repo_root, case_root)
    interim_path = case_root / validator.A6_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    target = interim
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = promoted_value
    interim_path.write_text(yaml.safe_dump(interim, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "EXPECTED_A6_INTERIM_SHA256",
        validator.sha256_file(interim_path),
    )

    _, _, registries = validator.load_bundle_documents(case_root)
    codes = _codes(
        validator.validate_a6_cpu_exact_registration(case_root, registries)
    )
    assert "A6_INTERIM_CANONICAL_HASH" not in codes
    assert "A6_INTERIM_SEMANTICS" in codes


def test_a6_claim_cell_cannot_establish_l3(validator, repo_root):
    _, _, registries = validator.load_bundle_documents(repo_root)
    mutated = deepcopy(registries)
    l3_claim = next(
        row
        for row in mutated["claim"]["claims"]
        if row["claim_id"] == "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW"
    )
    l3_claim["evidence_status"] = "PASS"
    l3_claim["claim_status"] = "ESTABLISHED"
    l3_claim["evidence_cells"][1]["establishes_formal_task_pass"] = True
    l3_claim["evidence_cells"][1]["establishes_a6_phase_pass"] = True
    l3_claim["evidence_cells"][1]["establishes_l3_claim"] = True
    l3_claim["evidence_cells"][1]["unlocks_a7"] = True

    codes = _codes(
        validator.validate_a6_cpu_exact_registration(repo_root, mutated)
    )
    assert "A6_CLAIM_CELL" in codes


def test_a6_gillespie_exact3_synchronized_manifest_rehash_is_rejected(
    validator,
    repo_root,
    tmp_path,
):
    case_root = tmp_path / "a6_exact3_synchronized_rehash"
    manifest = _copy_manifest_bundle(validator, repo_root, case_root)
    relative = validator.A6_GILLESPIE_PRODUCER_PATH
    producer_path = case_root / relative
    producer_path.write_bytes(producer_path.read_bytes() + b"\n# synchronized drift\n")
    next(row for row in manifest["files"] if row["path"] == relative)[
        "sha256"
    ] = validator.sha256_file(producer_path)
    manifest_path = case_root / validator.REGISTRY_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    codes = _codes(validator.validate_bundle(case_root))
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" not in codes
    assert "A6_STATIC_LEAF_DRIFT" in codes


def test_a6_registration_preserves_dec020_history_and_rebinds_only_current_claim(
    validator,
):
    claim_path = validator.REGISTRY_PATHS["claim"]
    assert (
        validator.DEC020_FROZEN_AUTHORITY_LEAF_SHA256[claim_path]
        == "9f5226ac78dd6c3848ba5ceb42742918de66ec459f951bb845ccaf21958a88f9"
    )
    assert (
        validator.DEC020_ACTIVE_AUTHORITY_LEAF_SHA256[claim_path]
        == "214279390d09c2857735c9cfa041ce38c45a7542142d4b0941ad7c035a7ee81a"
    )
    assert set(validator.DEC020_FROZEN_AUTHORITY_LEAF_SHA256) == set(
        validator.DEC020_ACTIVE_AUTHORITY_LEAF_SHA256
    )
    changed = {
        path
        for path in validator.DEC020_FROZEN_AUTHORITY_LEAF_SHA256
        if validator.DEC020_FROZEN_AUTHORITY_LEAF_SHA256[path]
        != validator.DEC020_ACTIVE_AUTHORITY_LEAF_SHA256[path]
    }
    assert changed == {claim_path}
    assert len(set(validator.DEC020_AUTHORITY_COMMIT_EXACT_CHANGED_PATHS)) == 14
