"""Negative tests for decision, manifest, supersession and sealed AST drift."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy

import pytest
import yaml


def _codes(issues):
    return {issue.code for issue in issues}


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
    assert preflight_binding["status"] == "UNKNOWN_NOT_ASSERTED"
    assert preflight_binding["implementation_commit"] == "UNKNOWN_NOT_ASSERTED"
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
    gse200304["blocking_requirements"].remove("REQUIRED_80S_ROLE_AUTHORITY_ABSENT")
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
        yaml.safe_dump({"updated_at": "2026-08-10T10:33:39+08:00"}),
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
        "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
        "base_commit": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
        "manifest_status": "A1_PLUMAGE_DEC018_MECHANICAL_REBIND",
        "initial_generated_at": "2026-08-10T10:10:05+08:00",
        "generated_at": "2026-08-10T10:41:50+08:00",
        "updated_at": "2026-08-10T10:41:50+08:00",
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
