"""Negative tests for decision, manifest, supersession and sealed AST drift."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy

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
    for index, relative in enumerate(sorted(validator.MANDATORY_REGISTRY_MANIFEST_PATHS)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"fixture-{index}-{relative}\n".encode("utf-8")
        path.write_bytes(data)
        entries.append({"path": relative, "role": "FIXTURE", "sha256": validator.sha256_bytes(data)})
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
        "active_amendment_decision_ids": ["V3-DEC-017"],
        "base_commit": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
        "manifest_status": "A1_SCHEME_A_AUTHORITY_REBIND",
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
