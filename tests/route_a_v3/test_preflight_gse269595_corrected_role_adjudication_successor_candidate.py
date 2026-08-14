from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/preflight_gse269595_corrected_role_adjudication_successor_candidate.py"
)
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json"
)
REPORT_PATH = (
    STAGING_ROOT
    / "reports/GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR_AGGREGATE_RECOMPUTE_V1.json"
)


def _load_runner():
    name = "gse269595_corrected_successor_candidate_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(runner):
    config = runner.load_config(CONFIG_PATH)
    runner.validate_protocol(config)
    return config


def _bound_config(runner):
    config = deepcopy(_config(runner))
    bindings = config["bindings"]
    own = bindings["implementation"]
    own["status"] = runner.BOUND
    own["implementation_commit"] = "a" * 40
    own["implementation_script_sha256"] = "b" * 64
    own["implementation_test_sha256"] = "c" * 64
    runner.validate_protocol(config)
    runner._require_production_bindings(config)
    return config


def _report(runner):
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    runner.validate_aggregate_only(report)
    return report


def _gates(report):
    return {gate["gate_id"]: gate for gate in report["gates"]}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_exact3_freezes_all_predecessor_histories_and_only_own_remains_unknown():
    runner = _load_runner()
    config = _config(runner)
    assert config["required_gate_ids_exactly"] == list(runner.GATE_IDS)
    assert len(runner.GATE_IDS) == 13
    assert config["decision_id"] == "V3-DEC-027"
    assert config["document_status"] == (
        "DRAFT_CORRECTED_SUCCESSOR_NOT_ACTIVE_PROTOCOL"
    )
    assert config["baseline"] == {
        "remote_branch": "routea-v3-a1-20260810",
        "dec027_authority_head": runner.AUTHORITY_COMMIT,
        "dec027_authority_parent": runner.AUTHORITY_PARENT,
        "pre_dec027_projection_event": "A1-EVT-058",
        "bound_runtime_event": "A1-EVT-059",
        "gse217518_final_binding_commit": runner.GSE217_FINAL_B,
        "encsr854ruf_final_binding_commit": runner.ENCSR_FINAL_B,
        "current_projection_status": "SETTLED_RUNTIME_CURRENT_PROJECTION",
    }
    assert config["bindings"]["authority"]["status"] == runner.BOUND
    assert config["bindings"]["runtime"]["frozen_i1_commit"] == (
        runner.RUNTIME_I1_COMMIT
    )
    assert config["bindings"]["runtime"]["implementation_commit"] == (
        runner.RUNTIME_I2_COMMIT
    )
    assert config["bindings"]["runtime"]["binding_commit"] == (
        runner.RUNTIME_B_COMMIT
    )
    assert config["bindings"]["gse217518_predecessor"] == {
        "status": runner.BOUND,
        "append_only_history": list(runner.GSE217_HISTORY),
        "terminal_binding_commit": runner.GSE217_FINAL_B,
    }
    assert config["bindings"]["encsr854ruf_predecessor"] == {
        "status": runner.BOUND,
        "append_only_history": list(runner.ENCSR_HISTORY),
        "terminal_binding_commit": runner.ENCSR_FINAL_B,
    }
    gse232 = config["bindings"]["gse232572_predecessor"]
    assert gse232["status"] == runner.BOUND
    assert gse232["terminal_binding_commit"] == (
        "0f2c00868b6581edd9a429c7a8a67bb43f6b7776"
    )
    assert [
        (step["step"], step["commit"], step["expected_parent"])
        for step in gse232["append_only_history"]
    ] == [
        (
            "I1",
            "d3dcae4c6ef53c52e942bb511946b52b952d3c7f",
            runner.ENCSR_FINAL_B,
        ),
        (
            "B1",
            "0f2c00868b6581edd9a429c7a8a67bb43f6b7776",
            "d3dcae4c6ef53c52e942bb511946b52b952d3c7f",
        ),
    ]
    assert gse232["append_only_history"][0]["blob_sha256_by_path"] == {
        runner.GSE232_CONFIG_PATH: "98f1e888e7466230767fb020ba48745f82faf4f2a3cb5e5c3e343d13fcb9a118",
        runner.GSE232_SCRIPT_PATH: "132a7cf7ea9008f87e2d77a4ba51b2e0701f6ccec2b6e8782a6f645c26cdb466",
        runner.GSE232_TEST_PATH: "ea46cfbbdc3f8f0fd74149b862deabe3ce2c551c94f3a46edf891c1917dfca66",
    }
    assert gse232["append_only_history"][1]["blob_sha256_by_path"] == {
        runner.GSE232_CONFIG_PATH: "c21821027d8a6806ee98d07177aacf5b7de3b007f15bd3d27bc6a6410bac3aab",
        runner.GSE232_SCRIPT_PATH: "132a7cf7ea9008f87e2d77a4ba51b2e0701f6ccec2b6e8782a6f645c26cdb466",
        runner.GSE232_TEST_PATH: "ea46cfbbdc3f8f0fd74149b862deabe3ce2c551c94f3a46edf891c1917dfca66",
    }
    gse113 = config["bindings"]["gse113849_predecessor"]
    assert gse113["status"] == runner.BOUND
    assert gse113["terminal_binding_commit"] == (
        "6372ddcb4b006d587a40ce628f9e193324c28b17"
    )
    assert [
        (step["step"], step["commit"], step["expected_parent"])
        for step in gse113["append_only_history"]
    ] == [
        (
            "I1",
            "8dfca85f3311ede01f594662d13b126bc8e2fef2",
            "0f2c00868b6581edd9a429c7a8a67bb43f6b7776",
        ),
        (
            "B1",
            "6372ddcb4b006d587a40ce628f9e193324c28b17",
            "8dfca85f3311ede01f594662d13b126bc8e2fef2",
        ),
    ]
    assert gse113["append_only_history"][0]["blob_sha256_by_path"] == {
        runner.GSE113_CONFIG_PATH: "ce591264d42dfe67a92da0d9f2e548eeaff1e037140bcdcfbee05ba3111fa986",
        runner.GSE113_SCRIPT_PATH: "44b934b828c8fa78aa37588006e7205d0f6277e463ebfe3b3834bbdfd022e23c",
        runner.GSE113_TEST_PATH: "98630bc8e48c5a07e5f17cf60addbf4382b7e7c49f66ff81245e76062e8d0001",
    }
    assert gse113["append_only_history"][1]["blob_sha256_by_path"] == {
        runner.GSE113_CONFIG_PATH: "464bf2da3988d3bce0a9edf978ac8fd2d88f07598e17d11cae7898ca3645758d",
        runner.GSE113_SCRIPT_PATH: "44b934b828c8fa78aa37588006e7205d0f6277e463ebfe3b3834bbdfd022e23c",
        runner.GSE113_TEST_PATH: "98630bc8e48c5a07e5f17cf60addbf4382b7e7c49f66ff81245e76062e8d0001",
    }
    own = config["bindings"]["implementation"]
    assert own["status"] == runner.UNKNOWN
    assert {own[field] for field in own["unknown_to_bound_fields"]} == {
        runner.UNKNOWN
    }
    assert config["production_activation_rule"]["predecessor_order"] == [
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
        "gse232572_predecessor",
        "gse113849_predecessor",
    ]


def test_real_aggregate_report_has_corrected_eight_pass_three_blocked_two_fail():
    runner = _load_runner()
    report = _report(runner)
    assert report["result_status"] == (
        "STOP_CORRECTED_ROLE_ADJUDICATION_GATES_NOT_CLOSED"
    )
    assert report["gate_counts"] == {
        "PASS": 8,
        "BLOCKED": 3,
        "FAIL": 2,
        "TOTAL": 13,
    }
    assert report["all_required_gates_pass"] is False
    assert report["terminal_state"]["qualified"] is False
    assert report["terminal_state"]["contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }


def test_single_two_candidate_family_is_excluded_without_failing_dense_universe():
    runner = _load_runner()
    report = _report(runner)
    family = _gates(report)[runner.GATE_IDS[2]]
    assert family["status"] == "PASS"
    assert family["aggregate_evidence"] == {
        "all_source_families": 373,
        "eligible_dense_source_families": 372,
        "pairwise_only_excluded_families": 1,
        "missing_source_anchor_families": 0,
        "ambiguous_source_anchor_families": 0,
    }
    corrected = report["corrected_predecessor_dispositions"]
    assert corrected["single_two_candidate_family_fails_all_dense_families"] is False


def test_all_candidate_design_diffs_replay_and_schema_coverage_remains_pass():
    runner = _load_runner()
    report = _report(runner)
    gates = _gates(report)
    replay = gates[runner.GATE_IDS[4]]
    assert replay["status"] == "PASS"
    assert replay["aggregate_evidence"]["candidate_design_count"] == 3429
    assert replay["aggregate_evidence"]["replayable_candidate_design_count"] == 3429
    assert replay["aggregate_evidence"]["source_unanchored_candidate_count"] == 0
    assert replay["aggregate_evidence"]["invalid_construct_candidate_count"] == 0
    assert replay["aggregate_evidence"]["zero_edit_candidate_count"] == 0

    schema = gates[runner.GATE_IDS[7]]
    assert schema["status"] == "PASS"
    assert schema["aggregate_evidence"]["joined_members"] == 6113
    assert schema["aggregate_evidence"]["unmatched_rows"] == 0
    assert schema["aggregate_evidence"]["unseen_publisher_members"] == 0
    assert schema["aggregate_evidence"][
        "declared_multiplicity_mismatch_designs"
    ] == 5
    assert schema["aggregate_evidence"][
        "multiplicity_discrepancy_is_asset_coverage_failure"
    ] is False


def test_publisher_two_biological_replicates_are_preserved_but_se_stays_blocked():
    runner = _load_runner()
    report = _report(runner)
    replicate = _gates(report)[runner.GATE_IDS[6]]
    assert replicate["status"] == "BLOCKED"
    evidence = replicate["aggregate_evidence"]
    assert evidence["publisher_biological_replicates"] == 2
    assert evidence["paired_replicate_endpoint_groups"] == 51405
    assert evidence["candidate_valid_se_group_count"] == 31887
    assert evidence["publisher_reported_se_field_present"] is False
    assert evidence["exact_valid_se_audit_closed"] is False
    assert report["corrected_predecessor_dispositions"][
        "publisher_two_biological_replicates_hardcoded_absent"
    ] is False


def test_finite_endpoint_is_hard_and_nonfinite_rows_are_not_zero_filled():
    runner = _load_runner()
    report = _report(runner)
    gates = _gates(report)
    endpoint = gates[runner.GATE_IDS[5]]
    missing = gates[runner.GATE_IDS[8]]
    assert endpoint["status"] == "BLOCKED"
    assert endpoint["aggregate_evidence"]["formula_mismatch_rows"] == 0
    assert endpoint["aggregate_evidence"]["nonfinite_or_undefined_rows"] == 82908
    assert endpoint["aggregate_evidence"]["publisher_censor_policy_present"] is True
    assert endpoint["aggregate_evidence"][
        "exact_publisher_pooling_and_censor_replay_closed"
    ] is False
    assert missing["status"] == "FAIL"
    assert report["corrected_predecessor_dispositions"][
        "nonfinite_endpoint_treated_as_zero"
    ] is False


def test_a1_true_a2_xor_is_role_eligibility_only_and_exposure_still_fails():
    runner = _load_runner()
    report = _report(runner)
    role = report["mutually_exclusive_role_disposition"]
    assert role == {
        "a1_geometry_eligible": False,
        "true_a2_dense_measured_neighborhood_geometry_eligible": True,
        "recommended_role_if_later_independently_qualified": "TRUE_A2_ONLY",
        "role_assigned": False,
        "double_credit_allowed": False,
    }
    exposure = _gates(report)[runner.GATE_IDS[9]]
    assert exposure["status"] == "FAIL"
    assert exposure["aggregate_evidence"]["aparent_guided_candidate_design"] is True
    assert exposure["aggregate_evidence"][
        "aparent_and_measured_response_guided_locus_selection"
    ] is True
    assert exposure["aggregate_evidence"]["future_model_input_route_closed"] is False


def test_post_dedup_n_and_power_are_reachable_without_formal_qualification_run():
    runner = _load_runner()
    config = _config(runner)
    report = _report(runner)
    reachability = report["post_dedup_power_reachability"]
    assert reachability["effective_source_group_n"] == 363
    assert reachability["required_effective_n"] == 156
    assert reachability["planning_power"] == pytest.approx(0.9977590398119174)
    assert reachability["planning_full_ci_width"] == pytest.approx(
        0.19610459396615834
    )
    assert reachability["formal_qualification_power_run"] is False
    assert reachability["verdict"] == (
        "REACHABLE_FOR_PREFLIGHT_INFORMATION_GEOMETRY_NOT_FORMAL_QUALIFICATION"
    )

    policy = config["split_and_power_policy"]
    required = next(
        n
        for n in range(4, 1001)
        if runner.fisher_power(
            n, policy["alternative_spearman_rho"], policy["alpha_two_sided"]
        )
        >= policy["target_power_minimum"]
        and runner.fisher_ci_width(
            n, policy["alternative_spearman_rho"], policy["confidence_level"]
        )
        <= policy["maximum_full_ci_width"]
    )
    assert required == 156


def test_own_grouped_unknown_stops_before_asset_or_output_io(tmp_path: Path):
    runner = _load_runner()
    calls = {"git": 0, "asset": 0, "output": 0}

    def poison(name):
        def callback(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} crossed grouped-UNKNOWN barrier")

        return callback

    asset_dir = tmp_path / "must-not-be-read-assets"
    output = tmp_path / "must-not-exist"
    with mock.patch.object(
        runner, "_audit_repository_bindings", poison("git")
    ), mock.patch.object(
        runner, "inspect_official_public_assets", poison("asset")
    ), mock.patch.object(runner, "_write_report", poison("output")):
        with pytest.raises(runner.BindingNotFrozen, match="grouped UNKNOWN"):
            runner.execute_production(
                config_path=CONFIG_PATH,
                asset_dir=asset_dir,
                output_dir=output,
                recorded_at="2026-08-15T13:00:00+08:00",
            )
    assert calls == {"git": 0, "asset": 0, "output": 0}
    assert not asset_dir.exists()
    assert not output.exists()


def test_later_unknown_predecessor_also_stops_before_all_io(tmp_path: Path):
    runner = _load_runner()
    config = deepcopy(_config(runner))
    gse113 = config["bindings"]["gse113849_predecessor"]
    for field in runner.FUTURE_PREDECESSOR_FIELDS:
        gse113[field] = runner.UNKNOWN
    runner.validate_protocol(config)
    calls = {"git": 0, "asset": 0, "output": 0}

    def poison(name):
        def callback(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} crossed later grouped-UNKNOWN barrier")

        return callback

    with mock.patch.object(
        runner, "load_config", return_value=config
    ), mock.patch.object(
        runner, "_audit_repository_bindings", poison("git")
    ), mock.patch.object(
        runner, "inspect_official_public_assets", poison("asset")
    ), mock.patch.object(runner, "_write_report", poison("output")):
        with pytest.raises(runner.BindingNotFrozen, match="grouped UNKNOWN"):
            runner.execute_production(
                config_path=CONFIG_PATH,
                asset_dir=tmp_path / "must-not-be-read-assets",
                output_dir=tmp_path / "must-not-exist",
                recorded_at="2026-08-15T13:00:00+08:00",
            )
    assert calls == {"git": 0, "asset": 0, "output": 0}


def test_partial_predecessor_binding_is_rejected():
    runner = _load_runner()
    config = deepcopy(runner.load_config(CONFIG_PATH))
    gse232 = config["bindings"]["gse232572_predecessor"]
    gse232["status"] = runner.UNKNOWN
    gse232["terminal_binding_commit"] = runner.UNKNOWN
    with pytest.raises(runner.CandidateContractError, match="partially populated"):
        runner.validate_protocol(config)


def test_clean_normalised_implementation_i_and_legal_disk_b_are_accepted():
    runner = _load_runner()
    bound = _bound_config(runner)
    implementation_i = runner._normalise_own_binding(bound)
    runner.validate_protocol(implementation_i)
    assert {
        implementation_i["bindings"]["implementation"][field]
        for field in runner.OWN_BINDING_FIELDS
    } == {runner.UNKNOWN}
    with pytest.raises(runner.BindingNotFrozen, match="grouped UNKNOWN"):
        runner._require_production_bindings(implementation_i)
    runner._require_production_bindings(bound)


def test_single_production_entry_has_no_public_analysis_or_loader_bypass():
    runner = _load_runner()
    assert set(inspect.signature(runner.execute_production).parameters) == {
        "config_path",
        "asset_dir",
        "output_dir",
        "recorded_at",
    }
    assert not hasattr(runner, "candidate_recompute")
    assert not hasattr(runner, "execute_public_analysis")
    parser_actions = {action.dest for action in runner._parser()._actions}
    assert "asset_dir" in parser_actions
    assert "processed_mpra" not in parser_actions
    assert "publisher_table_s5" not in parser_actions
    assert "mode" not in parser_actions


def _repository_audit_fixture(runner, tmp_path: Path, *, stale_copy: bool):
    config = _bound_config(runner)
    repo_root = tmp_path / "repo"
    script_path = repo_root / runner.SCRIPT_REPO_PATH
    test_path = repo_root / runner.TEST_REPO_PATH
    config_path = repo_root / runner.CONFIG_REPO_PATH
    for path in (script_path, test_path, config_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(SCRIPT_PATH.read_bytes())
    test_path.write_bytes(Path(__file__).read_bytes())
    disk_config = deepcopy(config)
    disk_config["repository_authority"]["production_repo_root"] = str(repo_root)
    config_path.write_text(
        json.dumps(disk_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    implementation_bytes = (
        json.dumps(
            runner._normalise_own_binding(disk_config),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    head = "d" * 40
    own_i = "a" * 40
    verified = []

    def fake_run_git(_repo_root, *arguments):
        return {
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "@{upstream}"): head,
            ("rev-parse", "--abbrev-ref", "HEAD"): runner.PRODUCTION_BRANCH,
            ("rev-parse", "--abbrev-ref", "@{upstream}"): runner.PRODUCTION_UPSTREAM,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }[arguments]

    def fake_verify(_repo_root, **kwargs):
        verified.append(
            (
                kwargs["label"],
                kwargs["commit"],
                kwargs["expected_parent"],
                tuple(kwargs["expected_paths"]),
            )
        )

    def fake_blob(_repo_root, commit, path):
        if commit == own_i and path == runner.CONFIG_REPO_PATH:
            return implementation_bytes
        if commit == head and path == runner.CONFIG_REPO_PATH:
            return config_path.read_bytes()
        if commit == own_i and path == runner.SCRIPT_REPO_PATH:
            return script_path.read_bytes()
        if commit == own_i and path == runner.TEST_REPO_PATH:
            return test_path.read_bytes()
        raise AssertionError(f"unexpected blob request: {commit}:{path}")

    executing = SCRIPT_PATH if stale_copy else script_path
    with mock.patch.object(runner, "_run_git", fake_run_git), mock.patch.object(
        runner, "_live_origin_head", return_value=head
    ), mock.patch.object(
        runner, "_verify_frozen_commit", fake_verify
    ), mock.patch.object(
        runner, "_git_blob", fake_blob
    ), mock.patch.object(
        runner, "__file__", str(executing)
    ):
        if stale_copy:
            with pytest.raises(runner.CandidateContractError, match="stale copy"):
                runner._audit_repository_bindings(
                    disk_config, config_path, repo_root
                )
            return verified
        result = runner._audit_repository_bindings(disk_config, config_path, repo_root)
    assert result["binding_commit"] == head
    return verified


def test_legal_full_i_b_chain_audits_in_exact_order(tmp_path: Path):
    runner = _load_runner()
    verified = _repository_audit_fixture(runner, tmp_path, stale_copy=False)
    assert [item[0] for item in verified] == [
        "DEC027_AUTHORITY_A",
        "DEC027_RUNTIME_I1",
        "DEC027_RUNTIME_I2",
        "DEC027_RUNTIME_B2",
        "GSE217518_I1",
        "GSE217518_I2",
        "GSE217518_B2",
        "GSE217518_I3",
        "GSE217518_B3",
        "ENCSR854RUF_I1",
        "ENCSR854RUF_I2",
        "ENCSR854RUF_B2",
        "ENCSR854RUF_I3",
        "ENCSR854RUF_B3",
        "ENCSR854RUF_I4",
        "ENCSR854RUF_B4",
        "GSE232572_I1",
        "GSE232572_B1",
        "GSE113849_I1",
        "GSE113849_B1",
        "GSE269595_I",
        "GSE269595_B",
    ]
    assert verified[4][2] == runner.RUNTIME_B_COMMIT
    assert verified[9][2] == runner.GSE217_FINAL_B
    assert verified[16][2] == runner.ENCSR_FINAL_B
    assert verified[18][2] == "0f2c00868b6581edd9a429c7a8a67bb43f6b7776"
    assert verified[20][2] == "6372ddcb4b006d587a40ce628f9e193324c28b17"
    assert verified[21][2] == "a" * 40


def test_stale_script_copy_is_rejected_before_asset_read(tmp_path: Path):
    runner = _load_runner()
    assert len(_repository_audit_fixture(runner, tmp_path, stale_copy=True)) == 22


def test_both_asset_identities_are_checked_before_any_parse(tmp_path: Path):
    runner = _load_runner()
    config = _config(runner)
    reads = []
    parse_count = 0

    def fake_read(path, spec):
        reads.append(path.name)
        if len(reads) == 1:
            return b"first-identity-verified"
        raise runner.PublicAssetError("second public asset identity differs")

    def forbidden_parse(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        raise AssertionError("parse crossed the two-asset identity barrier")

    with mock.patch.object(
        runner, "_read_bound_public_asset", fake_read
    ), mock.patch.object(runner, "_parse_table_s5", forbidden_parse):
        with pytest.raises(runner.PublicAssetError, match="second public asset"):
            runner.inspect_official_public_assets(
                config,
                tmp_path / config["official_asset_contract"]["processed_mpra"]["filename"],
                tmp_path
                / config["official_asset_contract"]["publisher_table_s5"]["filename"],
            )
    assert len(reads) == 2
    assert parse_count == 0


def test_assay_context_is_validated_before_asset_read_or_parse(tmp_path: Path):
    runner = _load_runner()
    config = deepcopy(runner.load_config(CONFIG_PATH))
    config["publisher_facts"]["endpoint_direction"] = "REVERSED"
    read_count = 0

    def forbidden_read(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        raise AssertionError("asset read crossed context validation barrier")

    with mock.patch.object(runner, "_read_bound_public_asset", forbidden_read):
        with pytest.raises(
            runner.CandidateContractError, match="publisher assay and exposure context"
        ):
            runner.inspect_official_public_assets(
                config, tmp_path / "mpra", tmp_path / "table"
            )
    assert read_count == 0


def test_report_is_aggregate_only_and_geometry_drift_fails_closed():
    runner = _load_runner()
    config = _config(runner)
    report = _report(runner)
    assert set(_walk_keys(report)).isdisjoint(runner.FORBIDDEN_OUTPUT_KEYS)
    assert report["internal_access_attestation"] == {
        "ordinary_public_asset_read_count": 2,
        "private_or_sealed_asset_read_count": 0,
        "raw_fastq_or_sra_member_payload_read_count": 0,
        "persistent_member_level_intermediate_count": 0,
        "member_identifier_sequence_row_effect_se_or_split_output_count": 0,
        "split_assignment_execution_count": 0,
        "training_run_count": 0,
        "gpu_run_count": 0,
        "model_selection_count": 0,
    }
    observation = deepcopy(report["aggregate_geometry"])
    observation["source_family_geometry"]["eligible_dense_source_family_count"] -= 1
    with pytest.raises(
        runner.CandidateContractError, match="aggregate geometry differs"
    ):
        runner.evaluate_observation(config, observation)


def test_power_values_are_finite_and_claim_locks_remain_unchanged():
    runner = _load_runner()
    report = _report(runner)
    reachability = report["post_dedup_power_reachability"]
    assert math.isfinite(reachability["planning_power"])
    assert math.isfinite(reachability["planning_full_ci_width"])
    terminal = report["terminal_state"]
    assert terminal["training_allowed"] is False
    assert terminal["gpu_work_allowed"] is False
    assert terminal["model_selection_allowed"] is False
    assert terminal["a7_allowed"] is False
    assert terminal["next_phase_authorized"] is False
    assert terminal["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_single_aggregate_output_is_atomic_no_replace_and_idempotent(
    tmp_path: Path,
):
    runner = _load_runner()
    report = _report(runner)
    output_dir = tmp_path / "result"
    output_path = runner._write_report(output_dir, report)
    assert output_path.name == runner.REPORT_FILENAME
    assert list(output_dir.iterdir()) == [output_path]
    assert runner._write_report(output_dir, report) == output_path
    different = deepcopy(report)
    different["observed_at_utc"] = "2026-08-15T13:00:01+08:00"
    with pytest.raises(runner.OutputError, match="different report already exists"):
        runner._write_report(output_dir, different)
