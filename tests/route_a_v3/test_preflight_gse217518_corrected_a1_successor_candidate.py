from __future__ import annotations

import csv
import gzip
import importlib.util
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path

import openpyxl
import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/preflight_gse217518_corrected_a1_successor_candidate.py"
)
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse217518_corrected_a1_successor_candidate_v1.json"
)


def _load_candidate():
    name = "gse217518_corrected_a1_successor_candidate_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(runner):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runner.validate_protocol(config)
    return config


def _gate_map(report):
    return {item["gate_id"]: item for item in report["gates"]}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_exact3_candidate_binds_dec027_a_and_evt059_but_keeps_own_four_unknown():
    runner = _load_candidate()
    config = _config(runner)
    assert config["required_gate_ids_exactly"] == list(runner.GATE_IDS)
    assert len(runner.GATE_IDS) == 11
    assert config["document_status"] == "DRAFT_CANDIDATE_NOT_ACTIVE_PROTOCOL"
    assert config["baseline"] == {
        "remote_branch": "routea-v3-a1-20260810",
        "dec027_authority_head": "3e0ad158a0b45b2f26ed82da3afe60667c712cd6",
        "dec027_authority_parent": "b1ca33d852bad111ff31b4f60493d8c43c63d1a3",
        "fresh_verified_clean_at_authority_freeze": True,
        "pre_dec027_projection_event": "A1-EVT-058",
        "dec027_runtime_event_expected_when_bound": "A1-EVT-059",
    }
    authority = config["bindings"]["authority"]
    assert authority["status"] == runner.BOUND
    assert authority["authority_commit"] == runner.AUTHORITY_COMMIT
    assert tuple(authority["authority_exact_changed_paths"]) == runner.AUTHORITY_EXACT12
    assert authority["authority_blob_sha256_by_path"] == runner.AUTHORITY_BLOBS
    runtime = config["bindings"]["runtime"]
    assert runtime["status"] == runner.BOUND
    assert runtime["runtime_event_id"] == "A1-EVT-059"
    assert runtime["implementation_commit"] == (
        "5d66e8dc83eb9966f7698ac0fc677f1b06af8ea6"
    )
    assert runtime["binding_commit"] == "e60956cf59cbddc0406c5d116fb9714906db36e1"
    assert runtime["binding_expected_parent"] == runtime["implementation_commit"]
    assert runtime["implementation_blob_sha256_by_path"] == {
        runner.RUNTIME_CONFIG_PATH: (
            "3d5af87e7512568ed663b211c24a8586eeb9f03936a397cf2d2ddaeb2a21f57b"
        ),
        runner.RUNTIME_SCRIPT_PATH: (
            "44dcda8897e747cfe363668ddc23d8dd9c53a7f3ffab692a1bb4e7cf738973ca"
        ),
        runner.RUNTIME_TEST_PATH: (
            "ff250d4f011d8526e9a4a7bf13049f1f47346faa1c7ea512cbf447a6fb59ba4a"
        ),
    }
    assert runtime["binding_blob_sha256_by_path"] == {
        runner.RUNTIME_CONFIG_PATH: (
            "e5c1f96ec57b220fd36ff4677deb37d6dc0be06e02f21af3837e17a51e91e5ee"
        ),
        runner.RUNTIME_SCRIPT_PATH: runtime["implementation_blob_sha256_by_path"][
            runner.RUNTIME_SCRIPT_PATH
        ],
        runner.RUNTIME_TEST_PATH: runtime["implementation_blob_sha256_by_path"][
            runner.RUNTIME_TEST_PATH
        ],
    }
    implementation = config["bindings"]["implementation"]
    assert implementation["status"] == runner.UNKNOWN
    assert {
        implementation[field] for field in implementation["unknown_to_bound_fields"]
    } == {runner.UNKNOWN}
    assert config["bindings"]["implementation"][
        "implementation_exact_changed_paths"
    ] == list(runner.EXACT3)
    assert config["bindings"]["implementation"]["unknown_to_bound_fields"] == list(
        runner.OWN_BINDING_FIELDS
    )


def test_current_public_aggregate_is_four_pass_four_blocked_three_not_run():
    runner = _load_candidate()
    config = _config(runner)
    report = runner.evaluate_observation(
        config, config["current_aggregate_observation"]
    )
    assert report["result_status"] == "STOP_CORRECTED_PREFLIGHT_GATES_NOT_CLOSED"
    assert report["gate_counts"] == {
        "PASS": 4,
        "BLOCKED": 4,
        "NOT_RUN": 3,
        "TOTAL": 11,
    }
    statuses = {item["gate_id"]: item["status"] for item in report["gates"]}
    assert {gate for gate, status in statuses.items() if status == "PASS"} == {
        runner.GATE_IDS[0],
        runner.GATE_IDS[3],
        runner.GATE_IDS[5],
        runner.GATE_IDS[6],
    }
    assert {gate for gate, status in statuses.items() if status == "BLOCKED"} == {
        runner.GATE_IDS[1],
        runner.GATE_IDS[2],
        runner.GATE_IDS[4],
        runner.GATE_IDS[7],
    }
    assert {gate for gate, status in statuses.items() if status == "NOT_RUN"} == {
        runner.GATE_IDS[8],
        runner.GATE_IDS[9],
        runner.GATE_IDS[10],
    }
    assert report["terminal_state"]["qualified"] is False
    assert report["terminal_state"]["contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }


def test_blanket_author_outlier_blocker_is_replaced_by_core_qc_scope_correction():
    runner = _load_candidate()
    config = _config(runner)
    report = runner.evaluate_observation(
        config, config["current_aggregate_observation"]
    )
    qc = _gate_map(report)[runner.GATE_IDS[5]]
    assert qc["status"] == "PASS"
    assert qc["reason_code"] == (
        "CORE_R2_MSE_QC_AND_MISSING_SELECTION_CLOSED_"
        "DOWNSTREAM_MOTIF_FILTER_NOT_APPLICABLE"
    )
    assert report["corrected_qc_disposition"] == {
        "core_half_life_qc": "R_SQUARED_GT_0_5_AND_MSE_LT_1",
        "downstream_reference_only_quantile_filter": (
            "NOT_APPLICABLE_TO_CORE_A1_PAIR_EFFECT"
        ),
        "blanket_author_qc_blocker_retained": False,
    }
    serialized = json.dumps(report, sort_keys=True)
    assert "AUTHOR_DEFINED_OUTLIER_POLICY_NOT_EXECUTABLE" not in serialized


def test_current_geometry_preserves_real_crosswalk_and_endpoint_denominators():
    runner = _load_candidate()
    config = _config(runner)
    report = runner.evaluate_observation(
        config, config["current_aggregate_observation"]
    )
    assert report["aggregate_geometry"] == {
        "supplement_rows": 5072,
        "supplement_region_counts": {"3UTR": 2580, "5UTR": 2492},
        "versioned_transcript_rows": 4337,
        "unique_versioned_transcripts": 1305,
        "complete_endpoint_rows_sh": 3678,
        "complete_endpoint_rows_hek": 3729,
        "complete_endpoint_rows_both": 2335,
        "complete_endpoint_rows_either": 5072,
        "syntactic_one_ref_one_mut_group_count": 5164,
        "allele_singleton_group_count": 2144,
        "post_dedup_independent_source_group_count": None,
    }
    crosswalk = _gate_map(report)[runner.GATE_IDS[1]]
    assert crosswalk["status"] == "BLOCKED"
    assert (
        crosswalk["aggregate_evidence"]["syntactic_one_ref_one_mut_group_count"] == 5164
    )
    assert crosswalk["aggregate_evidence"]["allele_singleton_group_count"] == 2144


def test_production_own_four_unknown_fails_before_git_asset_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _load_candidate()
    missing_asset_dir = tmp_path / "must-not-be-read"
    output_dir = tmp_path / "must-not-exist"
    calls = {"git": 0, "asset": 0, "output": 0}

    def poison(name):
        def callback(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} crossed grouped-UNKNOWN barrier")

        return callback

    monkeypatch.setattr(runner, "_audit_repository_bindings", poison("git"))
    monkeypatch.setattr(runner, "inspect_official_public_assets", poison("asset"))
    monkeypatch.setattr(runner, "write_report", poison("output"))
    with pytest.raises(runner.BindingNotFrozen, match="grouped UNKNOWN"):
        runner.execute(CONFIG_PATH, missing_asset_dir, output_dir)
    assert calls == {"git": 0, "asset": 0, "output": 0}
    assert not missing_asset_dir.exists()
    assert not output_dir.exists()


def _write_synthetic_public_assets(runner, config, asset_dir: Path):
    asset_dir.mkdir()
    contract = config["official_asset_contract"]
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = contract["supplement_sheet"]
    sheet.append(contract["supplement_headers"])
    sheet.append(
        [
            "synthetic_member_a",
            "NM_000001.1(SYN):c.*1A>G",
            "SYN",
            "chrSynthetic",
            1,
            1,
            "3'UTR",
            "A",
            "G",
            10.0,
            11.0,
            0.1,
            "No",
            None,
            None,
            None,
            None,
            0.5,
            0.5,
            0.1,
            0.2,
        ]
    )
    sheet.append(
        [
            "synthetic_member_b",
            "NM_000002.2(SYN):c.-1C>T",
            "SYN",
            "chrSynthetic",
            2,
            2,
            "5'UTR",
            "C",
            "T",
            None,
            None,
            None,
            None,
            12.0,
            13.0,
            0.2,
            "No",
            0.5,
            0.5,
            0.1,
            0.2,
        ]
    )
    workbook.save(asset_dir / contract["supplement_filename"])

    keys = {
        "3UTR": ["synthetic_group_a_ref", "synthetic_group_a_mut", "orphan_mut"],
        "5UTR": ["synthetic_group_b_ref", "synthetic_group_b_mut"],
    }
    for item in contract["processed_assets"]:
        with gzip.open(
            asset_dir / item["filename"],
            "wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + [f"measurement_{index}" for index in range(9)])
            for key in keys[item["region"]]:
                writer.writerow([key] + [float(index + 1) for index in range(9)])


def test_public_asset_reader_returns_only_aggregate_geometry(tmp_path: Path):
    runner = _load_candidate()
    config = _config(runner)
    asset_dir = tmp_path / "assets"
    _write_synthetic_public_assets(runner, config, asset_dir)
    observation = runner.inspect_official_public_assets(config, asset_dir)
    geometry = observation["asset_geometry"]
    assert geometry["supplement_rows"] == 2
    assert geometry["supplement_region_counts"] == {"3'UTR": 1, "5'UTR": 1}
    assert geometry["versioned_transcript_rows"] == 2
    assert geometry["complete_endpoint_rows_sh"] == 1
    assert geometry["complete_endpoint_rows_hek"] == 1
    assert geometry["complete_endpoint_rows_either"] == 2
    assert geometry["syntactic_one_ref_one_mut_group_count"] == 2
    assert geometry["allele_singleton_group_count"] == 1
    assert geometry["same_region_key_sets_identical_across_cell_lines"] is True
    assert geometry["cross_region_key_intersection_count"] == 0
    assert set(_walk_keys(observation)).isdisjoint(runner.FORBIDDEN_OUTPUT_KEYS)


def test_even_all_pass_candidate_only_requests_promotion_and_changes_no_state():
    runner = _load_candidate()
    config = _config(runner)
    observation = deepcopy(config["current_aggregate_observation"])
    observation["asset_geometry"]["allele_singleton_group_count"] = 0
    observation["crosswalk"][
        "shared_reference_and_supplement_to_raw_exact_crosswalk_closed"
    ] = True
    observation["construct_context"][
        "exact_boundary_aware_construct_replay_closed_for_all_rows"
    ] = True
    observation["replicate_and_se"][
        "valid_row_level_standard_error_replay_closed"
    ] = True
    observation["exposure"].update(
        {
            "internal_historical_analytic_record_count": 0,
            "internal_historical_checkpoint_training_row_count": 0,
            "successor_role_adjudication_closed": True,
        }
    )
    observation["split"].update(
        {
            "outcome_blind_source_group_split_readiness_closed": True,
            "near_duplicate_zero_leakage_audit_run": True,
            "split_assignment_execution_count": 0,
        }
    )
    observation["effective_n"].update(
        {
            "post_dedup_independent_source_group_count": 156,
            "audit_run": True,
        }
    )
    observation["power"].update({"formal_power_run": True, "full_ci_width_run": True})
    report = runner.evaluate_observation(config, observation)
    assert report["all_required_gates_pass"] is True
    assert report["result_status"] == (
        "ALL_ELEVEN_PREFLIGHT_GATES_PASS_PROMOTION_REQUEST_ONLY"
    )
    assert report["gate_counts"] == {
        "PASS": 11,
        "BLOCKED": 0,
        "NOT_RUN": 0,
        "TOTAL": 11,
    }
    assert report["terminal_state"]["qualified"] is False
    assert report["terminal_state"]["training_allowed"] is False
    assert report["terminal_state"]["gpu_work_allowed"] is False
    assert report["terminal_state"]["model_selection_allowed"] is False
    assert report["terminal_state"]["a7_allowed"] is False
    assert report["terminal_state"]["contribution"]["a1"] == 0


def test_report_contains_no_member_sequence_row_effect_se_or_split_payload():
    runner = _load_candidate()
    config = _config(runner)
    report = runner.evaluate_observation(
        config, config["current_aggregate_observation"]
    )
    assert set(_walk_keys(report)).isdisjoint(runner.FORBIDDEN_OUTPUT_KEYS)
    assert report["scope_attestation"]["member_identifier_output_count"] == 0
    assert report["scope_attestation"]["sequence_output_count"] == 0
    assert report["scope_attestation"]["row_effect_output_count"] == 0
    assert report["scope_attestation"]["row_standard_error_output_count"] == 0
    assert report["scope_attestation"]["split_assignment_output_count"] == 0


def _own_bound_config(runner):
    config = _config(runner)
    own = config["bindings"]["implementation"]
    own.update(
        {
            "status": runner.BOUND,
            "implementation_commit": "3" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
        }
    )
    runner.validate_protocol(config)
    return config


def test_gate_set_or_partial_runtime_or_own_binding_drift_fails_closed():
    runner = _load_candidate()
    config = _config(runner)
    wrong_gate = deepcopy(config)
    wrong_gate["required_gate_ids_exactly"][-1] = "ALTERNATE_POWER_GATE"
    with pytest.raises(runner.CandidateContractError, match="exact eleven"):
        runner.validate_protocol(wrong_gate)

    wrong_authority = deepcopy(config)
    wrong_authority["bindings"]["authority"]["authority_commit"] = "1" * 40
    with pytest.raises(runner.CandidateContractError, match="authority A"):
        runner.validate_protocol(wrong_authority)

    clean_runtime_unknown = deepcopy(config)
    runtime = clean_runtime_unknown["bindings"]["runtime"]
    for field in runtime["unknown_to_bound_fields"]:
        runtime[field] = runner.UNKNOWN
    runner.validate_protocol(clean_runtime_unknown)

    partial_runtime = deepcopy(clean_runtime_unknown)
    partial_runtime["bindings"]["runtime"]["implementation_commit"] = "1" * 40
    with pytest.raises(runner.CandidateContractError, match="partially bound"):
        runner.validate_protocol(partial_runtime)

    partial_own = deepcopy(config)
    partial_own["bindings"]["implementation"]["implementation_commit"] = "3" * 40
    with pytest.raises(runner.CandidateContractError, match="partially bound"):
        runner.validate_protocol(partial_own)


def test_repository_auditor_rejects_stale_executing_copy_and_freezes_full_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _load_candidate()
    config = _own_bound_config(runner)
    repo = tmp_path / "repo"
    config["repository_authority"]["production_repo_root"] = str(repo)
    config_path = repo / runner.CONFIG_REPO_PATH
    script_path = repo / runner.SCRIPT_REPO_PATH
    test_path = repo / runner.TEST_REPO_PATH
    for path in (config_path, script_path, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    script_path.write_bytes(b"stale copied producer")
    test_path.write_bytes(b"bound focused test")
    head = "4" * 40

    def fake_run_git(_repo, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == ("rev-parse", "@{upstream}"):
            return head
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return runner.PRODUCTION_BRANCH
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return runner.PRODUCTION_UPSTREAM
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    verified = []

    def fake_verify(_repo, **kwargs):
        verified.append(kwargs)

    implementation_protocol = runner._normalise_own_binding(config)

    def fake_blob(_repo, commit, path):
        if commit == "3" * 40 and path == runner.CONFIG_REPO_PATH:
            return json.dumps(implementation_protocol).encode("utf-8")
        if commit == head and path == runner.CONFIG_REPO_PATH:
            return config_path.read_bytes()
        if commit == "3" * 40 and path == runner.SCRIPT_REPO_PATH:
            return b"bound producer"
        if commit == "3" * 40 and path == runner.TEST_REPO_PATH:
            return b"bound focused test"
        raise AssertionError((commit, path))

    monkeypatch.setattr(runner, "_run_git", fake_run_git)
    monkeypatch.setattr(runner, "_live_origin_head", lambda *_: head)
    monkeypatch.setattr(runner, "_verify_frozen_commit", fake_verify)
    monkeypatch.setattr(runner, "_git_blob", fake_blob)
    monkeypatch.setattr(runner, "__file__", str(script_path))
    with pytest.raises(runner.CandidateContractError, match="differs from GSE217518 I"):
        runner._audit_repository_bindings(config, config_path, repo)
    assert [item["label"] for item in verified] == [
        "DEC027 authority A",
        "DEC027 runtime I",
        "DEC027 runtime B",
        "GSE217518 implementation I",
        "GSE217518 binding B",
    ]
    assert verified[1]["expected_parent"] == runner.AUTHORITY_COMMIT
    assert verified[2]["expected_parent"] == (
        "5d66e8dc83eb9966f7698ac0fc677f1b06af8ea6"
    )
    assert verified[3]["expected_parent"] == (
        "e60956cf59cbddc0406c5d116fb9714906db36e1"
    )
    assert verified[4]["expected_parent"] == "3" * 40
    assert tuple(verified[4]["expected_paths"]) == (runner.CONFIG_REPO_PATH,)


def test_atomic_fixed_name_publication_is_idempotent_and_never_replaces(
    tmp_path: Path,
):
    runner = _load_candidate()
    config = _config(runner)
    report = runner.evaluate_observation(
        config, config["current_aggregate_observation"]
    )
    output = tmp_path / "aggregate"
    first = runner.write_report(output, report)
    original = first.read_bytes()
    assert runner.write_report(output, report) == first
    different = deepcopy(report)
    different["result_status"] = "DIFFERENT_REPORT_FOR_NO_REPLACE_TEST"
    with pytest.raises(runner.OutputError, match="replacement refused"):
        runner.write_report(output, different)
    assert first.read_bytes() == original
    assert [path.name for path in output.iterdir()] == [runner.REPORT_FILENAME]


def test_atomic_publication_failure_removes_temp_partial_and_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _load_candidate()
    config = _config(runner)
    report = runner.evaluate_observation(
        config, config["current_aggregate_observation"]
    )
    output = tmp_path / "aggregate"
    real_fsync_directory = runner._fsync_directory
    fsync_calls = {"count": 0}

    def fail_after_link(path):
        fsync_calls["count"] += 1
        if fsync_calls["count"] == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(runner, "_fsync_directory", fail_after_link)
    with pytest.raises(runner.OutputError, match="atomically publish"):
        runner.write_report(output, report)
    assert not output.exists()


def test_main_has_no_public_asset_analysis_bypass():
    runner = _load_candidate()
    source = inspect.getsource(runner.main)
    assert "execute(args.config, args.asset_dir, args.output_dir)" in source
    assert "inspect_official_public_assets" not in source
    assert "evaluate_observation" not in source
