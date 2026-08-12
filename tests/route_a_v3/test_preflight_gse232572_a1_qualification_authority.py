from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse232572_a1_qualification_authority_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse232572_a1_qualification_authority.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gse232572_a1_qualification_authority_preflight", MODULE_PATH
)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _disk_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _synthetic_i_protocol(
    disk_protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    protocol = copy.deepcopy(disk_protocol or _disk_protocol())
    binding = protocol["implementation_binding"]
    for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS:
        binding[field] = PREFLIGHT.UNKNOWN
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _protocol() -> dict[str, object]:
    return _synthetic_i_protocol()


def _bound_protocol() -> dict[str, object]:
    protocol = _synthetic_i_protocol()
    binding = protocol["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _write_protocol(root: Path, protocol: object) -> Path:
    path = root / "configs" / PREFLIGHT.PROTOCOL_BASENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture_binding(
    protocol: object, protocol_path: Path, payload: bytes, repo_root: Path
) -> dict[str, str]:
    del protocol_path, payload, repo_root
    assert protocol["implementation_binding"]["status"] == PREFLIGHT.BOUND
    return {
        "status": "TEST_FIXTURE_BOUND_CONFIG_ONLY_LIFECYCLE_VERIFIED",
        "base_commit": "0" * 40,
        "implementation_commit": "1" * 40,
        "binding_commit": "2" * 40,
    }


def _fixture_authorities(
    protocol: dict[str, object], repo_root: Path
) -> list[dict[str, str]]:
    del repo_root
    return [
        {
            "path": item["path"],
            "role": item["role"],
            "sha256": item["sha256"],
        }
        for item in protocol["authority_inputs"]
    ]


def _materialization_report() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "dataset_id": "GSE232572",
        "study_id": "GSE232572",
        "status": "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED",
        "scientific_disposition": (
            "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
        ),
        "published_universe_row_count": 11929,
        "accepted_pair_complete_raw_endpoint_count": 8068,
        "accepted_pair_incomplete_raw_endpoint_count": 0,
        "rejected_published_row_count": 3861,
        "schema_valid_development_record_count": 8068,
        "canonical_record_count": 0,
        "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
        "license_boundary": {
            "private_derivative_use_fact": "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED",
            "public_redistribution_status": (
                "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
            ),
            "redistribution_allowed": False,
            "row_license_status": "UNKNOWN_BLOCKED",
            "verified_at_semantics": (
                "PUBLIC_RECOVERY_REPORT_RECORDED_AT_IS_STATUS_OBSERVATION_"
                "NOT_A_LICENSE_GRANT"
            ),
        },
        "rejection_reason_counts": {
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
            "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        },
    }


def _fixture_materialization(
    protocol: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    identity = protocol["materialization_report_identity"]
    return _materialization_report(), {
        "path": identity["absolute_path"],
        "role": identity["role"],
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def _execute_fixture(tmp_path: Path, *, output_name: str = "out") -> dict[str, object]:
    protocol_path = _write_protocol(tmp_path / "repo", _bound_protocol())
    return PREFLIGHT.execute(
        protocol_path,
        tmp_path / output_name,
        repo_root=tmp_path / "repo",
        binding_auditor=_fixture_binding,
        authority_auditor=_fixture_authorities,
        materialization_loader=_fixture_materialization,
        recorded_at="2026-08-13T00:00:00Z",
    )


def test_protocol_freezes_exact3_unknown_i_and_exact_public_inputs() -> None:
    protocol = _protocol()
    binding = protocol["implementation_binding"]
    assert binding["base_commit"] == PREFLIGHT.FROZEN_BASE_COMMIT
    assert [binding[field] for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS] == [
        PREFLIGHT.UNKNOWN
    ] * 4
    assert tuple(binding["implementation_commit_exact_changed_paths"]) == (
        PREFLIGHT.EXPECTED_EXACT3
    )
    assert binding["binding_commit_exact_changed_paths"] == [
        PREFLIGHT.EXPECTED_EXACT3[0]
    ]
    assert len(protocol["authority_inputs"]) == 5
    report_identity = protocol["materialization_report_identity"]
    assert report_identity["bytes"] == 3601
    assert report_identity["sha256"] == PREFLIGHT.EXPECTED_REPORT_SHA256
    assert protocol["registered_public_ledger_facts"]["runtime_event_id"] == (
        "A1-EVT-048"
    )
    assert protocol["registered_public_ledger_facts"][
        "runtime_output_count_after"
    ] == 203


def test_disk_i_or_b_normalizes_only_the_four_binding_scalars(tmp_path: Path) -> None:
    disk_protocol = _disk_protocol()
    synthetic_i = _synthetic_i_protocol(disk_protocol)
    assert PREFLIGHT._normalise_binding(disk_protocol) == synthetic_i
    assert [
        synthetic_i["implementation_binding"][field]
        for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS
    ] == [PREFLIGHT.UNKNOWN] * 4

    temporary_b_path = _write_protocol(tmp_path / "repo", _bound_protocol())
    temporary_b = _disk_protocol(temporary_b_path)
    temporary_i = _synthetic_i_protocol(temporary_b)
    assert temporary_b["implementation_binding"]["status"] == PREFLIGHT.BOUND
    assert PREFLIGHT._normalise_binding(temporary_b) == temporary_i
    assert [
        temporary_i["implementation_binding"][field]
        for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS
    ] == [PREFLIGHT.UNKNOWN] * 4


def test_unknown_i_stops_before_authority_report_or_output_io(tmp_path: Path) -> None:
    calls = {"authority": 0, "report": 0}

    def forbidden_authority(*args: object) -> list[dict[str, str]]:
        calls["authority"] += 1
        raise AssertionError("authority I/O must not occur")

    def forbidden_report(*args: object) -> tuple[dict[str, object], dict[str, object]]:
        calls["report"] += 1
        raise AssertionError("report I/O must not occur")

    repo_root = tmp_path / "repo"
    protocol_path = _write_protocol(repo_root, _synthetic_i_protocol())
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="config-only-B"):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=repo_root,
            authority_auditor=forbidden_authority,
            materialization_loader=forbidden_report,
        )
    assert calls == {"authority": 0, "report": 0}
    assert not output_dir.exists()


def test_bound_preflight_emits_one_public_blocked_aggregate(tmp_path: Path) -> None:
    report = _execute_fixture(tmp_path)
    output_dir = tmp_path / "out"
    assert [path.name for path in output_dir.iterdir()] == [
        PREFLIGHT.REPORT_FILENAME
    ]
    assert report["overall_decision"] == "BLOCKED_MISSING_EXTERNAL_AUTHORITY"
    assert report["terminal_status"] == (
        "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION"
    )
    assert report["registered_aggregate_pass_count"] == 3
    assert [item["gate_id"] for item in report["registered_aggregate_passes"]] == (
        list(PREFLIGHT.EXPECTED_PASS_IDS)
    )
    assert {item["status"] for item in report["registered_aggregate_passes"]} == {
        "PASS_FROM_REGISTERED_AGGREGATE"
    }
    assert report["open_qualification_blocker_count"] == 12
    assert [
        (item["gate_id"], item["status"])
        for item in report["qualification_blockers"]
    ] == list(PREFLIGHT.EXPECTED_BLOCKERS)
    assert report["sole_next_action"].startswith("OBTAIN_AND_USER_APPROVE")


def test_schema_valid_development_is_not_qualification_or_credit(tmp_path: Path) -> None:
    report = _execute_fixture(tmp_path)
    evidence = report["registered_aggregate_evidence"]
    assert evidence["schema_valid_development_record_count"] == 8068
    assert evidence["scientific_disposition"] == (
        "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
    )
    assert evidence["canonical_record_count"] == 0
    assert evidence["contribution"] == {"ordinary": 0, "a1": 0, "true_a2": 0}
    assert report["terminal_truth"] == {
        "qualified": False,
        "schema_valid_development_is_qualification": False,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    assert report["future_contribution_boundary"] == {
        "maximum_if_fully_qualified": {"ordinary": 1, "a1": 1, "true_a2": 0},
        "authorization_status": "NOT_AUTHORIZED",
        "current_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
    }


def test_exposure_rights_uncertainty_and_outer_gate_boundaries_are_explicit(
    tmp_path: Path,
) -> None:
    report = _execute_fixture(tmp_path)
    assert report["exposure_boundary"] == {
        "sequence_exposure": "SEQUENCE_EXPOSED",
        "label_exposure": "LABEL_EXPOSED",
        "untouched_confirmatory": False,
        "checkpoint_specific_exposure": PREFLIGHT.UNKNOWN,
    }
    assert report["rights_boundary"]["public_redistribution_status"] == (
        "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
    )
    assert report["rights_boundary"][
        "private_derivative_use_is_qualification_use_grant"
    ] is False
    assert report["rights_boundary"]["recommended_future_scope"] == (
        "PRIVATE_CANONICAL_ONLY"
    )
    assert report["rights_boundary"]["recommended_future_scope_approved"] is False
    assert report["historical_outer_recovery_boundary"] == {
        "gate_summary": {"PASS": 7, PREFLIGHT.UNKNOWN: 1},
        "is_dataset_qualification": False,
        "unknown_is_the_only_qualification_blocker": False,
    }


def test_pair_count_does_not_establish_candidate_pools_and_se_is_not_closed(
    tmp_path: Path,
) -> None:
    report = _execute_fixture(tmp_path)
    blockers = {item["gate_id"]: item for item in report["qualification_blockers"]}
    assert blockers["ELIGIBLE_MULTI_CANDIDATE_POOLS"] == {
        "gate_id": "ELIGIBLE_MULTI_CANDIDATE_POOLS",
        "status": PREFLIGHT.UNKNOWN,
        "accepted_pair_count_establishes_at_least_three_candidate_pools": False,
    }
    assert blockers["REPLICATE_OR_VALID_STANDARD_ERROR"] == {
        "gate_id": "REPLICATE_OR_VALID_STANDARD_ERROR",
        "status": "NOT_CLOSED",
        "public_replicate_count": 3,
        "primary_label_standard_error": None,
    }
    assert report["required_qualification_report_fields"][
        "accepted_pair_count_is_not_candidate_pool_eligibility"
    ] is True


def test_missing_or_mutated_authority_stops_before_output(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    protocol_path = _write_protocol(repo_root, _bound_protocol())
    output_dir = tmp_path / "out"
    with pytest.raises(PREFLIGHT.AuthorityError, match="cannot read"):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=repo_root,
            binding_auditor=_fixture_binding,
            materialization_loader=_fixture_materialization,
        )
    assert not output_dir.exists()

    first_authority = repo_root / PREFLIGHT.EXPECTED_AUTHORITIES[0][0]
    first_authority.parent.mkdir(parents=True, exist_ok=True)
    first_authority.write_text("mutated authority\n", encoding="utf-8")
    with pytest.raises(PREFLIGHT.AuthorityError, match="identity drifted"):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=repo_root,
            binding_auditor=_fixture_binding,
            materialization_loader=_fixture_materialization,
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("published_universe_row_count", 11928),
        ("schema_valid_development_record_count", 8067),
        ("canonical_record_count", 1),
        ("qualified", True),
    ],
)
def test_mutated_materialization_exact_fact_stops_before_output(
    tmp_path: Path, field: str, value: object
) -> None:
    protocol_path = _write_protocol(tmp_path / "repo", _bound_protocol())

    def mutated_loader(
        protocol: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        report, identity = _fixture_materialization(protocol)
        report[field] = value
        return report, identity

    output_dir = tmp_path / "out"
    with pytest.raises(PREFLIGHT.MaterializationReportError, match="fact drifted"):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=tmp_path / "repo",
            binding_auditor=_fixture_binding,
            authority_auditor=_fixture_authorities,
            materialization_loader=mutated_loader,
        )
    assert not output_dir.exists()


def test_missing_materialization_report_stops_before_output(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path / "repo", _bound_protocol())

    def missing_loader(
        protocol: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        del protocol
        raise PREFLIGHT.MaterializationReportError("registered report is missing")

    output_dir = tmp_path / "out"
    with pytest.raises(PREFLIGHT.MaterializationReportError, match="missing"):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=tmp_path / "repo",
            binding_auditor=_fixture_binding,
            authority_auditor=_fixture_authorities,
            materialization_loader=missing_loader,
        )
    assert not output_dir.exists()


def test_lifecycle_accepts_only_coherent_unknown_i_or_bound_b() -> None:
    unknown = _protocol()
    bound = _bound_protocol()
    assert PREFLIGHT._normalise_binding(bound) == PREFLIGHT._normalise_binding(
        unknown
    )

    mixed = copy.deepcopy(unknown)
    mixed["implementation_binding"]["status"] = PREFLIGHT.BOUND
    with pytest.raises(PREFLIGHT.ProtocolError, match="neither UNKNOWN-I nor BOUND-B"):
        PREFLIGHT._validate_protocol(mixed)

    extra_change = copy.deepcopy(bound)
    extra_change["output_contract"]["qualified"] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match="qualified"):
        PREFLIGHT._validate_protocol(extra_change)


def test_production_binding_accepts_fixed_i1_i2_b2_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    i1_protocol = _synthetic_i_protocol()
    i1_payload = (json.dumps(i1_protocol, indent=2) + "\n").encode("utf-8")
    assert PREFLIGHT._sha256(i1_payload) == PREFLIGHT.FROZEN_I1_SHA256[
        PREFLIGHT.EXPECTED_EXACT3[0]
    ]

    i2_commit = "4" * 40
    b2_commit = "5" * 40
    i2_script_payload = MODULE_PATH.read_bytes()
    i2_test_payload = Path(__file__).read_bytes()
    b2_protocol = copy.deepcopy(i1_protocol)
    binding = b2_protocol["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = i2_commit
    binding["implementation_script_sha256"] = PREFLIGHT._sha256(i2_script_payload)
    binding["implementation_test_sha256"] = PREFLIGHT._sha256(i2_test_payload)
    protocol_path = _write_protocol(repo_root, b2_protocol)
    b2_payload = protocol_path.read_bytes()

    working_script_path = repo_root / PREFLIGHT.EXPECTED_EXACT3[1]
    working_test_path = repo_root / PREFLIGHT.EXPECTED_EXACT3[2]
    working_script_path.parent.mkdir(parents=True, exist_ok=True)
    working_test_path.parent.mkdir(parents=True, exist_ok=True)
    working_script_path.write_bytes(i2_script_payload)
    working_test_path.write_bytes(i2_test_payload)

    command_results = {
        ("rev-parse", "HEAD"): b2_commit,
        ("rev-parse", "@{upstream}"): b2_commit,
        ("status", "--porcelain=v1", "--untracked-files=no"): "",
        ("rev-parse", f"{b2_commit}^"): i2_commit,
        ("rev-parse", f"{i2_commit}^"): PREFLIGHT.FROZEN_I1_COMMIT,
        ("rev-parse", f"{PREFLIGHT.FROZEN_I1_COMMIT}^"): (
            PREFLIGHT.FROZEN_BASE_COMMIT
        ),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            PREFLIGHT.FROZEN_I1_COMMIT,
        ): "\n".join(PREFLIGHT.EXPECTED_EXACT3),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            i2_commit,
        ): "\n".join(PREFLIGHT.EXPECTED_I2_EXACT2),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            b2_commit,
        ): PREFLIGHT.EXPECTED_EXACT3[0],
    }

    def fake_run_git(root: Path, *args: str) -> str:
        assert root == repo_root
        return command_results[args]

    historical_i1_script = b"FROZEN_I1_SCRIPT_BLOB"
    historical_i1_test = b"FROZEN_I1_TEST_BLOB"
    blob_results = {
        (PREFLIGHT.FROZEN_I1_COMMIT, PREFLIGHT.EXPECTED_EXACT3[0]): i1_payload,
        (
            PREFLIGHT.FROZEN_I1_COMMIT,
            PREFLIGHT.EXPECTED_EXACT3[1],
        ): historical_i1_script,
        (
            PREFLIGHT.FROZEN_I1_COMMIT,
            PREFLIGHT.EXPECTED_EXACT3[2],
        ): historical_i1_test,
        (i2_commit, PREFLIGHT.EXPECTED_EXACT3[0]): i1_payload,
        (i2_commit, PREFLIGHT.EXPECTED_EXACT3[1]): i2_script_payload,
        (i2_commit, PREFLIGHT.EXPECTED_EXACT3[2]): i2_test_payload,
        (b2_commit, PREFLIGHT.EXPECTED_EXACT3[0]): b2_payload,
        (b2_commit, PREFLIGHT.EXPECTED_EXACT3[1]): i2_script_payload,
        (b2_commit, PREFLIGHT.EXPECTED_EXACT3[2]): i2_test_payload,
    }

    def fake_git_blob_bytes(root: Path, commit: str, path: str) -> bytes:
        assert root == repo_root
        return blob_results[(commit, path)]

    real_sha256 = PREFLIGHT._sha256

    def fake_sha256(payload: bytes) -> str:
        if payload == historical_i1_script:
            return PREFLIGHT.FROZEN_I1_SHA256[PREFLIGHT.EXPECTED_EXACT3[1]]
        if payload == historical_i1_test:
            return PREFLIGHT.FROZEN_I1_SHA256[PREFLIGHT.EXPECTED_EXACT3[2]]
        return real_sha256(payload)

    monkeypatch.setattr(PREFLIGHT, "__file__", str(working_script_path))
    monkeypatch.setattr(PREFLIGHT, "_run_git", fake_run_git)
    monkeypatch.setattr(PREFLIGHT, "_git_blob_bytes", fake_git_blob_bytes)
    monkeypatch.setattr(PREFLIGHT, "_sha256", fake_sha256)

    result = PREFLIGHT._default_binding_auditor(
        b2_protocol,
        protocol_path,
        b2_payload,
        repo_root,
    )
    assert result == {
        "status": "BOUND_I1_I2_CONFIG_ONLY_B2_LIFECYCLE_VERIFIED",
        "base_commit": PREFLIGHT.FROZEN_BASE_COMMIT,
        "initial_implementation_commit": PREFLIGHT.FROZEN_I1_COMMIT,
        "implementation_commit": i2_commit,
        "binding_commit": b2_commit,
    }


@pytest.mark.parametrize(
    ("failure_mode", "error_match"),
    [
        ("DIRTY_SCRIPT", "working .* hash differs from I2 binding"),
        ("HEAD_NOT_UPSTREAM", "HEAD differs from the configured upstream"),
    ],
)
def test_production_binding_failure_stops_before_downstream_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    error_match: str,
) -> None:
    repo_root = tmp_path / "repo"
    working_script_path = repo_root / PREFLIGHT.EXPECTED_EXACT3[1]
    working_test_path = repo_root / PREFLIGHT.EXPECTED_EXACT3[2]
    working_script_path.parent.mkdir(parents=True, exist_ok=True)
    working_test_path.parent.mkdir(parents=True, exist_ok=True)

    implementation_script_payload = MODULE_PATH.read_bytes()
    implementation_test_payload = Path(__file__).read_bytes()
    working_script_payload = implementation_script_payload
    if failure_mode == "DIRTY_SCRIPT":
        working_script_payload += b"# uncommitted change\n"
    working_script_path.write_bytes(working_script_payload)
    working_test_path.write_bytes(implementation_test_payload)

    protocol = copy.deepcopy(_protocol())
    binding = protocol["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = PREFLIGHT._sha256(
        implementation_script_payload
    )
    binding["implementation_test_sha256"] = PREFLIGHT._sha256(
        implementation_test_payload
    )
    protocol_path = _write_protocol(repo_root, protocol)
    PREFLIGHT._validate_protocol(protocol)

    head_commit = "2" * 40
    upstream_commit = head_commit if failure_mode == "DIRTY_SCRIPT" else "3" * 40

    def fake_run_git(root: Path, *args: str) -> str:
        assert root == repo_root
        if args == ("rev-parse", "HEAD"):
            return head_commit
        if args == ("rev-parse", "@{upstream}"):
            return upstream_commit
        if args == ("status", "--porcelain=v1", "--untracked-files=no"):
            assert failure_mode == "DIRTY_SCRIPT"
            return f" M {PREFLIGHT.EXPECTED_EXACT3[1]}"
        raise AssertionError(f"unexpected git call after expected failure: {args}")

    calls = {"authority": 0, "report": 0}

    def forbidden_authority(*args: object) -> list[dict[str, str]]:
        calls["authority"] += 1
        raise AssertionError("authority I/O must not occur")

    def forbidden_report(*args: object) -> tuple[dict[str, object], dict[str, object]]:
        calls["report"] += 1
        raise AssertionError("report I/O must not occur")

    monkeypatch.setattr(PREFLIGHT, "__file__", str(working_script_path))
    monkeypatch.setattr(PREFLIGHT, "_run_git", fake_run_git)
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.ProtocolError, match=error_match):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=repo_root,
            authority_auditor=forbidden_authority,
            materialization_loader=forbidden_report,
        )
    assert calls == {"authority": 0, "report": 0}
    assert not output_dir.exists()


def test_split_power_row_and_model_work_remain_zero(tmp_path: Path) -> None:
    report = _execute_fixture(tmp_path)
    scope = report["scope_attestation"]
    assert scope == {
        "aggregate_only": True,
        "repository_public_authority_file_count": 5,
        "public_aggregate_report_count": 1,
        "private_row_artifact_read_count": 0,
        "row_record_read_count": 0,
        "sequence_value_read_count": 0,
        "effect_value_read_count": 0,
        "split_run_count": 0,
        "leakage_run_count": 0,
        "power_run_count": 0,
        "qualifier_run_count": 0,
        "canonical_materialization_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
    }
    blocker_status = {
        item["gate_id"]: item["status"] for item in report["qualification_blockers"]
    }
    assert blocker_status[
        "A1_GROUP_SPLIT_NEAR_DUPLICATE_GRAPH_SALT_AND_ZERO_LEAKAGE"
    ] == "NOT_RUN"
    assert blocker_status[
        "PREFROZEN_GROUP_EFFECTIVE_N_POWER_AND_FULL_CI_WIDTH"
    ] == "NOT_RUN"


def test_output_does_not_publish_row_artifact_locator_or_payload_tokens(
    tmp_path: Path,
) -> None:
    _execute_fixture(tmp_path)
    serialized = (tmp_path / "out" / PREFLIGHT.REPORT_FILENAME).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "development_v3_records",
        ".jsonl",
        "source_sequence",
        "candidate_sequence",
        "28b3dc927d47de9af36109b206804f286dccedaa4864b5b74442d1b59dc069bd",
    ):
        assert forbidden not in serialized


def test_exclusive_output_directory_refuses_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    protocol_path = _write_protocol(tmp_path / "repo", _bound_protocol())
    with pytest.raises(PREFLIGHT.OutputError, match="already exists"):
        PREFLIGHT.execute(
            protocol_path,
            existing,
            repo_root=tmp_path / "repo",
            binding_auditor=_fixture_binding,
            authority_auditor=_fixture_authorities,
            materialization_loader=_fixture_materialization,
        )
    assert list(existing.iterdir()) == []
