from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / (
        "route_a_v3_gse261709_public_identifier_asset_schema_aggregate_"
        "geometry_preflight_v1.json"
    )
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse261709_public_identifier_asset_schema_aggregate_geometry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gse261709_public_identifier_asset_schema_geometry_preflight", MODULE_PATH
)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _i_protocol() -> dict[str, object]:
    protocol = copy.deepcopy(_protocol())
    binding = protocol["implementation_binding"]
    for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS:
        binding[field] = PREFLIGHT.UNKNOWN
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _bound_protocol() -> dict[str, object]:
    protocol = _i_protocol()
    binding = protocol["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = "2" * 40
    binding["implementation_script_sha256"] = "3" * 64
    binding["implementation_test_sha256"] = "4" * 64
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _fixture_binding(*args: object) -> dict[str, str]:
    return {
        "status": "TEST_FIXTURE_BOUND_WITHOUT_GIT",
        "authority_commit": PREFLIGHT.AUTHORITY_COMMIT,
        "authority_runtime_implementation_i1_commit": PREFLIGHT.RUNTIME_I1_COMMIT,
        "authority_runtime_implementation_i2_commit": PREFLIGHT.RUNTIME_I2_COMMIT,
        "authority_runtime_binding_commit": PREFLIGHT.RUNTIME_B2_COMMIT,
        "preflight_implementation_i1_commit": PREFLIGHT.PREFLIGHT_I1_COMMIT,
        "preflight_implementation_i2_commit": PREFLIGHT.PREFLIGHT_I2_COMMIT,
        "preflight_binding_b2_commit": PREFLIGHT.PREFLIGHT_B2_COMMIT,
        "gse207584_binding_b4_commit": PREFLIGHT.GSE207_B4_COMMIT,
        "implementation_commit": "2" * 40,
        "binding_commit": "3" * 40,
    }


class StaticFetcher:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.whole_response_transport_and_decode_count = 0

    def fetch_text(self, url: str) -> str:
        self.urls.append(url)
        self.whole_response_transport_and_decode_count += 1
        if url == PREFLIGHT.GEO_URL:
            return " ".join(
                [
                    "GSE261709 PRJNA1088465 38773080 Homo sapiens",
                    "massively parallel reporter assay in AGS and SNU719 in triplicate",
                    "Samples (7) Platforms (1)",
                    "AGS, rep1 AGS, rep2 AGS, rep3",
                    "SNU719, rep1 SNU719, rep2 SNU719, rep3 plasmid pool",
                    "GSE261709_RAW.tar 690.0 Kb Raw data are available in SRA",
                    (
                        "MEMBER_BARCODE_POISON MEMBER_VARIANT_POISON "
                        "MEMBER_TRANSCRIPT_POISON MEMBER_SEQUENCE_POISON "
                        "ROW_EFFECT_POISON ROW_SE_POISON"
                    ),
                ]
            )
        if url == PREFLIGHT.PUBMED_SUMMARY_URL:
            return " ".join(
                [
                    "PMC11109163 PMID 38773080",
                    "10.1038/s41467-024-48436-5",
                ]
            )
        raise AssertionError(f"unexpected URL {url}")


class BombFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_text(self, url: str) -> str:
        self.calls += 1
        raise AssertionError(f"network must not be reached: {url}")


def _live_observation() -> dict[str, object]:
    observation, results = PREFLIGHT.build_live_observation(
        _protocol(), StaticFetcher()
    )
    assert [result["status"] for result in results] == ["PASS", "PASS"]
    PREFLIGHT._validate_observation(observation)
    return observation


def _all_pass_observation(*, existing_aggregate: bool = False) -> dict[str, object]:
    observation = _live_observation()
    observation["aggregate_role_geometry"]["run_count"] = 7
    observation["aggregate_role_geometry"]["run_role_mapping_complete"] = True
    observation["schema_and_rights"].update(
        {
            "header_name_count": 14,
            "header_role_class_presence": {
                role_class: True
                for role_class in PREFLIGHT.REQUIRED_HEADER_ROLE_CLASSES
            },
            "dimension_measure_count": 4,
            "exact_dimensions_complete": True,
            "asset_license_notice_visible": True,
            "asset_license_notice_applies_to_row_level_research": True,
        }
    )
    if existing_aggregate:
        observation["scope"] = PREFLIGHT._zero_scope(
            whole_response_transport_and_decode_count=0,
            archive_listing_metadata_parsed_count=0,
        )
    PREFLIGHT._validate_observation(observation)
    return observation


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


def test_protocol_freezes_dec023_exact3_and_supports_disk_i3_or_b3() -> None:
    protocol = _protocol()
    binding = protocol["implementation_binding"]
    assert binding["authority_commit"] == PREFLIGHT.AUTHORITY_COMMIT
    assert binding["authority_runtime_binding_commit"] == PREFLIGHT.RUNTIME_B2_COMMIT
    runtime = binding["authority_runtime_lineage"]
    assert tuple(runtime["paths"]) == PREFLIGHT.RUNTIME_PATHS
    assert runtime["implementation_i1_commit"] == PREFLIGHT.RUNTIME_I1_COMMIT
    assert runtime["implementation_i2_commit"] == PREFLIGHT.RUNTIME_I2_COMMIT
    assert runtime["implementation_i1_blob_sha256_by_path"] == (
        PREFLIGHT.RUNTIME_I1_BLOB_SHA256_BY_PATH
    )
    assert runtime["implementation_i2_blob_sha256_by_path"] == (
        PREFLIGHT.RUNTIME_I2_BLOB_SHA256_BY_PATH
    )
    assert runtime["binding_b2_blob_sha256_by_path"] == (
        PREFLIGHT.RUNTIME_B2_BLOB_SHA256_BY_PATH
    )
    predecessor = binding["predecessor_preflight_i1"]
    assert predecessor == {
        "status": "FROZEN_BOUND_EXACT3",
        "commit": PREFLIGHT.PREFLIGHT_I1_COMMIT,
        "expected_parent": PREFLIGHT.RUNTIME_B2_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.EXPECTED_EXACT3),
        "blob_sha256_by_path": PREFLIGHT.PREFLIGHT_I1_BLOB_SHA256_BY_PATH,
    }
    predecessor_i2_b2 = binding["predecessor_preflight_i2_b2"]
    assert predecessor_i2_b2 == {
        "status": "FROZEN_BOUND_EXACT3_CONFIG_ONLY_BINDING",
        "implementation_commit": PREFLIGHT.PREFLIGHT_I2_COMMIT,
        "implementation_expected_parent": PREFLIGHT.PREFLIGHT_I1_COMMIT,
        "implementation_exact_changed_paths": list(PREFLIGHT.EXPECTED_EXACT3),
        "implementation_blob_sha256_by_path": (
            PREFLIGHT.PREFLIGHT_I2_BLOB_SHA256_BY_PATH
        ),
        "binding_commit": PREFLIGHT.PREFLIGHT_B2_COMMIT,
        "binding_expected_parent": PREFLIGHT.PREFLIGHT_I2_COMMIT,
        "binding_exact_changed_paths": [PREFLIGHT.CONFIG_PATH],
        "binding_blob_sha256_by_path": PREFLIGHT.PREFLIGHT_B2_BLOB_SHA256_BY_PATH,
    }
    intervening = binding["intervening_gse207584_preflight_lifecycle"]
    assert intervening["protocol_id"] == PREFLIGHT.GSE207_PROTOCOL_ID
    assert tuple(intervening["paths"]) == PREFLIGHT.GSE207_PATHS
    assert [
        intervening["implementation_i1_commit"],
        intervening["implementation_i2_commit"],
        intervening["binding_b2_commit"],
        intervening["implementation_i3_commit"],
        intervening["binding_b3_commit"],
        intervening["implementation_i4_commit"],
        intervening["binding_b4_commit"],
    ] == [
        PREFLIGHT.GSE207_I1_COMMIT,
        PREFLIGHT.GSE207_I2_COMMIT,
        PREFLIGHT.GSE207_B2_COMMIT,
        PREFLIGHT.GSE207_I3_COMMIT,
        PREFLIGHT.GSE207_B3_COMMIT,
        PREFLIGHT.GSE207_I4_COMMIT,
        PREFLIGHT.GSE207_B4_COMMIT,
    ]
    assert [
        intervening["implementation_i1_expected_parent"],
        intervening["implementation_i2_expected_parent"],
        intervening["binding_b2_expected_parent"],
        intervening["implementation_i3_expected_parent"],
        intervening["binding_b3_expected_parent"],
        intervening["implementation_i4_expected_parent"],
        intervening["binding_b4_expected_parent"],
    ] == [
        PREFLIGHT.PREFLIGHT_B2_COMMIT,
        PREFLIGHT.GSE207_I1_COMMIT,
        PREFLIGHT.GSE207_I2_COMMIT,
        PREFLIGHT.GSE207_B2_COMMIT,
        PREFLIGHT.GSE207_I3_COMMIT,
        PREFLIGHT.GSE207_B3_COMMIT,
        PREFLIGHT.GSE207_I4_COMMIT,
    ]
    assert intervening["implementation_i1_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_I1_BLOB_SHA256_BY_PATH
    )
    assert intervening["implementation_i2_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_I2_BLOB_SHA256_BY_PATH
    )
    assert intervening["binding_b2_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_B2_BLOB_SHA256_BY_PATH
    )
    assert intervening["implementation_i3_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_I3_BLOB_SHA256_BY_PATH
    )
    assert intervening["binding_b3_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_B3_BLOB_SHA256_BY_PATH
    )
    assert intervening["implementation_i4_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_I4_BLOB_SHA256_BY_PATH
    )
    assert intervening["binding_b4_blob_sha256_by_path"] == (
        PREFLIGHT.GSE207_B4_BLOB_SHA256_BY_PATH
    )
    assert binding["status"] in {PREFLIGHT.UNKNOWN, PREFLIGHT.BOUND}
    normalised = PREFLIGHT._normalise_binding(protocol)
    assert [
        normalised["implementation_binding"][field]
        for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS
    ] == [
        PREFLIGHT.UNKNOWN
    ] * 4
    PREFLIGHT._validate_protocol(normalised)
    assert tuple(binding["implementation_commit_exact_changed_paths"]) == (
        PREFLIGHT.EXPECTED_EXACT3
    )
    assert binding["binding_commit_exact_changed_paths"] == [
        PREFLIGHT.CONFIG_PATH
    ]
    assert protocol["decision_authority"]["authorized_role"] == (
        "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY"
    )
    assert tuple(protocol["gate_contract"]["gate_ids_exactly"]) == (
        PREFLIGHT.EXPECTED_GATE_IDS
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("authority_commit", "0" * 40, "authority commit differs"),
        (
            "authority_runtime_binding_commit",
            "1" * 40,
            "authority-runtime B2 commit differs",
        ),
        ("implementation_commit", "2" * 40, "UNKNOWN group"),
    ],
)
def test_partial_binding_groups_are_rejected(
    field: str, value: str, message: str
) -> None:
    protocol = _i_protocol()
    protocol["implementation_binding"][field] = value
    with pytest.raises(PREFLIGHT.ProtocolError, match=message):
        PREFLIGHT._validate_protocol(protocol)


def test_runtime_lineage_identity_drift_is_rejected_without_disk_state() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["implementation_binding"]["authority_runtime_lineage"][
        "implementation_i2_commit"
    ] = "2" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="I2 commit differs"):
        PREFLIGHT._validate_protocol(protocol)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("input_contract", "archive_or_processed_asset_download_allowed"),
        ("input_contract", "archive_member_listing_or_open_allowed"),
        ("input_contract", "row_or_member_body_read_allowed"),
        ("input_contract", "sequence_value_allowed"),
        ("output_contract", "member_identifiers_included"),
        ("output_contract", "qualification_or_credit_included"),
    ],
)
def test_protocol_rejects_payload_or_promotion_flags(
    section: str, field: str
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][field] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match=field):
        PREFLIGHT._validate_protocol(protocol)


def test_unknown_binding_stops_before_network_or_output(tmp_path: Path) -> None:
    fetcher = BombFetcher()
    protocol_path = _write_json(tmp_path / PREFLIGHT.PROTOCOL_BASENAME, _i_protocol())
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="not BOUND"):
        PREFLIGHT.execute(protocol_path, output_dir, fetcher=fetcher)
    assert fetcher.calls == 0
    assert not output_dir.exists()


def test_live_metadata_only_reports_real_schema_rights_blockers_without_poison(
    tmp_path: Path,
) -> None:
    fetcher = StaticFetcher()
    output_dir = tmp_path / "output"
    report = PREFLIGHT.execute(
        PROTOCOL_PATH,
        output_dir,
        fetcher=fetcher,
        binding_auditor=_fixture_binding,
        recorded_at="2026-08-14T08:00:00Z",
    )
    assert fetcher.urls == list(PREFLIGHT.LIVE_URLS)
    assert fetcher.whole_response_transport_and_decode_count == 2
    assert report["status"] == "STOP_PREFLIGHT_GATES_NOT_CLOSED"
    assert report["gate_counts"] == {
        "PASS": 1,
        "BLOCKED": 2,
        "UNKNOWN_NOT_ASSERTED": 0,
    }
    assert {gate["gate_id"]: gate["status"] for gate in report["gates"]} == {
        "OFFICIAL_IDENTIFIER_AND_CONTEXT": "PASS",
        "ASSET_SAMPLE_AND_RUN_ROLE_AGGREGATE_GEOMETRY": "BLOCKED",
        "HEADER_DIMENSION_AND_ASSET_LICENSE_NOTICE": "BLOCKED",
    }
    scope = report["scope_attestation"]
    assert scope["whole_small_metadata_response_transport_and_decode_count"] == 2
    assert scope["archive_listing_metadata_parsed_count"] == 1
    assert scope["archive_endpoint_access_count"] == 0
    assert scope["member_endpoint_access_count"] == 0
    assert scope["payload_endpoint_access_count"] == 0
    assert scope["archive_download_count"] == 0
    assert scope["archive_member_listing_count"] == 0
    assert scope["archive_member_open_count"] == 0
    assert scope["forbidden_value_parsed_or_extracted_count"] == 0
    assert scope["forbidden_value_persistently_stored_count"] == 0
    assert scope["forbidden_value_output_count"] == 0
    assert "row_or_member_body_read_count" not in scope
    assert not any(key.endswith("_value_read_count") for key in scope)
    files = list(output_dir.iterdir())
    assert [path.name for path in files] == [PREFLIGHT.REPORT_FILENAME]
    serialized = files[0].read_text(encoding="utf-8")
    for poison in (
        "MEMBER_BARCODE_POISON",
        "MEMBER_VARIANT_POISON",
        "MEMBER_TRANSCRIPT_POISON",
        "MEMBER_SEQUENCE_POISON",
        "ROW_EFFECT_POISON",
        "ROW_SE_POISON",
        "GSM8149344",
    ):
        assert poison not in serialized
    assert "transports and decodes exactly two whole allowlisted" in report[
        "claim_boundary"
    ]


def test_all_three_pass_only_requests_separate_row_level_authority() -> None:
    report = PREFLIGHT.evaluate_observation(
        _protocol(),
        _all_pass_observation(),
        binding=_fixture_binding(),
        source_mode="EXISTING_ORDINARY_PUBLIC_METADATA_SCHEMA_AGGREGATE_ONLY",
        source_results=[],
        recorded_at="2026-08-14T08:00:00Z",
    )
    assert report["gate_counts"] == {
        "PASS": 3,
        "BLOCKED": 0,
        "UNKNOWN_NOT_ASSERTED": 0,
    }
    assert report["status"].endswith("ROW_LEVEL_AUTHORITY_REQUIRED")
    assert report["sole_next_action"] == (
        "GO_REQUEST_SEPARATE_ROW_LEVEL_QUALIFICATION_AUTHORITY"
    )
    assert report["terminal_truth"] == PREFLIGHT.EXPECTED_OUTER_TRUTH
    assert report["terminal_truth"]["gse261709_qualified"] is False
    assert report["terminal_truth"]["gse261709_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert report["terminal_truth"]["training_allowed"] is False
    assert report["terminal_truth"]["gpu_work_allowed"] is False
    assert report["terminal_truth"]["model_selection_allowed"] is False
    assert report["terminal_truth"]["a7_unlocked"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "IDENTITY",
        "RUN_ROLE",
        "HEADER",
        "HEADER_ROLE",
        "DIMENSIONS",
        "ASSET_LICENSE",
    ],
)
def test_any_nonpass_gate_stops_without_state_change(mutation: str) -> None:
    observation = _all_pass_observation()
    if mutation == "IDENTITY":
        observation["identity_context"]["context_verified"] = False
    elif mutation == "RUN_ROLE":
        observation["aggregate_role_geometry"]["run_role_mapping_complete"] = False
    elif mutation == "HEADER":
        observation["schema_and_rights"]["header_name_count"] = 0
    elif mutation == "HEADER_ROLE":
        observation["schema_and_rights"]["header_role_class_presence"][
            "SOURCE_CANDIDATE_ROLE"
        ] = False
    elif mutation == "DIMENSIONS":
        observation["schema_and_rights"]["exact_dimensions_complete"] = False
    else:
        observation["schema_and_rights"][
            "asset_license_notice_applies_to_row_level_research"
        ] = False
    report = PREFLIGHT.evaluate_observation(
        _protocol(),
        observation,
        binding=_fixture_binding(),
        source_mode="TEST",
        source_results=[],
        recorded_at="2026-08-14T08:00:00Z",
    )
    assert report["status"] == "STOP_PREFLIGHT_GATES_NOT_CLOSED"
    assert report["sole_next_action"] == "STOP_PREFLIGHT_GATES_NOT_CLOSED"
    assert report["terminal_truth"] == PREFLIGHT.EXPECTED_OUTER_TRUTH


def test_poison_member_fields_are_rejected_without_output(tmp_path: Path) -> None:
    observation = _all_pass_observation(existing_aggregate=True)
    observation["member_rows"] = [
        {
            "barcode": "MEMBER_BARCODE_POISON",
            "sequence": "MEMBER_SEQUENCE_POISON",
            "effect": 9.9,
        }
    ]
    observation_path = _write_json(tmp_path / "poison.json", observation)
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.ProtocolError, match="fields differ"):
        PREFLIGHT.execute(
            PROTOCOL_PATH,
            output_dir,
            observation_path=observation_path,
            binding_auditor=_fixture_binding,
        )
    assert not output_dir.exists()


def test_existing_aggregate_all_pass_publishes_one_atomic_aggregate(
    tmp_path: Path,
) -> None:
    observation_path = _write_json(
        tmp_path / "aggregate.json", _all_pass_observation(existing_aggregate=True)
    )
    output_dir = tmp_path / "output"
    report = PREFLIGHT.execute(
        PROTOCOL_PATH,
        output_dir,
        observation_path=observation_path,
        binding_auditor=_fixture_binding,
        recorded_at="2026-08-14T08:00:00Z",
    )
    assert report["sole_next_action"] == (
        "GO_REQUEST_SEPARATE_ROW_LEVEL_QUALIFICATION_AUTHORITY"
    )
    assert [path.name for path in output_dir.iterdir()] == [
        PREFLIGHT.REPORT_FILENAME
    ]
    assert not list(output_dir.glob("*.tmp"))
    with pytest.raises(PREFLIGHT.OutputError, match="not empty"):
        PREFLIGHT.execute(
            PROTOCOL_PATH,
            output_dir,
            observation_path=observation_path,
            binding_auditor=_fixture_binding,
        )
    assert [path.name for path in output_dir.iterdir()] == [
        PREFLIGHT.REPORT_FILENAME
    ]


def test_atomic_write_failure_leaves_no_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(path: Path, payload: bytes) -> None:
        del path, payload
        raise OSError("injected write failure")

    monkeypatch.setattr(PREFLIGHT, "_write_temp_payload", fail_write)
    output_dir = tmp_path / "output"
    with pytest.raises(PREFLIGHT.OutputError, match="cannot publish"):
        PREFLIGHT._write_exclusive(output_dir, {"aggregate": True})
    assert not (output_dir / PREFLIGHT.REPORT_FILENAME).exists()
    assert list(output_dir.iterdir()) == []


def test_default_binding_auditor_is_real_disk_i_b_future_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    script_blob = b"gse261709 preflight implementation at I\n"
    test_blob = b"gse261709 focused tests at I\n"
    protocol = _bound_protocol()
    binding = protocol["implementation_binding"]
    binding["implementation_script_sha256"] = hashlib.sha256(script_blob).hexdigest()
    binding["implementation_test_sha256"] = hashlib.sha256(test_blob).hexdigest()
    PREFLIGHT._validate_protocol(protocol)

    i_protocol = PREFLIGHT._normalise_binding(protocol)
    i_payload = (json.dumps(i_protocol, indent=2) + "\n").encode()
    b_payload = (json.dumps(protocol, indent=2) + "\n").encode()
    protocol_path = repo_root / PREFLIGHT.CONFIG_PATH
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes(b_payload)
    script_path = repo_root / PREFLIGHT.SCRIPT_PATH
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(script_blob)
    test_path = repo_root / PREFLIGHT.TEST_PATH
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_bytes(test_blob)

    authority_a = PREFLIGHT.AUTHORITY_COMMIT
    runtime_i1 = PREFLIGHT.RUNTIME_I1_COMMIT
    runtime_i2 = PREFLIGHT.RUNTIME_I2_COMMIT
    runtime_b2 = PREFLIGHT.RUNTIME_B2_COMMIT
    preflight_i1 = PREFLIGHT.PREFLIGHT_I1_COMMIT
    preflight_i2 = PREFLIGHT.PREFLIGHT_I2_COMMIT
    preflight_b2 = PREFLIGHT.PREFLIGHT_B2_COMMIT
    gse207_i1 = PREFLIGHT.GSE207_I1_COMMIT
    gse207_i2 = PREFLIGHT.GSE207_I2_COMMIT
    gse207_b2 = PREFLIGHT.GSE207_B2_COMMIT
    gse207_i3 = PREFLIGHT.GSE207_I3_COMMIT
    gse207_b3 = PREFLIGHT.GSE207_B3_COMMIT
    gse207_i4 = PREFLIGHT.GSE207_I4_COMMIT
    gse207_b4 = PREFLIGHT.GSE207_B4_COMMIT
    preflight_i3 = "2" * 40
    preflight_b3 = "8" * 40
    git_text = {
        ("rev-parse", "HEAD"): preflight_b3,
        ("rev-parse", f"{preflight_b3}^"): preflight_i3,
        ("rev-parse", f"{preflight_i3}^"): gse207_b4,
        ("rev-parse", f"{gse207_b4}^"): gse207_i4,
        ("rev-parse", f"{gse207_i4}^"): gse207_b3,
        ("rev-parse", f"{gse207_b3}^"): gse207_i3,
        ("rev-parse", f"{gse207_i3}^"): gse207_b2,
        ("rev-parse", f"{gse207_b2}^"): gse207_i2,
        ("rev-parse", f"{gse207_i2}^"): gse207_i1,
        ("rev-parse", f"{gse207_i1}^"): preflight_b2,
        ("rev-parse", f"{preflight_b2}^"): preflight_i2,
        ("rev-parse", f"{preflight_i2}^"): preflight_i1,
        ("rev-parse", f"{preflight_i1}^"): runtime_b2,
        ("rev-parse", f"{runtime_b2}^"): runtime_i2,
        ("rev-parse", f"{runtime_i2}^"): runtime_i1,
        ("rev-parse", f"{runtime_i1}^"): authority_a,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            runtime_i1,
        ): "\n".join(PREFLIGHT.RUNTIME_PATHS),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            runtime_i2,
        ): "\n".join(PREFLIGHT.RUNTIME_PATHS),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            runtime_b2,
        ): PREFLIGHT.RUNTIME_PATHS[0],
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_i1,
        ): "\n".join(PREFLIGHT.EXPECTED_EXACT3),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_i2,
        ): "\n".join(PREFLIGHT.EXPECTED_EXACT3),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_b2,
        ): PREFLIGHT.CONFIG_PATH,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_i1,
        ): "\n".join(PREFLIGHT.GSE207_PATHS),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_i2,
        ): "\n".join(PREFLIGHT.GSE207_PATHS),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_b2,
        ): PREFLIGHT.GSE207_PATHS[0],
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_i3,
        ): "\n".join(PREFLIGHT.GSE207_PATHS),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_b3,
        ): PREFLIGHT.GSE207_PATHS[0],
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_i4,
        ): "\n".join(PREFLIGHT.GSE207_PATHS),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            gse207_b4,
        ): PREFLIGHT.GSE207_PATHS[0],
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_i3,
        ): "\n".join(PREFLIGHT.EXPECTED_EXACT3),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_b3,
        ): PREFLIGHT.CONFIG_PATH,
    }
    git_blobs = {
        (preflight_i3, PREFLIGHT.CONFIG_PATH): i_payload,
        (preflight_i3, PREFLIGHT.SCRIPT_PATH): script_blob,
        (preflight_i3, PREFLIGHT.TEST_PATH): test_blob,
        (preflight_b3, PREFLIGHT.CONFIG_PATH): b_payload,
    }

    git_calls: list[tuple[str, ...]] = []

    def fake_git_text(root: Path, *args: str) -> str:
        assert root == repo_root
        git_calls.append(args)
        return git_text[args]

    def fake_git_blob(root: Path, commit: str, path: str) -> bytes:
        assert root == repo_root
        return git_blobs[(commit, path)]

    verified_runtime_blobs: list[tuple[str, dict[str, str]]] = []

    def fake_verify_blob_map(
        root: Path, commit: str, expected: dict[str, str]
    ) -> None:
        assert root == repo_root
        verified_runtime_blobs.append((commit, dict(expected)))

    monkeypatch.setattr(PREFLIGHT, "_run_git_text", fake_git_text)
    monkeypatch.setattr(PREFLIGHT, "_git_blob", fake_git_blob)
    monkeypatch.setattr(PREFLIGHT, "_verify_blob_map", fake_verify_blob_map)
    monkeypatch.setattr(PREFLIGHT, "__file__", str(script_path))
    result = PREFLIGHT._default_binding_auditor(
        protocol, protocol_path, b_payload, repo_root
    )
    assert result == {
        "status": "BOUND_GLOBAL_RUNTIME_GSE261_GSE207_B4_GSE261_CHAIN_VERIFIED",
        "authority_commit": authority_a,
        "authority_runtime_implementation_i1_commit": runtime_i1,
        "authority_runtime_implementation_i2_commit": runtime_i2,
        "authority_runtime_binding_commit": runtime_b2,
        "preflight_implementation_i1_commit": preflight_i1,
        "preflight_implementation_i2_commit": preflight_i2,
        "preflight_binding_b2_commit": preflight_b2,
        "gse207584_implementation_i1_commit": gse207_i1,
        "gse207584_implementation_i2_commit": gse207_i2,
        "gse207584_binding_b2_commit": gse207_b2,
        "gse207584_implementation_i3_commit": gse207_i3,
        "gse207584_binding_b3_commit": gse207_b3,
        "gse207584_implementation_i4_commit": gse207_i4,
        "gse207584_binding_b4_commit": gse207_b4,
        "implementation_commit": preflight_i3,
        "binding_commit": preflight_b3,
    }
    assert verified_runtime_blobs == [
        (runtime_i1, PREFLIGHT.RUNTIME_I1_BLOB_SHA256_BY_PATH),
        (runtime_i2, PREFLIGHT.RUNTIME_I2_BLOB_SHA256_BY_PATH),
        (runtime_b2, PREFLIGHT.RUNTIME_B2_BLOB_SHA256_BY_PATH),
        (preflight_i1, PREFLIGHT.PREFLIGHT_I1_BLOB_SHA256_BY_PATH),
        (preflight_i2, PREFLIGHT.PREFLIGHT_I2_BLOB_SHA256_BY_PATH),
        (preflight_b2, PREFLIGHT.PREFLIGHT_B2_BLOB_SHA256_BY_PATH),
        (gse207_i1, PREFLIGHT.GSE207_I1_BLOB_SHA256_BY_PATH),
        (gse207_i2, PREFLIGHT.GSE207_I2_BLOB_SHA256_BY_PATH),
        (gse207_i3, PREFLIGHT.GSE207_I3_BLOB_SHA256_BY_PATH),
        (gse207_i4, PREFLIGHT.GSE207_I4_BLOB_SHA256_BY_PATH),
        (gse207_b2, PREFLIGHT.GSE207_B2_BLOB_SHA256_BY_PATH),
        (gse207_b3, PREFLIGHT.GSE207_B3_BLOB_SHA256_BY_PATH),
        (gse207_b4, PREFLIGHT.GSE207_B4_BLOB_SHA256_BY_PATH),
    ]
    assert [
        ("rev-parse", f"{preflight_b3}^"),
        ("rev-parse", f"{preflight_i3}^"),
        ("rev-parse", f"{gse207_b4}^"),
        ("rev-parse", f"{gse207_i4}^"),
        ("rev-parse", f"{gse207_b3}^"),
        ("rev-parse", f"{gse207_i3}^"),
        ("rev-parse", f"{gse207_b2}^"),
        ("rev-parse", f"{gse207_i2}^"),
        ("rev-parse", f"{gse207_i1}^"),
        ("rev-parse", f"{preflight_b2}^"),
        ("rev-parse", f"{preflight_i2}^"),
        ("rev-parse", f"{preflight_i1}^"),
        ("rev-parse", f"{runtime_b2}^"),
        ("rev-parse", f"{runtime_i2}^"),
        ("rev-parse", f"{runtime_i1}^"),
    ] == [call for call in git_calls if call[:1] == ("rev-parse",)][1:]


def test_stale_copy_stops_before_metadata_fetch_or_output(tmp_path: Path) -> None:
    repo_root = tmp_path / "production-repo"
    protocol = _bound_protocol()
    protocol_path = repo_root / PREFLIGHT.CONFIG_PATH
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    production_script = repo_root / PREFLIGHT.SCRIPT_PATH
    production_script.parent.mkdir(parents=True, exist_ok=True)
    production_script.write_text("bound production producer\n")
    production_test = repo_root / PREFLIGHT.TEST_PATH
    production_test.parent.mkdir(parents=True, exist_ok=True)
    production_test.write_text("bound production test\n")

    fetcher = BombFetcher()
    output_dir = tmp_path / "must-not-exist"
    assert Path(PREFLIGHT.__file__).resolve() != production_script.resolve()
    with pytest.raises(PREFLIGHT.ProtocolError, match="bound production path"):
        PREFLIGHT.execute(
            protocol_path,
            output_dir,
            repo_root=repo_root,
            fetcher=fetcher,
        )
    assert fetcher.calls == 0
    assert not output_dir.exists()


def test_existing_observation_and_live_fetch_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    observation_path = _write_json(
        tmp_path / "aggregate.json", _all_pass_observation(existing_aggregate=True)
    )
    fetcher = BombFetcher()
    with pytest.raises(PREFLIGHT.ProtocolError, match="exclusive"):
        PREFLIGHT.execute(
            PROTOCOL_PATH,
            tmp_path / "output",
            observation_path=observation_path,
            fetcher=fetcher,
            binding_auditor=_fixture_binding,
        )
    assert fetcher.calls == 0
    assert not (tmp_path / "output").exists()
