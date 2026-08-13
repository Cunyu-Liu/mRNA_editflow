from __future__ import annotations

import copy
import gzip
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
    / "route_a_v3_gse256185_public_identifier_pool_geometry_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse256185_public_identifier_pool_geometry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gse256185_public_identifier_pool_geometry_preflight", MODULE_PATH
)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _disk_lifecycle_state(protocol: dict[str, object]) -> str:
    binding = protocol["implementation_binding"]
    authority_values = [
        binding["authority_commit"],
        binding["authority_runtime_binding_commit"],
    ]
    normal_values = [
        binding[field] for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS
    ]
    authority_unknown = authority_values == [PREFLIGHT.UNKNOWN] * 2
    authority_bound = all(
        isinstance(value, str) and PREFLIGHT.HEX40_RE.fullmatch(value)
        for value in authority_values
    )
    normal_unknown = normal_values == [PREFLIGHT.UNKNOWN] * 4
    if authority_unknown:
        assert normal_unknown
        return "TEMPLATE"
    assert authority_bound
    if normal_unknown:
        return "I"

    assert binding["status"] == PREFLIGHT.BOUND
    assert PREFLIGHT.HEX40_RE.fullmatch(binding["implementation_commit"])
    assert hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest() == (
        binding["implementation_script_sha256"]
    )
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == (
        binding["implementation_test_sha256"]
    )
    return "B"


def _bound_protocol() -> dict[str, object]:
    protocol = _i_protocol()
    binding = protocol["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = "2" * 40
    binding["implementation_script_sha256"] = "3" * 64
    binding["implementation_test_sha256"] = "4" * 64
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _i_protocol() -> dict[str, object]:
    protocol = copy.deepcopy(_protocol())
    binding = protocol["implementation_binding"]
    for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS:
        binding[field] = PREFLIGHT.UNKNOWN
    binding["authority_commit"] = "0" * 40
    binding["authority_runtime_binding_commit"] = "1" * 40
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _write_protocol(path: Path, protocol: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture_binding(*args: object) -> dict[str, str]:
    return {
        "status": "TEST_FIXTURE_BOUND_WITHOUT_GIT",
        "authority_commit": "0" * 40,
        "authority_runtime_binding_commit": "1" * 40,
        "implementation_commit": "2" * 40,
        "binding_commit": "3" * 40,
    }


def _fixture_identity(*args: object) -> dict[str, object]:
    return {
        "filename": PREFLIGHT.OFFICIAL_ASSET["filename"],
        "compressed_bytes": PREFLIGHT.OFFICIAL_ASSET["compressed_bytes"],
        "compressed_sha256": PREFLIGHT.OFFICIAL_ASSET["compressed_sha256"],
        "identity_status": "PASS_FROZEN_ORDINARY_PUBLIC_ASSET",
    }


def _official_observation() -> dict[str, object]:
    total = PREFLIGHT.EXPECTED_GEOMETRY["total_body_row_count"]
    return {
        "header_observation": {
            "status": "PASS_EXACT_HEADER_NAMES",
            "column_name_count": 10,
            "identifier_column_index": 0,
            "forbidden_body_value_column_count": 9,
        },
        **copy.deepcopy(PREFLIGHT.EXPECTED_GEOMETRY),
        "body_access_attestation": {
            "whole_asset_stream_transport_and_decompression_performed": True,
            "full_row_bytes_discarded_after_first_tab_without_decoding_or_tokenizing_forbidden_fields": True,
            "identifier_body_cell_decoded_count": total,
            "identifier_body_cell_parsed_count": total,
            "role_token_derived_from_identifier_count": total,
            "body_columns_decoded_or_parsed": ["ID"],
            "forbidden_body_cells": {
                field_class: {
                    "decoded_count": 0,
                    "parsed_count": 0,
                    "stored_count": 0,
                    "output_count": 0,
                }
                for field_class in ("SEQUENCE", "EFFECT", "CPM", "OTHER")
            },
        },
    }


def _write_identifier_only_fixture(path: Path) -> list[str]:
    rows: list[tuple[str, bytes]] = []
    serial = 0

    def add(group: str, role: str, *, missing_delimiter: bool = False) -> None:
        nonlocal serial
        serial += 1
        suffix = f"B{serial:011d}"
        identifier = (
            f"{group}{role}.{suffix}"
            if missing_delimiter
            else f"{group}.{role}.{suffix}"
        )
        rows.append((identifier, f"FORBIDDEN_POISON_{serial}".encode("ascii")))

    group_a = "ENSG1-ENST1-10"
    for role in ("parent", "win0", "rand1", "+1CCC"):
        add(group_a, role)

    group_b = "ENSG2-ENST2-10"
    for role in ("parent", "win0", "rand1"):
        add(group_b, role)

    group_c = "ENSG3-ENST3-10"
    for role in ("parent", "parent", "win0", "rand1", "+1CCC"):
        add(group_c, role)

    group_d = "ENSG4-ENST4-10"
    for role in ("parent", "-1CCC", "-3CCC", "-4CCC", "-5CCC", "2CCC"):
        add(group_d, role)

    group_e = "ENSG5-ENST5-10"
    for role in ("parent", "-1CCC", "-3CCC", "-4CCC", "-5CCC"):
        add(group_e, role)
    add(group_e, "-2CCC", missing_delimiter=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(("\t".join(PREFLIGHT.EXPECTED_HEADER) + "\n").encode("ascii"))
        for identifier, poison in rows:
            forbidden_cells = [
                poison,
                b"\xff\xfeNOT_UTF8_CPM",
                b"CPM_POISON",
                b"CPM_POISON",
                b"CPM_POISON",
                b"CPM_POISON",
                b"CPM_POISON",
                b"CPM_POISON",
                b"SEQUENCE_POISON",
            ]
            handle.write(
                identifier.encode("ascii")
                + b"\t"
                + b"\t".join(forbidden_cells)
                + b"\n"
            )
    return [identifier for identifier, _ in rows]


def test_protocol_freezes_dec021_scope_two_stage_parent_and_exact3() -> None:
    protocol = _protocol()
    binding = protocol["implementation_binding"]
    assert _disk_lifecycle_state(protocol) in {"TEMPLATE", "I", "B"}
    assert tuple(binding["implementation_commit_exact_changed_paths"]) == (
        PREFLIGHT.EXPECTED_EXACT3
    )
    assert binding["binding_commit_exact_changed_paths"] == [PREFLIGHT.CONFIG_PATH]
    assert binding["pre_implementation_authority_scalar_paths"] == [
        "implementation_binding.authority_commit",
        "implementation_binding.authority_runtime_binding_commit",
    ]
    assert protocol["decision_authority"]["authorized_role"] == (
        "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY"
    )
    assert protocol["official_processed_asset"] == {
        "filename": "GSE256185_CPMandRRS_VCE_Var.tsv.gz",
        "locator": (
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE256185&file="
            "GSE256185_CPMandRRS_VCE_Var.tsv.gz&format=file"
        ),
        "compressed_bytes": 952533,
        "compressed_sha256": (
            "71a8476a76e9a47a03bc69a2e0cbf79d92019249fba2049f57b7aa60f3f25aeb"
        ),
        "identity_mismatch_action": "STOP_BEFORE_DECOMPRESSION_OR_AGGREGATION",
    }

    i_protocol = _i_protocol()
    i_binding = i_protocol["implementation_binding"]
    assert i_binding["authority_commit"] == "0" * 40
    assert i_binding["authority_runtime_binding_commit"] == "1" * 40
    assert [i_binding[field] for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS] == [
        PREFLIGHT.UNKNOWN
    ] * 4

    b_protocol = _bound_protocol()
    b_binding = b_protocol["implementation_binding"]
    assert b_binding["status"] == PREFLIGHT.BOUND
    assert b_binding["authority_commit"] == "0" * 40
    assert b_binding["authority_runtime_binding_commit"] == "1" * 40
    assert b_binding["implementation_commit"] == "2" * 40
    assert b_binding["implementation_script_sha256"] == "3" * 64
    assert b_binding["implementation_test_sha256"] == "4" * 64


def test_partial_authority_or_normal_binding_group_is_rejected() -> None:
    protocol = copy.deepcopy(_protocol())
    binding = protocol["implementation_binding"]
    binding["authority_commit"] = PREFLIGHT.UNKNOWN
    binding["authority_runtime_binding_commit"] = PREFLIGHT.UNKNOWN
    binding["authority_commit"] = "0" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="partially known"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = copy.deepcopy(_protocol())
    binding = protocol["implementation_binding"]
    for field in PREFLIGHT.UNKNOWN_BINDING_SCALARS:
        binding[field] = PREFLIGHT.UNKNOWN
    binding["implementation_commit"] = "2" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="initial-I binding scalars"):
        PREFLIGHT._validate_protocol(protocol)


def test_default_binding_auditor_verifies_real_i_and_b_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    script_blob = b"preflight implementation at I\n"
    test_blob = b"focused test at I\n"
    protocol = _bound_protocol()
    binding = protocol["implementation_binding"]
    binding["implementation_script_sha256"] = hashlib.sha256(script_blob).hexdigest()
    binding["implementation_test_sha256"] = hashlib.sha256(test_blob).hexdigest()
    PREFLIGHT._validate_protocol(protocol)

    i_protocol = PREFLIGHT._normalise_binding(protocol)
    i_payload = (json.dumps(i_protocol, indent=2) + "\n").encode("utf-8")
    b_payload = (json.dumps(protocol, indent=2) + "\n").encode("utf-8")
    protocol_path = repo_root / PREFLIGHT.CONFIG_PATH
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes(b_payload)
    script_path = repo_root / PREFLIGHT.SCRIPT_PATH
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(script_blob)
    test_path = repo_root / PREFLIGHT.TEST_PATH
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_bytes(test_blob)

    authority_a = "0" * 40
    authority_runtime_i = "9" * 40
    authority_runtime_b = "1" * 40
    preflight_i = "2" * 40
    preflight_b = "8" * 40
    git_text = {
        ("rev-parse", "HEAD"): preflight_b,
        ("rev-parse", f"{preflight_b}^"): preflight_i,
        ("rev-parse", f"{preflight_i}^"): authority_runtime_b,
        ("rev-parse", f"{authority_runtime_b}^"): authority_runtime_i,
        ("rev-parse", f"{authority_runtime_i}^"): authority_a,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_i,
        ): "\n".join(PREFLIGHT.EXPECTED_EXACT3),
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preflight_b,
        ): PREFLIGHT.CONFIG_PATH,
    }
    git_blobs = {
        (preflight_i, PREFLIGHT.CONFIG_PATH): i_payload,
        (preflight_i, PREFLIGHT.SCRIPT_PATH): script_blob,
        (preflight_i, PREFLIGHT.TEST_PATH): test_blob,
        (preflight_b, PREFLIGHT.CONFIG_PATH): b_payload,
    }

    def fake_git_text(root: Path, *args: str) -> str:
        assert root == repo_root
        return git_text[args]

    def fake_git_blob(root: Path, commit: str, relative_path: str) -> bytes:
        assert root == repo_root
        return git_blobs[(commit, relative_path)]

    monkeypatch.setattr(PREFLIGHT, "_run_git_text", fake_git_text)
    monkeypatch.setattr(PREFLIGHT, "_git_blob", fake_git_blob)
    result = PREFLIGHT._default_binding_auditor(
        protocol, protocol_path, b_payload, repo_root
    )
    assert result == {
        "status": "BOUND_EXACT3_I_CONFIG_ONLY_B_VERIFIED",
        "authority_commit": authority_a,
        "authority_runtime_binding_commit": authority_runtime_b,
        "implementation_commit": preflight_i,
        "binding_commit": preflight_b,
    }


def test_protocol_rejects_scope_or_terminal_promotion() -> None:
    protocol = _protocol()
    protocol["input_contract"]["sequence_body_values_allowed"] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match="must remain false"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _protocol()
    protocol["frozen_outer_truth"]["training_allowed"] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match="outer truth"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _protocol()
    grammar = protocol["identifier_role_grammar"]
    grammar["family_closure_axis_status"] = "PUBLISHER_EXPLICIT"
    with pytest.raises(PREFLIGHT.ProtocolError, match="inference label"):
        PREFLIGHT._validate_protocol(protocol)


def test_unknown_preflight_binding_stops_before_asset_or_output_io(
    tmp_path: Path,
) -> None:
    calls = {"asset": 0, "geometry": 0}

    def forbidden_asset(*args: object) -> dict[str, object]:
        calls["asset"] += 1
        raise AssertionError("asset identity must not be read")

    def forbidden_geometry(*args: object) -> dict[str, object]:
        calls["geometry"] += 1
        raise AssertionError("asset body must not be read")

    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="not BOUND"):
        PREFLIGHT.execute(
            PROTOCOL_PATH,
            tmp_path / "missing-asset.gz",
            output_dir,
            asset_identity_auditor=forbidden_asset,
            geometry_aggregator=forbidden_geometry,
        )
    assert calls == {"asset": 0, "geometry": 0}
    assert not output_dir.exists()


def test_binary_stream_parser_decodes_only_id_and_separates_strict_from_inference(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "fixture.tsv.gz"
    member_identifiers = _write_identifier_only_fixture(asset)
    observation = PREFLIGHT.aggregate_asset_geometry(asset)

    assert observation["total_body_row_count"] == 24
    assert observation["strict_grammar_row_count"] == 22
    assert observation["strict_role_family_row_counts"] == {
        "parent": 6,
        "win": 3,
        "+CCC": 2,
        "-CCC": 8,
        "rand": 3,
    }
    assert observation["identifier_grammar_anomaly_counts"] == {
        "MISSING_GROUP_ROLE_DELIMITER": 1,
        "UNSIGNED_CCC_ROLE": 1,
        "OTHER_IDENTIFIER_GRAMMAR": 0,
        "OTHER_ROLE_GRAMMAR": 0,
    }
    assert observation["strict_axis"] == {
        "group_count": 5,
        "groups_with_parent": 5,
        "single_parent_group_count": 4,
        "dual_parent_group_count": 1,
        "other_parent_multiplicity_group_count": 0,
        "single_parent_groups_with_at_least_3_strict_candidate_rows": 3,
        "single_parent_groups_with_exactly_2_strict_candidate_rows": 1,
        "strict_candidate_rows_in_at_least_3_candidate_groups": 11,
    }
    assert observation["reasoned_family_closure_axis"] == {
        "status": PREFLIGHT.FAMILY_CLOSURE_STATUS,
        "group_count": 5,
        "groups_with_parent": 5,
        "single_parent_group_count": 4,
        "dual_parent_group_count": 1,
        "other_parent_multiplicity_group_count": 0,
        "single_parent_groups_with_at_least_3_candidate_rows": 3,
        "single_parent_groups_with_exactly_2_candidate_rows": 1,
        "candidate_rows_in_at_least_3_candidate_groups": 13,
    }
    assert observation["body_access_attestation"] == {
        "whole_asset_stream_transport_and_decompression_performed": True,
        "full_row_bytes_discarded_after_first_tab_without_decoding_or_tokenizing_forbidden_fields": True,
        "identifier_body_cell_decoded_count": 24,
        "identifier_body_cell_parsed_count": 24,
        "role_token_derived_from_identifier_count": 24,
        "body_columns_decoded_or_parsed": ["ID"],
        "forbidden_body_cells": {
            field_class: {
                "decoded_count": 0,
                "parsed_count": 0,
                "stored_count": 0,
                "output_count": 0,
            }
            for field_class in ("SEQUENCE", "EFFECT", "CPM", "OTHER")
        },
    }
    serialized = json.dumps(observation, sort_keys=True)
    assert "FORBIDDEN_POISON" not in serialized
    assert "SEQUENCE_POISON" not in serialized
    assert all(identifier not in serialized for identifier in member_identifiers)


def test_complete_report_is_aggregate_only_not_qualified_and_preserves_outer_counts(
    tmp_path: Path,
) -> None:
    protocol_path = _write_protocol(tmp_path / "repo" / PREFLIGHT.CONFIG_PATH, _bound_protocol())
    output_dir = tmp_path / "output"
    report = PREFLIGHT.execute(
        protocol_path,
        tmp_path / PREFLIGHT.OFFICIAL_ASSET["filename"],
        output_dir,
        repo_root=tmp_path / "repo",
        binding_auditor=_fixture_binding,
        asset_identity_auditor=_fixture_identity,
        geometry_aggregator=lambda path: _official_observation(),
        recorded_at="2026-08-13T12:00:00Z",
    )

    assert [path.name for path in output_dir.iterdir()] == [
        PREFLIGHT.REPORT_FILENAME
    ]
    assert report["status"] == PREFLIGHT.COMPLETION_STATUS
    assert report["preflight_complete"] is True
    assert report["aggregate_pool_geometry"] == PREFLIGHT.EXPECTED_GEOMETRY
    assert report["terminal_truth"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert report["terminal_truth"]["gse256185_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert report["terminal_truth"]["gse256185_qualified"] is False
    assert report["terminal_truth"]["gse256185_true_a2_established"] is False
    assert report["terminal_truth"]["a1_complete"] is False
    assert report["terminal_truth"]["training_allowed"] is False
    assert report["terminal_truth"]["model_selection_allowed"] is False
    assert report["terminal_truth"]["next_phase_authorized"] is False
    assert report["scope_attestation"][
        "compressed_asset_bytes_verified_before_decompression"
    ] is True
    assert report["scope_attestation"][
        "whole_asset_stream_transport_and_decompression_performed"
    ] is True
    assert report["scope_attestation"][
        "full_row_bytes_discarded_after_first_tab_without_decoding_or_tokenizing_forbidden_fields"
    ] is True
    assert report["scope_attestation"]["identifier_body_cell_decoded_count"] == 11404
    assert report["scope_attestation"]["identifier_body_cell_parsed_count"] == 11404
    assert report["scope_attestation"]["forbidden_body_cells"] == {
        field_class: {
            "decoded_count": 0,
            "parsed_count": 0,
            "stored_count": 0,
            "output_count": 0,
        }
        for field_class in ("SEQUENCE", "EFFECT", "CPM", "OTHER")
    }
    assert report["interpretation_boundary"][
        "strict_axis_is_frozen_observed_identifier_grammar"
    ] is True
    assert report["interpretation_boundary"][
        "publisher_identifier_grammar_documented"
    ] is False
    assert report["interpretation_boundary"][
        "family_closure_axis_is_publisher_explicit"
    ] is False
    assert report["interpretation_boundary"]["sequence_edit_semantics"] == (
        "NOT_EVALUATED_OUT_OF_SCOPE"
    )
    assert report["sole_next_action"] == (
        "STOP_NO_FURTHER_ACTION_AUTHORIZED_BY_V3_DEC_021"
    )

    serialized = (output_dir / PREFLIGHT.REPORT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "ENSG" not in serialized
    assert "ENST" not in serialized
    assert "SEQUENCE_POISON" not in serialized
    assert "FORBIDDEN_POISON" not in serialized


def test_wrong_public_asset_identity_stops_before_decompression_or_output(
    tmp_path: Path,
) -> None:
    protocol_path = _write_protocol(tmp_path / "repo" / PREFLIGHT.CONFIG_PATH, _bound_protocol())
    asset = tmp_path / PREFLIGHT.OFFICIAL_ASSET["filename"]
    asset.write_bytes(b"not-the-frozen-asset")
    calls = {"geometry": 0}

    def forbidden_geometry(*args: object) -> dict[str, object]:
        calls["geometry"] += 1
        raise AssertionError("geometry parser must not run after identity mismatch")

    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.AssetIdentityError, match="byte count"):
        PREFLIGHT.execute(
            protocol_path,
            asset,
            output_dir,
            repo_root=tmp_path / "repo",
            binding_auditor=_fixture_binding,
            geometry_aggregator=forbidden_geometry,
        )
    assert calls["geometry"] == 0
    assert not output_dir.exists()


def test_geometry_drift_stops_without_output(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path / "repo" / PREFLIGHT.CONFIG_PATH, _bound_protocol())
    observation = _official_observation()
    observation["strict_axis"][
        "single_parent_groups_with_at_least_3_strict_candidate_rows"
    ] = 633
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.GeometryError, match="geometry differs"):
        PREFLIGHT.execute(
            protocol_path,
            tmp_path / PREFLIGHT.OFFICIAL_ASSET["filename"],
            output_dir,
            repo_root=tmp_path / "repo",
            binding_auditor=_fixture_binding,
            asset_identity_auditor=_fixture_identity,
            geometry_aggregator=lambda path: observation,
        )
    assert not output_dir.exists()


def test_exclusive_output_write_failure_leaves_no_final_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"

    def injected_partial_write(temp_path: Path, payload: bytes) -> None:
        temp_path.write_bytes(payload[:7])
        raise OSError("injected write failure")

    monkeypatch.setattr(PREFLIGHT, "_write_temp_payload", injected_partial_write)
    with pytest.raises(PREFLIGHT.OutputError, match="cannot write"):
        PREFLIGHT._write_exclusive(output_dir, {"aggregate": True})
    assert not (output_dir / PREFLIGHT.REPORT_FILENAME).exists()
    assert list(output_dir.iterdir()) == []


def test_exclusive_output_does_not_overwrite_existing_final(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_path = output_dir / PREFLIGHT.REPORT_FILENAME
    original = b"existing aggregate report\n"
    output_path.write_bytes(original)

    with pytest.raises(PREFLIGHT.OutputError, match="not empty"):
        PREFLIGHT._write_exclusive(output_dir, {"replacement": True})
    assert output_path.read_bytes() == original
