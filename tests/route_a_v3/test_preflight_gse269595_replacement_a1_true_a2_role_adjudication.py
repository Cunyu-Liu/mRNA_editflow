from __future__ import annotations

import copy
import gzip
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse269595_replacement_a1_true_a2_role_adjudication_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse269595_replacement_a1_true_a2_role_adjudication.py"
)
REAL_ASSET_DIRECTORY = Path("/private/tmp/aparent-perturb-audit.ycCcHc")
REAL_MPRA_ASSET = (
    REAL_ASSET_DIRECTORY
    / "GSE269595_mpra_constructs_all_samples_proximal_site_usage.txt.gz"
)
REAL_TABLE_S5_ASSET = REAL_ASSET_DIRECTORY / "Table_S5.xlsx"

SPEC = importlib.util.spec_from_file_location("gse269595_role_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _protocol() -> dict[str, object]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(value)
    return value


def _bound_protocol() -> dict[str, object]:
    value = copy.deepcopy(_protocol())
    own = value["implementation_binding"]["preflight_group"]
    own.update(
        {
            "status": PREFLIGHT.BOUND,
            "implementation_commit": "3" * 40,
            "implementation_script_sha256": "e" * 64,
            "implementation_test_sha256": "f" * 64,
        }
    )
    PREFLIGHT._validate_protocol(value)
    return value


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _require_real_assets() -> None:
    if not REAL_MPRA_ASSET.is_file() or not REAL_TABLE_S5_ASSET.is_file():
        pytest.skip("the two frozen local official assets are not available")


@pytest.fixture(scope="module")
def real_report() -> dict[str, object]:
    _require_real_assets()
    return PREFLIGHT.aggregate(_protocol(), REAL_MPRA_ASSET, REAL_TABLE_S5_ASSET)


def test_static_protocol_binds_exact_assets_and_predecessor_but_keeps_own_unknown() -> None:
    protocol = _protocol()
    assert protocol["dataset_id"] == "GSE269595"
    assert protocol["bioproject_id"] == "PRJNA1122592"
    assert tuple(protocol["gate_ids"]) == PREFLIGHT.GATE_IDS
    assert len(protocol["gate_ids"]) == 13
    assert protocol["implementation_binding"]["authority_group"][
        "authority_commit"
    ] == PREFLIGHT.AUTHORITY_COMMIT
    assert protocol["implementation_binding"]["nonauthoritative_a6_g0_group"] == {
        "status": PREFLIGHT.A6_G0_STATUS,
        "commit": PREFLIGHT.A6_G0_COMMIT,
        "expected_parent": PREFLIGHT.AUTHORITY_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.A6_G0_EXACT4),
        "blob_sha256_by_path": PREFLIGHT.A6_G0_BLOBS,
        "authority_or_scientific_state_changed": False,
    }
    runtime = protocol["implementation_binding"]["authority_runtime_group"]
    assert runtime["implementation_commit"] == PREFLIGHT.RUNTIME_IMPLEMENTATION_COMMIT
    assert runtime["implementation_expected_parent"] == PREFLIGHT.A6_G0_COMMIT
    assert runtime["binding_commit"] == PREFLIGHT.RUNTIME_BINDING_COMMIT
    assert runtime["binding_expected_parent"] == PREFLIGHT.RUNTIME_IMPLEMENTATION_COMMIT
    projection = protocol["implementation_binding"]["current_projection_group"]
    assert projection["commit"] == PREFLIGHT.PROJECTION_COMMIT
    assert projection["expected_parent"] == PREFLIGHT.RUNTIME_BINDING_COMMIT
    predecessor = protocol["implementation_binding"]["gse261709_predecessor_group"]
    assert predecessor["implementation_commit"] == PREFLIGHT.PREDECESSOR_IMPLEMENTATION_COMMIT
    assert predecessor["implementation_expected_parent"] == PREFLIGHT.PROJECTION_COMMIT
    assert predecessor["binding_commit"] == PREFLIGHT.PREDECESSOR_BINDING_COMMIT
    assert predecessor["binding_expected_parent"] == PREFLIGHT.PREDECESSOR_IMPLEMENTATION_COMMIT
    own = protocol["implementation_binding"]["preflight_group"]
    assert all(
        value == PREFLIGHT.UNKNOWN
        for value in PREFLIGHT._preflight_dynamic_values(own)
    )

    inputs = protocol["ordinary_public_input_contract"]
    assert inputs == PREFLIGHT.ORDINARY_PUBLIC_INPUT_CONTRACT
    assert inputs["caller_supplied_context_allowed"] is False
    assert inputs["both_asset_identities_must_close_before_any_parse"] is True
    assert inputs["official_mpra_processed_asset"]["sha256"] == (
        "6527ea54257b3f17ddb9df5977637e41e2ef16d926a27be4e29f56165acaa1de"
    )
    assert inputs["publisher_table_s5_asset"]["sha256"] == (
        "d350be818d87120216052645a5ffa97afeee898d32a877c74033dba0d0fa151a"
    )
    semantics = protocol["bound_role_context_semantics"]
    assert semantics == PREFLIGHT.BOUND_ROLE_CONTEXT_SEMANTICS
    assert semantics["source_replay_protocol"]["source_family_key_exactly"] == [
        "gene_id",
        "pas_id",
    ]
    assert semantics["join_protocol"]["subtype_subaim_equality_required"] is False
    assert protocol["no_promotion_locks"] == PREFLIGHT.NO_PROMOTION_LOCKS
    assert protocol["no_promotion_locks"]["dataset_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }


def test_protocol_rejects_asset_or_join_semantic_substitution() -> None:
    wrong_digest = _protocol()
    wrong_digest["ordinary_public_input_contract"]["official_mpra_processed_asset"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(PREFLIGHT.ProtocolError, match="input boundary"):
        PREFLIGHT._validate_protocol(wrong_digest)

    caller_context = _protocol()
    caller_context["ordinary_public_input_contract"]["context_format"] = "JSON_OBJECT"
    with pytest.raises(PREFLIGHT.ProtocolError, match="input boundary"):
        PREFLIGHT._validate_protocol(caller_context)

    wrong_join = _protocol()
    wrong_join["bound_role_context_semantics"]["join_protocol"][
        "subtype_subaim_equality_required"
    ] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match="role/context semantics"):
        PREFLIGHT._validate_protocol(wrong_join)


def test_runtime_projection_and_predecessor_parent_chain_is_literal_and_fail_closed() -> None:
    wrong_runtime = _protocol()
    wrong_runtime["implementation_binding"]["authority_runtime_group"][
        "implementation_commit"
    ] = "1" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="runtime I exact3"):
        PREFLIGHT._validate_protocol(wrong_runtime)

    wrong_projection = _protocol()
    wrong_projection["implementation_binding"]["current_projection_group"][
        "expected_parent"
    ] = PREFLIGHT.RUNTIME_IMPLEMENTATION_COMMIT
    with pytest.raises(PREFLIGHT.ProtocolError, match="projection P"):
        PREFLIGHT._validate_protocol(wrong_projection)

    partial_predecessor = _protocol()
    partial_predecessor["implementation_binding"]["gse261709_predecessor_group"][
        "binding_commit"
    ] = PREFLIGHT.UNKNOWN
    with pytest.raises(PREFLIGHT.ProtocolError, match="partially bound"):
        PREFLIGHT._validate_protocol(partial_predecessor)

    wrong_predecessor = _protocol()
    group = wrong_predecessor["implementation_binding"]["gse261709_predecessor_group"]
    group["implementation_commit"] = "1" * 40
    group["binding_expected_parent"] = "1" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="predecessor I exact3"):
        PREFLIGHT._validate_protocol(wrong_predecessor)


def test_current_candidate_stops_before_any_asset_or_output_path_io(tmp_path: Path) -> None:
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="grouped UNKNOWN"):
        PREFLIGHT.execute(
            PROTOCOL_PATH,
            tmp_path / "DOES_NOT_EXIST_OFFICIAL.txt.gz",
            tmp_path / "DOES_NOT_EXIST_TABLE_S5.xlsx",
            tmp_path / "WRONG_OUTPUT_NAME.json",
            repo_root=tmp_path / "DOES_NOT_EXIST_REPO",
        )
    assert list(tmp_path.iterdir()) == []


def test_official_execute_has_only_fixed_builtin_producer_path() -> None:
    parameters = inspect.signature(PREFLIGHT.execute).parameters
    assert tuple(parameters) == (
        "protocol_path",
        "official_mpra_asset",
        "publisher_table_s5_asset",
        "output",
        "repo_root",
    )
    assert "binding_auditor" not in parameters
    assert "aggregator" not in parameters
    assert "privacy_validator" not in parameters
    source = inspect.getsource(PREFLIGHT.execute)
    assert "_default_binding_auditor(" in source
    assert "report = aggregate(" in source
    assert "_default_privacy_validator(" in source
    assert "_atomic_publish(" in source


def test_same_schema_substitution_stops_before_parse() -> None:
    # A structurally plausible, exact-size gzip must still close the fixed digest.
    substitute = gzip.compress(
        (" ".join(PREFLIGHT.MPRA_HEADER) + "\n" + " ".join(["x"] * 15) + "\n").encode()
    )
    substitute += b"\0" * (
        PREFLIGHT.OFFICIAL_MPRA_SPEC["byte_count"] - len(substitute)
    )
    assert len(substitute) == PREFLIGHT.OFFICIAL_MPRA_SPEC["byte_count"]
    with pytest.raises(PREFLIGHT.AssetError, match="digest identity differs"):
        PREFLIGHT.aggregate(
            _protocol(),
            _BytesPath(substitute),
            _UnreadablePath(),
        )


class _BytesPath:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read_bytes(self) -> bytes:
        return self.payload


class _UnreadablePath:
    def read_bytes(self) -> bytes:
        raise AssertionError("second asset must not be read after first identity failure")


def test_one_byte_table_change_stops_both_parsers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_real_assets()
    changed = bytearray(REAL_TABLE_S5_ASSET.read_bytes())
    changed[len(changed) // 2] ^= 1
    changed_path = tmp_path / "one-byte-changed-Table-S5.xlsx"
    changed_path.write_bytes(changed)
    parser_calls: list[str] = []
    monkeypatch.setattr(
        PREFLIGHT,
        "_parse_publisher_table_s5",
        lambda _: parser_calls.append("table"),
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_parse_official_mpra",
        lambda *_: parser_calls.append("mpra"),
    )
    with pytest.raises(PREFLIGHT.AssetError, match="digest identity differs"):
        PREFLIGHT.aggregate(_protocol(), REAL_MPRA_ASSET, changed_path)
    assert parser_calls == []


def test_wrong_asset_identity_stops_before_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_real_assets()
    parser_calls: list[str] = []
    monkeypatch.setattr(
        PREFLIGHT,
        "_parse_publisher_table_s5",
        lambda _: parser_calls.append("table"),
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_parse_official_mpra",
        lambda *_: parser_calls.append("mpra"),
    )
    with pytest.raises(PREFLIGHT.AssetError, match="byte identity differs"):
        PREFLIGHT.aggregate(_protocol(), REAL_TABLE_S5_ASSET, REAL_MPRA_ASSET)
    assert parser_calls == []


def test_identity_failure_in_execute_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_real_assets()
    protocol_path = _write_json(tmp_path / "bound-protocol.json", _bound_protocol())
    substitute = tmp_path / "same-schema-substitution.txt.gz"
    with gzip.open(substitute, "wt", encoding="utf-8") as handle:
        handle.write(" ".join(PREFLIGHT.MPRA_HEADER) + "\n")
        handle.write(" ".join(["x"] * len(PREFLIGHT.MPRA_HEADER)) + "\n")
    parser_calls: list[str] = []
    publication_calls: list[str] = []
    monkeypatch.setattr(
        PREFLIGHT,
        "_default_binding_auditor",
        lambda *_: {"status": "TEST_BOUND"},
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_parse_publisher_table_s5",
        lambda _: parser_calls.append("table"),
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_parse_official_mpra",
        lambda *_: parser_calls.append("mpra"),
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_atomic_publish",
        lambda *_: publication_calls.append("publish"),
    )
    output = tmp_path / PREFLIGHT.REPORT_FILENAME
    with pytest.raises(PREFLIGHT.AssetError, match="identity differs"):
        PREFLIGHT.execute(
            protocol_path,
            substitute,
            REAL_TABLE_S5_ASSET,
            output,
            repo_root=tmp_path,
        )
    assert parser_calls == []
    assert publication_calls == []
    assert not output.exists()


def test_real_assets_close_join_and_formula_but_nonfinite_endpoint_stops_gate(
    real_report: dict[str, object],
) -> None:
    report = real_report
    assert report["status"] == PREFLIGHT.STATUS_STOP
    assert report["mutually_exclusive_role_status"] == (
        "NEITHER_CURRENTLY_JUSTIFIED_NOT_ASSIGNED_NOT_QUALIFIED"
    )
    assert report["geometry_compatibility_observation"] == (
        "PROVISIONAL_TRUE_A2_CANDIDATE_DENSE_NEIGHBORHOOD_ONLY_"
        "NOT_PRIMARY_ROLE_EVIDENCE_NOT_CREDIT"
    )
    assert report["aggregate_gate_summary"]["status_counts"] == {
        PREFLIGHT.FAIL: 5,
        PREFLIGHT.PASS: 1,
        PREFLIGHT.UNKNOWN: 7,
    }
    gates = report["gates"]
    assert gates[PREFLIGHT.GATE_IDS[0]]["status"] == PREFLIGHT.UNKNOWN
    assert gates[PREFLIGHT.GATE_IDS[1]]["status"] == PREFLIGHT.PASS
    assert gates[PREFLIGHT.GATE_IDS[2]]["status"] == PREFLIGHT.FAIL
    assert gates[PREFLIGHT.GATE_IDS[3]]["status"] == PREFLIGHT.UNKNOWN
    assert gates[PREFLIGHT.GATE_IDS[4]]["status"] == PREFLIGHT.FAIL
    assert gates[PREFLIGHT.GATE_IDS[5]] == {
        "status": PREFLIGHT.FAIL,
        "reason": (
            "NONFINITE_OR_UNDEFINED_ENDPOINT_PRESENT_WITHOUT_AUTHORITATIVE_CENSOR_RULE"
        ),
    }
    assert gates[PREFLIGHT.GATE_IDS[6]]["status"] == PREFLIGHT.FAIL
    assert gates[PREFLIGHT.GATE_IDS[7]]["status"] == PREFLIGHT.FAIL
    assert all(
        gates[PREFLIGHT.GATE_IDS[index]]["status"] == PREFLIGHT.UNKNOWN
        for index in range(8, 13)
    )


def test_real_asset_geometry_is_exact_and_does_not_gate_subtype_equality(
    real_report: dict[str, object],
) -> None:
    asset = real_report["aggregate_observation"]["asset_schema_geometry"]
    assert asset["verified_asset_identity_count"] == 2
    assert asset["publisher_library_member_count"] == 6113
    assert asset["publisher_unique_member_key_count"] == 6113
    assert asset["processed_measurement_row_count"] == 366780
    assert asset["processed_distinct_joined_member_count"] == 6113
    assert asset["processed_unmatched_row_count"] == 0
    assert asset["publisher_unseen_member_count"] == 0
    assert asset["join_crosscheck_mismatch_row_count"] == 0
    assert asset["complete_context_member_count"] == 6113
    assert asset["incomplete_context_member_count"] == 0
    assert asset["context_duplicate_row_count"] == 0
    assert asset["publisher_design_count"] == 3802
    assert asset["declared_member_total"] == 6118
    assert asset["observed_member_total"] == 6113
    assert asset["declared_multiplicity_mismatch_design_count"] == 5
    assert asset["observed_member_count_by_design_histogram"] == {
        "1": 3117,
        "2": 2,
        "3": 210,
        "4": 3,
        "5": 470,
    }
    assert asset["generic_design_class_refinement_row_count"] == 47880
    assert asset["generic_design_class_refinement_member_count"] == 798
    assert asset["generic_design_class_refinement_class_count"] == 3


def test_real_source_replay_and_measurement_geometry_is_exact(
    real_report: dict[str, object],
) -> None:
    family = real_report["aggregate_observation"]["source_family_geometry"]
    assert family["candidate_source_family_count"] == 373
    assert family["literal_source_anchored_family_count"] == 373
    assert family["literal_source_missing_family_count"] == 0
    assert family["literal_source_ambiguous_family_count"] == 0
    assert family["candidate_design_count"] == 3429
    assert family["below_minimum_candidate_family_count"] == 1
    assert family["invalid_source_family_count"] == 1

    replay = real_report["aggregate_observation"]["legal_substitution_replay"]
    assert replay["candidate_design_count"] == 3429
    assert replay["source_unanchored_candidate_count"] == 0
    assert replay["sequence_diff_replayable_candidate_count"] == 3429
    assert replay["invalid_length_or_alphabet_candidate_count"] == 0
    assert replay["zero_edit_candidate_count"] == 0
    assert replay["declared_legal_edit_annotation_candidate_count"] == 0
    assert replay["row_order_inference_count"] == 0

    assay = real_report["aggregate_observation"][
        "assay_endpoint_and_replicate_geometry"
    ]
    assert assay["sample_label_count"] == 12
    assert assay["perturbation_label_count"] == 3
    assert assay["distal_reporter_context_count"] == 5
    assert assay["sample_context_label_mismatch_row_count"] == 0
    assert assay["count_and_endpoint_formula_mismatch_row_count"] == 0
    assert assay["endpoint_formula_replay_status"] == PREFLIGHT.PASS
    assert assay["nonfinite_or_undefined_endpoint_row_count"] == 82908
    assert assay["endpoint_value_class_counts"] == {
        "FINITE": 283872,
        "NEGATIVE_INFINITY": 44795,
        "POSITIVE_INFINITY": 26971,
        "UNDEFINED_ZERO_TOTAL": 11142,
    }
    assert assay["biological_independence_authority_present"] is False
    assert assay["reported_standard_error_field_present"] is False


def test_real_report_is_aggregate_only_and_delta_zero(
    real_report: dict[str, object],
) -> None:
    PREFLIGHT._default_privacy_validator(real_report)
    serialized = json.dumps(real_report, sort_keys=True)
    assert "ACTCGCCTATACCTAGAACA" not in serialized
    assert "CSTF3gA-rep1" not in serialized
    assert "barcoded_seq_184bp" not in serialized
    assert '"barcode"' not in serialized
    assert real_report["internal_access_attestation"][
        "member_identifier_sequence_row_effect_se_or_split_output_count"
    ] == 0
    assert real_report["no_promotion_state"]["registry_role"] == "AUDIT_ONLY"
    assert real_report["no_promotion_state"]["dataset_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }


def test_fixed_privacy_validator_rejects_member_payload(
    real_report: dict[str, object],
) -> None:
    poisoned = copy.deepcopy(real_report)
    poisoned["barcode"] = "ACTCGCCTATACCTAGAACA"
    with pytest.raises(PREFLIGHT.OutputError, match="forbidden member field"):
        PREFLIGHT._default_privacy_validator(poisoned)


def test_atomic_publisher_creates_one_report_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / PREFLIGHT.REPORT_FILENAME
    payload = PREFLIGHT._json_bytes({"status": PREFLIGHT.STATUS_STOP})
    PREFLIGHT._atomic_publish(output, payload)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == PREFLIGHT.STATUS_STOP
    with pytest.raises(PREFLIGHT.OutputError, match="overwrite"):
        PREFLIGHT._atomic_publish(output, payload)


def test_atomic_publisher_preserves_late_target_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / PREFLIGHT.REPORT_FILENAME
    late_bytes = b"late-owner-bytes-must-survive\n"
    real_link = PREFLIGHT.os.link

    def create_late_target_then_link(source: str, destination: Path) -> None:
        Path(destination).write_bytes(late_bytes)
        real_link(source, destination)

    monkeypatch.setattr(PREFLIGHT.os, "link", create_late_target_then_link)
    with pytest.raises(PREFLIGHT.OutputError, match="appeared during atomic publish"):
        PREFLIGHT._atomic_publish(output, b"candidate-bytes-must-not-overwrite\n")
    assert output.read_bytes() == late_bytes
    assert not tuple(tmp_path.glob(f".{PREFLIGHT.REPORT_FILENAME}.*.tmp"))


def test_power_boundary_remains_prefrozen_at_source_group_not_rows() -> None:
    assert PREFLIGHT.required_effective_n(
        rho=0.25,
        alpha=0.05,
        target_power=0.8,
        confidence=0.95,
        max_width=0.3,
    ) == 156
    assert PREFLIGHT.fisher_power(156, 0.25, 0.05) >= 0.8
    assert PREFLIGHT.fisher_ci_width(156, 0.25, 0.95) <= 0.3
    assert PREFLIGHT.fisher_ci_width(155, 0.25, 0.95) > 0.3
    assert math.isfinite(PREFLIGHT.fisher_power(156, 0.25, 0.05))
