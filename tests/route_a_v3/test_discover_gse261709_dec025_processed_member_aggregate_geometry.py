from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs/route_a_v3_gse261709_dec025_processed_member_aggregate_discovery_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts/route_a_v3/discover_gse261709_dec025_processed_member_aggregate_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("gse261709_dec025_discovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DISCOVERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DISCOVERY
SPEC.loader.exec_module(DISCOVERY)


def _protocol() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    DISCOVERY._validate_protocol(value)
    return value


def _bound_protocol() -> dict:
    value = copy.deepcopy(_protocol())
    candidate = value["implementation_binding"]["candidate_group"]
    candidate.update(
        {
            "status": DISCOVERY.BOUND,
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    DISCOVERY._validate_protocol(value)
    return value


def _payload(index: int) -> bytes:
    rows = [
        "barcode\tcount",
        "COMMON_SECRET\t1",
        f"UNIQUE_SECRET_{index}\t{index + 2}",
    ]
    if index < 3:
        rows.append("THREE_MEMBER_SECRET\t4")
    if index == 0:
        rows.append("COMMON_SECRET\t5")
    if index == 1:
        rows.append("\t6")
    if index == 2:
        rows.append("MISSING_COUNT_SECRET\tNA")
    if index == 3:
        rows.append("INVALID_COUNT_SECRET\tnegative")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _profiles() -> list:
    return [DISCOVERY._parse_member_payload(_payload(index)) for index in range(7)]


def test_reviewed_successor_freezes_bound_owner_exact_asset_directory_and_lineage() -> None:
    protocol = _protocol()
    assert protocol["protocol_status"] == DISCOVERY.ACTIVE_DISCOVERY_STATUS
    assert protocol["owner_decision"]["status"] == DISCOVERY.BOUND
    assert protocol["owner_decision"]["exact_approval_text"] == (
        DISCOVERY.OWNER_APPROVAL_TEXT
    )
    assert protocol["owner_decision"]["activation_instruction_exact"] == (
        DISCOVERY.OWNER_ACTIVATION_INSTRUCTION
    )
    predecessor = protocol["implementation_binding"]["current_predecessor"]
    assert predecessor["implementation_commit"] == DISCOVERY.GSE269_IMPLEMENTATION_COMMIT
    assert predecessor["binding_commit"] == DISCOVERY.GSE269_BINDING_COMMIT
    failed = protocol["implementation_binding"]["failed_outer_identity_attempt_group"]
    assert failed["implementation_commit"] == DISCOVERY.DEC025_I1_COMMIT
    assert failed["binding_commit"] == DISCOVERY.DEC025_B1_COMMIT
    assert failed["member_payload_open_count"] == 0
    assert failed["report_publication_count"] == 0
    candidate = protocol["implementation_binding"]["candidate_group"]
    assert [candidate[field] for field in DISCOVERY.OWN_BINDING_FIELDS] == [
        DISCOVERY.UNKNOWN
    ] * 4
    assert tuple(candidate["implementation_changed_paths_exactly"]) == DISCOVERY.EXACT3
    assert candidate["binding_changed_paths_exactly"] == [DISCOVERY.CONFIG_REPO_PATH]

    asset = protocol["ordinary_public_asset"]
    assert asset["filename"] == DISCOVERY.OUTER_FILENAME
    assert asset["byte_count"] == 706560
    assert asset["sha256"] == DISCOVERY.OUTER_SHA256
    assert [
        (item["filename"], item["gzip_byte_count"])
        for item in asset["tar_members_exactly"]
    ] == list(DISCOVERY.MEMBERS)
    assert asset["network_access_allowed"] is False
    assert asset["archive_or_member_access_before_owner_and_binding_bound"] is False
    assert protocol["identity_correction"] == DISCOVERY.IDENTITY_CORRECTION


def test_unbound_implementation_stops_before_repository_archive_or_output_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"repository": 0}

    def bomb_repository(_protocol: dict) -> dict:
        calls["repository"] += 1
        raise AssertionError("repository audit must not run")

    monkeypatch.setattr(DISCOVERY, "_audit_repository", bomb_repository)
    archive = tmp_path / DISCOVERY.OUTER_FILENAME
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(DISCOVERY.AuthorityNotBound, match="implementation binding"):
        DISCOVERY.execute(CONFIG_PATH, archive, output_dir)
    assert calls == {"repository": 0}
    assert not archive.exists()
    assert not output_dir.exists()


def test_partial_or_in_place_binding_is_rejected() -> None:
    protocol = _protocol()
    protocol["implementation_binding"]["candidate_group"][
        "implementation_commit"
    ] = "1" * 40
    with pytest.raises(DISCOVERY.ProtocolError, match="partially bound"):
        DISCOVERY._validate_protocol(protocol)

    protocol = _protocol()
    protocol["owner_decision"]["current_candidate_may_be_activated_in_place"] = True
    with pytest.raises(DISCOVERY.ProtocolError, match="must not activate in place"):
        DISCOVERY._validate_protocol(protocol)

    protocol = _protocol()
    protocol["owner_decision"]["status"] = DISCOVERY.NOT_GRANTED
    with pytest.raises(DISCOVERY.ProtocolError, match="must remain DRAFT"):
        DISCOVERY._validate_protocol(protocol)


def test_fixed_in_memory_parser_and_aggregate_geometry_are_exact() -> None:
    profiles = _profiles()
    first = profiles[0]
    assert first.row_count == 4
    assert first.column_count == 2
    assert first.numeric_column_count == 1
    assert first.unique_token_count == 3
    assert first.duplicate_token_row_count == 1
    assert profiles[1].missing_barcode_row_count == 1
    assert profiles[2].missing_numeric_cell_count == 1
    assert profiles[3].invalid_numeric_cell_count == 1

    aggregate = DISCOVERY._aggregate_profiles(profiles)
    assert aggregate["aggregate_schema_geometry"] == {
        "member_schema_observation_count": 7,
        "distinct_normalized_header_schema_count": 1,
        "members_per_normalized_header_schema_histogram": {"7": 1},
        "same_normalized_header_schema_across_all_members": True,
        "barcode_role_column_count_per_member_histogram": {"1": 7},
        "numeric_count_column_count_per_member_histogram": {"1": 7},
        "actual_header_name_output_count": 0,
    }
    cross = aggregate["aggregate_cross_sample_join_geometry"]
    assert cross["union_token_count"] == 11
    assert cross["all_member_intersection_token_count"] == 1
    assert cross["member_presence_cardinality_histogram"] == {
        "1": 9,
        "3": 1,
        "7": 1,
    }
    assert cross["pairwise_comparison_count"] == 21
    assert cross["pairwise_intersection_size_histogram"] == {"1": 18, "2": 3}
    assert cross["member_or_barcode_value_output_count"] == 0


def test_report_contains_only_aggregate_geometry_and_all_science_remains_stop() -> None:
    report = DISCOVERY._build_report(
        _bound_protocol(),
        _profiles(),
        binding={
            "status": "SYNTHETIC_TEST_BINDING_ONLY",
            "implementation_commit": "1" * 40,
            "binding_commit": "4" * 40,
            "predecessor_binding_commit": DISCOVERY.GSE269_BINDING_COMMIT,
        },
        generated_at_utc="2026-08-14T13:00:00Z",
    )
    serialized = json.dumps(report, sort_keys=True)
    for poison in (
        "COMMON_SECRET",
        "THREE_MEMBER_SECRET",
        "UNIQUE_SECRET_0",
        "MISSING_COUNT_SECRET",
        "INVALID_COUNT_SECRET",
        *[name for name, _ in DISCOVERY.MEMBERS],
    ):
        assert poison not in serialized
    assert report["status"] == DISCOVERY.TERMINAL_STATUS
    stop = report["downstream_stop_state"]
    assert set(stop["scientific_gates"]) == set(DISCOVERY.SCIENTIFIC_GATE_IDS)
    assert set(stop["scientific_gates"].values()) == {DISCOVERY.STOP_GATE}
    assert stop["qualifier_input_emission_count"] == 0
    assert stop["qualifier_invocation_count"] == 0
    assert stop["credit_or_canonical_delta"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert list(inspect.signature(DISCOVERY.execute).parameters) == [
        "protocol_path",
        "archive_path",
        "output_dir",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        b"count\tvalue\n1\t2\n",
        b"barcode\tbc\tcount\nA\tB\t1\n",
        b"barcode\tcount\nA\t1\tEXTRA\n",
        b"barcode\tbarcode\nA\t1\n",
    ],
)
def test_fixed_parser_rejects_unsupported_or_ambiguous_schema(payload: bytes) -> None:
    with pytest.raises(DISCOVERY.AssetError):
        DISCOVERY._parse_member_payload(payload)


def test_atomic_publication_never_replaces_existing_report(tmp_path: Path) -> None:
    report = {"status": DISCOVERY.TERMINAL_STATUS, "value": 1}
    final_path = DISCOVERY._publish_atomic_no_replace(tmp_path / "out", report)
    first_bytes = final_path.read_bytes()
    with pytest.raises(DISCOVERY.PublicationError, match="refusing replacement"):
        DISCOVERY._publish_atomic_no_replace(
            tmp_path / "out", {"status": DISCOVERY.TERMINAL_STATUS, "value": 2}
        )
    assert final_path.read_bytes() == first_bytes
    assert not list(final_path.parent.glob(f".{DISCOVERY.OUTPUT_FILENAME}.*"))
