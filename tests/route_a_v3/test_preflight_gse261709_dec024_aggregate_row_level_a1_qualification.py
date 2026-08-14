from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from statistics import stdev

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs/route_a_v3_gse261709_dec024_aggregate_row_level_a1_qualification_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts/route_a_v3/preflight_gse261709_dec024_aggregate_row_level_a1_qualification.py"
)
SPEC = importlib.util.spec_from_file_location("gse261709_dec024_row_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _protocol() -> dict:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(value)
    return value


def _runtime_bound(protocol: dict) -> dict:
    value = copy.deepcopy(protocol)
    runtime = value["implementation_binding"]["authority_group"][
        "authority_runtime_lifecycle"
    ]
    assert runtime["status"] == PREFLIGHT.BOUND
    PREFLIGHT._validate_protocol(value)
    return value


def _implementation_bound(protocol: dict) -> dict:
    value = _runtime_bound(protocol)
    binding = value["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = "3" * 40
    binding["implementation_script_sha256"] = "c" * 64
    binding["implementation_test_sha256"] = "d" * 64
    PREFLIGHT._validate_protocol(value)
    return value


def _source(index: int, width: int = 8) -> str:
    alphabet = "ACGT"
    chars = []
    value = index
    for _ in range(width):
        chars.append(alphabet[value % 4])
        value //= 4
    return "".join(chars)


def _mutate(source: str, position: int) -> str:
    alt = {"A": "C", "C": "G", "G": "T", "T": "A"}
    chars = list(source)
    chars[position] = alt[chars[position]]
    return "".join(chars)


def _record(group: int, candidate_index: int) -> dict:
    source = _source(group)
    candidate = _mutate(source, candidate_index)
    desired_effects = [
        float(candidate_index - 1),
        float(candidate_index),
        float(candidate_index + 1),
    ]
    dna = [99.0, 99.0, 99.0]
    rna = [100.0 * (2.0**effect) - 1.0 for effect in desired_effects]
    effects = [
        PREFLIGHT._replicate_effect(left, right, 1.0)
        for left, right in zip(rna, dna)
    ]
    cell = "AGS" if group % 2 == 0 else "SNU719"
    return {
        "source_group_token": f"SOURCE_POISON_{group}",
        "candidate_token": f"CANDIDATE_POISON_{group}_{candidate_index}",
        "near_duplicate_component_token": f"COMPONENT_POISON_{group}",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "construct_context": "FULL_REPORTER_CONTEXT",
        "cell_context": cell,
        "barcode_to_allele_join_closed": True,
        "allele_to_transcript_join_closed": True,
        "transcript_to_source_join_closed": True,
        "full_construct_join_closed": True,
        "replicate_roles": ["BIOLOGICAL_1", "BIOLOGICAL_2", "BIOLOGICAL_3"],
        "replicate_sample_provenance_tokens": [
            f"{cell}_BIOLOGICAL_1",
            f"{cell}_BIOLOGICAL_2",
            f"{cell}_BIOLOGICAL_3",
        ],
        "replicate_independence_closed": True,
        "rna_counts": rna,
        "dna_counts": dna,
        "replicate_effects": effects,
        "reported_effect": sum(effects) / 3.0,
        "reported_standard_error": stdev(effects) / math.sqrt(3.0),
        "endpoint_direction": "HIGHER_REPORTED_EFFECT_IS_GREATER_REPORTER_ACTIVITY",
        "missing": False,
        "censored": False,
        "qc_pass": True,
    }


def _records(group_count: int = 156) -> list[dict]:
    return [
        _record(group, candidate)
        for group in range(group_count)
        for candidate in range(3)
    ]


def _split() -> dict:
    return {
        "status": PREFLIGHT.PASS,
        "outcome_blind": True,
        "components_indivisible": True,
        "split_executed": False,
        "assignment_output_count": 0,
        "source_group_leakage_count": 0,
        "exact_sequence_leakage_count": 0,
        "near_duplicate_leakage_count": 0,
        "reverse_edge_leakage_count": 0,
        "candidate_leakage_count": 0,
        "study_context_leakage_count": 0,
    }


def _evidence() -> dict:
    return {
        "processed_asset_identity_status": PREFLIGHT.PASS,
        "processed_asset_role_and_primary_measurement_route_status": PREFLIGHT.PASS,
        "processed_asset_schema_role_binding_status": PREFLIGHT.PASS,
        "endpoint_formula_and_primary_documentation_status": PREFLIGHT.PASS,
        "biological_replicate_sample_role_provenance_status": PREFLIGHT.PASS,
        "license_and_reuse_rights_status": PREFLIGHT.PASS,
        "historical_analytic_or_checkpoint_exposure_status": PREFLIGHT.PASS,
        "split_readiness": _split(),
    }


def _gate_map(report: dict) -> dict[str, str]:
    return {item["gate_id"]: item["status"] for item in report["gate_results"]}


def _synthetic_schema_manifest(row_count: int) -> dict:
    headers = [f"HEADER_{field}" for field in PREFLIGHT.ASSET_FIELD_KEYS]
    return {
        "header_names_exactly": headers,
        "column_count": len(headers),
        "row_count_excluding_header": row_count,
        "field_columns_exactly": dict(zip(PREFLIGHT.ASSET_FIELD_KEYS, headers)),
        "endpoint_pseudocount": 1.0,
        "primary_measurement_route_status": PREFLIGHT.PASS,
        "endpoint_formula_and_primary_documentation_status": PREFLIGHT.PASS,
        "biological_replicate_sample_role_provenance_status": PREFLIGHT.PASS,
        "license_and_reuse_rights_status": PREFLIGHT.PASS,
        "historical_analytic_or_checkpoint_exposure_status": PREFLIGHT.PASS,
        "split_readiness": _split(),
    }


def _tsv_text() -> str:
    manifest = _synthetic_schema_manifest(1)
    fields = {field: "VALUE" for field in PREFLIGHT.ASSET_FIELD_KEYS}
    fields.update(
        {
            "barcode_token": "BARCODE_POISON",
            "allele_token": "ALLELE_POISON",
            "transcript_token": "TRANSCRIPT_POISON",
            "source_join_token": "SOURCE_JOIN_POISON",
            "full_construct_token": "FULL_CONSTRUCT_POISON",
            "source_group_token": "SOURCE_POISON_0",
            "candidate_token": "CANDIDATE_POISON_0_0",
            "near_duplicate_component_token": "COMPONENT_POISON_0",
            "source_sequence": "AAAAAAAA",
            "candidate_sequence": "CAAAAAAA",
            "construct_context": "FULL_REPORTER_CONTEXT",
            "cell_context": "AGS",
            "biological_1_role": "BIOLOGICAL_1",
            "biological_2_role": "BIOLOGICAL_2",
            "biological_3_role": "BIOLOGICAL_3",
            "biological_1_sample_provenance": "AGS_BIOLOGICAL_1",
            "biological_2_sample_provenance": "AGS_BIOLOGICAL_2",
            "biological_3_sample_provenance": "AGS_BIOLOGICAL_3",
            "biological_1_rna_count": "49",
            "biological_1_dna_count": "99",
            "biological_2_rna_count": "99",
            "biological_2_dna_count": "99",
            "biological_3_rna_count": "199",
            "biological_3_dna_count": "99",
            "reported_effect": "0",
            "reported_standard_error": str(1.0 / math.sqrt(3.0)),
            "endpoint_direction": "HIGHER_REPORTED_EFFECT_IS_GREATER_REPORTER_ACTIVITY",
            "missing": "false",
            "censored": "false",
            "qc_pass": "true",
        }
    )
    columns = manifest["field_columns_exactly"]
    row = [fields[field] for field in PREFLIGHT.ASSET_FIELD_KEYS]
    assert [columns[field] for field in PREFLIGHT.ASSET_FIELD_KEYS] == manifest[
        "header_names_exactly"
    ]
    return "\t".join(manifest["header_names_exactly"]) + "\n" + "\t".join(row) + "\n"


def test_protocol_freezes_exact_authority_chain_and_outer_archive_directory_only() -> None:
    protocol = _protocol()
    binding = protocol["implementation_binding"]
    authority = binding["authority_group"]
    assert authority["authority_commit"] == PREFLIGHT.AUTHORITY_COMMIT
    assert authority["authority_expected_parent"] == PREFLIGHT.AUTHORITY_PARENT
    assert tuple(authority["authority_exact_changed_paths"]) == PREFLIGHT.AUTHORITY_EXACT12
    runtime = authority["authority_runtime_lifecycle"]
    assert runtime["status"] == PREFLIGHT.BOUND
    predecessor = runtime["mandatory_non_authoritative_predecessor"]
    assert predecessor["commit"] == PREFLIGHT.A6_G0_ENGINEERING_COMMIT
    assert predecessor["expected_parent"] == PREFLIGHT.AUTHORITY_COMMIT
    assert tuple(predecessor["exact_changed_paths"]) == PREFLIGHT.A6_G0_ENGINEERING_EXACT4
    assert predecessor["changes_dec024_authority"] is False
    assert predecessor["changes_scientific_state"] is False
    assert runtime["implementation_commit"] == (
        "f955ca5a1714af57f706ee2ddf0a6825ad4737de"
    )
    assert runtime["implementation_expected_parent"] == PREFLIGHT.A6_G0_ENGINEERING_COMMIT
    assert runtime["implementation_blob_sha256_by_path"] == {
        "configs/route_a_v3_dec024_authority_runtime_sync_v1.json": (
            "3dbbdaed8458c6eb68af1c11d890b66c4116457b84df96dd84ef8622be0fd669"
        ),
        "scripts/route_a_v3/dec024_authority_runtime_sync.py": (
            "6d9614b53e160fe38bbf280c310d6adec773b790b48f93374448ff6b29e5bd3b"
        ),
        "tests/route_a_v3/test_dec024_authority_runtime_sync.py": (
            "b22d92aeea7f3c0dc754dd739daa1d5025af64e96e86a84e7b0458b1788c3799"
        ),
    }
    assert runtime["binding_commit"] == "e3c3416e24e0298ab792a1e0998018125c907ffa"
    assert runtime["binding_expected_parent"] == runtime["implementation_commit"]
    assert runtime["binding_blob_sha256_by_path"] == {
        "configs/route_a_v3_dec024_authority_runtime_sync_v1.json": (
            "1b1a3159fc7b08aeb967983f3b651bdb8cba182829c22e767b6a9e9dad1fb7e1"
        ),
        "scripts/route_a_v3/dec024_authority_runtime_sync.py": (
            "6d9614b53e160fe38bbf280c310d6adec773b790b48f93374448ff6b29e5bd3b"
        ),
        "tests/route_a_v3/test_dec024_authority_runtime_sync.py": (
            "b22d92aeea7f3c0dc754dd739daa1d5025af64e96e86a84e7b0458b1788c3799"
        ),
    }
    projection = binding["mandatory_non_science_current_projection_predecessor"]
    assert projection["commit"] == PREFLIGHT.EVT058_PROJECTION_COMMIT
    assert tuple(projection["exact_changed_paths"]) == PREFLIGHT.EVT058_PROJECTION_EXACT4
    assert projection["blob_sha256_by_path"] == PREFLIGHT.EVT058_PROJECTION_BLOBS
    assert projection["changes_dec024_authority"] is False
    assert projection["changes_scientific_state"] is False
    assert tuple(binding["unknown_to_bound_scalar_paths"]) == PREFLIGHT.BINDING_SCALAR_PATHS

    manifest = protocol["processed_asset_contract"]["official_processed_asset_manifest"]
    assert manifest["status"] == PREFLIGHT.ARCHIVE_DIRECTORY_BOUND
    assert manifest["byte_count"] == 667648
    assert manifest["sha256"] == (
        "3024746ce25f4b795daa376ac6dbafd3d53f6d30be8aed9fb14db0f118c6f434"
    )
    assert [
        (item["filename"], item["gzip_byte_count"])
        for item in manifest["tar_member_directory_exactly"]
    ] == list(PREFLIGHT.ARCHIVE_MEMBER_DIRECTORY)
    assert manifest["member_body_access_authority_status"] == (
        "EXPLICIT_USER_AUTHORITY_REQUIRED_NOT_GRANTED"
    )
    assert all(manifest[field] == PREFLIGHT.UNKNOWN for field in PREFLIGHT.MEMBER_SCHEMA_FIELDS)


def test_partial_runtime_own_or_member_schema_group_is_rejected() -> None:
    protocol = _protocol()
    runtime = protocol["implementation_binding"]["authority_group"][
        "authority_runtime_lifecycle"
    ]
    runtime["status"] = PREFLIGHT.UNKNOWN
    with pytest.raises(PREFLIGHT.ProtocolError, match="partial UNKNOWN authority-runtime"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _runtime_bound(_protocol())
    protocol["implementation_binding"]["implementation_commit"] = "3" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="partial UNKNOWN exact3"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _protocol()
    manifest = protocol["processed_asset_contract"]["official_processed_asset_manifest"]
    manifest["encoding"] = "UTF-8"
    with pytest.raises(PREFLIGHT.ProtocolError, match="partial processed member schema"):
        PREFLIGHT._validate_protocol(protocol)


def test_repository_auditor_orders_projection_between_runtime_b_and_preflight_i() -> None:
    source = inspect.getsource(PREFLIGHT._default_binding_auditor)
    runtime_position = source.index('label="DEC024 authority-runtime B"')
    projection_position = source.index('label="EVT058 non-science current projection P"')
    preflight_position = source.index('label="GSE261709 preflight I"')
    assert runtime_position < projection_position < preflight_position
    assert "expected_parent=runtime_b" in source[projection_position:preflight_position]
    assert "expected_parent=EVT058_PROJECTION_COMMIT" in source[preflight_position:]


def test_runtime_and_preflight_binding_still_stop_before_member_or_output_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _implementation_bound(_protocol())
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="manifest"):
        PREFLIGHT._ensure_ready_before_asset_or_output_io(protocol)

    calls = {"binding": 0, "reader": 0, "publisher": 0}

    def poison(name: str):
        def callback(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} crossed the member authority barrier")

        return callback

    monkeypatch.setattr(PREFLIGHT, "_default_binding_auditor", poison("binding"))
    monkeypatch.setattr(PREFLIGHT, "read_bound_processed_asset", poison("reader"))
    monkeypatch.setattr(PREFLIGHT, "_write_exclusive_aggregate", poison("publisher"))
    bound_path = tmp_path / "bound.json"
    bound_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="manifest"):
        PREFLIGHT.execute(
            bound_path,
            tmp_path / "GSE261709_RAW.tar",
            output,
            repo_root=tmp_path,
        )
    assert calls == {"binding": 0, "reader": 0, "publisher": 0}
    assert not output.exists()


def test_synthetic_all_pass_is_permanently_implementation_only() -> None:
    protocol = _protocol()
    report = PREFLIGHT.aggregate_canonical_records(_records(), _evidence(), protocol)
    assert report["status"] == protocol["gate_contract"]["synthetic_all_pass_action"]
    assert report["evidence_status"] == PREFLIGHT.SYNTHETIC_EVIDENCE
    assert report["gate_status_counts"] == {
        PREFLIGHT.PASS: 12,
        PREFLIGHT.BLOCKED: 0,
        PREFLIGHT.UNKNOWN: 0,
    }
    assert list(_gate_map(report)) == list(PREFLIGHT.GATE_IDS)
    assert report["qualification_changed"] is False
    assert report["credit_changed"] is False
    assert report["canonical_changed"] is False
    assert report["frozen_outer_truth"]["gse261709_qualified"] is False
    power = report["aggregate_observations"][
        "post_dedup_effective_n_and_power_readiness"
    ]
    assert power["effective_n"] == 156
    assert power["required_effective_n"] == 156
    assert power["formal_qualification_power_gate_executed"] is False

    with pytest.raises(TypeError):
        PREFLIGHT.aggregate_canonical_records(
            _records(),
            _evidence(),
            protocol,
            evidence_class=PREFLIGHT.PUBLIC_PROCESSED_EVIDENCE,
        )
    with pytest.raises(PREFLIGHT.ObservationError, match="production asset path"):
        PREFLIGHT._aggregate_records(
            _records(),
            _evidence(),
            protocol,
            evidence_class=PREFLIGHT.PUBLIC_PROCESSED_EVIDENCE,
            production_asset_verified=False,
        )


@pytest.mark.parametrize(
    ("mutator", "gate_index", "expected_status"),
    [
        (
            lambda rows, evidence: evidence.update(
                processed_asset_identity_status=PREFLIGHT.UNKNOWN
            ),
            0,
            PREFLIGHT.UNKNOWN,
        ),
        (
            lambda rows, evidence: rows[0].update(barcode_to_allele_join_closed=False),
            1,
            PREFLIGHT.BLOCKED,
        ),
        (lambda rows, evidence: rows.__setitem__(slice(None), rows[:2]), 2, PREFLIGHT.BLOCKED),
        (
            lambda rows, evidence: rows[0].update(
                candidate_sequence=rows[0]["candidate_sequence"] + "A"
            ),
            3,
            PREFLIGHT.BLOCKED,
        ),
        (lambda rows, evidence: rows[0].update(reported_effect=9.0), 4, PREFLIGHT.BLOCKED),
        (
            lambda rows, evidence: rows[0].update(
                replicate_sample_provenance_tokens=["SAME", "SAME", "SAME"]
            ),
            5,
            PREFLIGHT.BLOCKED,
        ),
        (lambda rows, evidence: rows[0].update(missing=True), 6, PREFLIGHT.BLOCKED),
        (
            lambda rows, evidence: evidence.update(
                license_and_reuse_rights_status=PREFLIGHT.UNKNOWN
            ),
            7,
            PREFLIGHT.UNKNOWN,
        ),
        (
            lambda rows, evidence: evidence.update(
                historical_analytic_or_checkpoint_exposure_status=PREFLIGHT.UNKNOWN
            ),
            8,
            PREFLIGHT.UNKNOWN,
        ),
        (
            lambda rows, evidence: evidence["split_readiness"].update(
                source_group_leakage_count=1
            ),
            9,
            PREFLIGHT.BLOCKED,
        ),
        (lambda rows, evidence: rows.clear(), 10, PREFLIGHT.BLOCKED),
        (lambda rows, evidence: rows.__setitem__(slice(None), rows[:3]), 11, PREFLIGHT.BLOCKED),
    ],
)
def test_all_twelve_gate_axes_fail_closed(mutator, gate_index: int, expected_status: str) -> None:
    rows = _records()
    evidence = _evidence()
    mutator(rows, evidence)
    report = PREFLIGHT.aggregate_canonical_records(rows, evidence, _protocol())
    assert report["gate_results"][gate_index]["status"] == expected_status
    assert report["status"] == _protocol()["gate_contract"]["nonpass_action"]
    assert report["qualification_changed"] is False


def test_effects_are_replayed_from_counts_and_biological_provenance_is_separate() -> None:
    rows = _records()
    rows[0]["rna_counts"][0] += 10.0
    report = PREFLIGHT.aggregate_canonical_records(rows, _evidence(), _protocol())
    assert report["gate_results"][4]["status"] == PREFLIGHT.BLOCKED
    assert report["gate_results"][5]["status"] == PREFLIGHT.BLOCKED

    evidence = _evidence()
    evidence["biological_replicate_sample_role_provenance_status"] = PREFLIGHT.UNKNOWN
    report = PREFLIGHT.aggregate_canonical_records(_records(), evidence, _protocol())
    assert report["gate_results"][5]["status"] == PREFLIGHT.UNKNOWN


def test_schema_parser_recomputes_effects_and_rejects_header_drift() -> None:
    manifest = _synthetic_schema_manifest(1)
    records, evidence = PREFLIGHT._parse_bound_tsv_text(_tsv_text(), manifest, _protocol())
    assert len(records) == 1
    record = records[0]
    assert record["replicate_effects"] == [-1.0, 0.0, 1.0]
    assert record["reported_effect"] == 0.0
    assert record["replicate_independence_closed"] is True
    assert len(set(record["replicate_sample_provenance_tokens"])) == 3
    assert evidence["biological_replicate_sample_role_provenance_status"] == PREFLIGHT.PASS

    drifted = _tsv_text().replace("HEADER_barcode_token", "WRONG_HEADER", 1)
    with pytest.raises(PREFLIGHT.ObservationError, match="header differs"):
        PREFLIGHT._parse_bound_tsv_text(drifted, manifest, _protocol())


def test_aggregate_output_drops_all_member_sequence_effect_and_provenance_payloads() -> None:
    protocol = _protocol()
    records = _records()
    report = PREFLIGHT.aggregate_canonical_records(records, _evidence(), protocol)
    payload = PREFLIGHT.json_bytes(report, protocol).decode("utf-8")
    for poison in (
        "SOURCE_POISON_0",
        "CANDIDATE_POISON_0_0",
        "COMPONENT_POISON_0",
        "AGS_BIOLOGICAL_1",
        records[0]["source_sequence"],
        str(records[0]["reported_standard_error"]),
    ):
        assert poison not in payload
    assert '"row_count"' not in payload
    assert '"split_assignment"' not in payload

    report["aggregate_observations"]["source_family_size_histogram"]["member_id"] = "x"
    with pytest.raises(PREFLIGHT.OutputError, match="payload key"):
        PREFLIGHT.validate_public_report(report, protocol)


def test_atomic_fixed_name_single_json_publication_is_no_replace(tmp_path: Path) -> None:
    protocol = _protocol()
    report = PREFLIGHT.aggregate_canonical_records(_records(1), _evidence(), protocol)
    payload = PREFLIGHT.json_bytes(report, protocol)
    output_directory = tmp_path / "one-output"
    path = PREFLIGHT._write_exclusive_aggregate(output_directory, payload, protocol)
    assert path.name == protocol["output_contract"]["filename"]
    assert path.read_bytes() == payload
    assert [item.name for item in output_directory.iterdir()] == [path.name]
    with pytest.raises(PREFLIGHT.OutputError, match="not empty"):
        PREFLIGHT._write_exclusive_aggregate(output_directory, b"replacement", protocol)
    assert path.read_bytes() == payload
    assert not list(output_directory.glob("*.tmp"))


def test_execute_has_no_injectable_reader_evidence_or_publisher_callbacks() -> None:
    parameters = inspect.signature(PREFLIGHT.execute).parameters
    assert set(parameters) == {
        "protocol_path",
        "processed_asset_path",
        "output_directory",
        "repo_root",
    }
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "processed_record_reader" not in source
    assert "evidence_reader" not in source
    assert "publisher:" not in source


def test_module_has_no_network_or_filesystem_archive_extraction_path() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)
    assert imported_roots.isdisjoint(
        {"urllib", "requests", "http", "socket", "ftplib", "aiohttp"}
    )
    assert "extract" not in called_attributes
    assert "extractall" not in called_attributes
    assert {"subprocess", "tarfile", "gzip", "hashlib"} <= imported_roots
