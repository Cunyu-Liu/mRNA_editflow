from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping
from xml.etree import ElementTree

import pytest
from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "route_a_v3" / "qualify_gse200304_published_endpoint_a1.py"
CONFIG = ROOT / "configs" / "route_a_v3_gse200304_published_endpoint_a1.json"

SPEC = importlib.util.spec_from_file_location("gse200304_published_endpoint_a1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALIFY
SPEC.loader.exec_module(QUALIFY)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _encode_source(code: int, central_base: str) -> str:
    alphabet = "ACGT"
    values = ["A"] * 201
    cursor = 0
    remaining = code
    while cursor < 16:
        position = cursor if cursor < 100 else cursor + 1
        values[position] = alphabet[remaining % 4]
        remaining //= 4
        cursor += 1
    values[100] = central_base
    return "".join(values)


def _base_changes() -> list[str]:
    counts = Counter(QUALIFY.EXPECTED_S2["central_base_change_counts"])
    first = ["A>C", "A>G", "C>A", "C>G", "G>A", "G>C"]
    for value in first:
        counts[value] -= 1
    result = list(first)
    for value in sorted(counts):
        result.extend([value] * counts[value])
    assert len(result) == 6885 and not any(value < 0 for value in counts.values())
    return result


def _orientations() -> list[str]:
    joined = ["FORWARD"] * 3451 + ["REVERSE_COMPLEMENT"] * 3321
    absent = ["FORWARD"] * 46 + ["REVERSE_COMPLEMENT"] * 67
    result = joined + absent
    assert Counter(result) == {"FORWARD": 3497, "REVERSE_COMPLEMENT": 3388}
    return result


def _build_s2() -> tuple[bytes, list[str]]:
    header = QUALIFY.EXPECTED_S2["exact_header"]
    changes = _base_changes()
    orientations = _orientations()
    rows: list[list[str]] = []
    keys: list[str] = []
    pair_rows: list[tuple[list[str], list[str]]] = []
    complement = QUALIFY.BASE_COMPLEMENT
    for index, (change, orientation) in enumerate(zip(changes, orientations)):
        wt_base, mutant_base = change.split(">")
        if index < 6:
            source_code = index // 2
        else:
            source_code = index - 3
        wt = _encode_source(source_code, wt_base)
        mutant_values = list(wt)
        mutant_values[100] = mutant_base
        mutant = "".join(mutant_values)
        if orientation == "FORWARD":
            ref, alt = wt_base, mutant_base
        else:
            ref, alt = complement[wt_base], complement[mutant_base]
        key = f"synthetic:{index + 1}_{ref}-{alt}"
        keys.append(key)
        five_prime = "A" * 25
        three_prime = "C" * 24
        wt_row = [key, "WT", wt, five_prime, three_prime, five_prime + wt + three_prime]
        mutant_row = [
            key,
            "Mutant",
            mutant,
            five_prime,
            three_prime,
            five_prime + mutant + three_prime,
        ]
        rows.extend([wt_row, mutant_row])
        pair_rows.append((wt_row, mutant_row))
    for index in range(66):
        control = _encode_source(7000 + index, "T")
        rows.append(
            [
                f"synthetic_control_{index + 1}",
                "Control",
                control,
                "A" * 25,
                "C" * 24,
                "A" * 25 + control + "C" * 24,
            ]
        )
    for wt_row, mutant_row in pair_rows[:7]:
        rows.extend([list(wt_row), list(mutant_row)])
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    assert len(rows) == 13850
    return payload, keys


def _patch_formula_caches(
    payload: bytes, cache_for_row: Callable[[int], str]
) -> bytes:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ElementTree.register_namespace("", namespace)
    source = zipfile.ZipFile(io.BytesIO(payload), "r")
    infos = source.infolist()
    members = {info.filename: source.read(info.filename) for info in infos}
    source.close()
    worksheet_name = "xl/worksheets/sheet1.xml"
    root = ElementTree.fromstring(members[worksheet_name])
    for cell in root.findall(f".//{{{namespace}}}c"):
        reference = cell.attrib.get("r", "")
        if not reference.startswith("G"):
            continue
        row_number = int(reference[1:])
        if row_number < 2:
            continue
        formula = cell.find(f"{{{namespace}}}f")
        assert formula is not None
        value = cell.find(f"{{{namespace}}}v")
        if value is None:
            value = ElementTree.SubElement(cell, f"{{{namespace}}}v")
        cell.set("t", "str")
        value.text = cache_for_row(row_number)
    members[worksheet_name] = ElementTree.tostring(
        root, encoding="utf-8", xml_declaration=True
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    return output.getvalue()


def _remove_first_translation_formula(payload: bytes) -> bytes:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ElementTree.register_namespace("", namespace)
    source = zipfile.ZipFile(io.BytesIO(payload), "r")
    infos = source.infolist()
    members = {info.filename: source.read(info.filename) for info in infos}
    source.close()
    worksheet_name = "xl/worksheets/sheet1.xml"
    root = ElementTree.fromstring(members[worksheet_name])
    for cell in root.findall(f".//{{{namespace}}}c"):
        reference = cell.attrib.get("r", "")
        if not reference.startswith("G") or int(reference[1:]) < 2:
            continue
        formula = cell.find(f"{{{namespace}}}f")
        assert formula is not None
        cell.remove(formula)
        cell.set("t", "str")
        break
    members[worksheet_name] = ElementTree.tostring(
        root, encoding="utf-8", xml_declaration=True
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    return output.getvalue()


def _replace_opaque_control_measurement(payload: bytes) -> bytes:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ElementTree.register_namespace("", namespace)
    source = zipfile.ZipFile(io.BytesIO(payload), "r")
    infos = source.infolist()
    members = {info.filename: source.read(info.filename) for info in infos}
    source.close()
    worksheet_name = "xl/worksheets/sheet2.xml"
    root = ElementTree.fromstring(members[worksheet_name])
    cell = next(
        value
        for value in root.findall(f".//{{{namespace}}}c")
        if value.attrib.get("r") == "B2"
    )
    for child in list(cell):
        cell.remove(child)
    cell.set("t", "inlineStr")
    inline = ElementTree.SubElement(cell, f"{{{namespace}}}is")
    text = ElementTree.SubElement(inline, f"{{{namespace}}}t")
    text.text = "NOT_A_MEASUREMENT_AND_MUST_NOT_BE_READ"
    members[worksheet_name] = ElementTree.tostring(
        root, encoding="utf-8", xml_declaration=True
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    return output.getvalue()


def _build_s3(keys: list[str]) -> bytes:
    workbook = Workbook(write_only=False)
    primary = workbook.active
    primary.title = QUALIFY.EXPECTED_S3["primary_sheet"]["name"]
    primary.append(QUALIFY.EXPECTED_S3["primary_sheet"]["exact_header"])
    for index, key in enumerate(keys[:6772]):
        annotation = f"annotation_{index % 1947}"
        high_finite = index < 6538
        total_finite = index < 6547
        primary.append(
            [
                key,
                annotation,
                "HighPoly:RNA",
                float(index % 17) / 10 if high_finite else "NA",
                0.25 if high_finite else "NA",
                0.5 if high_finite else "NA",
                '="cached"',
            ]
        )
        primary.append(
            [
                key,
                annotation,
                "TotalPoly:RNA",
                float(index % 19) / 10 if total_finite else "NA",
                0.2 if total_finite else "NA",
                0.4 if total_finite else "NA",
                '="cached"',
            ]
        )
    control = workbook.create_sheet(QUALIFY.EXPECTED_S3["control_sheet"]["name"])
    control.append(QUALIFY.EXPECTED_S3["control_sheet"]["exact_header"])
    measurement_index = 0
    for row_index in range(29):
        values: list[Any] = [f"opaque_control_{row_index + 1}"]
        for _ in range(12):
            values.append("NA" if measurement_index < 5 else float(measurement_index))
            measurement_index += 1
        control.append(values)
    raw = io.BytesIO()
    workbook.save(raw)
    workbook.close()

    def cache_for_row(row_number: int) -> str:
        data_index = row_number - 2
        pair_index = data_index // 2
        if data_index % 2 == 0:
            return "Significant" if pair_index < 58 else "Not Significant"
        return "Significant" if pair_index < 174 else "Not Significant"

    return _patch_formula_caches(raw.getvalue(), cache_for_row)


def _member(
    template: Mapping[str, Any], payload: bytes
) -> dict[str, Any]:
    value = copy.deepcopy(dict(template))
    value["bytes"] = len(payload)
    value["sha256"] = _sha256(payload)
    return value


def _write_source_bundle(root: Path, s2: bytes, s3: bytes) -> tuple[dict[str, Any], ...]:
    root.mkdir(parents=True)
    templates = {asset["asset_id"]: asset for asset in QUALIFY.EXPECTED_ASSETS}
    runinfo = b"Run,Study\nSYNTHETIC,NO_REAL_DATA\n"
    opaque_archive_buffer = io.BytesIO()
    with zipfile.ZipFile(opaque_archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", "opaque synthetic archive; never opened by qualifier\n")
    archive_payload = opaque_archive_buffer.getvalue()
    scientific_payloads = {
        "NCBI_PRJNA824033_RUNINFO": runinfo,
        "PMC10540565_TABLE_S2": s2,
        "PMC10540565_TABLE_S3": s3,
        "ZENODO_8007705_V1_2": archive_payload,
    }
    members: dict[str, dict[str, Any]] = {}
    for asset_id, payload in scientific_payloads.items():
        members[asset_id] = _member(templates[asset_id], payload)
        (root / members[asset_id]["relative_path"]).write_bytes(payload)

    manifest = {
        "record_type": "ROUTE_A_V3_PUBLIC_ASSET_ACQUISITION_MANIFEST",
        "contract_id": QUALIFY.CONTRACT_ID,
        "dataset_id": QUALIFY.DATASET_ID,
        "status": "ASSETS_ACQUIRED_NOT_QUALIFIED",
        "assets": [
            {
                "filename": members[asset_id]["relative_path"],
                "bytes": members[asset_id]["bytes"],
                "sha256": members[asset_id]["sha256"],
            }
            for asset_id in scientific_payloads
        ],
        "scientific_boundaries": {
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "qualified": False,
            "training_started": False,
            "next_phase_authorized": False,
        },
    }
    manifest_payload = _json_bytes(manifest)
    members["SOURCE_BUNDLE_MANIFEST"] = _member(
        templates["SOURCE_BUNDLE_MANIFEST"], manifest_payload
    )
    (root / "ASSET_ACQUISITION_MANIFEST.json").write_bytes(manifest_payload)

    checksum_ids = {
        "SOURCE_BUNDLE_MANIFEST",
        "NCBI_PRJNA824033_RUNINFO",
        "PMC10540565_TABLE_S2",
        "PMC10540565_TABLE_S3",
        "ZENODO_8007705_V1_2",
    }
    checksum_payload = "".join(
        f"{members[asset_id]['sha256']}  {members[asset_id]['relative_path']}\n"
        for asset_id in sorted(
            checksum_ids, key=lambda value: members[value]["relative_path"]
        )
    ).encode("ascii")
    members["SOURCE_BUNDLE_SHA256SUMS"] = _member(
        templates["SOURCE_BUNDLE_SHA256SUMS"], checksum_payload
    )
    (root / "SHA256SUMS").write_bytes(checksum_payload)

    marker_members = sorted(
        set(QUALIFY.EXPECTED_SOURCE_CLOSURE["exact_member_names"])
        - {"PUBLICATION_COMMIT.json"}
    )
    marker = {
        "record_type": "ROUTE_A_V3_PUBLIC_ASSET_ACQUISITION_COMMIT",
        "contract_id": QUALIFY.CONTRACT_ID,
        "dataset_id": QUALIFY.DATASET_ID,
        "intended_final_path": str(root),
        "member_files": marker_members,
        "member_file_count": 6,
        "sha256sums_sha256": members["SOURCE_BUNDLE_SHA256SUMS"]["sha256"],
        "commit_marker_written_last": True,
        "committed": True,
        "scientific_status": "ASSETS_ACQUIRED_NOT_QUALIFIED",
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "training_started": False,
        "next_phase_authorized": False,
    }
    marker_payload = _json_bytes(marker)
    members["SOURCE_BUNDLE_PUBLICATION_COMMIT"] = _member(
        templates["SOURCE_BUNDLE_PUBLICATION_COMMIT"], marker_payload
    )
    (root / "PUBLICATION_COMMIT.json").write_bytes(marker_payload)
    return tuple(members[asset["asset_id"]] for asset in QUALIFY.EXPECTED_ASSETS)


def _write_protocol(base: Path, root: Path, assets: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol["input_contract"]["data_root"] = str(root)
    protocol["input_contract"]["source_bundle_members"] = list(assets)
    protocol["implementation_binding"] = {
        **copy.deepcopy(QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN),
        "status": "BOUND",
        "implementation_commit": "1" * 40,
        "qualifier_blob_sha256": "2" * 64,
        "test_blob_sha256": "3" * 64,
    }
    protocol["unresolved_blockers"] = QUALIFY._expected_blockers(
        protocol["implementation_binding"]
    )
    config_dir = base / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / QUALIFY.PROTOCOL_BASENAME
    payload = _json_bytes(protocol)
    path.write_bytes(payload)
    return {
        "base": base,
        "root": root,
        "assets": assets,
        "protocol": protocol,
        "protocol_path": path,
        "protocol_sha256": _sha256(payload),
    }


@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("published_endpoint_fixture")
    root = base / "GSE200304_SOURCE_BUNDLE"
    s2, keys = _build_s2()
    s3 = _build_s3(keys)
    assets = _write_source_bundle(root, s2, s3)
    fixture = _write_protocol(base, root, assets)
    fixture["s2"] = s2
    fixture["s3"] = s3
    return fixture


def _bind_fixture(monkeypatch: pytest.MonkeyPatch, fixture: Mapping[str, Any]) -> None:
    monkeypatch.setattr(QUALIFY, "EXPECTED_DATA_ROOT", Path(fixture["root"]))
    monkeypatch.setattr(QUALIFY, "EXPECTED_ASSETS", tuple(fixture["assets"]))

    def synthetic_verified_binding(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "PASS_BOUND_IMPLEMENTATION",
            "verified": True,
            "implementation_commit": "1" * 40,
            "binding_commit": "4" * 40,
            "clean_worktree": True,
            "config_only_direct_child": True,
            "authority_blobs_match": True,
            "implementation_blobs_match": True,
            "running_script_matches_bound_blob": True,
        }

    monkeypatch.setattr(
        QUALIFY, "_verify_implementation_binding", synthetic_verified_binding
    )


def _clone_fixture(tmp_path: Path, fixture: Mapping[str, Any]) -> dict[str, Any]:
    root = tmp_path / "GSE200304_SOURCE_BUNDLE"
    shutil.copytree(fixture["root"], root)
    marker_path = root / "PUBLICATION_COMMIT.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["intended_final_path"] = str(root)
    marker_payload = _json_bytes(marker)
    marker_path.write_bytes(marker_payload)
    assets = copy.deepcopy(list(fixture["assets"]))
    for asset in assets:
        if asset["asset_id"] == "SOURCE_BUNDLE_PUBLICATION_COMMIT":
            asset["bytes"] = len(marker_payload)
            asset["sha256"] = _sha256(marker_payload)
    clone = _write_protocol(tmp_path, root, tuple(assets))
    clone["s2"] = (root / "NIHMS1928233-supplement-3.csv").read_bytes()
    clone["s3"] = (root / "NIHMS1928233-supplement-4.xlsx").read_bytes()
    return clone


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_production_protocol_freezes_dec018_closure_and_immutable_blockers() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    QUALIFY._validate_protocol(protocol)
    assert protocol["authority"] == QUALIFY.EXPECTED_AUTHORITY
    assert protocol["authority"]["active_authority_commit"] == (
        "d328bf04c394d4960ac11058e079c063e09280af"
    )
    assert protocol["input_contract"]["source_bundle_member_count"] == 7
    assert protocol["input_contract"]["parsed_scientific_asset_ids"] == [
        "PMC10540565_TABLE_S2",
        "PMC10540565_TABLE_S3",
    ]
    assert protocol["table_contract"]["table_s3"]["primary_sheet"][
        "canonical_compact_header_json_sha256"
    ] == "d204a821928cb76b2fbc29201d3bcd103e6f4d3fa9cc526bd669604d74ef2ea5"
    assert protocol["table_contract"]["table_s3"]["control_sheet"][
        "canonical_compact_header_json_sha256"
    ] == "9008ea2fd8533da367e9dacad56f7089130574eea3b9829dc3fce2a76ad5d292"
    assert protocol["table_contract"]["table_s3"]["control_sheet"][
        "data_access_policy"
    ] == "HEADER_AND_DIMENSIONS_ONLY"
    assert protocol["table_contract"]["table_s3"]["control_sheet"][
        "data_cells_must_not_be_read"
    ] is True
    primary = protocol["table_contract"]["table_s3"]["primary_sheet"]
    assert primary["cell_type_counts"]["translation_formula"] == 13544
    assert primary["cached_translation_counts_role"] == (
        "DESCRIPTIVE_ONLY_NOT_MEMBERSHIP_OR_GATE"
    )
    assert primary["cached_translation_values_used_for_gate"] is False
    boundary = protocol["decision_neutral_boundary"]
    assert boundary["qualified"] is False
    assert boundary["ordinary_study_contribution"] == 0
    assert boundary["a1_intervention_study_contribution"] == 0
    assert boundary["true_a2_dense_study_contribution"] == 0
    assert boundary["canonical_record_count"] == 0
    assert boundary["training_allowed"] is False
    assert boundary["model_selection_allowed"] is False
    assert boundary["next_phase_authorized"] is False
    endpoint = protocol["endpoint_boundary"]
    assert endpoint["primary_complete_distinct_wt_201nt_proxy_group_count"] == 6544
    assert endpoint["wt_201nt_grouping_authority"] is False
    assert endpoint["wt_201nt_grouping_proxy_only"] is True
    assert endpoint["standard_error"] is None
    assert endpoint["study_level_reported_biological_replicate_count"] == 6
    assert endpoint["row_level_effective_replicate_count"] is None
    assert endpoint["power_effective_n"] is None
    assert protocol["unresolved_blockers"] == QUALIFY._expected_blockers(
        protocol["implementation_binding"]
    )
    assert "OWNER_POLICY_FOR_PUBLISHED_ENDPOINT_USE_NOT_FROZEN" in protocol[
        "unresolved_blockers"
    ]
    assert "CHECKPOINT_SPECIFIC_ENDPOINT_USE_NOT_CLEARED" in protocol[
        "unresolved_blockers"
    ]
    assert "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_NOT_CLOSED" in protocol[
        "unresolved_blockers"
    ]


def test_unknown_i_stops_before_official_asset_or_output_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Keep this regression independent of the checked-in protocol's lifecycle
    # state.  The production file is UNKNOWN in implementation commit I and
    # becomes BOUND in the direct config-only commit B; both repository states
    # must retain an explicit UNKNOWN fixture proving stop-before-data.
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol["implementation_binding"] = copy.deepcopy(
        QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN
    )
    protocol["unresolved_blockers"] = QUALIFY._expected_blockers(
        protocol["implementation_binding"]
    )
    QUALIFY._validate_protocol(protocol)
    monkeypatch.setattr(
        QUALIFY,
        "_load_protocol",
        lambda path, expected_sha256: (
            protocol,
            {"sha256": "0" * 64, "bytes": 0, "launch_expected_sha256": "0" * 64},
        ),
    )
    monkeypatch.setattr(QUALIFY, "EXPECTED_DATA_ROOT", tmp_path / "source")
    source_or_output_accessed = False

    def forbidden_directory_access(*args: Any, **kwargs: Any) -> Any:
        nonlocal source_or_output_accessed
        source_or_output_accessed = True
        raise AssertionError("official source or output was accessed")

    monkeypatch.setattr(QUALIFY, "_open_directory_no_symlinks", forbidden_directory_access)
    with pytest.raises(QUALIFY.ProtocolError, match="must be BOUND"):
        QUALIFY.execute_qualification(
            protocol_path=tmp_path / "configs" / QUALIFY.PROTOCOL_BASENAME,
            protocol_sha256="0" * 64,
            data_root=tmp_path / "source",
            output_directory=tmp_path / "output",
        )
    assert source_or_output_accessed is False
    assert not (tmp_path / "output").exists()


def test_full_synthetic_run_is_aggregate_blocked_and_atomically_committed(
    synthetic_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bind_fixture(monkeypatch, synthetic_fixture)
    output = tmp_path / "blocked_bundle"
    result = QUALIFY.qualify_gse200304_published_endpoint_a1(
        protocol_path=Path(synthetic_fixture["protocol_path"]),
        protocol_sha256=str(synthetic_fixture["protocol_sha256"]),
        data_root=Path(synthetic_fixture["root"]),
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.SUCCESS_OUTCOME
    assert result["committed"] is True and result["accepted"] is True
    assert result["terminal_publication_operation"] == (
        "FSYNCED_STAGED_HARDLINK_NO_REPLACE"
    )
    assert result["no_acceptance_critical_read_after_commit"] is True
    assert QUALIFY.validate_published_bundle(output)["accepted"] is True
    assert set(path.name for path in output.iterdir()) == {
        *QUALIFY.SUCCESS_JSON_FILES,
        QUALIFY.SHA256SUMS_FILENAME,
        QUALIFY.PUBLICATION_MARKER,
    }
    assert (output / QUALIFY.PUBLICATION_MARKER).stat().st_nlink == 1

    integrity = _read_json(output / "INPUT_INTEGRITY_AUDIT.json")
    audit = _read_json(output / "PUBLISHED_ENDPOINT_AUDIT.json")
    report = _read_json(output / "QUALIFICATION_REPORT.json")
    assert integrity["source_bundle_closure"]["exact_member_count"] == 7
    assert integrity["source_bundle_closure"]["endpoint_statistics_parsed_asset_count"] == 2
    assert integrity["source_bundle_closure"]["opaque_zenodo_code_executed"] is False
    assert audit["table_s2"]["duplicated_pair_count"] == 7
    assert audit["table_s2"]["all_pair_orientation_counts"] == {
        "FORWARD": 3497,
        "REVERSE_COMPLEMENT": 3388,
        "UNRESOLVED": 0,
    }
    assert audit["table_s3"]["translation_formula_count"] == 13544
    assert audit["table_s3"]["translation_formula_executed"] is False
    assert audit["table_s3"]["cached_translation_values_used_for_gate"] is False
    assert audit["table_s3"]["both_comparisons_finite_pair_count"] == 6538
    assert audit["table_s3"]["primary_only_finite_pair_count"] == 9
    assert audit["table_s3"]["secondary_only_finite_pair_count"] == 0
    assert audit["table_s3"]["neither_comparison_finite_pair_count"] == 225
    endpoint = audit["endpoint_boundary"]
    assert endpoint["joined_pair_count"] == 6772
    assert endpoint["joined_pair_orientation_counts"] == {
        "FORWARD": 3451,
        "REVERSE_COMPLEMENT": 3321,
        "UNRESOLVED": 0,
    }
    assert endpoint["primary_absent_pair_count"] == 113
    assert endpoint["primary_na_pair_count"] == 225
    assert endpoint["primary_finite_effect_pair_count"] == 6547
    assert endpoint["primary_complete_distinct_wt_201nt_proxy_group_count"] == 6544
    assert endpoint["primary_complete_wt_201nt_proxy_pool_size_counts"] == {
        "1": 6541,
        "2": 3,
    }
    assert endpoint["wt_201nt_grouping_authority"] is False
    assert endpoint["standard_error"] is None
    assert endpoint["study_level_reported_biological_replicate_count"] == 6
    assert endpoint["row_level_effective_replicate_count"] is None
    assert endpoint["power_effective_n"] is None
    assert endpoint["true_a2_dense_candidate_count"] == 0
    assert report["qualification_status"] == "BLOCKED_NOT_QUALIFIED"
    for key in (
        "qualified",
        "canonical_materialization_allowed",
        "training_allowed",
        "model_selection_allowed",
        "next_phase_authorized",
    ):
        assert report[key] is False
    assert report["ordinary_study_contribution"] == 0
    assert report["a1_intervention_study_contribution"] == 0
    assert report["true_a2_dense_study_contribution"] == 0
    assert report["canonical_record_count"] == 0
    assert report["unresolved_blockers"] == QUALIFY._expected_blockers(
        report["implementation_binding"]
    )
    closed_payloads = {
        name: _read_json(output / name) for name in QUALIFY.SUCCESS_JSON_FILES
    }
    closed_payloads["PUBLISHED_ENDPOINT_AUDIT.json"]["table_s3"][
        "row_effects"
    ] = [{"row_id": "hidden", "effect": 1.0}]
    with pytest.raises(QUALIFY.PublicationError, match="closed schema"):
        QUALIFY._validate_success_payloads(closed_payloads)


def test_forbidden_scope_is_rejected_before_protocol_read_or_output(
    synthetic_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bind_fixture(monkeypatch, synthetic_fixture)
    read_called = False

    def forbidden_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal read_called
        read_called = True
        raise AssertionError("protocol read occurred")

    monkeypatch.setattr(QUALIFY, "_read_path_verified_snapshot", forbidden_read)
    output = tmp_path / "forbidden_fastq_output"
    with pytest.raises(QUALIFY.ScopeViolation):
        QUALIFY.execute_qualification(
            protocol_path=Path(synthetic_fixture["protocol_path"]),
            protocol_sha256=str(synthetic_fixture["protocol_sha256"]),
            data_root=Path(synthetic_fixture["root"]),
            output_directory=output,
        )
    assert read_called is False
    assert not output.exists()


@pytest.mark.parametrize("case", ["extra_member", "symlink", "snapshot_rewrite"])
def test_source_snapshot_guards_fail_before_table_parse(
    case: str,
    synthetic_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _clone_fixture(tmp_path / case, synthetic_fixture)
    _bind_fixture(monkeypatch, fixture)
    table_parse_called = False

    def forbidden_table_parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal table_parse_called
        table_parse_called = True
        raise AssertionError("table parse occurred")

    monkeypatch.setattr(QUALIFY, "_audit_table_s2", forbidden_table_parse)
    root = Path(fixture["root"])
    if case == "extra_member":
        (root / "unexpected.txt").write_text("not in closure", encoding="utf-8")
    elif case == "symlink":
        target = tmp_path / "same_s2_bytes.csv"
        target.write_bytes(fixture["s2"])
        leaf = root / "NIHMS1928233-supplement-3.csv"
        leaf.unlink()
        leaf.symlink_to(target)
    else:
        leaf = root / "NIHMS1928233-supplement-3.csv"

        def rewrite_same_bytes() -> None:
            leaf.write_bytes(fixture["s2"])

        monkeypatch.setattr(
            QUALIFY, "_POST_VERIFIED_INPUT_SNAPSHOT_HOOK", rewrite_same_bytes
        )
    output = tmp_path / f"failure_{case}"
    result = QUALIFY.execute_qualification(
        protocol_path=Path(fixture["protocol_path"]),
        protocol_sha256=str(fixture["protocol_sha256"]),
        data_root=root,
        output_directory=output,
    )
    assert result["execution_outcome"] == QUALIFY.FAILURE_OUTCOME
    assert result["committed"] is True
    assert table_parse_called is False
    failure = _read_json(output / "FAILURE_REPORT.json")
    assert failure["qualified"] is False
    assert failure["training_allowed"] is False
    assert failure["next_phase_authorized"] is False


def test_s2_duplicate_and_central_snv_mutations_fail_closed(
    synthetic_fixture: Mapping[str, Any],
) -> None:
    rows = list(csv.reader(io.StringIO(synthetic_fixture["s2"].decode("utf-8"))))
    missing_duplicate = io.StringIO(newline="")
    writer = csv.writer(missing_duplicate, lineterminator="\n")
    writer.writerows(rows[:-1])
    with pytest.raises(QUALIFY.TableAuditError):
        QUALIFY._audit_table_s2(
            missing_duplicate.getvalue().encode("utf-8"), QUALIFY.EXPECTED_S2
        )

    changed = copy.deepcopy(rows)
    mutant_index = next(
        index for index, row in enumerate(changed[1:], start=1) if row[1] == "Mutant"
    )
    values = list(changed[mutant_index][2])
    values[0] = "C" if values[0] != "C" else "A"
    changed[mutant_index][2] = "".join(values)
    changed[mutant_index][5] = (
        changed[mutant_index][3]
        + changed[mutant_index][2]
        + changed[mutant_index][4]
    )
    malformed = io.StringIO(newline="")
    writer = csv.writer(malformed, lineterminator="\n")
    writer.writerows(changed)
    with pytest.raises(QUALIFY.TableAuditError):
        QUALIFY._audit_table_s2(
            malformed.getvalue().encode("utf-8"), QUALIFY.EXPECTED_S2
        )


def test_cached_translation_values_are_descriptive_not_membership_gate(
    synthetic_fixture: Mapping[str, Any],
) -> None:
    changed = _patch_formula_caches(
        synthetic_fixture["s3"], lambda row_number: "Descriptive Only"
    )
    state = QUALIFY._audit_table_s3(changed, QUALIFY.EXPECTED_S3)
    assert len(state.pair_keys) == 6772
    assert state.aggregate["cached_translation_values_used_for_gate"] is False
    assert state.aggregate["cached_translation_values_used_for_membership"] is False
    assert state.aggregate["significant_rows"] == {}
    assert state.aggregate["nonsignificant_rows"] == {}
    assert state.aggregate["other_cached_translation_rows"] == {
        "HighPoly:RNA": 6772,
        "TotalPoly:RNA": 6772,
    }


def test_translation_formula_cell_type_drift_fails_closed(
    synthetic_fixture: Mapping[str, Any],
) -> None:
    changed = _remove_first_translation_formula(synthetic_fixture["s3"])
    with pytest.raises(QUALIFY.TableAuditError, match="formula cell"):
        QUALIFY._audit_table_s3(changed, QUALIFY.EXPECTED_S3)


def test_opaque_control_data_cells_are_never_read_or_used_as_gate(
    synthetic_fixture: Mapping[str, Any],
) -> None:
    changed = _replace_opaque_control_measurement(synthetic_fixture["s3"])
    state = QUALIFY._audit_table_s3(changed, QUALIFY.EXPECTED_S3)
    assert state.aggregate["opaque_control_data_cell_read_count"] == 0
    assert state.aggregate["opaque_control_data_access_policy"] == (
        "HEADER_AND_DIMENSIONS_ONLY"
    )
    assert state.aggregate["opaque_control_excluded_from_qualification_counts"] is True


def test_closed_failure_output_rejects_extra_fields_and_embedded_sequence(
    tmp_path: Path,
) -> None:
    payloads = QUALIFY._failure_payload(QUALIFY.InputIntegrityError.code)
    payloads["FAILURE_REPORT.json"]["row_effects"] = [{"effect": 1.0}]
    with pytest.raises(QUALIFY.PublicationError, match="closed schema"):
        QUALIFY._publish_closed_bundle(
            tmp_path / "must_not_exist", payloads, outcome=QUALIFY.FAILURE_OUTCOME
        )
    assert not (tmp_path / "must_not_exist").exists()
    with pytest.raises(QUALIFY.PublicationError, match="nucleotide payload"):
        QUALIFY._assert_aggregate_safe_document(
            {"status": "prefix_" + "ACGT" * 6 + "_suffix"}
        )


def test_publisher_has_no_post_visibility_acceptance_read_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "publication"
    visible = False
    original_link = QUALIFY.os.link
    original_read = QUALIFY._read_member_snapshot_at

    def tracked_link(*args: Any, **kwargs: Any) -> None:
        nonlocal visible
        original_link(*args, **kwargs)
        visible = True

    def reject_post_visibility_read(
        *args: Any, **kwargs: Any
    ) -> tuple[bytes, QUALIFY.FileIdentity]:
        if visible:
            raise AssertionError("acceptance-critical read followed marker visibility")
        return original_read(*args, **kwargs)

    def postcommit_fault(phase: str) -> None:
        if phase == "postcommit_parent_fsync":
            raise OSError("synthetic postcommit durability fault")

    monkeypatch.setattr(QUALIFY.os, "link", tracked_link)
    monkeypatch.setattr(
        QUALIFY, "_read_member_snapshot_at", reject_post_visibility_read
    )
    monkeypatch.setattr(QUALIFY, "_PUBLICATION_FAULT_HOOK", postcommit_fault)
    payloads = QUALIFY._failure_payload(QUALIFY.InputIntegrityError.code)
    result = QUALIFY._publish_closed_bundle(
        output, payloads, outcome=QUALIFY.FAILURE_OUTCOME
    )
    assert result["committed"] is True and result["accepted"] is True
    assert result["publication_state"] == "COMMITTED_WITH_DURABILITY_WARNING"
    assert "POSTCOMMIT_PARENT_FSYNC_WARNING" in result["postcommit_warning_codes"]
    assert (output / QUALIFY.PUBLICATION_MARKER).stat().st_nlink == 1
    assert not list(tmp_path.glob(".publication-*.stage"))
    before = {
        path.name: _sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()
    }
    with pytest.raises(QUALIFY.PublicationContention):
        QUALIFY._publish_closed_bundle(
            output, payloads, outcome=QUALIFY.FAILURE_OUTCOME
        )
    after = {
        path.name: _sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()
    }
    assert after == before


def test_terminal_link_fault_mutation_is_rejected_before_marker_visibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "mutated_precommit"

    def mutate_before_final_validation(phase: str) -> None:
        if phase == "terminal_hardlink":
            (output / "FAILURE_REPORT.json").write_bytes(b"corrupted\n")

    monkeypatch.setattr(
        QUALIFY, "_PUBLICATION_FAULT_HOOK", mutate_before_final_validation
    )
    with pytest.raises(QUALIFY.PartialPrecommitError):
        QUALIFY._publish_closed_bundle(
            output,
            QUALIFY._failure_payload(QUALIFY.InputIntegrityError.code),
            outcome=QUALIFY.FAILURE_OUTCOME,
        )
    assert not (output / QUALIFY.PUBLICATION_MARKER).exists()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def test_unknown_i_and_future_direct_config_only_b_binding(tmp_path: Path) -> None:
    unknown = QUALIFY._verify_implementation_binding(
        QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN,
        QUALIFY.EXPECTED_AUTHORITY,
        tmp_path / "not_a_repository",
    )
    assert unknown["status"] == "UNKNOWN_NOT_ASSERTED"
    assert unknown["verified"] is False

    repository = tmp_path / "binding_repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Synthetic Test")
    _git(repository, "config", "user.email", "synthetic@example.invalid")
    authority_payloads = {
        "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md": b"contract authority\n",
        "docs/execution/route_a_v3_data_role_registry.yaml": b"data role authority\n",
        "docs/execution/route_a_v3_decision_log.yaml": b"decision authority\n",
    }
    for relative, payload in authority_payloads.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "authority")
    active = _git(repository, "rev-parse", "HEAD")
    (repository / "staging_parent.txt").write_text("staging\n", encoding="utf-8")
    _git(repository, "add", "staging_parent.txt")
    _git(repository, "commit", "-qm", "staging parent")
    staging_parent = _git(repository, "rev-parse", "HEAD")

    binding = copy.deepcopy(QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN)
    qualifier = repository / binding["qualifier_path"]
    test_path = repository / binding["test_path"]
    config_path = repository / binding["post_implementation_allowed_changed_paths"][0]
    qualifier.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    qualifier.write_bytes(b"print('bound synthetic qualifier')\n")
    test_path.write_bytes(b"def test_bound():\n    assert True\n")
    implementation_protocol = {
        "core": "unchanged",
        "implementation_binding": copy.deepcopy(
            QUALIFY.IMPLEMENTATION_BINDING_UNKNOWN
        ),
        "unresolved_blockers": [
            *QUALIFY.BASE_BLOCKERS,
            QUALIFY.IMPLEMENTATION_BINDING_BLOCKER,
        ],
    }
    config_path.write_text(
        json.dumps(implementation_protocol, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "implementation I")
    implementation = _git(repository, "rev-parse", "HEAD")
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation,
            "qualifier_blob_sha256": _sha256(qualifier.read_bytes()),
            "test_blob_sha256": _sha256(test_path.read_bytes()),
        }
    )
    binding_protocol = {
        "core": "unchanged",
        "implementation_binding": binding,
        "unresolved_blockers": list(QUALIFY.BASE_BLOCKERS),
    }
    tampered = copy.deepcopy(binding_protocol)
    tampered["unresolved_blockers"][0] = "SCIENTIFIC_BLOCKER_TAMPERED"
    with pytest.raises(QUALIFY.ProtocolError, match="scientific blockers drifted"):
        QUALIFY._validate_i_to_b_protocol_transition(
            implementation_protocol, tampered, binding
        )
    config_path.write_text(
        json.dumps(binding_protocol, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(repository, "add", binding["post_implementation_allowed_changed_paths"][0])
    _git(repository, "commit", "-qm", "binding B config only")
    authority = {
        "contract_path": "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
        "contract_sha256": _sha256(
            authority_payloads["docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md"]
        ),
        "data_role_registry_path": "docs/execution/route_a_v3_data_role_registry.yaml",
        "data_role_registry_sha256": _sha256(
            authority_payloads[
                "docs/execution/route_a_v3_data_role_registry.yaml"
            ]
        ),
        "decision_log_path": "docs/execution/route_a_v3_decision_log.yaml",
        "decision_log_sha256": _sha256(
            authority_payloads["docs/execution/route_a_v3_decision_log.yaml"]
        ),
        "active_authority_commit": active,
        "staging_parent_head": staging_parent,
    }
    verified = QUALIFY._verify_implementation_binding(
        binding, authority, repository, running_script_path=qualifier
    )
    assert verified["status"] == "PASS_BOUND_IMPLEMENTATION"
    assert verified["verified"] is True
    assert verified["implementation_commit"] == implementation
    assert verified["config_only_direct_child"] is True
