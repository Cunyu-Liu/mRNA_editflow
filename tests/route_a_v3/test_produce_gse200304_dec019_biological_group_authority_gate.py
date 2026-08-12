from __future__ import annotations

import copy
import importlib.util
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
SCRIPT = ROOT / "scripts/route_a_v3/produce_gse200304_dec019_biological_group_authority_gate.py"
CONFIG = ROOT / "configs/route_a_v3_gse200304_dec019_biological_group_authority_gate_v1.json"
INTEGRATED_CONSUMER = (
    ROOT / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
CONSUMER_ROOT = ROOT if INTEGRATED_CONSUMER.exists() else (
    WORK / "g200_group_consumer_commitment_upgrade_staging"
)
CONSUMER_CONFIG = CONSUMER_ROOT / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
CONSUMER_SCRIPT = CONSUMER_ROOT / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
CONFIGURED_PUBLIC_ASSETS = Path(
    json.loads(CONFIG.read_text(encoding="utf-8"))["source_authority"]["data_root"]
)
PUBLIC_ASSETS = CONFIGURED_PUBLIC_ASSETS if CONFIGURED_PUBLIC_ASSETS.is_dir() else (
    WORK / "gse200304_public_assets_20260810T143731P0800"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCER = _load(SCRIPT, "gse200304_group_producer")
CONSUMER = _load(CONSUMER_SCRIPT, "gse200304_group_consumer")


def _producer_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
            "consumer_upgrade_binding_commit": "4" * 40,
            "consumer_upgrade_config_sha256": "5" * 64,
            "consumer_upgrade_script_sha256": "6" * 64,
        }
    )
    return config


def _consumer_config() -> dict[str, Any]:
    config = json.loads(CONSUMER_CONFIG.read_text(encoding="utf-8"))
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "7" * 40,
            "implementation_script_sha256": "8" * 64,
            "implementation_test_sha256": "9" * 64,
        }
    )
    CONSUMER.validate_static_config(config)
    return config


def _authority_payloads(config: dict[str, Any]) -> tuple[bytes, bytes]:
    upstream = {
        "schema_version": "route_a_v3_gse200304_upstream_authority_viability.v1",
        "dataset_id": PRODUCER.DATASET_ID,
        "status": "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE",
        "geo_soft_authority": {
            "series_accession": "GSE200302",
            "subseries_of_gse200304": True,
        },
        "biological_group_authority": {
            "alternate_allele_in_group_key": False,
            "repair_route_group_key_fields": copy.deepcopy(
                config["mapping_contract"]["group_key_fields"]
            ),
        },
    }
    lineage = {
        "status": "PASS",
        "gate_id": "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE",
        "facts": {
            "canonical_record_count": 6547,
            "deterministic_row_locator_frozen": True,
            "multi_asset_lineage_closed": True,
            "s2_s3_join_rule_frozen": True,
        },
    }
    return PRODUCER.json_bytes(upstream), PRODUCER.json_bytes(lineage)


def _minimal_s3_xlsx() -> bytes:
    strings = [
        "barcode",
        "Comparison",
        "xtail_log2FC_TE",
        "xtail_pvalue",
        "xtail_FDR",
        "chr1:42_C-A",
        "chr2:99_G-T",
        "HighPoly:RNA",
        "TotalPoly:RNA",
        "NA",
    ]
    index = {value: position for position, value in enumerate(strings)}

    def shared(column: str, row: int, value: str) -> str:
        return f'<c r="{column}{row}" t="s"><v>{index[value]}</v></c>'

    def numeric(column: str, row: int, value: str) -> str:
        return f'<c r="{column}{row}"><v>{value}</v></c>'

    rows = [
        '<row r="1">'
        + "".join(
            shared(column, 1, value)
            for column, value in (
                ("A", "barcode"),
                ("C", "Comparison"),
                ("D", "xtail_log2FC_TE"),
                ("E", "xtail_pvalue"),
                ("F", "xtail_FDR"),
            )
        )
        + "</row>"
    ]
    row_number = 2
    for pair_id, totalpoly_is_finite in (
        ("chr1:42_C-A", True),
        ("chr2:99_G-T", False),
    ):
        rows.append(
            f'<row r="{row_number}">'
            + shared("A", row_number, pair_id)
            + shared("C", row_number, "HighPoly:RNA")
            + numeric("D", row_number, "0.1")
            + numeric("E", row_number, "0.2")
            + numeric("F", row_number, "0.3")
            + "</row>"
        )
        row_number += 1
        if totalpoly_is_finite:
            statistics = (
                numeric("D", row_number, "-0.1")
                + numeric("E", row_number, "0.4")
                + numeric("F", row_number, "0.5")
            )
        else:
            statistics = "".join(
                shared(column, row_number, "NA") for column in ("D", "E", "F")
            )
        rows.append(
            f'<row r="{row_number}">'
            + shared("A", row_number, pair_id)
            + shared("C", row_number, "TotalPoly:RNA")
            + statistics
            + "</row>"
        )
        row_number += 1

    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheets><sheet name="S2A_Polysome_MPRA_Mut_Stats" sheetId="1"/>'
        '<sheet name="S2B_Poly_MPRA_Control_Stats" sheetId="2"/></sheets>'
        "</workbook>"
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(rows)
        + "</sheetData></worksheet>"
    )
    shared_strings = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + "".join(f"<si><t>{value}</t></si>" for value in strings)
        + "</sst>"
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheetData/></worksheet>',
        )
        archive.writestr("xl/sharedStrings.xml", shared_strings)
    return output.getvalue()


def test_standard_library_s3_parser_preserves_finite_totalpoly_membership() -> None:
    eligible, audit = PRODUCER.finite_totalpoly_pair_ids(_minimal_s3_xlsx())
    assert eligible == {"chr1:42_C-A"}
    assert audit == {
        "pair_count": 2,
        "highpoly_row_count": 2,
        "totalpoly_row_count": 2,
        "finite_totalpoly_pair_count": 1,
    }
    assert "openpyxl" not in SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def real_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("g200-group-real")
    config = _producer_config()
    upstream, lineage = _authority_payloads(config)
    upstream_path = root / "upstream.json"
    lineage_path = root / "lineage.json"
    upstream_path.write_bytes(upstream)
    lineage_path.write_bytes(lineage)
    config["source_authority"]["upstream_audit"].update(
        {"bytes": len(upstream), "sha256": PRODUCER.sha256(upstream)}
    )
    config["source_authority"]["canonical_lineage_gate"].update(
        {"bytes": len(lineage), "sha256": PRODUCER.sha256(lineage)}
    )
    consumer_config = _consumer_config()
    consumer_path = root / "consumer.json"
    consumer_path.write_bytes(CONSUMER.json_bytes(consumer_config))
    payloads = PRODUCER.produce(
        config,
        source_root=PUBLIC_ASSETS,
        upstream_audit_path=upstream_path,
        lineage_gate_path=lineage_path,
        consumer_config_path=consumer_path,
        consumer_script_path=CONSUMER_SCRIPT,
    )
    return {
        "config": config,
        "consumer_config": consumer_config,
        "payloads": payloads,
        "mapping": json.loads(
            payloads[config["output_contract"]["private_mapping_basename"]]
        ),
        "audit": json.loads(
            payloads[config["output_contract"]["aggregate_audit_basename"]]
        ),
        "gate": json.loads(
            payloads[config["output_contract"]["allowed_basename"]]
        ),
    }


def test_real_data_observed_counts_and_mapping_commitment(real_result: dict[str, Any]) -> None:
    mapping = real_result["mapping"]
    audit = real_result["audit"]
    assert (mapping["record_count"], mapping["group_count"]) == (6547, 6544)
    assert audit["group_size_histogram"] == {"1": 6541, "2": 3}
    assert audit["orientation_counts"] == {
        "FORWARD": 3336,
        "REVERSE_COMPLEMENT": 3211,
    }
    mapping_payload = real_result["payloads"][
        real_result["config"]["output_contract"]["private_mapping_basename"]
    ]
    commitment = PRODUCER.sha256(mapping_payload)
    assert audit["mapping_commitment_sha256"] == commitment
    assert real_result["gate"]["provenance"][PRODUCER.MAPPING_COMMITMENT_KEY] == commitment


def test_three_double_candidate_groups_are_not_overwritten(real_result: dict[str, Any]) -> None:
    counts = Counter(
        item["biological_group_id"] for item in real_result["mapping"]["mappings"]
    )
    assert Counter(counts.values()) == Counter({1: 6541, 2: 3})
    assert len(real_result["mapping"]["mappings"]) == 6547


def test_forward_and_reverse_orientation_are_author_reference_normalized() -> None:
    forward = "A" * 100 + "C" + "G" * 100
    reverse_author = "T" * 100 + "A" + "C" * 100
    reverse_input = PRODUCER.reverse_complement(reverse_author)
    assert forward[100] == "C"
    assert reverse_input[100] != "A"
    assert PRODUCER.reverse_complement(reverse_input) == reverse_author
    assert PRODUCER.reverse_complement(reverse_input)[100] == "A"


def test_alternate_allele_is_excluded_from_group_key() -> None:
    config = _producer_config()
    config["mapping_contract"]["expected_canonical_locator_count"] = 2
    wt = "A" * 100 + "C" + "G" * 100
    pairs = {
        "chr1:42_C-A": {
            "chromosome": "chr1", "position": "42", "reference": "C",
            "alternate": "A", "wt201": wt,
        },
        "chr1:42_C-G": {
            "chromosome": "chr1", "position": "42", "reference": "C",
            "alternate": "G", "wt201": wt,
        },
    }
    mapping, audit = PRODUCER.build_mapping(pairs, set(pairs), config)
    assert mapping["record_count"] == 2
    assert mapping["group_count"] == 1
    assert audit["group_size_histogram"] == {"2": 1}
    assert audit["alternate_allele_in_group_key"] is False


def test_upgraded_consumer_actually_accepts_exact_group_pass(real_result: dict[str, Any]) -> None:
    config = real_result["consumer_config"]
    gate_payload = real_result["payloads"][
        real_result["config"]["output_contract"]["allowed_basename"]
    ]
    slot = next(
        item
        for item in config["evidence_contract"]["slots"]
        if item["slot_id"] == PRODUCER.GATE_ID
    )
    record = CONSUMER._validate_gate_record(gate_payload, slot, config)
    assert CONSUMER._slot_gate_pass(PRODUCER.GATE_ID, record["facts"]) is True
    forbidden = {"row_id", "row_ids", "sequence", "gene", "effect", "effect_value"}
    assert not (forbidden & {key.casefold() for key in PRODUCER._walk_keys(record)})


def _synthetic_publication_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    output = config["output_contract"]
    return {
        output["private_mapping_basename"]: b"private mapping\n",
        output["aggregate_audit_basename"]: b"aggregate audit\n",
        output["allowed_basename"]: b"aggregate gate\n",
    }


def test_publication_writes_three_fsynced_members_then_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _producer_config()
    payloads = _synthetic_publication_payloads(config)
    output = config["output_contract"]
    written_names: list[str] = []
    fsync_calls: list[int] = []
    real_write = PRODUCER._write_fsynced
    real_fsync = PRODUCER.os.fsync

    def record_write(path: Path, payload: bytes) -> None:
        written_names.append(path.name)
        real_write(path, payload)

    def record_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(PRODUCER, "_write_fsynced", record_write)
    monkeypatch.setattr(PRODUCER.os, "fsync", record_fsync)
    destination = tmp_path / "published"
    assert PRODUCER.write_outputs(destination, payloads, config) == "PUBLISHED"
    assert written_names == output["exact_final_member_names"]
    assert len(fsync_calls) == 5
    assert {path.name for path in destination.iterdir()} == set(
        output["exact_final_member_names"]
    )
    marker = json.loads((destination / output["terminal_commit_marker"]).read_bytes())
    assert marker == {
        "schema_version": PRODUCER.PUBLICATION_COMMIT_SCHEMA,
        "committed": True,
        "members": [
            {
                "name": name,
                "bytes": len(payloads[name]),
                "sha256": PRODUCER.sha256(payloads[name]),
            }
            for name in output["data_member_names"]
        ],
    }


def test_partial_publication_is_stopped_and_preserved(tmp_path: Path) -> None:
    config = _producer_config()
    payloads = _synthetic_publication_payloads(config)
    first_name = config["output_contract"]["data_member_names"][0]
    destination = tmp_path / "partial"
    destination.mkdir()
    (destination / first_name).write_bytes(payloads[first_name])

    with pytest.raises(PRODUCER.ProducerError, match="partial or mismatched"):
        PRODUCER.write_outputs(destination, payloads, config)

    assert {path.name for path in destination.iterdir()} == {first_name}
    assert (destination / first_name).read_bytes() == payloads[first_name]


def test_exact_four_member_publication_is_idempotent(tmp_path: Path) -> None:
    config = _producer_config()
    payloads = _synthetic_publication_payloads(config)
    destination = tmp_path / "published"
    assert PRODUCER.write_outputs(destination, payloads, config) == "PUBLISHED"
    first_bytes = {path.name: path.read_bytes() for path in destination.iterdir()}

    assert PRODUCER.write_outputs(destination, payloads, config) == "IDEMPOTENT"
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == first_bytes
