#!/usr/bin/env python3
"""Build the DEC-019 GSE200304 author-anchored biological-group PASS record.

The private mapping stores only canonical locator and biological-group IDs.
Raw author IDs and WT sequence values are used in memory and never persisted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

UNKNOWN = "UNKNOWN_NOT_ASSERTED"
PASS = "PASS"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
GATE_ID = "BIOLOGICAL_GROUP_AUTHORITY"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE_V1"
EVIDENCE_SCHEMA = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
MAPPING_COMMITMENT_KEY = "group_mapping_commitment_sha256"
PUBLICATION_COMMIT_SCHEMA = (
    "route_a_v3_gse200304_biological_group_publication_commit.v1"
)
JOIN_KEY_DOMAIN = b"route-a-v3/gse200304/dec019/join-key/v1"
LOCATOR_DOMAIN = b"route-a-v3/gse200304/dec019/canonical-row-locator/v1"
GROUP_ID_DOMAIN = b"route-a-v3/gse200304/dec019/biological-group-id/v1"
IMPLEMENTATION_BASE_COMMIT = "d725fd5ae4fd8d66f339c3d400210f02ead69bc4"
CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_biological_group_authority_gate_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_biological_group_authority_gate.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_biological_group_authority_gate.py"
)
EXPECTED_GROUP_KEY_FIELDS = [
    "GSE200304_STUDY",
    "GSE200302_SUBSERIES",
    "AUTHOR_LOCUS_TOKEN",
    "CANONICAL_DECIMAL_POSITION",
    "REFERENCE_ALLELE",
    "ORIENTATION_NORMALIZED_WT201_DIGEST",
]
MAPPING_COMMITMENT_ALGORITHM = (
    "DOMAIN_SEPARATED_LENGTH_PREFIXED_SHA256_LEAF_SORT_ODD_DUPLICATE"
)
MAPPING_COMMITMENT_LEAF_DOMAIN = (
    "route-a-v3/gse200304/dec019/biological-group-mapping-leaf/v1"
)
MAPPING_COMMITMENT_PARENT_DOMAIN = (
    "route-a-v3/gse200304/dec019/biological-group-mapping-parent/v1"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DNA201 = re.compile(r"^[ACGT]{201}$")
CELL_REFERENCE = re.compile(r"^(?P<column>[A-Z]+)(?P<row>[1-9][0-9]*)$")
SPREADSHEETML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class ProducerError(RuntimeError):
    """The observed public authority cannot support a PASS record."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProducerError(f"invalid JSON: {label}") from exc
    if type(value) is not dict:
        raise ProducerError(f"JSON root is not an object: {label}")
    return value


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _framed(parts: Iterable[bytes]) -> bytes:
    output = bytearray()
    for part in parts:
        output.extend(_u64(len(part)))
        output.extend(part)
    return bytes(output)


def _domain_hash(domain: bytes, parts: Iterable[bytes]) -> bytes:
    return hashlib.sha256(_framed((domain, *parts))).digest()


def canonical_locator(pair_id: str) -> str:
    join_key = _domain_hash(JOIN_KEY_DOMAIN, (pair_id.encode("utf-8"),))
    return _domain_hash(
        LOCATOR_DOMAIN,
        (DATASET_ID.encode("ascii"), join_key, b"TotalPoly:RNA"),
    ).hex()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def normalized_wt201_digest(normalized_wt201: str) -> str:
    if DNA201.fullmatch(normalized_wt201) is None:
        raise ProducerError("orientation-normalized WT201 is not exact DNA201")
    return sha256(normalized_wt201.encode("ascii"))


def biological_group_id(
    study: str,
    subseries: str,
    chromosome: str,
    position: str,
    reference: str,
    normalized_wt201_sha256: str,
) -> str:
    if HEX64.fullmatch(normalized_wt201_sha256) is None:
        raise ProducerError("orientation-normalized WT201 digest is not HEX64")
    return _domain_hash(
        GROUP_ID_DOMAIN,
        (
            study.encode("ascii"),
            subseries.encode("ascii"),
            chromosome.encode("utf-8"),
            position.encode("ascii"),
            reference.encode("ascii"),
            normalized_wt201_sha256.encode("ascii"),
        ),
    ).hex()


def biological_group_mapping_commitment(
    entries: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> str:
    mapping = config["mapping_contract"]
    leaf_domain = mapping["mapping_commitment_leaf_domain"].encode("ascii")
    parent_domain = mapping["mapping_commitment_parent_domain"].encode("ascii")
    level = sorted(
        _domain_hash(
            leaf_domain,
            (
                entry["canonical_locator"].encode("ascii"),
                entry["biological_group_id"].encode("ascii"),
            ),
        )
        for entry in entries
    )
    if not level:
        raise ProducerError("biological-group mapping is empty")
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            _domain_hash(parent_domain, (level[index], level[index + 1]))
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _bound_source(path: Path, spec: Mapping[str, Any], *, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProducerError(f"cannot read {label}") from exc
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise ProducerError(f"{label} differs from its accepted source binding")
    return payload


def validate_static_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != (
        "route_a_v3_gse200304_dec019_biological_group_authority_gate.v1"
    ):
        raise ProducerError("config schema differs")
    for key, expected in {
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        if config.get(key) != expected:
            raise ProducerError(f"config {key} differs")
    mapping = config["mapping_contract"]
    if mapping["study_id"] != DATASET_ID or mapping["subseries_id"] != "GSE200302":
        raise ProducerError("study/subseries authority differs")
    if mapping["group_key_fields"] != EXPECTED_GROUP_KEY_FIELDS:
        raise ProducerError("biological-group key fields differ")
    if mapping["normalized_wt201_digest_algorithm"] != "SHA256":
        raise ProducerError("orientation-normalized WT201 digest algorithm differs")
    if mapping["mapping_commitment_algorithm"] != MAPPING_COMMITMENT_ALGORITHM:
        raise ProducerError("mapping commitment algorithm differs")
    if mapping["mapping_commitment_leaf_domain"] != MAPPING_COMMITMENT_LEAF_DOMAIN:
        raise ProducerError("mapping commitment leaf domain differs")
    if (
        mapping["mapping_commitment_parent_domain"]
        != MAPPING_COMMITMENT_PARENT_DOMAIN
    ):
        raise ProducerError("mapping commitment parent domain differs")
    if mapping["alternate_allele_in_group_key"] is not False:
        raise ProducerError("alternate allele entered the group key")
    if mapping["observed_group_count_must_be_hardcoded"] is not False:
        raise ProducerError("observed group count must not be prefrozen")
    if mapping["expected_canonical_locator_count"] != 6547:
        raise ProducerError("accepted canonical membership count differs")
    if mapping["wt_window_length"] != 201 or mapping["reference_index_zero_based"] != 100:
        raise ProducerError("WT source-window geometry differs")
    if config["consumer_contract"]["mapping_commitment_provenance_key"] != (
        MAPPING_COMMITMENT_KEY
    ):
        raise ProducerError("consumer commitment key differs")
    repository = config["repository_authority"]
    if repository["implementation_base_commit"] != IMPLEMENTATION_BASE_COMMIT:
        raise ProducerError("producer implementation base differs")
    if repository["implementation_commit_exact_changed_paths"] != [
        CONFIG_REPO_PATH,
        SCRIPT_REPO_PATH,
        TEST_REPO_PATH,
    ]:
        raise ProducerError("producer implementation path set differs")
    if repository["binding_commit_exact_changed_paths"] != [CONFIG_REPO_PATH]:
        raise ProducerError("producer binding path set differs")
    output = config["output_contract"]
    data_member_names = [
        output["private_mapping_basename"],
        output["aggregate_audit_basename"],
        output["allowed_basename"],
    ]
    if output["data_member_names"] != data_member_names:
        raise ProducerError("publication data member order differs")
    if output["terminal_commit_marker"] != "PUBLICATION_COMMIT.json":
        raise ProducerError("publication commit marker differs")
    if output["exact_final_member_names"] != [
        *data_member_names,
        output["terminal_commit_marker"],
    ]:
        raise ProducerError("publication exact-four member contract differs")
    for key, expected in {
        "commit_marker_written_last": True,
        "existing_exact_is_idempotent": True,
        "partial_or_mismatched_publication_requires_manual_intervention": True,
        "automatic_recovery": False,
    }.items():
        if output[key] is not expected:
            raise ProducerError(f"publication contract {key} differs")


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    if binding["status"] != "BOUND":
        raise ProducerError("producer implementation binding is UNKNOWN")
    if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
        raise ProducerError("producer implementation commit is not bound")
    for key in (
        "implementation_script_sha256",
        "implementation_test_sha256",
        "consumer_upgrade_config_sha256",
        "consumer_upgrade_script_sha256",
    ):
        if HEX64.fullmatch(str(binding[key])) is None:
            raise ProducerError(f"{key} is not bound")
    if HEX40.fullmatch(str(binding["consumer_upgrade_binding_commit"])) is None:
        raise ProducerError("consumer upgrade binding commit is not bound")


def validate_authorities(
    upstream: Mapping[str, Any],
    lineage: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if (
        upstream.get("schema_version")
        != "route_a_v3_gse200304_upstream_authority_viability.v1"
        or upstream.get("dataset_id") != DATASET_ID
        or upstream.get("status")
        != "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE"
    ):
        raise ProducerError("upstream authority is not the accepted closed audit")
    soft = upstream.get("geo_soft_authority")
    if type(soft) is not dict or soft.get("series_accession") != "GSE200302" or (
        soft.get("subseries_of_gse200304") is not True
    ):
        raise ProducerError("GSE200302 subseries authority is absent")
    prior = upstream.get("biological_group_authority")
    if type(prior) is not dict or prior.get("alternate_allele_in_group_key") is not False:
        raise ProducerError("upstream repair-route group semantics differ")
    if prior.get("repair_route_group_key_fields") != config["mapping_contract"][
        "group_key_fields"
    ]:
        raise ProducerError("upstream repair-route group key differs")
    if prior.get("repair_route_commitment") != config["mapping_contract"][
        "mapping_commitment_algorithm"
    ]:
        raise ProducerError("upstream repair-route mapping commitment differs")
    facts = lineage.get("facts")
    if (
        lineage.get("status") != PASS
        or lineage.get("gate_id") != "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE"
        or type(facts) is not dict
        or facts.get("canonical_record_count")
        != config["mapping_contract"]["expected_canonical_locator_count"]
        or facts.get("deterministic_row_locator_frozen") is not True
        or facts.get("multi_asset_lineage_closed") is not True
        or facts.get("s2_s3_join_rule_frozen") is not True
    ):
        raise ProducerError("canonical lineage gate is not accepted")


def parse_s2(payload: bytes, pair_pattern: re.Pattern[str]) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ProducerError("Table S2 is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    expected_header = ["ID", "Type", "201bp", "5' End", "3'End", "Full_Oligo"]
    if reader.fieldnames != expected_header:
        raise ProducerError("Table S2 header differs")
    rows_by_pair: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    physical_counts: Counter[str] = Counter()
    raw_rows = 0
    controls = 0
    for row in reader:
        raw_rows += 1
        role = row["Type"]
        if role == "Control":
            controls += 1
            continue
        if role not in {"WT", "Mutant"} or pair_pattern.fullmatch(row["ID"]) is None:
            raise ProducerError("Table S2 mutation row violates author ID grammar")
        sequence = row["201bp"].upper()
        if DNA201.fullmatch(sequence) is None:
            raise ProducerError("Table S2 mutation window is not exact 201-nt DNA")
        rows_by_pair[row["ID"]][role].add(sequence)
        physical_counts[row["ID"]] += 1
    pairs: dict[str, dict[str, str]] = {}
    duplicated_pairs = 0
    for pair_id, roles in rows_by_pair.items():
        if set(roles) != {"WT", "Mutant"} or any(len(values) != 1 for values in roles.values()):
            raise ProducerError("Table S2 pair lacks one unique WT and Mutant design")
        multiplicity = physical_counts[pair_id]
        if multiplicity == 4:
            duplicated_pairs += 1
        elif multiplicity != 2:
            raise ProducerError("Table S2 pair physical multiplicity differs")
        match = pair_pattern.fullmatch(pair_id)
        assert match is not None
        pairs[pair_id] = {
            "chromosome": match.group("chromosome"),
            "position": str(int(match.group("position"))),
            "reference": match.group("reference"),
            "alternate": match.group("alternate"),
            "wt201": next(iter(roles["WT"])),
        }
    return pairs, {
        "raw_row_count": raw_rows,
        "control_row_count": controls,
        "deduplicated_pair_count": len(pairs),
        "duplicated_pair_count": duplicated_pairs,
    }


def _xlsx_shared_strings(payload: bytes) -> list[str]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ProducerError("Table S3 shared strings XML is invalid") from exc
    return [
        "".join(
            child.text or ""
            for child in item.iter()
            if child.tag == f"{SPREADSHEETML}t"
        )
        for item in root.findall(f"{SPREADSHEETML}si")
    ]


def _xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str | float:
    values = [child.text or "" for child in cell if child.tag == f"{SPREADSHEETML}v"]
    if len(values) != 1:
        raise ProducerError("Table S3 selected cell lacks one cached value")
    value = values[0]
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "s":
        if not value.isdigit() or int(value) >= len(shared_strings):
            raise ProducerError("Table S3 shared-string index is invalid")
        return shared_strings[int(value)]
    if cell_type in {"n", ""}:
        try:
            return float(value)
        except ValueError as exc:
            raise ProducerError("Table S3 selected numeric cell is invalid") from exc
    raise ProducerError("Table S3 selected cell type differs")


def _xlsx_selected_s3_rows(
    worksheet: bytes,
    shared_strings: Sequence[str],
) -> list[dict[str, str | float]]:
    selected_columns = {"A", "C", "D", "E", "F"}
    rows: list[dict[str, str | float]] = []
    try:
        iterator = ET.iterparse(io.BytesIO(worksheet), events=("end",))
        for _event, element in iterator:
            if element.tag != f"{SPREADSHEETML}row":
                continue
            row_text = element.attrib.get("r", "")
            if not row_text.isdigit() or int(row_text) != len(rows) + 1:
                raise ProducerError("Table S3 row sequence differs")
            row_number = int(row_text)
            selected: dict[str, str | float] = {}
            for cell in element:
                if cell.tag != f"{SPREADSHEETML}c":
                    continue
                match = CELL_REFERENCE.fullmatch(cell.attrib.get("r", ""))
                if match is None or int(match.group("row")) != row_number:
                    raise ProducerError("Table S3 cell reference differs")
                column = match.group("column")
                if column not in selected_columns:
                    continue
                if column in selected:
                    raise ProducerError("Table S3 selected cell is duplicated")
                selected[column] = _xlsx_cell_value(cell, shared_strings)
            if set(selected) != selected_columns:
                raise ProducerError("Table S3 selected row width differs")
            rows.append(selected)
            element.clear()
    except ET.ParseError as exc:
        raise ProducerError("Table S3 primary worksheet XML is invalid") from exc
    if not rows:
        raise ProducerError("Table S3 primary worksheet is empty")
    return rows


def finite_totalpoly_pair_ids(payload: bytes) -> tuple[set[str], dict[str, int]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            required = {
                "xl/workbook.xml",
                "xl/worksheets/sheet1.xml",
                "xl/worksheets/sheet2.xml",
                "xl/sharedStrings.xml",
            }
            if not required <= set(archive.namelist()):
                raise ProducerError("Table S3 XLSX required members are absent")
            workbook_payload = archive.read("xl/workbook.xml")
            worksheet = archive.read("xl/worksheets/sheet1.xml")
            shared_payload = archive.read("xl/sharedStrings.xml")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProducerError("Table S3 cannot be opened as XLSX") from exc
    try:
        workbook_root = ET.fromstring(workbook_payload)
    except ET.ParseError as exc:
        raise ProducerError("Table S3 workbook XML is invalid") from exc
    sheet_names = [
        sheet.attrib.get("name")
        for sheet in workbook_root.findall(f".//{SPREADSHEETML}sheet")
    ]
    if sheet_names != [
        "S2A_Polysome_MPRA_Mut_Stats",
        "S2B_Poly_MPRA_Control_Stats",
    ]:
        raise ProducerError("Table S3 sheet names/order differ")
    rows = _xlsx_selected_s3_rows(worksheet, _xlsx_shared_strings(shared_payload))
    expected_header = {
        "A": "barcode",
        "C": "Comparison",
        "D": "xtail_log2FC_TE",
        "E": "xtail_pvalue",
        "F": "xtail_FDR",
    }
    if rows[0] != expected_header:
        raise ProducerError("Table S3 selected primary header differs")

    eligible: set[str] = set()
    comparisons: Counter[str] = Counter()
    pair_comparisons: dict[str, set[str]] = defaultdict(set)
    for row in rows[1:]:
        pair_id = row["A"]
        comparison = row["C"]
        if type(pair_id) is not str or comparison not in {"HighPoly:RNA", "TotalPoly:RNA"}:
            raise ProducerError("Table S3 pair/comparison differs")
        if comparison in pair_comparisons[pair_id]:
            raise ProducerError("Table S3 duplicates a pair/comparison")
        pair_comparisons[pair_id].add(comparison)
        comparisons[comparison] += 1
        if comparison == "TotalPoly:RNA":
            statistics = tuple(row[column] for column in ("D", "E", "F"))
            finite = all(
                type(value) in {int, float} and math.isfinite(float(value))
                for value in statistics
            )
            exact_na = all(value == "NA" for value in statistics)
            if not (finite or exact_na):
                raise ProducerError("Table S3 TotalPoly statistics are mixed or invalid")
            if finite:
                eligible.add(pair_id)
    if any(values != {"HighPoly:RNA", "TotalPoly:RNA"} for values in pair_comparisons.values()):
        raise ProducerError("Table S3 pair comparison geometry differs")
    return eligible, {
        "pair_count": len(pair_comparisons),
        "highpoly_row_count": comparisons["HighPoly:RNA"],
        "totalpoly_row_count": comparisons["TotalPoly:RNA"],
        "finite_totalpoly_pair_count": len(eligible),
    }


def build_mapping(
    pairs: Mapping[str, Mapping[str, str]],
    eligible: set[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not eligible <= set(pairs):
        raise ProducerError("a finite TotalPoly pair lacks its exact S2 pair")
    expected = config["mapping_contract"]["expected_canonical_locator_count"]
    if len(eligible) != expected:
        raise ProducerError("finite TotalPoly membership differs from accepted canonical lineage")
    study = config["mapping_contract"]["study_id"]
    subseries = config["mapping_contract"]["subseries_id"]
    orientations: Counter[str] = Counter()
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entries: list[dict[str, str]] = []
    seen_locators: set[str] = set()
    for pair_id in sorted(eligible):
        pair = pairs[pair_id]
        reference = pair["reference"]
        wt201 = pair["wt201"]
        if wt201[100] == reference:
            normalized = wt201
            orientations["FORWARD"] += 1
        elif reverse_complement(wt201)[100] == reference:
            normalized = reverse_complement(wt201)
            orientations["REVERSE_COMPLEMENT"] += 1
        else:
            raise ProducerError("WT201 cannot be oriented to the author reference allele")
        normalized_digest = normalized_wt201_digest(normalized)
        group_id = biological_group_id(
            study,
            subseries,
            pair["chromosome"],
            pair["position"],
            reference,
            normalized_digest,
        )
        locator = canonical_locator(pair_id)
        if locator in seen_locators:
            raise ProducerError("canonical locator is not unique")
        seen_locators.add(locator)
        groups[group_id].append((pair["alternate"], locator))
        entries.append(
            {"canonical_locator": locator, "biological_group_id": group_id}
        )
    histogram = Counter(len(members) for members in groups.values())
    multi = [members for members in groups.values() if len(members) > 1]
    if any(len({alternate for alternate, _locator in members}) != len(members) for members in multi):
        raise ProducerError("a multi-candidate group does not preserve distinct alternates")
    sorted_entries = sorted(entries, key=lambda item: item["canonical_locator"])
    mapping_commitment = biological_group_mapping_commitment(sorted_entries, config)
    private_mapping = {
        "schema_version": "route_a_v3_gse200304_private_biological_group_mapping.v1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "mapping_commitment_algorithm": config["mapping_contract"][
            "mapping_commitment_algorithm"
        ],
        "mapping_commitment_leaf_domain": config["mapping_contract"][
            "mapping_commitment_leaf_domain"
        ],
        "mapping_commitment_parent_domain": config["mapping_contract"][
            "mapping_commitment_parent_domain"
        ],
        "mapping_commitment_sha256": mapping_commitment,
        "record_count": len(entries),
        "group_count": len(groups),
        "mappings": sorted_entries,
    }
    audit = {
        "schema_version": "route_a_v3_gse200304_biological_group_mapping_audit.v1",
        "record_type": "GSE200304_AUTHOR_ANCHORED_BIOLOGICAL_GROUP_MAPPING_AUDIT_V1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": "PASS_AUTHOR_ANCHORED_GROUP_MAPPING",
        "eligible_canonical_locator_count": len(entries),
        "mapped_locator_count": len(entries),
        "unmapped_locator_count": expected - len(entries),
        "distinct_biological_group_count": len(groups),
        "group_size_histogram": {
            str(size): count for size, count in sorted(histogram.items())
        },
        "maximum_group_size": max(histogram),
        "multi_candidate_group_count": len(multi),
        "multi_candidate_groups_have_distinct_alternates": True,
        "orientation_counts": dict(sorted(orientations.items())),
        "alternate_allele_in_group_key": False,
        "mapping_commitment_sha256": mapping_commitment,
        "raw_id_sequence_gene_effect_persisted": False,
    }
    return private_mapping, audit


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gse200304_consumer_upgrade", path)
    if spec is None or spec.loader is None:
        raise ProducerError("consumer module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_gate_record(
    config: Mapping[str, Any],
    mapping_commitment: str,
    consumer_config: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor = consumer_config["evidence_contract"]["required_predecessor_authority"]
    binding = config["implementation_binding"]
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "record_type": EVIDENCE_RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "dataset_id": DATASET_ID,
        "gate_id": GATE_ID,
        "status": PASS,
        "accepted": True,
        "aggregate_only": True,
        "privacy": {
            "contains_row_level_payload": False,
            "contains_sequence": False,
            "contains_row_identifier": False,
            "contains_raw_label_or_effect": False,
            "contains_member_identifiers_or_hashes": False,
        },
        "provenance": {
            "producer_protocol_id": PROTOCOL_ID,
            "producer_commit": binding["implementation_commit"],
            "producer_script_sha256": binding["implementation_script_sha256"],
            "source_bundle_id": predecessor["bundle_id"],
            "source_bundle_root_or_target_sha256": predecessor[
                "terminal_marker_final_output_target_sha256"
            ],
            "predecessor_authority": copy.deepcopy(predecessor),
            "acceptance_authority": copy.deepcopy(
                consumer_config["evidence_contract"]["gate_record_provenance_contract"]
                ["acceptance_authority"]
            ),
            MAPPING_COMMITMENT_KEY: mapping_commitment,
        },
        "facts": copy.deepcopy(config["consumer_contract"]["facts"]),
        "unknown_fields": [],
        "reason_codes": [],
    }


def validate_with_consumer(
    gate_payload: bytes,
    consumer_config: Mapping[str, Any],
    consumer_module: ModuleType,
) -> None:
    consumer_module.validate_static_config(consumer_config)
    slot = next(
        item
        for item in consumer_config["evidence_contract"]["slots"]
        if item["slot_id"] == GATE_ID
    )
    record = consumer_module._validate_gate_record(
        gate_payload,
        slot,
        consumer_config,
    )
    if consumer_module._slot_gate_pass(GATE_ID, record["facts"]) is not True:
        raise ProducerError("upgraded consumer did not accept the group PASS")


def produce(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    upstream_audit_path: Path,
    lineage_gate_path: Path,
    consumer_config_path: Path,
    consumer_script_path: Path,
) -> dict[str, bytes]:
    validate_static_config(config)
    validate_implementation_binding(config)
    source = config["source_authority"]
    s2 = _bound_source(source_root / source["table_s2"]["relative_path"], source["table_s2"], label="Table S2")
    s3 = _bound_source(source_root / source["table_s3"]["relative_path"], source["table_s3"], label="Table S3")
    upstream = strict_json(
        _bound_source(upstream_audit_path, source["upstream_audit"], label="upstream audit"),
        label="upstream audit",
    )
    lineage = strict_json(
        _bound_source(lineage_gate_path, source["canonical_lineage_gate"], label="lineage gate"),
        label="lineage gate",
    )
    validate_authorities(upstream, lineage, config)
    pattern = re.compile(config["mapping_contract"]["pair_id_grammar"])
    pairs, s2_audit = parse_s2(s2, pattern)
    eligible, s3_audit = finite_totalpoly_pair_ids(s3)
    private_mapping, audit = build_mapping(pairs, eligible, config)
    audit["table_s2_aggregates"] = s2_audit
    audit["table_s3_aggregates"] = s3_audit
    private_payload = json_bytes(private_mapping)

    consumer_config = strict_json(consumer_config_path.read_bytes(), label="consumer config")
    consumer_module = _load_module(consumer_script_path)
    gate = build_gate_record(
        config,
        private_mapping["mapping_commitment_sha256"],
        consumer_config,
    )
    gate_payload = json_bytes(gate)
    validate_with_consumer(gate_payload, consumer_config, consumer_module)
    audit["consumer_actual_acceptance"] = True
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_gate_keys"]}
    if any(key.casefold() in forbidden for key in _walk_keys(gate)):
        raise ProducerError("aggregate gate contains forbidden output content")
    output = config["output_contract"]
    return {
        output["private_mapping_basename"]: private_payload,
        output["aggregate_audit_basename"]: json_bytes(audit),
        output["allowed_basename"]: gate_payload,
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_keys(child)


def publication_commit_payload(payloads: Mapping[str, bytes], names: Sequence[str]) -> bytes:
    return json_bytes(
        {
            "schema_version": PUBLICATION_COMMIT_SCHEMA,
            "committed": True,
            "members": [
                {
                    "name": name,
                    "bytes": len(payloads[name]),
                    "sha256": sha256(payloads[name]),
                }
                for name in names
            ],
        }
    )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        written = handle.write(payload)
        if written != len(payload):
            raise ProducerError(f"short publication write: {path.name}")
        handle.flush()
        os.fsync(handle.fileno())


def write_outputs(
    output_directory: Path,
    payloads: Mapping[str, bytes],
    config: Mapping[str, Any],
) -> str:
    output = config["output_contract"]
    data_names = list(output["data_member_names"])
    if len(payloads) != len(data_names) or set(payloads) != set(data_names):
        raise ProducerError("publication payload names differ from the exact data members")
    marker_name = output["terminal_commit_marker"]
    marker_payload = publication_commit_payload(payloads, data_names)
    expected_payloads = {name: payloads[name] for name in data_names}
    expected_payloads[marker_name] = marker_payload

    if output_directory.exists():
        try:
            existing_names = {path.name for path in output_directory.iterdir()}
        except OSError as exc:
            raise ProducerError("existing publication cannot be inspected") from exc
        if existing_names != set(output["exact_final_member_names"]):
            raise ProducerError(
                "existing publication is partial or mismatched; preserve it for manual intervention"
            )
        for name, expected in expected_payloads.items():
            try:
                observed = (output_directory / name).read_bytes()
            except OSError as exc:
                raise ProducerError(
                    "existing publication cannot be inspected; preserve it for manual intervention"
                ) from exc
            if observed != expected:
                raise ProducerError(
                    "existing publication is partial or mismatched; preserve it for manual intervention"
                )
        return "IDEMPOTENT"

    output_directory.mkdir(parents=True, exist_ok=False)
    for name in data_names:
        _write_fsynced(output_directory / name, payloads[name])
    _write_fsynced(output_directory / marker_name, marker_payload)
    directory_fd = os.open(output_directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return "PUBLISHED"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--consumer-config", type=Path, required=True)
    parser.add_argument("--consumer-script", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    config = strict_json(args.config.read_bytes(), label="producer config")
    source = config["source_authority"]
    payloads = produce(
        config,
        source_root=Path(source["data_root"]),
        upstream_audit_path=Path(source["upstream_audit"]["absolute_path"]),
        lineage_gate_path=Path(source["canonical_lineage_gate"]["absolute_path"]),
        consumer_config_path=args.consumer_config,
        consumer_script_path=args.consumer_script,
    )
    publication_status = write_outputs(args.output_directory, payloads, config)
    print(
        json.dumps(
            {
                "status": "PASS",
                "publication_status": publication_status,
                "outputs": config["output_contract"]["exact_final_member_names"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
