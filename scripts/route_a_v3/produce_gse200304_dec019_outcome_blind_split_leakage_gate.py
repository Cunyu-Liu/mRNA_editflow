#!/usr/bin/env python3
"""Build the fixed, outcome-blind DEC-019 GSE200304 split graph and folds."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
PASS = "PASS"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
GATE_ID = "OUTCOME_BLIND_SPLIT_LEAKAGE"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE_V1"
EVIDENCE_SCHEMA = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
SPLIT_COMMITMENT_KEY = "split_assignment_commitment_sha256"
EXPECTED_GROUP_MAPPING_ROOT = (
    "1900fa1085043aea9ff1aa96469ce6c8ae1795c7febf814eba04a2d3a259ff59"
)
EXPECTED_UNIVERSE = (
    "6547_ACCEPTED_CANONICAL_FINITE_TOTALPOLY_RECORDS_"
    "PREMODEL_ENDPOINT_AVAILABILITY"
)
JOIN_KEY_DOMAIN = b"route-a-v3/gse200304/dec019/join-key/v1"
LOCATOR_DOMAIN = b"route-a-v3/gse200304/dec019/canonical-row-locator/v1"
GROUP_MAPPING_LEAF_DOMAIN = (
    b"route-a-v3/gse200304/dec019/biological-group-mapping-leaf/v1"
)
GROUP_MAPPING_PARENT_DOMAIN = (
    b"route-a-v3/gse200304/dec019/biological-group-mapping-parent/v1"
)
COMPONENT_ID_DOMAIN = b"route-a-v3/gse200304/dec019/split-component-id/v1"
CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_outcome_blind_split_leakage_gate_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_outcome_blind_split_leakage_gate.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_outcome_blind_split_leakage_gate.py"
)
PUBLICATION_COMMIT_SCHEMA = "route_a_v3_gse200304_split_publication_commit.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DNA201 = re.compile(r"^[ACGT]{201}$")
CELL_REFERENCE = re.compile(r"^(?P<column>[A-Z]+)(?P<row>[1-9][0-9]*)$")
SPREADSHEETML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EDGE_GENE = "EXACT_GENE_TOKEN"
EDGE_HAMMING = "ORIENTATION_NORMALIZED_WT201_HAMMING_LE_10"
EDGE_JACCARD = "CANONICAL_15MER_SET_JACCARD_GE_0_80"


class ProducerError(RuntimeError):
    """A bound input or closed protocol invariant differs."""


class ProtocolStop(ProducerError):
    """The fixed graph cannot support the prefrozen PASS conditions."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


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


def _bound_source(path: Path, spec: Mapping[str, Any], *, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProducerError(f"cannot read {label}") from exc
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise ProducerError(f"{label} differs from its accepted source binding")
    return payload


def validate_static_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": (
            "route_a_v3_gse200304_dec019_outcome_blind_split_leakage_gate.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ProducerError(f"config {key} differs")
    repository = config["repository_authority"]
    if repository["implementation_commit_exact_changed_paths"] != [
        CONFIG_REPO_PATH,
        SCRIPT_REPO_PATH,
        TEST_REPO_PATH,
    ]:
        raise ProducerError("implementation exact-three path set differs")
    if repository["binding_commit_exact_changed_paths"] != [CONFIG_REPO_PATH]:
        raise ProducerError("binding path set differs")
    base = repository["implementation_base_commit"]
    if base != UNKNOWN and HEX40.fullmatch(str(base)) is None:
        raise ProducerError("implementation base is neither UNKNOWN nor HEX40")
    source = config["source_authority"]["biological_group_publication"]
    if source["mapping_commitment_sha256"] != EXPECTED_GROUP_MAPPING_ROOT:
        raise ProducerError("group mapping root differs")
    if (source["record_count"], source["biological_group_count"]) != (6547, 6544):
        raise ProducerError("group record/group counts differ")
    member_names = [item["name"] for item in source["members"]]
    if member_names != [
        "GSE200304_DEC019_BIOLOGICAL_GROUP_MAPPING_PRIVATE.json",
        "GSE200304_DEC019_BIOLOGICAL_GROUP_MAPPING_AUDIT.json",
        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json",
        "PUBLICATION_COMMIT.json",
    ]:
        raise ProducerError("group exact-four member order differs")
    split = config["split_contract"]
    expected_split = {
        "universe": EXPECTED_UNIVERSE,
        "universe_is_not_a2_final_membership": True,
        "outcome_columns_read": [],
        "study_id": DATASET_ID,
        "study_is_partition_scope_not_graph_edge": True,
        "node_id": "BIOLOGICAL_GROUP_ID",
        "wt_window_length": 201,
        "reference_index_zero_based": 100,
        "hamming_max_distance": 10,
        "hamming_candidate_fixed_block_count": 11,
        "kmer_length": 15,
        "jaccard_threshold_numerator": 4,
        "jaccard_threshold_denominator": 5,
        "components_are_indivisible_fold_atoms": True,
        "split_method": "STRICT_NESTED_GROUP_CV",
        "outer_fold_count": 5,
        "inner_fold_count": 4,
        "split_salt": "GSE200304_DEC019_A1_SPLIT_V1",
        "component_order": (
            "GROUP_COUNT_DESC_THEN_SHA256_SALT_NUL_COMPONENT_ID_ASC"
        ),
        "fold_choice": "CURRENT_GROUP_COUNT_ASC_THEN_FOLD_INDEX_ASC",
        "assignment_commitment_algorithm": (
            "DOMAIN_SEPARATED_LENGTH_PREFIXED_SHA256_LEAF_SORT_ODD_DUPLICATE"
        ),
    }
    for key, value in expected_split.items():
        if split.get(key) != value:
            raise ProducerError(f"split contract {key} differs")
    if split["edge_rules"] != [EDGE_GENE, EDGE_HAMMING, EDGE_JACCARD]:
        raise ProducerError("split edge rules differ")
    if config["pass_conditions"] != {
        "all_6547_records_mapped_to_6544_frozen_groups": True,
        "all_outer_folds_nonempty": True,
        "all_outer_train_inner_folds_nonempty": True,
        "all_required_cross_fold_leakage_counts_zero": True,
        "salt_or_threshold_retry_allowed": False,
    }:
        raise ProducerError("PASS conditions differ")


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
    if HEX40.fullmatch(str(config["repository_authority"]["implementation_base_commit"])) is None:
        raise ProducerError("producer implementation base is not bound")


def parse_s2_wt(payload: bytes, pair_pattern: re.Pattern[str]) -> dict[str, str]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ProducerError("Table S2 is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != ["ID", "Type", "201bp", "5' End", "3'End", "Full_Oligo"]:
        raise ProducerError("Table S2 header differs")
    roles: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    counts: Counter[str] = Counter()
    for row in reader:
        if row["Type"] == "Control":
            continue
        pair_id = row["ID"]
        role = row["Type"]
        if pair_pattern.fullmatch(pair_id) is None or role not in {"WT", "Mutant"}:
            raise ProducerError("Table S2 mutation row differs")
        sequence = row["201bp"].upper()
        if DNA201.fullmatch(sequence) is None:
            raise ProducerError("Table S2 mutation window is not DNA201")
        roles[pair_id][role].add(sequence)
        counts[pair_id] += 1
    result: dict[str, str] = {}
    for pair_id, values in roles.items():
        if set(values) != {"WT", "Mutant"} or any(len(item) != 1 for item in values.values()):
            raise ProducerError("Table S2 pair lacks one unique WT and Mutant design")
        if counts[pair_id] not in {2, 4}:
            raise ProducerError("Table S2 physical pair multiplicity differs")
        result[pair_id] = next(iter(values["WT"]))
    return result


def _xlsx_shared_strings(payload: bytes) -> list[str]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ProducerError("Table S3 shared strings XML is invalid") from exc
    return [
        "".join(child.text or "" for child in item.iter() if child.tag == f"{SPREADSHEETML}t")
        for item in root.findall(f"{SPREADSHEETML}si")
    ]


def _xlsx_text_cell(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    values = [child.text or "" for child in cell if child.tag == f"{SPREADSHEETML}v"]
    if len(values) != 1 or cell.attrib.get("t") != "s" or not values[0].isdigit():
        raise ProducerError("Table S3 selected text cell differs")
    index = int(values[0])
    if index >= len(shared_strings):
        raise ProducerError("Table S3 shared-string index is invalid")
    return shared_strings[index]


def _xlsx_abc_rows(worksheet: bytes, shared_strings: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        for _event, element in ET.iterparse(io.BytesIO(worksheet), events=("end",)):
            if element.tag != f"{SPREADSHEETML}row":
                continue
            row_number_text = element.attrib.get("r", "")
            if not row_number_text.isdigit() or int(row_number_text) != len(rows) + 1:
                raise ProducerError("Table S3 row sequence differs")
            row_number = int(row_number_text)
            selected: dict[str, str] = {}
            for cell in element:
                if cell.tag != f"{SPREADSHEETML}c":
                    continue
                match = CELL_REFERENCE.fullmatch(cell.attrib.get("r", ""))
                if match is None or int(match.group("row")) != row_number:
                    raise ProducerError("Table S3 cell reference differs")
                column = match.group("column")
                if column not in {"A", "B", "C"}:
                    continue
                if column in selected:
                    raise ProducerError("Table S3 selected cell is duplicated")
                selected[column] = _xlsx_text_cell(cell, shared_strings)
            if set(selected) != {"A", "B", "C"}:
                raise ProducerError("Table S3 A/B/C row width differs")
            rows.append(selected)
            element.clear()
    except ET.ParseError as exc:
        raise ProducerError("Table S3 worksheet XML is invalid") from exc
    return rows


def parse_s3_pair_genes(payload: bytes) -> dict[str, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            required = {
                "xl/workbook.xml",
                "xl/worksheets/sheet1.xml",
                "xl/sharedStrings.xml",
            }
            if not required <= set(archive.namelist()):
                raise ProducerError("Table S3 XLSX required members are absent")
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            worksheet = archive.read("xl/worksheets/sheet1.xml")
            shared = _xlsx_shared_strings(archive.read("xl/sharedStrings.xml"))
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise ProducerError("Table S3 cannot be opened") from exc
    sheet_names = [
        sheet.attrib.get("name")
        for sheet in workbook.findall(f".//{SPREADSHEETML}sheet")
    ]
    if sheet_names != [
        "S2A_Polysome_MPRA_Mut_Stats",
        "S2B_Poly_MPRA_Control_Stats",
    ]:
        raise ProducerError("Table S3 sheet names/order differ")
    rows = _xlsx_abc_rows(worksheet, shared)
    if not rows or rows[0] != {"A": "barcode", "B": "Gene", "C": "Comparison"}:
        raise ProducerError("Table S3 A/B/C header differs")
    observed: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows[1:]:
        pair_id, gene, comparison = row["A"], row["B"], row["C"]
        if comparison not in {"HighPoly:RNA", "TotalPoly:RNA"}:
            raise ProducerError("Table S3 comparison differs")
        if not gene or gene != gene.strip():
            raise ProducerError("Table S3 exact gene token is empty or padded")
        if comparison in observed[pair_id]:
            raise ProducerError("Table S3 duplicates a pair/comparison")
        observed[pair_id][comparison] = gene
    result: dict[str, str] = {}
    for pair_id, comparisons in observed.items():
        if set(comparisons) != {"HighPoly:RNA", "TotalPoly:RNA"}:
            raise ProducerError("Table S3 pair lacks one of the two comparisons")
        if len(set(comparisons.values())) != 1:
            raise ProducerError("Table S3 gene token differs between comparisons")
        result[pair_id] = comparisons["TotalPoly:RNA"]
    return result


def group_mapping_commitment(entries: Sequence[Mapping[str, str]]) -> str:
    level = sorted(
        _domain_hash(
            GROUP_MAPPING_LEAF_DOMAIN,
            (
                item["canonical_locator"].encode("ascii"),
                item["biological_group_id"].encode("ascii"),
            ),
        )
        for item in entries
    )
    if not level:
        raise ProducerError("group mapping is empty")
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            _domain_hash(GROUP_MAPPING_PARENT_DOMAIN, (level[index], level[index + 1]))
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def load_group_mapping(
    directory: Path,
    source: Mapping[str, Any],
) -> list[dict[str, str]]:
    expected_names = [item["name"] for item in source["members"]]
    try:
        observed_names = sorted(path.name for path in directory.iterdir())
    except OSError as exc:
        raise ProducerError("group publication cannot be inspected") from exc
    if observed_names != sorted(expected_names):
        raise ProducerError("group publication is not exact-four")
    payloads = {
        item["name"]: _bound_source(
            directory / item["name"], item, label=f"group {item['name']}"
        )
        for item in source["members"]
    }
    marker = strict_json(payloads["PUBLICATION_COMMIT.json"], label="group marker")
    if marker.get("committed") is not True:
        raise ProducerError("group publication is not committed")
    marker_members = marker.get("members")
    expected_data = source["members"][:3]
    if marker_members != [
        {"name": item["name"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in expected_data
    ]:
        raise ProducerError("group marker member binding differs")
    mapping = strict_json(
        payloads["GSE200304_DEC019_BIOLOGICAL_GROUP_MAPPING_PRIVATE.json"],
        label="group private mapping",
    )
    entries = mapping.get("mappings")
    if type(entries) is not list or any(
        type(item) is not dict
        or set(item) != {"canonical_locator", "biological_group_id"}
        or HEX64.fullmatch(str(item["canonical_locator"])) is None
        or HEX64.fullmatch(str(item["biological_group_id"])) is None
        for item in entries
    ):
        raise ProducerError("group private mapping schema differs")
    if (
        mapping.get("record_count") != source["record_count"]
        or mapping.get("group_count") != source["biological_group_count"]
        or len(entries) != source["record_count"]
        or len({item["canonical_locator"] for item in entries}) != len(entries)
        or len({item["biological_group_id"] for item in entries})
        != source["biological_group_count"]
    ):
        raise ProducerError("group mapping counts differ")
    root = group_mapping_commitment(entries)
    if (
        root != EXPECTED_GROUP_MAPPING_ROOT
        or root != source["mapping_commitment_sha256"]
        or mapping.get("mapping_commitment_sha256") != root
    ):
        raise ProducerError("group mapping commitment differs")
    audit = strict_json(
        payloads["GSE200304_DEC019_BIOLOGICAL_GROUP_MAPPING_AUDIT.json"],
        label="group mapping audit",
    )
    gate = strict_json(
        payloads["GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json"],
        label="group gate",
    )
    if (
        audit.get("mapping_commitment_sha256") != root
        or audit.get("eligible_canonical_locator_count") != 6547
        or audit.get("distinct_biological_group_count") != 6544
        or gate.get("gate_id") != "BIOLOGICAL_GROUP_AUTHORITY"
        or gate.get("status") != PASS
        or gate.get("provenance", {}).get("group_mapping_commitment_sha256") != root
    ):
        raise ProducerError("group aggregate authority differs")
    return entries


def build_nodes(
    mappings: Sequence[Mapping[str, str]],
    wt_by_pair: Mapping[str, str],
    gene_by_pair: Mapping[str, str],
    pair_pattern: re.Pattern[str],
) -> dict[str, dict[str, Any]]:
    pair_by_locator: dict[str, str] = {}
    for pair_id in sorted(set(wt_by_pair) & set(gene_by_pair)):
        locator = canonical_locator(pair_id)
        if locator in pair_by_locator:
            raise ProducerError("canonical locator collision")
        pair_by_locator[locator] = pair_id
    mapping_locators = {item["canonical_locator"] for item in mappings}
    if not mapping_locators <= set(pair_by_locator):
        raise ProducerError("a frozen group locator lacks S2/S3 design authority")
    nodes: dict[str, dict[str, Any]] = {}
    for item in mappings:
        pair_id = pair_by_locator[item["canonical_locator"]]
        match = pair_pattern.fullmatch(pair_id)
        if match is None:
            raise ProducerError("accepted pair violates author grammar")
        reference = match.group("reference")
        wt201 = wt_by_pair[pair_id]
        if wt201[100] == reference:
            normalized = wt201
        else:
            normalized = reverse_complement(wt201)
            if normalized[100] != reference:
                raise ProducerError("WT201 cannot orient to author reference")
        group_id = item["biological_group_id"]
        candidate = {
            "sequence": normalized,
            "gene": gene_by_pair[pair_id],
            "record_count": 1,
        }
        if group_id in nodes:
            current = nodes[group_id]
            if (
                current["sequence"] != candidate["sequence"]
                or current["gene"] != candidate["gene"]
            ):
                raise ProducerError("one biological group has inconsistent sequence or gene")
            current["record_count"] += 1
        else:
            nodes[group_id] = candidate
    if len(nodes) != 6544 or sum(node["record_count"] for node in nodes.values()) != 6547:
        raise ProducerError("observed node universe differs from 6547-to-6544 authority")
    return nodes


def _pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _fixed_blocks(sequence: str, count: int) -> list[str]:
    return [
        sequence[(index * len(sequence)) // count : ((index + 1) * len(sequence)) // count]
        for index in range(count)
    ]


def _canonical_kmers(sequence: str, length: int) -> frozenset[str]:
    return frozenset(
        min(token, reverse_complement(token))
        for token in (
            sequence[index : index + length]
            for index in range(len(sequence) - length + 1)
        )
    )


def build_edges(
    nodes: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[tuple[str, str], set[str]]:
    split = config["split_contract"]
    ids = sorted(nodes)
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)

    genes: dict[str, list[str]] = defaultdict(list)
    for group_id in ids:
        genes[str(nodes[group_id]["gene"])].append(group_id)
    for members in genes.values():
        for left, right in combinations(sorted(members), 2):
            reasons[(left, right)].add(EDGE_GENE)

    block_count = split["hamming_candidate_fixed_block_count"]
    hamming_max = split["hamming_max_distance"]
    block_index: dict[tuple[int, str], list[str]] = defaultdict(list)
    for group_id in ids:
        sequence = str(nodes[group_id]["sequence"])
        for index, block in enumerate(_fixed_blocks(sequence, block_count)):
            block_index[(index, block)].append(group_id)
    hamming_candidates: set[tuple[str, str]] = set()
    for members in block_index.values():
        hamming_candidates.update(combinations(sorted(members), 2))
    for left, right in hamming_candidates:
        distance = sum(
            a != b
            for a, b in zip(nodes[left]["sequence"], nodes[right]["sequence"])
        )
        if distance <= hamming_max:
            reasons[(left, right)].add(EDGE_HAMMING)

    kmer_length = split["kmer_length"]
    numerator = split["jaccard_threshold_numerator"]
    denominator = split["jaccard_threshold_denominator"]
    kmer_sets = {
        group_id: _canonical_kmers(str(nodes[group_id]["sequence"]), kmer_length)
        for group_id in ids
    }
    kmer_index: dict[str, list[str]] = defaultdict(list)
    for group_id in ids:
        for token in kmer_sets[group_id]:
            kmer_index[token].append(group_id)
    intersection_counts: Counter[tuple[str, str]] = Counter()
    for members in kmer_index.values():
        intersection_counts.update(combinations(sorted(members), 2))
    for (left, right), intersection in intersection_counts.items():
        union = len(kmer_sets[left] | kmer_sets[right])
        if denominator * intersection >= numerator * union:
            reasons[(left, right)].add(EDGE_JACCARD)
    return dict(reasons)


class UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        root = member
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[member] != member:
            parent = self.parent[member]
            self.parent[member] = root
            member = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def build_components(
    node_ids: Iterable[str],
    edges: Mapping[tuple[str, str], set[str]],
) -> dict[str, list[str]]:
    ids = sorted(node_ids)
    union = UnionFind(ids)
    for left, right in edges:
        union.union(left, right)
    by_root: dict[str, list[str]] = defaultdict(list)
    for group_id in ids:
        by_root[union.find(group_id)].append(group_id)
    components: dict[str, list[str]] = {}
    for members in by_root.values():
        ordered = sorted(members)
        component_id = _domain_hash(
            COMPONENT_ID_DOMAIN,
            (member.encode("ascii") for member in ordered),
        ).hex()
        if component_id in components:
            raise ProducerError("component ID collision")
        components[component_id] = ordered
    return components


def assign_component_folds(
    components: Mapping[str, Sequence[str]],
    fold_count: int,
    salt: str,
) -> tuple[dict[str, int], list[dict[str, int]]]:
    counts = [0] * fold_count
    component_counts = [0] * fold_count
    order = sorted(
        components,
        key=lambda component_id: (
            -len(components[component_id]),
            sha256(salt.encode("utf-8") + b"\0" + component_id.encode("ascii")),
        ),
    )
    assignments: dict[str, int] = {}
    for component_id in order:
        fold = min(range(fold_count), key=lambda index: (counts[index], index))
        assignments[component_id] = fold
        counts[fold] += len(components[component_id])
        component_counts[fold] += 1
    return assignments, [
        {"fold": index, "group_count": counts[index], "component_count": component_counts[index]}
        for index in range(fold_count)
    ]


def assignment_commitment(
    assignments: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> str:
    split = config["split_contract"]
    leaf_domain = split["assignment_commitment_leaf_domain"].encode("ascii")
    parent_domain = split["assignment_commitment_parent_domain"].encode("ascii")
    level = sorted(
        _domain_hash(
            leaf_domain,
            (
                item["biological_group_id"].encode("ascii"),
                item["component_id"].encode("ascii"),
                str(item["outer_fold"]).encode("ascii"),
                *(
                    ("NA" if value is None else str(value)).encode("ascii")
                    for value in item["inner_folds_by_outer"]
                ),
            ),
        )
        for item in assignments
    )
    if not level:
        raise ProducerError("assignment commitment input is empty")
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            _domain_hash(parent_domain, (level[index], level[index + 1]))
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _cross_fold_counts(
    edges: Mapping[tuple[str, str], set[str]],
    group_folds: Mapping[str, int],
) -> dict[str, int]:
    counts = {
        "component_cross_fold_count": 0,
        "exact_gene_edge_cross_fold_count": 0,
        "hamming_edge_cross_fold_count": 0,
        "jaccard_edge_cross_fold_count": 0,
    }
    reason_keys = {
        EDGE_GENE: "exact_gene_edge_cross_fold_count",
        EDGE_HAMMING: "hamming_edge_cross_fold_count",
        EDGE_JACCARD: "jaccard_edge_cross_fold_count",
    }
    for (left, right), reasons in edges.items():
        if group_folds[left] == group_folds[right]:
            continue
        for reason in reasons:
            counts[reason_keys[reason]] += 1
    return counts


def _component_cross_fold_count(
    components: Mapping[str, Sequence[str]],
    group_folds: Mapping[str, int],
) -> int:
    return sum(
        len({group_folds[group_id] for group_id in members}) > 1
        for members in components.values()
    )


def build_split(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Mapping[tuple[str, str], set[str]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    split = config["split_contract"]
    components = build_components(nodes, edges)
    outer_components, outer_fold_counts = assign_component_folds(
        components,
        split["outer_fold_count"],
        split["split_salt"],
    )
    group_component = {
        group_id: component_id
        for component_id, members in components.items()
        for group_id in members
    }
    outer_group = {
        group_id: outer_components[group_component[group_id]] for group_id in nodes
    }
    inner_components_by_outer: dict[int, dict[str, int]] = {}
    inner_fold_counts: list[dict[str, Any]] = []
    inner_leakage: list[dict[str, Any]] = []
    for outer_fold in range(split["outer_fold_count"]):
        training = {
            component_id: members
            for component_id, members in components.items()
            if outer_components[component_id] != outer_fold
        }
        inner_assignments, fold_counts = assign_component_folds(
            training,
            split["inner_fold_count"],
            f"{split['split_salt']}/outer={outer_fold}/inner",
        )
        inner_components_by_outer[outer_fold] = inner_assignments
        inner_fold_counts.append({"outer_fold": outer_fold, "folds": fold_counts})
        inner_group = {
            group_id: inner_assignments[component_id]
            for component_id, members in training.items()
            for group_id in members
        }
        training_edges = {
            pair: reasons
            for pair, reasons in edges.items()
            if pair[0] in inner_group and pair[1] in inner_group
        }
        leakage = _cross_fold_counts(training_edges, inner_group)
        leakage["component_cross_fold_count"] = _component_cross_fold_count(
            training,
            inner_group,
        )
        leakage["outer_fold"] = outer_fold
        inner_leakage.append(leakage)

    assignments = []
    for group_id in sorted(nodes):
        outer_fold = outer_group[group_id]
        component_id = group_component[group_id]
        inner = [
            None
            if candidate_outer == outer_fold
            else inner_components_by_outer[candidate_outer][component_id]
            for candidate_outer in range(split["outer_fold_count"])
        ]
        assignments.append(
            {
                "biological_group_id": group_id,
                "component_id": component_id,
                "outer_fold": outer_fold,
                "inner_folds_by_outer": inner,
            }
        )
    commitment = assignment_commitment(assignments, config)
    outer_leakage = _cross_fold_counts(edges, outer_group)
    outer_leakage["component_cross_fold_count"] = _component_cross_fold_count(
        components,
        outer_group,
    )
    reason_counts = Counter(reason for values in edges.values() for reason in values)
    outer_nonempty = all(item["group_count"] > 0 for item in outer_fold_counts)
    inner_nonempty = all(
        fold["group_count"] > 0
        for item in inner_fold_counts
        for fold in item["folds"]
    )
    all_leakage = [
        value
        for key, value in outer_leakage.items()
        if key.endswith("_cross_fold_count")
    ] + [
        value
        for item in inner_leakage
        for key, value in item.items()
        if key.endswith("_cross_fold_count")
    ]
    go = outer_nonempty and inner_nonempty and all(value == 0 for value in all_leakage)
    private = {
        "schema_version": "route_a_v3_gse200304_private_split_assignment.v1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "universe": EXPECTED_UNIVERSE,
        "split_method": split["split_method"],
        "split_salt": split["split_salt"],
        "assignment_commitment_algorithm": split["assignment_commitment_algorithm"],
        "assignment_commitment_leaf_domain": split["assignment_commitment_leaf_domain"],
        "assignment_commitment_parent_domain": split["assignment_commitment_parent_domain"],
        "assignment_commitment_sha256": commitment,
        "group_count": len(nodes),
        "component_count": len(components),
        "assignments": assignments,
    }
    component_sizes = Counter(len(members) for members in components.values())
    audit = {
        "schema_version": "route_a_v3_gse200304_split_leakage_audit.v1",
        "record_type": "GSE200304_OUTCOME_BLIND_STRICT_NESTED_GROUP_CV_AUDIT_V1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": "GO_PASS_CONDITIONS_MET" if go else "STOP_PASS_CONDITIONS_NOT_MET",
        "universe": EXPECTED_UNIVERSE,
        "final_benchmark_membership_deferred_to_a2": True,
        "outcome_columns_read": [],
        "record_count": sum(node["record_count"] for node in nodes.values()),
        "biological_group_node_count": len(nodes),
        "connected_component_count": len(components),
        "component_size_histogram": {
            str(size): count for size, count in sorted(component_sizes.items())
        },
        "largest_component_group_count": max(component_sizes),
        "edge_counts_by_reason": {
            EDGE_GENE: reason_counts[EDGE_GENE],
            EDGE_HAMMING: reason_counts[EDGE_HAMMING],
            EDGE_JACCARD: reason_counts[EDGE_JACCARD],
            "UNION_DISTINCT_EDGE_COUNT": len(edges),
        },
        "outer_fold_counts": outer_fold_counts,
        "inner_fold_counts_by_outer": inner_fold_counts,
        "outer_cross_fold_leakage_counts": outer_leakage,
        "inner_cross_fold_leakage_counts_by_outer": inner_leakage,
        "all_outer_folds_nonempty": outer_nonempty,
        "all_outer_train_inner_folds_nonempty": inner_nonempty,
        "all_required_cross_fold_leakage_counts_zero": all(value == 0 for value in all_leakage),
        "assignment_commitment_sha256": commitment,
        "aggregate_contains_gene_sequence_pair_or_group_identifiers": False,
    }
    if not go:
        raise ProtocolStop("fixed split graph does not meet PASS conditions", audit)
    return private, audit


def compute_split(
    config: Mapping[str, Any],
    *,
    public_root: Path,
    group_publication: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_static_config(config)
    source = config["source_authority"]
    s2 = _bound_source(
        public_root / source["table_s2"]["relative_path"],
        source["table_s2"],
        label="Table S2",
    )
    s3 = _bound_source(
        public_root / source["table_s3"]["relative_path"],
        source["table_s3"],
        label="Table S3",
    )
    mappings = load_group_mapping(group_publication, source["biological_group_publication"])
    pattern = re.compile(config["split_contract"]["pair_id_grammar"])
    nodes = build_nodes(
        mappings,
        parse_s2_wt(s2, pattern),
        parse_s3_pair_genes(s3),
        pattern,
    )
    edges = build_edges(nodes, config)
    return build_split(nodes, edges, config)


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gse200304_split_consumer", path)
    if spec is None or spec.loader is None:
        raise ProducerError("consumer module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_gate_record(
    config: Mapping[str, Any],
    assignment_commitment_sha256: str,
    consumer_config: Mapping[str, Any],
) -> dict[str, Any]:
    if HEX64.fullmatch(assignment_commitment_sha256) is None:
        raise ProducerError("split assignment commitment is not HEX64")
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
            SPLIT_COMMITMENT_KEY: assignment_commitment_sha256,
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
        raise ProducerError("upgraded consumer did not accept the split PASS")


def _walk_keys(value: Any) -> Iterable[str]:
    if type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_keys(child)


def produce(
    config: Mapping[str, Any],
    *,
    public_root: Path,
    group_publication: Path,
    consumer_config_path: Path,
    consumer_script_path: Path,
) -> dict[str, bytes]:
    validate_static_config(config)
    validate_implementation_binding(config)
    binding = config["implementation_binding"]
    try:
        consumer_config_payload = consumer_config_path.read_bytes()
        consumer_script_payload = consumer_script_path.read_bytes()
    except OSError as exc:
        raise ProducerError("consumer upgrade input is unavailable") from exc
    if sha256(consumer_config_payload) != binding["consumer_upgrade_config_sha256"]:
        raise ProducerError("consumer upgrade config differs from its binding")
    if sha256(consumer_script_payload) != binding["consumer_upgrade_script_sha256"]:
        raise ProducerError("consumer upgrade script differs from its binding")
    consumer_config = strict_json(consumer_config_payload, label="consumer config")
    consumer_module = _load_module(consumer_script_path)
    private, audit = compute_split(
        config,
        public_root=public_root,
        group_publication=group_publication,
    )
    gate = build_gate_record(
        config,
        private["assignment_commitment_sha256"],
        consumer_config,
    )
    gate_payload = json_bytes(gate)
    validate_with_consumer(gate_payload, consumer_config, consumer_module)
    audit["consumer_actual_acceptance"] = True
    forbidden = {
        key.casefold() for key in config["output_contract"]["aggregate_forbidden_keys"]
    }
    if any(
        key.casefold() in forbidden
        for payload in (audit, gate)
        for key in _walk_keys(payload)
    ):
        raise ProducerError("aggregate output contains a forbidden key")
    output = config["output_contract"]
    return {
        output["private_assignment_basename"]: json_bytes(private),
        output["aggregate_audit_basename"]: json_bytes(audit),
        output["allowed_gate_basename"]: gate_payload,
    }


def publication_commit_payload(
    payloads: Mapping[str, bytes],
    names: Sequence[str],
) -> bytes:
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


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        written = handle.write(payload)
        if written != len(payload):
            raise ProducerError(f"short publication write: {path.name}")


def write_outputs(
    output_directory: Path,
    payloads: Mapping[str, bytes],
    config: Mapping[str, Any],
) -> str:
    output = config["output_contract"]
    data_names = list(output["data_member_names"])
    if set(payloads) != set(data_names) or len(payloads) != len(data_names):
        raise ProducerError("publication payload names differ from exact data members")
    marker_name = output["terminal_commit_marker"]
    expected = {name: payloads[name] for name in data_names}
    expected[marker_name] = publication_commit_payload(payloads, data_names)

    if output_directory.exists():
        try:
            existing_names = {path.name for path in output_directory.iterdir()}
        except OSError as exc:
            raise ProducerError("existing publication cannot be inspected") from exc
        if existing_names != set(output["exact_final_member_names"]):
            raise ProducerError(
                "existing publication is partial or mismatched; preserve it for manual intervention"
            )
        for name, payload in expected.items():
            try:
                observed = (output_directory / name).read_bytes()
            except OSError as exc:
                raise ProducerError("existing publication cannot be read") from exc
            if observed != payload:
                raise ProducerError(
                    "existing publication is partial or mismatched; preserve it for manual intervention"
                )
        return "EXISTING_EXACT"

    output_directory.mkdir()
    for name in data_names:
        _write_new(output_directory / name, payloads[name])
    _write_new(output_directory / marker_name, expected[marker_name])
    return "PUBLISHED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--group-publication", type=Path, required=True)
    parser.add_argument("--consumer-config", type=Path)
    parser.add_argument("--consumer-script", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = strict_json(args.config.read_bytes(), label="config")
    if args.dry_run:
        try:
            private, audit = compute_split(
                config,
                public_root=args.public_root,
                group_publication=args.group_publication,
            )
        except ProtocolStop as exc:
            print(json.dumps(exc.audit, sort_keys=True))
            return 2
        print(json.dumps(audit, sort_keys=True))
        if private["assignment_commitment_sha256"] != audit["assignment_commitment_sha256"]:
            raise ProducerError("private/audit commitment differs")
        return 0
    if args.consumer_config is None or args.consumer_script is None or args.output_directory is None:
        raise ProducerError("production requires consumer config, consumer script, and output directory")
    payloads = produce(
        config,
        public_root=args.public_root,
        group_publication=args.group_publication,
        consumer_config_path=args.consumer_config,
        consumer_script_path=args.consumer_script,
    )
    status = write_outputs(args.output_directory, payloads, config)
    print(json.dumps({"status": status, "output_directory": os.fspath(args.output_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
