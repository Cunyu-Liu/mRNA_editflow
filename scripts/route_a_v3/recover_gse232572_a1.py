#!/usr/bin/env python3
"""Audit a private development reconstruction of public GSE232572 MapUTR data.

All row-level material is held in memory until every global gate closes.  A
failed run writes one aggregate STOP report and no reconstruction or reject
rows.  A structurally closed run remains development-only, AUDIT_PENDING, and
not qualified; it cannot award ordinary/A1 credit or materialize canonical V3.
The tracked legacy GSE232572 helper remains the authority for FASTA header and
adapter interpretation; this producer adds the official-universe pairing,
matrix, published-result, grouping, and rights gates.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import statistics
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


DATASET_ID = "GSE232572"
STUDY_ID = "GSE232572"
REGION = "3UTR"
PROTOCOL_ID = "ROUTE_A_V3_GSE232572_PUBLIC_RECOVERY_AUDIT_V1"
INSPECTED_HEAD = "99b1fc1ffd65f1a1e45b4390d6d7ab32bdd0d06e"
SUCCESS_STATUS = "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED"
STOP_STATUS = "STOP_BEFORE_DEVELOPMENT_RECONSTRUCTION_ROW_PRODUCTION"
REPORT_FILENAME = "GSE232572_A1_RECOVERY_REPORT.json"
RECONSTRUCTION_FILENAME = "development_reconstruction_records.private.jsonl"
REJECTION_FILENAME = "rejection_aggregates.private.jsonl"
REQUIRED_RIGHTS_STATUS = "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED"
FORBIDDEN_PATH_TOKENS = ("gse246381", "restricted", "sealed", "access_log")
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class RecoveryError(RuntimeError):
    """A fail-closed, aggregate-reportable recovery error."""

    def __init__(self, gate: str, code: str):
        super().__init__(f"{gate}: {code}")
        self.gate = gate
        self.code = code


def _stop(gate: str, code: str) -> None:
    raise RecoveryError(gate, code)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _stop("CONFIG", "CONFIG_NOT_READABLE_JSON")
    if not isinstance(document, dict):
        _stop("CONFIG", "CONFIG_ROOT_NOT_OBJECT")
    return document


def _require_mapping(value: Any, gate: str, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _stop(gate, code)
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    expected_scalars = {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "study_id": STUDY_ID,
        "region": REGION,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            _stop("CONFIG", f"CONFIG_{key.upper()}_NOT_FROZEN")

    authority = _require_mapping(
        config.get("authority"), "CONFIG", "AUTHORITY_NOT_OBJECT"
    )
    if authority.get("inspected_head") != INSPECTED_HEAD:
        _stop("CONFIG", "INSPECTED_HEAD_NOT_FROZEN")
    if authority.get("generic_helper_path") != (
        "d1_staging/scripts/d1/reconstruct_gse232572_sequences.py"
    ):
        _stop("CONFIG", "GENERIC_HELPER_PATH_NOT_FROZEN")

    scope = _require_mapping(config.get("scope"), "CONFIG", "SCOPE_NOT_OBJECT")
    frozen_scope = {
        "ordinary_public_inputs_only": True,
        "registry_role": "AUDIT_ONLY",
        "registry_qualification_status": "AUDIT_PENDING",
        "qualified": False,
        "candidate_role": "PUBLIC_RECOVERY_AUDIT_ONLY",
        "independent_study_count": 1,
        "subpools_are_independent_studies": False,
        "replicates_are_independent_studies": False,
        "ordinary_study_contribution_if_all_gates_pass": 0,
        "a1_study_contribution_if_all_gates_pass": 0,
        "true_a2_study_contribution": 0,
        "development_record_release_mode": (
            "PRIVATE_RECONSTRUCTION_ONLY_NOT_CANONICAL"
        ),
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
    }
    for key, expected in frozen_scope.items():
        if scope.get(key) != expected:
            _stop("CONFIG", f"SCOPE_{key.upper()}_NOT_FROZEN")

    pairing = _require_mapping(
        config.get("pairing"), "CONFIG", "PAIRING_NOT_OBJECT"
    )
    frozen_pairing = {
        "universe": "UNIQUE_OFFICIAL_SHEET_5_ROWS",
        "header_allele_rule": (
            "USE_FASTA_HEADER_ALLELES_AS_WRITTEN_NEVER_COMPLEMENT_RC"
        ),
        "sequence_normalization": (
            "TRACKED_HELPER_ORIENTATION_NORMALIZED_165NT_INSERT"
        ),
        "hamming_neighbor_prefilter_fields": ["gene", "strand", "orientation"],
        "hamming_neighbor_rule": (
            "EXACTLY_ONE_REFERENCE_AT_DISTANCE_ONE_PER_ALTERNATE"
        ),
        "published_key_source": (
            "SELECTED_REFERENCE_CHR_GENE_STRAND_REF_HEADER_ALLELE_PLUS_ALT_HEADER_ALLELE"
        ),
        "distinct_sequence_pair_identity": [
            "normalized_reference_insert",
            "normalized_alternate_insert",
        ],
        "accepted_rule": (
            "EXACTLY_ONE_PHYSICAL_PAIR_AND_ONE_DISTINCT_SEQUENCE_PAIR_PER_SHEET_5_KEY"
        ),
    }
    for key, expected in frozen_pairing.items():
        if pairing.get(key) != expected:
            _stop("CONFIG", f"PAIRING_{key.upper()}_NOT_FROZEN")
    expected_pairing_counts = _require_mapping(
        pairing.get("expected_counts"),
        "CONFIG",
        "PAIRING_EXPECTED_COUNTS_NOT_OBJECT",
    )
    if set(expected_pairing_counts) != {
        "published_universe",
        "accepted",
        "NO_UNIQUE_SEQUENCE_PAIR",
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS",
    }:
        _stop("CONFIG", "PAIRING_EXPECTED_COUNT_KEYS_NOT_CLOSED")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in expected_pairing_counts.values()
    ):
        _stop("CONFIG", "PAIRING_EXPECTED_COUNTS_INVALID")
    if expected_pairing_counts["published_universe"] != (
        expected_pairing_counts["accepted"]
        + expected_pairing_counts["NO_UNIQUE_SEQUENCE_PAIR"]
        + expected_pairing_counts["AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS"]
    ):
        _stop("CONFIG", "PAIRING_EXPECTED_COUNTS_DO_NOT_PARTITION_UNIVERSE")

    matrix_contract = _require_mapping(
        config.get("matrix_contract"),
        "CONFIG",
        "MATRIX_CONTRACT_NOT_OBJECT",
    )
    frozen_matrix_contract = {
        "subpools": [1, 2, 3],
        "molecules": ["DNA", "RNA"],
        "biological_replicates": [1, 2, 3],
        "required_matrix_count_per_subpool": 6,
        "required_total_matrix_count": 18,
        "matrix_header": ["gene", "count"],
        "matrix_identifier": "EXACT_FASTA_HEADER",
        "full_fasta_matrix_identifier_set_equality_required": False,
        "accepted_pair_endpoint_contract": (
            "REF_AND_ALT_PRESENT_IN_DNA_RNA_X_THREE_REPLICATES_OF_SELECTED_REFERENCE_SUBPOOL"
        ),
        "required_endpoint_count_per_accepted_pair": 12,
        "missing_endpoint_policy": "STRUCTURAL_STOP_NEVER_IMPUTE_ZERO",
        "finite_nonnegative_counts_required": True,
    }
    for key, expected in frozen_matrix_contract.items():
        if matrix_contract.get(key) != expected:
            _stop("CONFIG", f"MATRIX_{key.upper()}_NOT_FROZEN")
    if matrix_contract.get("expected_complete_accepted_pair_count") != (
        expected_pairing_counts["accepted"]
    ):
        _stop("CONFIG", "MATRIX_EXPECTED_COMPLETE_PAIR_COUNT_NOT_FROZEN")

    endpoint = _require_mapping(
        config.get("endpoint"), "CONFIG", "ENDPOINT_NOT_OBJECT"
    )
    frozen_endpoint = {
        "activity_definition": "A=RNA/DNA",
        "label_name": "ln_activity_ratio_alt_over_ref",
        "primary_label_source": (
            "MOESM4_OFFICIAL_PUBLISHED_RELATIVE_ACTIVITY_LNFC"
        ),
        "raw_auxiliary_replicate_formula": (
            "ln[(RNA_alt/DNA_alt)/(RNA_ref/DNA_ref)]"
        ),
        "direction": (
            "POSITIVE_MEANS_ALT_HAS_HIGHER_RNA_PER_DNA_THAN_REF"
        ),
        "pseudocount": None,
        "zero_count_policy": (
            "MARK_AUXILIARY_ZERO_COUNT_ENDPOINT_UNDEFINED_NO_PSEUDOCOUNT_AND_KEEP_PUBLISHED_LABEL"
        ),
        "raw_auxiliary_summary": (
            "ARITHMETIC_MEAN_OF_THREE_LOG_RATIOS_ONLY_WHEN_ALL_THREE_ARE_DEFINED"
        ),
        "raw_auxiliary_may_replace_primary_label": False,
        "unpublished_standard_error_may_be_claimed": False,
        "mpranalyze_role": (
            "INFERENTIAL_SIGNIFICANCE_CROSSCHECK_ONLY_NOT_LABEL"
        ),
    }
    for key, expected in frozen_endpoint.items():
        if endpoint.get(key) != expected:
            _stop("CONFIG", f"ENDPOINT_{key.upper()}_NOT_FROZEN")

    grouping = _require_mapping(
        config.get("grouping"), "CONFIG", "GROUPING_NOT_OBJECT"
    )
    if grouping.get("source_group_definition") != (
        "gene + source_context_cluster"
    ):
        _stop("CONFIG", "SOURCE_GROUP_DEFINITION_NOT_FROZEN")
    if grouping.get("split_or_bootstrap_execution") != "NOT_ALLOWED_AUDIT_ONLY":
        _stop("CONFIG", "SPLIT_BOOTSTRAP_BOUNDARY_NOT_FROZEN")
    if grouping.get("subpool_not_an_independent_group") is not True:
        _stop("CONFIG", "SUBPOOL_GROUP_BOUNDARY_NOT_FROZEN")
    if grouping.get("biological_replicate_not_an_independent_group") is not True:
        _stop("CONFIG", "REPLICATE_GROUP_BOUNDARY_NOT_FROZEN")

    results = _require_mapping(
        config.get("published_result_contract"),
        "CONFIG",
        "PUBLISHED_RESULT_CONTRACT_NOT_OBJECT",
    )
    if results.get("fasta_and_sheet_key_set_equality_required") is not False:
        _stop("CONFIG", "FASTA_SHEET_EQUALITY_BOUNDARY_NOT_FROZEN")
    if results.get("published_ln_activity_is_primary_label") is not True:
        _stop("CONFIG", "PUBLISHED_LNFC_NOT_PRIMARY_LABEL")
    if results.get("raw_auxiliary_direction_role") != (
        "DIAGNOSTIC_ONLY_NOT_AN_ELIGIBILITY_GATE"
    ):
        _stop("CONFIG", "RAW_DIRECTION_ROLE_NOT_FROZEN")
    columns = _require_mapping(
        results.get("columns"), "CONFIG", "PUBLISHED_COLUMNS_NOT_OBJECT"
    )
    required_column_roles = {
        "chromosome_position",
        "gene",
        "gene_strand",
        "reference_allele",
        "alternate_allele",
        "cell_line",
        "published_ln_activity",
        "mpranalyze_fdr",
    }
    if set(columns) != required_column_roles:
        _stop("CONFIG", "PUBLISHED_COLUMN_ROLES_NOT_CLOSED")
    frozen_published_result_values = {
        "sheet_name": "Sheet 5",
        "sheet_semantics": (
            "ALL_COSMIC_SOMATIC_MUTATIONS_TESTED_WITH_MAPUTR_IN_HELA"
        ),
        "header_row_1_based": 4,
        "columns": {
            "chromosome_position": {
                "columns": ["chromosome", "position (hg19, 1-based)"],
                "separator": ":",
            },
            "gene": "gene",
            "gene_strand": "gene_strand",
            "reference_allele": "ref",
            "alternate_allele": "alt",
            "cell_line": None,
            "published_ln_activity": "lnFC",
            "mpranalyze_fdr": "FDR",
        },
        "required_cell_line": "HeLa",
        "cell_line_authority": "EXACT_SHEET_5_SEMANTICS_NOT_A_ROW_COLUMN",
        "join_key": [
            "chromosome_position",
            "gene",
            "gene_strand",
            "reference_allele",
            "alternate_allele",
        ],
        "fasta_and_sheet_key_set_equality_required": False,
        "duplicate_join_keys_allowed": False,
        "published_ln_activity_is_primary_label": True,
        "raw_auxiliary_direction_role": (
            "DIAGNOSTIC_ONLY_NOT_AN_ELIGIBILITY_GATE"
        ),
        "mpranalyze_fdr_range": [0.0, 1.0],
    }
    for key, expected in frozen_published_result_values.items():
        if results.get(key) != expected:
            _stop("CONFIG", f"PUBLISHED_RESULT_{key.upper()}_NOT_FROZEN")

    rights = _require_mapping(
        config.get("rights"), "CONFIG", "RIGHTS_NOT_OBJECT"
    )
    if rights.get("required_status_for_private_development_reconstruction") != (
        REQUIRED_RIGHTS_STATUS
    ):
        _stop("CONFIG", "RIGHTS_MATERIALIZATION_THRESHOLD_NOT_FROZEN")
    if rights.get("unknown_or_unusable_policy") != "STOP_WHOLE_RUN":
        _stop("CONFIG", "RIGHTS_UNKNOWN_POLICY_NOT_FROZEN")

    outputs = _require_mapping(
        config.get("outputs"), "CONFIG", "OUTPUTS_NOT_OBJECT"
    )
    expected_outputs = {
        "aggregate_report_filename": REPORT_FILENAME,
        "development_reconstruction_private_filename": RECONSTRUCTION_FILENAME,
        "rejection_aggregates_private_filename": REJECTION_FILENAME,
        "stop_status": STOP_STATUS,
        "success_status": SUCCESS_STATUS,
    }
    for key, expected in expected_outputs.items():
        if outputs.get(key) != expected:
            _stop("CONFIG", f"OUTPUT_{key.upper()}_NOT_FROZEN")


def _reject_forbidden_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    hits = [token for token in FORBIDDEN_PATH_TOKENS if token in lowered]
    if hits:
        _stop("SCOPE", f"FORBIDDEN_{label.upper()}_PATH")


def _require_input(
    path: Path,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    _reject_forbidden_path(path, label)
    expected_basename = contract.get("filename")
    if path.name != expected_basename:
        _stop("INPUTS", f"{label.upper()}_BASENAME_MISMATCH")
    if not path.is_file():
        _stop("INPUTS", f"{label.upper()}_MISSING")
    expected_bytes = contract.get("bytes")
    expected_sha256 = contract.get("sha256")
    if (
        not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        _stop("CONFIG", f"{label.upper()}_IDENTITY_NOT_FROZEN")
    try:
        payload = path.read_bytes()
    except OSError:
        _stop("INPUTS", f"{label.upper()}_NOT_READABLE")
    if len(payload) != expected_bytes:
        _stop("INPUTS", f"{label.upper()}_BYTE_COUNT_MISMATCH")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        _stop("INPUTS", f"{label.upper()}_SHA256_MISMATCH")


def _load_generic_helper(repo_root: Path, relative_path: str) -> Any:
    _reject_forbidden_path(repo_root, "repository_root")
    helper_path = repo_root / PurePosixPath(relative_path)
    if not helper_path.is_file():
        _stop("HELPER", "GENERIC_HELPER_MISSING")
    spec = importlib.util.spec_from_file_location(
        "route_a_v3_gse232572_generic_helper", helper_path
    )
    if spec is None or spec.loader is None:
        _stop("HELPER", "GENERIC_HELPER_NOT_LOADABLE")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _stop("HELPER", "GENERIC_HELPER_IMPORT_FAILED")
    for name in ("parse_fasta_header", "extract_insert"):
        if not callable(getattr(module, name, None)):
            _stop("HELPER", "GENERIC_HELPER_API_MISSING")
    return module


def _read_fasta_records(path: Path, subpool: int, helper: Any) -> list[dict[str, Any]]:
    try:
        handle = gzip.open(path, "rt", encoding="utf-8")
        lines = handle.readlines()
        handle.close()
    except (OSError, UnicodeError):
        _stop("FASTA", "FASTA_NOT_READABLE_GZIP_TEXT")

    raw_records: list[tuple[str, str]] = []
    current_header: str | None = None
    sequence_parts: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                raw_records.append((current_header, "".join(sequence_parts)))
            current_header = line[1:]
            sequence_parts = []
        elif current_header is None:
            _stop("FASTA", "FASTA_SEQUENCE_BEFORE_HEADER")
        else:
            sequence_parts.append(line.upper())
    if current_header is not None:
        raw_records.append((current_header, "".join(sequence_parts)))
    if not raw_records:
        _stop("FASTA", "FASTA_HAS_NO_RECORDS")

    records: list[dict[str, Any]] = []
    seen_headers: set[str] = set()
    for header, sequence in raw_records:
        if header in seen_headers:
            _stop("FASTA", "DUPLICATE_FASTA_HEADER")
        seen_headers.add(header)
        parts = header.split("|")
        if len(parts) < 8:
            _stop("FASTA", "INVALID_FASTA_HEADER")
        parsed = helper.parse_fasta_header(header)
        if not isinstance(parsed, Mapping):
            _stop("FASTA", "GENERIC_HELPER_REJECTED_HEADER")
        expected_subpool = f"subpool{subpool}"
        if parsed.get("subpool") != expected_subpool:
            _stop("FASTA", "FASTA_SUBPOOL_DISAGREES_WITH_FILE")
        allele_type = parsed.get("allele_type")
        if allele_type not in {"reference", "alternate"}:
            _stop("FASTA", "FASTA_ALLELE_TYPE_INVALID")
        if not sequence or set(sequence) - set("ACGT"):
            _stop("FASTA", "FASTA_SEQUENCE_NOT_ACGT")
        insert = helper.extract_insert(sequence, parsed.get("orientation"))
        if not isinstance(insert, str) or len(insert) != 165:
            _stop("FASTA", "GENERIC_HELPER_INSERT_NOT_165NT")
        insert = insert.upper()
        if set(insert) - set("ACGT"):
            _stop("FASTA", "GENERIC_HELPER_INSERT_NOT_ACGT")
        required_text = ("source", "chr_pos", "gene", "strand", "orientation", "allele")
        if any(not isinstance(parsed.get(key), str) or not parsed.get(key) for key in required_text):
            _stop("FASTA", "PARSED_FASTA_METADATA_INCOMPLETE")
        records.append(
            {
                "header": header,
                "subpool": expected_subpool,
                "subpool_number": subpool,
                "source": str(parsed["source"]),
                "chr_pos": str(parsed["chr_pos"]),
                "gene": str(parsed["gene"]),
                "strand": str(parsed["strand"]),
                "orientation": str(parsed["orientation"]),
                "allele_type": str(allele_type),
                "allele": str(parsed["allele"]).upper(),
                "insert": insert,
            }
        )
    return records


def _hamming_distance(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def _map_published_universe(
    records: Sequence[Mapping[str, Any]],
    published: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    references = [record for record in records if record["allele_type"] == "reference"]
    alternates = [record for record in records if record["allele_type"] == "alternate"]
    if not references or not alternates:
        _stop("PAIRING", "FASTA_REFERENCE_OR_ALTERNATE_UNIVERSE_EMPTY")

    split_at = 82
    first_half: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)
    second_half: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)
    for index, reference in enumerate(references):
        sequence = str(reference["insert"])
        metadata = (
            str(reference["gene"]),
            str(reference["strand"]),
            str(reference["orientation"]),
        )
        first_half[metadata + (sequence[:split_at],)].append(index)
        second_half[metadata + (sequence[split_at:],)].append(index)

    physical_pairs_by_key: dict[
        tuple[str, ...], list[dict[str, Any]]
    ] = collections.defaultdict(list)
    for alternate in alternates:
        sequence = str(alternate["insert"])
        metadata = (
            str(alternate["gene"]),
            str(alternate["strand"]),
            str(alternate["orientation"]),
        )
        possible = set(first_half.get(metadata + (sequence[:split_at],), []))
        possible.update(second_half.get(metadata + (sequence[split_at:],), []))
        neighbors = [
            references[index]
            for index in possible
            if _hamming_distance(str(references[index]["insert"]), sequence) == 1
        ]
        if len(neighbors) != 1:
            continue
        reference = neighbors[0]
        key = _join_key(
            reference["chr_pos"],
            reference["gene"],
            reference["strand"],
            reference["allele"],
            alternate["allele"],
        )
        if key not in published:
            continue
        differences = [
            index
            for index, (ref_base, alt_base) in enumerate(
                zip(str(reference["insert"]), sequence)
            )
            if ref_base != alt_base
        ]
        physical_pairs_by_key[key].append(
            {
                "ref": reference,
                "alt": alternate,
                "edit_position": differences[0],
                "edit_ref": str(reference["insert"])[differences[0]],
                "edit_alt": sequence[differences[0]],
            }
        )

    accepted: list[dict[str, Any]] = []
    rejection_counts = {
        "NO_UNIQUE_SEQUENCE_PAIR": 0,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
    }
    for key in sorted(published):
        physical_pairs = physical_pairs_by_key.get(key, [])
        if not physical_pairs:
            rejection_counts["NO_UNIQUE_SEQUENCE_PAIR"] += 1
            continue
        distinct_sequence_pairs = {
            (str(pair["ref"]["insert"]), str(pair["alt"]["insert"]))
            for pair in physical_pairs
        }
        if len(distinct_sequence_pairs) > 1:
            rejection_counts["AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS"] += 1
            continue
        if len(physical_pairs) != 1:
            _stop("PAIRING", "MULTIPLE_PHYSICAL_PAIRS_FOR_ONE_SEQUENCE_PAIR")
        pair = physical_pairs[0]
        pair["published_key"] = key
        accepted.append(pair)
    return accepted, rejection_counts


def _parse_matrix_payload(payload: bytes, expected_header: Sequence[str]) -> dict[str, float]:
    try:
        lines = gzip.decompress(payload).decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        _stop("MATRICES", "MATRIX_NOT_READABLE_GZIP_TEXT")
    if not lines or lines[0].split() != list(expected_header):
        _stop("MATRICES", "MATRIX_HEADER_MISMATCH")
    counts: dict[str, float] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            _stop("MATRICES", "MATRIX_ROW_SHAPE_INVALID")
        identifier, rendered = fields
        if identifier in counts:
            _stop("MATRICES", "MATRIX_IDENTIFIER_DUPLICATED")
        try:
            count = float(rendered)
        except ValueError:
            _stop("MATRICES", "MATRIX_COUNT_NOT_NUMERIC")
        if not math.isfinite(count) or count < 0:
            _stop("MATRICES", "MATRIX_COUNT_NOT_FINITE_NONNEGATIVE")
        counts[identifier] = count
    return counts


def _read_matrices(
    raw_tar: Path,
    matrix_contract: Mapping[str, Any],
) -> dict[tuple[int, str, int], dict[str, float]]:
    try:
        archive = tarfile.open(raw_tar, mode="r:")
    except (OSError, tarfile.TarError):
        _stop("MATRICES", "RAW_TAR_NOT_READABLE")
    try:
        member_pattern = re.compile(str(matrix_contract["member_name_regex"]))
    except (KeyError, re.error):
        _stop("CONFIG", "MATRIX_MEMBER_REGEX_INVALID")

    matrices: dict[tuple[int, str, int], dict[str, float]] = {}
    with archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            match = member_pattern.fullmatch(member.name)
            if match is None:
                continue
            subpool = int(match.group(1))
            molecule = "DNA" if match.group(2) == "D" else "RNA"
            replicate = int(match.group(3))
            key = (subpool, molecule, replicate)
            if key in matrices:
                _stop("MATRICES", "DUPLICATE_REQUIRED_MATRIX")
            extracted = archive.extractfile(member)
            if extracted is None:
                _stop("MATRICES", "REQUIRED_MATRIX_NOT_EXTRACTABLE")
            matrices[key] = _parse_matrix_payload(
                extracted.read(), list(matrix_contract["matrix_header"])
            )

    expected_keys = {
        (subpool, molecule, replicate)
        for subpool in (1, 2, 3)
        for molecule in ("DNA", "RNA")
        for replicate in (1, 2, 3)
    }
    if set(matrices) != expected_keys:
        _stop("MATRICES", "REQUIRED_EXACT_18_MATRIX_SET_NOT_CLOSED")
    return matrices


def _accepted_pairs_with_complete_raw_endpoints(
    pairs: Sequence[Mapping[str, Any]],
    matrices: Mapping[tuple[int, str, int], Mapping[str, float]],
) -> tuple[list[Mapping[str, Any]], int]:
    complete: list[Mapping[str, Any]] = []
    incomplete_count = 0
    for pair in pairs:
        subpool = int(pair["ref"]["subpool_number"])
        ref_header = str(pair["ref"]["header"])
        alt_header = str(pair["alt"]["header"])
        endpoint_present = all(
            ref_header in matrices[(subpool, molecule, replicate)]
            and alt_header in matrices[(subpool, molecule, replicate)]
            for molecule in ("DNA", "RNA")
            for replicate in (1, 2, 3)
        )
        if endpoint_present:
            complete.append(pair)
        else:
            incomplete_count += 1
    return complete, incomplete_count


def _xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{{{NS_MAIN}}}t")
        )
    value = cell.find(f"{{{NS_MAIN}}}v")
    if value is None or value.text is None:
        return ""
    rendered = value.text
    if cell_type == "s":
        try:
            return shared_strings[int(rendered)]
        except (IndexError, ValueError):
            _stop("PUBLISHED_RESULTS", "XLSX_SHARED_STRING_REFERENCE_INVALID")
    if cell_type in {"str", "e"}:
        return rendered
    if cell_type == "b":
        return rendered == "1"
    try:
        numeric = float(rendered)
    except ValueError:
        return rendered
    return int(numeric) if numeric.is_integer() else numeric


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    if not letters:
        _stop("PUBLISHED_RESULTS", "XLSX_CELL_REFERENCE_INVALID")
    result = 0
    for character in letters.upper():
        result = result * 26 + (ord(character) - ord("A") + 1)
    return result - 1


def _read_xlsx_sheet(path: Path, sheet_name: str) -> dict[int, dict[int, Any]]:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        _stop("PUBLISHED_RESULTS", "PUBLISHED_RESULTS_NOT_READABLE_XLSX")
    with archive:
        try:
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in shared_root.findall(f"{{{NS_MAIN}}}si"):
                    shared_strings.append(
                        "".join(
                            node.text or ""
                            for node in item.iter(f"{{{NS_MAIN}}}t")
                        )
                    )
            rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {
                rel.attrib["Id"]: rel.attrib["Target"]
                for rel in rels_root.findall(f"{{{NS_PKG_REL}}}Relationship")
            }
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        except (KeyError, ET.ParseError):
            _stop("PUBLISHED_RESULTS", "XLSX_WORKBOOK_STRUCTURE_INVALID")

        target: str | None = None
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                relation_id = sheet.attrib.get(f"{{{NS_REL}}}id")
                target = targets.get(str(relation_id))
                break
        if target is None:
            _stop("PUBLISHED_RESULTS", "REQUIRED_XLSX_SHEET_MISSING")
        if target.startswith("/"):
            worksheet_path = target.lstrip("/")
        else:
            worksheet_path = str(PurePosixPath("xl") / PurePosixPath(target))
        try:
            worksheet = ET.fromstring(archive.read(worksheet_path))
        except (KeyError, ET.ParseError):
            _stop("PUBLISHED_RESULTS", "REQUIRED_XLSX_SHEET_INVALID")

        rows: dict[int, dict[int, Any]] = {}
        for row in worksheet.findall(
            f".//{{{NS_MAIN}}}sheetData/{{{NS_MAIN}}}row"
        ):
            row_reference = row.attrib.get("r")
            try:
                row_number = int(str(row_reference))
            except (TypeError, ValueError):
                _stop("PUBLISHED_RESULTS", "XLSX_ROW_REFERENCE_INVALID")
            if row_number < 1 or row_number in rows:
                _stop("PUBLISHED_RESULTS", "XLSX_ROW_REFERENCE_INVALID")
            rendered: dict[int, Any] = {}
            for cell in row.findall(f"{{{NS_MAIN}}}c"):
                reference = cell.attrib.get("r", "")
                rendered[_column_index(reference)] = _xlsx_cell_value(
                    cell, shared_strings
                )
            rows[row_number] = rendered
        return rows


def _column_value(
    row: Mapping[int, Any],
    headers: Mapping[str, int],
    specification: Any,
) -> Any:
    if isinstance(specification, str):
        if specification not in headers:
            _stop("PUBLISHED_RESULTS", "REQUIRED_PUBLISHED_COLUMN_MISSING")
        return row.get(headers[specification], "")
    if isinstance(specification, Mapping):
        columns = specification.get("columns")
        separator = specification.get("separator")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(item, str) and item for item in columns)
            or not isinstance(separator, str)
        ):
            _stop("CONFIG", "COMPOSITE_PUBLISHED_COLUMN_SPEC_INVALID")
        values = []
        for column in columns:
            if column not in headers:
                _stop("PUBLISHED_RESULTS", "REQUIRED_PUBLISHED_COLUMN_MISSING")
            value = row.get(headers[column], "")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            values.append(str(value).strip())
        return separator.join(values)
    if specification is None:
        return None
    _stop("CONFIG", "PUBLISHED_COLUMN_SPEC_INVALID")


def _normalize_chr_pos(value: Any) -> str:
    rendered = str(value).strip().replace(" ", "")
    if ":" not in rendered:
        _stop("PUBLISHED_RESULTS", "PUBLISHED_CHROMOSOME_POSITION_INVALID")
    chromosome, position = rendered.split(":", 1)
    if not chromosome.lower().startswith("chr"):
        chromosome = "chr" + chromosome
    if position.endswith(".0"):
        position = position[:-2]
    if not position.isdigit():
        _stop("PUBLISHED_RESULTS", "PUBLISHED_CHROMOSOME_POSITION_INVALID")
    return chromosome.lower() + ":" + position


def _join_key(
    chr_pos: Any,
    gene: Any,
    gene_strand: Any,
    reference: Any,
    alternate: Any,
) -> tuple[str, ...]:
    rendered_gene = str(gene).strip().upper()
    rendered_strand = str(gene_strand).strip()
    rendered_ref = str(reference).strip().upper()
    rendered_alt = str(alternate).strip().upper()
    if (
        not rendered_gene
        or rendered_strand not in {"+", "-"}
        or rendered_ref not in set("ACGT")
        or rendered_alt not in set("ACGT")
    ):
        _stop("PUBLISHED_RESULTS", "PUBLISHED_JOIN_KEY_INVALID")
    if len(rendered_ref) != 1 or len(rendered_alt) != 1:
        _stop("PUBLISHED_RESULTS", "PUBLISHED_JOIN_ALLELE_NOT_SNV")
    return (
        _normalize_chr_pos(chr_pos),
        rendered_gene,
        rendered_strand,
        rendered_ref,
        rendered_alt,
    )


def _as_finite_float(value: Any, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        _stop("PUBLISHED_RESULTS", code)
    if not math.isfinite(result):
        _stop("PUBLISHED_RESULTS", code)
    return result


def _read_published_results(
    path: Path, contract: Mapping[str, Any]
) -> dict[tuple[str, ...], dict[str, float]]:
    sheet_name = contract.get("sheet_name")
    header_row_number = contract.get("header_row_1_based")
    if not isinstance(sheet_name, str) or not sheet_name:
        _stop("CONFIG", "PUBLISHED_SHEET_NAME_INVALID")
    if not isinstance(header_row_number, int) or header_row_number < 1:
        _stop("CONFIG", "PUBLISHED_HEADER_ROW_INVALID")
    rows = _read_xlsx_sheet(path, sheet_name)
    if header_row_number not in rows:
        _stop("PUBLISHED_RESULTS", "PUBLISHED_HEADER_ROW_MISSING")
    raw_header = rows[header_row_number]
    headers: dict[str, int] = {}
    for index, value in raw_header.items():
        rendered = str(value).strip()
        if not rendered:
            continue
        if rendered in headers:
            _stop("PUBLISHED_RESULTS", "PUBLISHED_HEADER_DUPLICATED")
        headers[rendered] = index
    columns = _require_mapping(
        contract.get("columns"), "CONFIG", "PUBLISHED_COLUMNS_NOT_OBJECT"
    )
    required_cell_line = contract.get("required_cell_line")
    if required_cell_line is not None and not isinstance(required_cell_line, str):
        _stop("CONFIG", "REQUIRED_CELL_LINE_INVALID")

    result: dict[tuple[str, ...], dict[str, float]] = {}
    for row_number in sorted(rows):
        if row_number <= header_row_number:
            continue
        row = rows[row_number]
        if not row or not any(str(value).strip() for value in row.values()):
            continue
        if columns["cell_line"] is not None:
            cell_line_value = _column_value(row, headers, columns["cell_line"])
            if (
                required_cell_line is not None
                and str(cell_line_value).strip() != required_cell_line
            ):
                continue
        elif required_cell_line != "HeLa":
            _stop("CONFIG", "SHEET_FIXED_CELL_LINE_NOT_FROZEN")
        key = _join_key(
            _column_value(row, headers, columns["chromosome_position"]),
            _column_value(row, headers, columns["gene"]),
            _column_value(row, headers, columns["gene_strand"]),
            _column_value(row, headers, columns["reference_allele"]),
            _column_value(row, headers, columns["alternate_allele"]),
        )
        if key in result:
            _stop("PUBLISHED_RESULTS", "PUBLISHED_JOIN_KEY_DUPLICATED")
        published_lnfc = _as_finite_float(
            _column_value(row, headers, columns["published_ln_activity"]),
            "PUBLISHED_LNFC_NOT_FINITE",
        )
        fdr = _as_finite_float(
            _column_value(row, headers, columns["mpranalyze_fdr"]),
            "MPRANALYZE_FDR_NOT_FINITE",
        )
        fdr_range = contract.get("mpranalyze_fdr_range")
        if not (
            isinstance(fdr_range, list)
            and len(fdr_range) == 2
            and float(fdr_range[0]) <= fdr <= float(fdr_range[1])
        ):
            _stop("PUBLISHED_RESULTS", "MPRANALYZE_FDR_OUT_OF_RANGE")
        result[key] = {"published_lnfc": published_lnfc, "mpranalyze_fdr": fdr}
    if not result:
        _stop("PUBLISHED_RESULTS", "NO_RELEVANT_PUBLISHED_RESULT_ROWS")
    return result


def _direction(value: float) -> int:
    if abs(value) <= 1e-12:
        return 0
    return 1 if value > 0 else -1


def _build_development_reconstruction_records(
    pairs: Sequence[Mapping[str, Any]],
    matrices: Mapping[tuple[int, str, int], Mapping[str, float]],
    published: Mapping[tuple[str, ...], Mapping[str, float]],
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    auxiliary_defined = 0
    auxiliary_zero_undefined = 0
    seen_record_ids: set[str] = set()
    for pair in pairs:
        key = pair["published_key"]
        ref = pair["ref"]
        alt = pair["alt"]
        subpool = int(ref["subpool_number"])
        replicate_values: list[float] = []
        has_zero = False
        for replicate in (1, 2, 3):
            dna_ref = matrices[(subpool, "DNA", replicate)][str(ref["header"])]
            dna_alt = matrices[(subpool, "DNA", replicate)][str(alt["header"])]
            rna_ref = matrices[(subpool, "RNA", replicate)][str(ref["header"])]
            rna_alt = matrices[(subpool, "RNA", replicate)][str(alt["header"])]
            required = (dna_ref, dna_alt, rna_ref, rna_alt)
            if any(value == 0 for value in required):
                has_zero = True
                break
            replicate_values.append(
                math.log((rna_alt / dna_alt) / (rna_ref / dna_ref))
            )

        published_row = published[key]
        published_lnfc = float(published_row["published_lnfc"])
        if has_zero:
            auxiliary_status = "ZERO_COUNT_ENDPOINT_UNDEFINED_NO_PSEUDOCOUNT"
            auxiliary_mean: float | None = None
            auxiliary_zero_undefined += 1
        else:
            if len(replicate_values) != 3:
                _stop("ENDPOINT", "RAW_AUXILIARY_REPLICATE_SET_INCOMPLETE")
            auxiliary_mean = statistics.fmean(replicate_values)
            auxiliary_status = (
                "DEFINED_DIRECTION_AGREES_WITH_PUBLISHED_LABEL"
                if _direction(auxiliary_mean) == _direction(published_lnfc)
                else "DEFINED_DIRECTION_DISAGREES_WITH_PUBLISHED_LABEL_DIAGNOSTIC_ONLY"
            )
            auxiliary_defined += 1

        source_context_cluster = "|".join(
            [
                str(ref["source"]),
                str(ref["chr_pos"]),
                str(ref["strand"]),
                str(ref["orientation"]),
                str(ref["insert"]),
            ]
        )
        source_group_id = f"{ref['gene']}|{source_context_cluster}"
        record_id = "|".join(
            [
                DATASET_ID,
                str(ref["subpool"]),
                str(ref["source"]),
                str(ref["chr_pos"]),
                str(ref["gene"]),
                str(alt["allele"]),
            ]
        )
        if record_id in seen_record_ids:
            _stop("GROUPING", "DEVELOPMENT_RECORD_ID_DUPLICATED")
        seen_record_ids.add(record_id)
        records.append(
            {
                "record_id": record_id,
                "dataset_id": DATASET_ID,
                "study_id": STUDY_ID,
                "region": REGION,
                "data_role": "AUDIT_ONLY",
                "qualification_status": "AUDIT_PENDING",
                "qualified": False,
                "claim_boundary": (
                    "DEVELOPMENT_PRIVATE_RECONSTRUCTION_NOT_CANONICAL"
                ),
                "source_sequence": ref["insert"],
                "candidate_sequence": alt["insert"],
                "edit": {
                    "position_zero_based": pair["edit_position"],
                    "reference": pair["edit_ref"],
                    "alternate": pair["edit_alt"],
                },
                "label": {
                    "name": "ln_activity_ratio_alt_over_ref",
                    "value": published_lnfc,
                    "source": "MOESM4_OFFICIAL_PUBLISHED_RELATIVE_ACTIVITY_LNFC",
                    "direction": (
                        "POSITIVE_MEANS_ALT_HAS_HIGHER_RNA_PER_DNA_THAN_REF"
                    ),
                    "pseudocount": None,
                },
                "raw_count_auxiliary": {
                    "role": "DIAGNOSTIC_ONLY_NOT_LABEL_OR_ELIGIBILITY_GATE",
                    "status": auxiliary_status,
                    "replicate_log_ratios": (
                        replicate_values if auxiliary_mean is not None else None
                    ),
                    "mean_log_ratio": auxiliary_mean,
                    "pseudocount": None,
                },
                "mpranalyze_crosscheck": {
                    "role": "INFERENTIAL_SIGNIFICANCE_ONLY_NOT_LABEL",
                    "fdr": published_row["mpranalyze_fdr"],
                },
                "grouping": {
                    "gene": ref["gene"],
                    "source_context_cluster": source_context_cluster,
                    "source_group_id": source_group_id,
                    "split_or_bootstrap_assignment": "NOT_CREATED_AUDIT_ONLY",
                },
                "replicate_structure": {
                    "subpool": ref["subpool"],
                    "biological_replicate_count": 3,
                    "independent_study_count": 1,
                },
                "provenance": {
                    "published_universe": "UNIQUE_OFFICIAL_SHEET_5_ROW",
                    "hamming_neighbor_prefilter_fields": [
                        "gene",
                        "strand",
                        "orientation",
                    ],
                    "hamming_neighbor_rule": (
                        "EXACTLY_ONE_REFERENCE_AT_DISTANCE_ONE_PER_ALTERNATE"
                    ),
                    "published_key_source": (
                        "SELECTED_REFERENCE_CHR_GENE_STRAND_REF_HEADER_ALLELE_PLUS_ALT_HEADER_ALLELE"
                    ),
                    "header_alleles_complemented_for_rc": False,
                    "distinct_sequence_pair_count": 1,
                    "physical_pair_count": 1,
                    "published_result_authority": (
                        "41467_2024_46795_MOESM4_ESM.xlsx"
                    ),
                },
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_allowed": False,
                "true_a2": False,
            }
        )
    return records, auxiliary_defined, auxiliary_zero_undefined


def _gate(name: str, status: str, code: str) -> dict[str, str]:
    return {"gate": name, "status": status, "code": code}


def _base_report(recorded_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "study_id": STUDY_ID,
        "recorded_at": recorded_at,
        "status": STOP_STATUS,
        "scientific_disposition": "NOT_QUALIFIED",
        "registry_role": "AUDIT_ONLY",
        "qualification_status": "AUDIT_PENDING",
        "qualified": False,
        "scope": {
            "region": REGION,
            "ordinary_public_inputs_only": True,
            "independent_study_count": 1,
            "subpools_are_independent_studies": False,
            "replicates_are_independent_studies": False,
            "development_record_release_mode": (
                "PRIVATE_RECONSTRUCTION_ONLY_NOT_CANONICAL"
            ),
            "canonical_materialization_allowed": False,
        },
        "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
        "published_universe_row_count": 0,
        "accepted_pair_count": 0,
        "rejected_published_row_count": 0,
        "rejection_reason_counts": {
            "NO_UNIQUE_SEQUENCE_PAIR": 0,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
        },
        "accepted_pair_complete_raw_endpoint_count": 0,
        "accepted_pair_incomplete_raw_endpoint_count": 0,
        "development_reconstruction_record_count": 0,
        "rejection_aggregate_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
        "gates": [],
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def recover(
    *,
    repo_root: Path,
    config_path: Path,
    fasta_paths: Mapping[int, Path],
    raw_tar: Path,
    published_results: Path,
    output_dir: Path,
    recorded_at: str,
) -> tuple[int, dict[str, Any]]:
    _reject_forbidden_path(output_dir, "output")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RuntimeError("output directory already exists")
    report = _base_report(recorded_at)
    gates: list[dict[str, str]] = report["gates"]

    try:
        config = _read_json(config_path)
        _validate_config(config)
        gates.append(_gate("CONFIG", "PASS", "SCIENTIFIC_CONTRACT_FROZEN"))

        inputs = _require_mapping(
            config.get("inputs"), "CONFIG", "INPUTS_NOT_OBJECT"
        )
        fasta_contract = _require_mapping(
            inputs.get("fasta_by_subpool"), "CONFIG", "FASTA_INPUTS_NOT_OBJECT"
        )
        for subpool in (1, 2, 3):
            entry = _require_mapping(
                fasta_contract.get(str(subpool)),
                "CONFIG",
                "FASTA_INPUT_ENTRY_NOT_OBJECT",
            )
            _require_input(
                fasta_paths[subpool],
                entry,
                f"fasta{subpool}",
            )
        raw_contract = _require_mapping(
            inputs.get("raw_tar"), "CONFIG", "RAW_TAR_INPUT_NOT_OBJECT"
        )
        results_input = _require_mapping(
            inputs.get("published_results"),
            "CONFIG",
            "PUBLISHED_RESULTS_INPUT_NOT_OBJECT",
        )
        _require_input(
            raw_tar,
            raw_contract,
            "raw_tar",
        )
        _require_input(
            published_results,
            results_input,
            "published_results",
        )
        gates.append(_gate("INPUTS", "PASS", "EXACT_FIVE_PUBLIC_INPUTS_PRESENT"))

        authority = _require_mapping(
            config.get("authority"), "CONFIG", "AUTHORITY_NOT_OBJECT"
        )
        helper = _load_generic_helper(
            repo_root, str(authority["generic_helper_path"])
        )
        gates.append(_gate("HELPER", "PASS", "TRACKED_GENERIC_HELPER_REUSED"))

        records: list[dict[str, Any]] = []
        for subpool in (1, 2, 3):
            subpool_records = _read_fasta_records(
                fasta_paths[subpool], subpool, helper
            )
            records.extend(subpool_records)
        if len({record["header"] for record in records}) != len(records):
            _stop("FASTA", "FASTAS_SHARE_DUPLICATE_HEADERS")
        gates.append(_gate("FASTA", "PASS", "THREE_FASTAS_PARSED_BY_GENERIC_HELPER"))

        matrix_contract = _require_mapping(
            config.get("matrix_contract"),
            "CONFIG",
            "MATRIX_CONTRACT_NOT_OBJECT",
        )
        matrices = _read_matrices(raw_tar, matrix_contract)
        gates.append(
            _gate("MATRICES", "PASS", "EXACT_18_RAW_MATRICES_PARSED")
        )

        result_contract = _require_mapping(
            config.get("published_result_contract"),
            "CONFIG",
            "PUBLISHED_RESULT_CONTRACT_NOT_OBJECT",
        )
        published = _read_published_results(published_results, result_contract)
        report["published_universe_row_count"] = len(published)
        expected_counts = _require_mapping(
            _require_mapping(
                config.get("pairing"), "CONFIG", "PAIRING_NOT_OBJECT"
            ).get("expected_counts"),
            "CONFIG",
            "PAIRING_EXPECTED_COUNTS_NOT_OBJECT",
        )
        pairs, rejection_counts = _map_published_universe(records, published)
        report["accepted_pair_count"] = len(pairs)
        report["rejection_reason_counts"] = rejection_counts
        report["rejected_published_row_count"] = sum(rejection_counts.values())
        actual_counts = {
            "published_universe": len(published),
            "accepted": len(pairs),
            **rejection_counts,
        }
        if actual_counts != dict(expected_counts):
            _stop("PAIRING", "PREFROZEN_MAPPING_COUNTS_MISMATCH")
        gates.append(
            _gate(
                "PAIRING",
                "PASS",
                "SHEET_5_UNIVERSE_MAPPED_BY_UNIQUE_HAMMING_ONE_PAIR",
            )
        )

        complete_pairs, incomplete_pair_count = (
            _accepted_pairs_with_complete_raw_endpoints(pairs, matrices)
        )
        report["accepted_pair_complete_raw_endpoint_count"] = len(complete_pairs)
        report["accepted_pair_incomplete_raw_endpoint_count"] = (
            incomplete_pair_count
        )
        if (
            incomplete_pair_count != 0
            or len(complete_pairs)
            != matrix_contract["expected_complete_accepted_pair_count"]
        ):
            _stop("MATRICES", "ACCEPTED_PAIR_RAW_ENDPOINTS_INCOMPLETE")
        gates.append(
            _gate(
                "MATRICES",
                "PASS",
                "ALL_ACCEPTED_PAIRS_HAVE_12_REQUIRED_RAW_ENDPOINTS",
            )
        )

        reconstruction_records, auxiliary_defined, auxiliary_zero_undefined = (
            _build_development_reconstruction_records(
                complete_pairs, matrices, published
            )
        )
        gates.append(
            _gate(
                "PUBLISHED_RESULTS",
                "PASS",
                "OFFICIAL_MOESM4_LNFC_PRIMARY_LABEL_CLOSED",
            )
        )
        gates.append(
            _gate(
                "ENDPOINT",
                "PASS",
                "PUBLISHED_LNFC_PRIMARY_RAW_DIRECTION_AUXILIARY_ONLY",
            )
        )
        gates.append(
            _gate(
                "GROUPING",
                "PASS",
                "GENE_PLUS_SOURCE_CONTEXT_CLUSTER_FROZEN",
            )
        )

        rights = _require_mapping(
            config.get("rights"), "CONFIG", "RIGHTS_NOT_OBJECT"
        )
        actual_rights = rights.get("asset_level_private_derivative_use_status")
        if actual_rights != REQUIRED_RIGHTS_STATUS:
            gates.append(
                _gate(
                    "RIGHTS",
                    "FAIL",
                    "ASSET_LEVEL_PRIVATE_DERIVATIVE_USE_NOT_VERIFIED",
                )
            )
            report["raw_auxiliary_defined_pair_count"] = auxiliary_defined
            report["raw_auxiliary_zero_undefined_pair_count"] = (
                auxiliary_zero_undefined
            )
            _write_json(output_dir / REPORT_FILENAME, report)
            return 2, report
        gates.append(
            _gate(
                "RIGHTS",
                "PASS",
                "PRIVATE_DERIVATIVE_USE_VERIFIED",
            )
        )

        if not reconstruction_records:
            _stop("OUTPUT", "ALL_GATES_PASS_BUT_DEVELOPMENT_SET_EMPTY")
        rejection_aggregates = [
            {"reason": reason, "count": rejection_counts[reason]}
            for reason in (
                "NO_UNIQUE_SEQUENCE_PAIR",
                "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS",
            )
        ]
        _write_jsonl(output_dir / RECONSTRUCTION_FILENAME, reconstruction_records)
        _write_jsonl(output_dir / REJECTION_FILENAME, rejection_aggregates)
        report.update(
            {
                "status": SUCCESS_STATUS,
                "scientific_disposition": (
                    "DEVELOPMENT_RECONSTRUCTION_ONLY_AUDIT_PENDING_NOT_QUALIFIED"
                ),
                "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
                "development_reconstruction_record_count": len(
                    reconstruction_records
                ),
                "rejection_aggregate_count": len(rejection_aggregates),
                "raw_auxiliary_defined_pair_count": auxiliary_defined,
                "raw_auxiliary_zero_undefined_pair_count": (
                    auxiliary_zero_undefined
                ),
                "outputs": {
                    "development_reconstruction_private": RECONSTRUCTION_FILENAME,
                    "rejection_aggregates_private": REJECTION_FILENAME,
                    "aggregate_report": REPORT_FILENAME,
                },
            }
        )
        _write_json(output_dir / REPORT_FILENAME, report)
        return 0, report
    except RecoveryError as error:
        gates.append(_gate(error.gate, "FAIL", error.code))
        _write_json(output_dir / REPORT_FILENAME, report)
        return 2, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fasta-subpool-1", required=True, type=Path)
    parser.add_argument("--fasta-subpool-2", required=True, type=Path)
    parser.add_argument("--fasta-subpool-3", required=True, type=Path)
    parser.add_argument("--raw-tar", required=True, type=Path)
    parser.add_argument("--published-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recorded-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, report = recover(
        repo_root=args.repo_root,
        config_path=args.config,
        fasta_paths={
            1: args.fasta_subpool_1,
            2: args.fasta_subpool_2,
            3: args.fasta_subpool_3,
        },
        raw_tar=args.raw_tar,
        published_results=args.published_results,
        output_dir=args.output_dir,
        recorded_at=args.recorded_at,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_reconstruction_record_count": report[
                    "development_reconstruction_record_count"
                ],
                "rejected_published_row_count": report[
                    "rejected_published_row_count"
                ],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
