#!/usr/bin/env python3
"""One-shot DEC028 SS3 GSE200304 private materialization and conformance."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_dec028_gse200304_ss3_materialization_v1.json"
PROTOCOL_ID = "ROUTE_A_V3_DEC028_GSE200304_SS3_MATERIALIZATION_V1"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
PAIR_RE = re.compile(r"^(?P<locus>[^:]+):(?P<position>[1-9][0-9]*)_(?P<ref>[ACGT])-(?P<alt>[ACGT])$")
BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGT", "TGCA")


class MaterializationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, "config root must be object")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol differs")
    _require(config.get("decision_id") == "V3-DEC-028", "decision differs")
    _require(config.get("runtime_authority_event_id") == "A1-EVT-061", "runtime authority differs")
    _require(config.get("document_status") == "ACTIVE_FOR_ONE_SS3_MATERIALIZATION_ONLY", "authority inactive")
    authority = config["authority"]
    _require(authority["authorized_execution_count"] == 1, "execution count differs")
    for key in ("split_authorized", "model_authorized", "cuda_authorized", "optimizer_authorized", "training_authorized", "g1_authorized", "sealed_access_authorized"):
        _require(authority[key] is False, f"forbidden authority enabled: {key}")
    row = config["row_contract"]
    _require((row["expected_public_join_count"], row["expected_na_exclusion_count"], row["expected_materialized_count"]) == (6772, 225, 6547), "membership geometry differs")
    _require(row["membership_frozen_before_replicate_effect_calculation"] is True, "membership timing differs")
    expected_definitions = {
        "membership_definition": "INNER_JOIN_TABLE_S2_EXACT_WT_MUTANT_PAIR_TABLE_S3_FINITE_TOTALPOLY_AND_PROCESSED_MATRIX_KEY",
        "source_group_definition": "GSE200304_GSE200302_AUTHOR_LOCUS_CANONICAL_POSITION_REFERENCE_ALLELE_ORIENTATION_NORMALIZED_WT201",
        "context_vector_definition": "SIXTEEN_CONTIGUOUS_201NT_POSITION_BINS_TIMES_ACGT_PAIR_MEAN_FRACTIONS_SOURCE_CANDIDATE_SWAP_INVARIANT",
        "edit_feature_definition": "THREE_POSITIONS_CENTER_MINUS_ONE_CENTER_CENTER_PLUS_ONE_TIMES_ACGT_CANDIDATE_ONEHOT_MINUS_SOURCE_ONEHOT",
        "effect_definition": "MEAN_OVER_SIX_PAIRED_BIOLOGICAL_REPLICATES_OF_LOG2_SUM_HIGH_LOW_MINUS_TOTAL_RNA_MUTANT_MINUS_WT",
        "standard_error_definition": "SAMPLE_STANDARD_DEVIATION_OF_SIX_PAIRED_REPLICATE_DELTAS_DIVIDED_BY_SQRT_SIX_FINITE_POSITIVE",
        "missing_nonfinite_or_nonpositive_se_action": "STOP_NO_PRIVATE_SUCCESS_ASSET_NO_SPLIT_MODEL_OR_CUDA",
    }
    for key, expected in expected_definitions.items():
        _require(row.get(key) == expected, f"row-contract definition differs: {key}")
    truth = config["current_truth"]
    for key in ("materialization_execution_count", "data_rows_read", "private_rows_written", "split_assignments_written", "model_constructions", "cuda_touches", "optimizer_constructions", "parameter_updates"):
        _require(truth[key] == 0, f"current truth nonzero: {key}")
    _require(truth["g1_launched"] is False, "G1 already launched")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def audit_repository(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    _require(binding["status"] == "BOUND", "implementation binding is not BOUND")
    _require(_git("status", "--porcelain") == "", "repository is not clean")
    head = _git("rev-parse", "HEAD")
    implementation = binding["implementation_commit"]
    _require(_git("rev-parse", f"{head}^") == implementation, "binding parent differs")
    _require(sorted(_git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()) == sorted(binding["binding_exact_changed_paths"]), "binding paths differ")
    _require(_git("rev-parse", f"{implementation}^") == binding["implementation_expected_parent"], "implementation parent differs")
    _require(sorted(_git("diff-tree", "--no-commit-id", "--name-only", "-r", implementation).splitlines()) == sorted(binding["implementation_exact_changed_paths"]), "implementation paths differ")
    for path in binding["implementation_exact_changed_paths"][1:]:
        committed = subprocess.run(["git", "show", f"{implementation}:{path}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        _require((REPO_ROOT / path).read_bytes() == committed, f"working bytes differ: {path}")


def _asset_identity(path: Path, spec: Mapping[str, Any]) -> None:
    _require(path.is_file(), f"asset absent: {path}")
    _require(path.stat().st_size == spec["bytes"], f"asset bytes differ: {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(digest == spec["sha256"], f"asset digest differs: {path.name}")


def _column_index(reference: str) -> int:
    letters = "".join(ch for ch in reference if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + ord(ch.upper()) - 64
    return value - 1


def _xlsx_rows(path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships", "p": "http://schemas.openxmlformats.org/package/2006/relationships"}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels.findall("p:Relationship", ns)}
        sheet = next((item for item in workbook.findall("m:sheets/m:sheet", ns) if item.attrib["name"] == sheet_name), None)
        _require(sheet is not None, f"xlsx sheet absent: {sheet_name}")
        target = targets[sheet.attrib[f"{{{ns['r']}}}id"]]
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target.lstrip("./")
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{ns['m']}}}t")))
        root = ET.fromstring(archive.read(member))
        rows: list[list[str]] = []
        for row in root.findall("m:sheetData/m:row", ns):
            values: list[str] = []
            for cell in row.findall("m:c", ns):
                index = _column_index(cell.attrib["r"])
                while len(values) <= index:
                    values.append("")
                if cell.attrib.get("t") == "inlineStr":
                    text = "".join(node.text or "" for node in cell.iter(f"{{{ns['m']}}}t"))
                else:
                    node = cell.find("m:v", ns)
                    text = "" if node is None or node.text is None else node.text
                    if cell.attrib.get("t") == "s" and text:
                        text = shared[int(text)]
                values[index] = text
            rows.append(values)
        return rows


def _read_s2(path: Path) -> dict[str, tuple[str, str]]:
    by_id: dict[str, dict[str, tuple[str, ...]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ["ID", "Type", "201bp", "5' End", "3'End", "Full_Oligo"], "S2 header differs")
        for row in reader:
            if row["Type"] not in {"WT", "Mutant"}:
                continue
            value = tuple(row[key] for key in reader.fieldnames)
            existing = by_id.setdefault(row["ID"], {}).get(row["Type"])
            _require(existing is None or existing == value, f"S2 nonidentical duplicate arm: {row['ID']}")
            by_id.setdefault(row["ID"], {})[row["Type"]] = value
    pairs: dict[str, tuple[str, str]] = {}
    for key, arms in by_id.items():
        _require(set(arms) == {"WT", "Mutant"}, f"S2 pair incomplete: {key}")
        pairs[key] = (arms["WT"][2].upper(), arms["Mutant"][2].upper())
    _require(len(pairs) == 6885, "S2 pair count differs")
    return pairs


def _read_s3(path: Path) -> tuple[set[str], int]:
    rows = _xlsx_rows(path, "S2A_Polysome_MPRA_Mut_Stats")
    header = rows[0]
    expected = ["barcode", "Gene", "Comparison", "xtail_log2FC_TE", "xtail_pvalue", "xtail_FDR", "Translation_Sig"]
    _require(header[:7] == expected, "S3 header differs")
    finite: set[str] = set()
    total = 0
    na = 0
    for values in rows[1:]:
        values += [""] * (7 - len(values))
        if values[2] != "TotalPoly:RNA":
            continue
        total += 1
        try:
            effect = float(values[3])
        except ValueError:
            effect = math.nan
        if math.isfinite(effect):
            finite.add(values[0])
        else:
            na += 1
    _require((total, len(finite), na) == (6772, 6547, 225), "S3 membership geometry differs")
    return finite, total


def _read_matrix(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows: dict[str, list[float]] = {}
        for raw in reader:
            _require(len(raw) == len(header), "matrix row width differs")
            key = raw[0]
            _require(key not in rows, "matrix duplicate key")
            values = [float(item) for item in raw[1:]]
            _require(all(math.isfinite(item) for item in values), "matrix nonfinite value")
            rows[key] = values
    _require(len(rows) == 6772 and len(header) == 61, "matrix geometry differs")
    return header, rows


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _normalize_pair(key: str, source: str, candidate: str) -> tuple[str, str, str]:
    match = PAIR_RE.match(key)
    _require(match is not None, f"pair ID grammar differs: {key}")
    _require(len(source) == len(candidate) == 201, "sequence length differs")
    differences = [i for i, (left, right) in enumerate(zip(source, candidate)) if left != right]
    _require(differences == [100], "pair is not exact central SNV")
    ref, alt = match.group("ref"), match.group("alt")
    if source[100] == ref and candidate[100] == alt:
        return source, candidate, "FORWARD"
    if source[100] == ref.translate(COMPLEMENT) and candidate[100] == alt.translate(COMPLEMENT):
        return _reverse_complement(source), _reverse_complement(candidate), "REVERSE_COMPLEMENT"
    raise MaterializationError("declared allele orientation does not replay")


def _log2_sum(left: float, right: float) -> float:
    largest = max(left, right)
    return largest + math.log2(2.0 ** (left - largest) + 2.0 ** (right - largest))


def _replicate_effects(header: list[str], values: list[float]) -> list[float]:
    index = {name: position - 1 for position, name in enumerate(header) if position > 0}
    effects: list[float] = []
    for replicate in range(1, 7):
        def value(role: str, arm: str) -> float:
            prefix = f"{role}_{replicate}_"
            name = next(name for name in index if name.startswith(prefix) and name.endswith(f"_{arm}"))
            return values[index[name]]
        mutant = _log2_sum(value("High_Poly", "Mutant"), value("Low_Poly", "Mutant")) - value("Total_RNA", "Mutant")
        source = _log2_sum(value("High_Poly", "WT"), value("Low_Poly", "WT")) - value("Total_RNA", "WT")
        effects.append(mutant - source)
    return effects


def _context_vector(source: str, candidate: str) -> list[float]:
    vector: list[float] = []
    for bin_index in range(16):
        start = math.floor(bin_index * 201 / 16)
        end = math.floor((bin_index + 1) * 201 / 16)
        denominator = 2.0 * (end - start)
        for base in BASES:
            vector.append((source[start:end].count(base) + candidate[start:end].count(base)) / denominator)
    _require(len(vector) == 64, "context width differs")
    return vector


def _edit_features(source: str, candidate: str) -> list[float]:
    vector: list[float] = []
    for position in (99, 100, 101):
        for base in BASES:
            vector.append(float(candidate[position] == base) - float(source[position] == base))
    _require(len(vector) == 12, "edit width differs")
    return vector


def build_rows(s2: Mapping[str, tuple[str, str]], finite_keys: set[str], matrix_header: list[str], matrix: Mapping[str, list[float]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    membership = sorted(set(s2) & finite_keys & set(matrix))
    rejects = {"s2_not_in_s3": len(set(s2) - finite_keys), "s3_na_or_nonfinite": 225, "nonpositive_se": 0}
    _require(len(set(matrix) & set(s2)) == 6772, "public join count differs")
    _require(len(membership) == 6547, "membership count differs before effect calculation")
    rows: list[dict[str, Any]] = []
    for key in membership:
        source, candidate, _ = _normalize_pair(key, *s2[key])
        match = PAIR_RE.match(key)
        assert match is not None
        effects = _replicate_effects(matrix_header, matrix[key])
        mean = statistics.fmean(effects)
        se = statistics.stdev(effects) / math.sqrt(6.0)
        if not math.isfinite(mean) or not math.isfinite(se) or se <= 0.0:
            rejects["nonpositive_se"] += 1
            continue
        group = "|".join(("GSE200304", "GSE200302", match.group("locus"), match.group("position"), match.group("ref"), source))
        rows.append({
            "record_key": key,
            "source_group": group,
            "source_sequence": source,
            "candidate_sequence": candidate,
            "context_vector": _context_vector(source, candidate),
            "edit_features": _edit_features(source, candidate),
            "direction_normalized_effect": mean,
            "biological_standard_error": se,
        })
    return rows, rejects


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(config: Mapping[str, Any], public_dir: Path, matrix_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), "output directory already exists; one-shot authority exhausted")
    p0 = json.loads(Path(config["authority"]["p0_correction_report"]).read_text(encoding="utf-8"))
    _require(p0["overall_status"] == config["authority"]["required_p0_status"] and p0["status_counts"] == {"PASS": 11, "NONPASS": 0}, "P0 correction is not 11/11")
    specs = {item["name"]: item for item in config["inputs"]["assets"]}
    s2_path = public_dir / "NIHMS1928233-supplement-3.csv"
    s3_path = public_dir / "NIHMS1928233-supplement-4.xlsx"
    for path in (s2_path, s3_path, matrix_path):
        _asset_identity(path, specs[path.name])
    s2 = _read_s2(s2_path)
    finite, joined = _read_s3(s3_path)
    header, matrix = _read_matrix(matrix_path)
    rows, rejects = build_rows(s2, finite, header, matrix)
    success = len(rows) == 6547 and rejects["nonpositive_se"] == 0
    result = {
        "protocol_id": PROTOCOL_ID,
        "overall_status": "PASS_SS3_MATERIALIZATION_CONFORMANCE" if success else "STOP_SS3_MATERIALIZATION_CONFORMANCE_FAILED",
        "public_join_count": joined,
        "na_exclusion_count": 225,
        "materialized_count": len(rows) if success else 0,
        "private_candidate_row_count_before_conformance": len(rows),
        "reject_counts": rejects,
        "split_assignment_count": 0,
        "model_construction_count": 0,
        "cuda_touch_count": 0,
        "optimizer_construction_count": 0,
        "parameter_update_count": 0,
        "g1_launched": False,
        "qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        _write_json(temporary / config["outputs"]["aggregate_reject_summary_filename"], {"protocol_id": PROTOCOL_ID, "reject_counts": rejects, "contains_member_payload": False})
        _write_json(temporary / config["outputs"]["conformance_filename"], result)
        if success:
            private_path = temporary / config["outputs"]["private_rows_filename"]
            with private_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            manifest = {
                "protocol_id": PROTOCOL_ID,
                "private_row_count": len(rows),
                "private_rows_bytes": private_path.stat().st_size,
                "private_rows_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
                "required_fields_exactly": config["row_contract"]["required_fields_exactly"],
                "member_payload_in_aggregate": False,
                "split_assignment_count": 0,
            }
            _write_json(temporary / config["outputs"]["aggregate_manifest_filename"], manifest)
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--public-asset-dir", type=Path, required=True)
    parser.add_argument("--processed-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    audit_repository(config)
    result = execute(config, args.public_asset_dir, args.processed_matrix, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["overall_status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
