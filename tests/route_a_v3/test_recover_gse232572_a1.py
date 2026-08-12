from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = STAGING_ROOT / "scripts/route_a_v3/recover_gse232572_a1.py"
CONFIG = STAGING_ROOT / "configs/route_a_v3_gse232572_a1_recovery_v1.json"
RECORDED_AT = "2026-08-12T21:00:00+08:00"


def _column_name(index: int) -> str:
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _write_xlsx(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    rendered_rows: list[str] = []
    worksheet_rows = [
        (1, ["Sheet 5. All COSMIC somatic mutations tested with mapUTR in HeLa cells."]),
        (2, ["lnFC is the published natural-log relative activity; FDR is inferential only."]),
        (4, headers),
        *((row_index, values) for row_index, values in enumerate(rows, start=5)),
    ]
    for row_index, values in worksheet_rows:
        cells: list[str] = []
        for column_index, value in enumerate(values):
            reference = f"{_column_name(column_index)}{row_index}"
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                )
        rendered_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rendered_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet 5" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _write_generic_helper(repo_root: Path) -> None:
    helper = (
        repo_root
        / "d1_staging/scripts/d1/reconstruct_gse232572_sequences.py"
    )
    helper.parent.mkdir(parents=True)
    helper.write_text(
        """
def parse_fasta_header(header):
    parts = header.split("|")
    if len(parts) < 8:
        return None
    orientation = parts[7].strip().lower()
    if orientation.startswith("orig"):
        orientation = "orig"
    elif orientation.startswith("rc"):
        orientation = "rc"
    else:
        return None
    return {
        "subpool": parts[0],
        "source": parts[1],
        "chr_pos": parts[2],
        "gene": parts[3],
        "strand": parts[4],
        "allele_type": parts[5].strip().lower(),
        "allele": parts[6].strip().upper(),
        "orientation": orientation,
    }

def extract_insert(sequence, orientation):
    if orientation == "orig":
        return sequence[21:186]
    if orientation == "rc":
        table = str.maketrans("ACGT", "TGCA")
        return sequence[14:179].translate(table)[::-1]
    return ""
""".lstrip(),
        encoding="utf-8",
    )


def _headers_and_sequences(
    subpool: int, *, no_unique_first_pair: bool
) -> tuple[str, str, str, str]:
    gene = f"GENE{subpool}"
    chromosome_position = f"chr{subpool}:{100 + subpool}"
    reference_insert = "A" * 165
    alternate_bases = list(reference_insert)
    alternate_bases[82] = "T"
    if no_unique_first_pair and subpool == 1:
        alternate_bases[83] = "C"
    alternate_insert = "".join(alternate_bases)
    reference_header = (
        f"subpool{subpool}|COSMIC|{chromosome_position}|{gene}|+|reference|A|orig"
    )
    alternate_chr_pos = "chr99:999" if subpool == 2 else chromosome_position
    alternate_header = (
        f"subpool{subpool}|COSMIC|{alternate_chr_pos}|{gene}|+|alternate|T|orig"
    )
    reference_sequence = "G" * 21 + reference_insert + "C" * 14
    alternate_sequence = "G" * 21 + alternate_insert + "C" * 14
    return reference_header, reference_sequence, alternate_header, alternate_sequence


def _write_assets(
    root: Path,
    *,
    mixed_rejections: bool = False,
    missing_endpoint: bool = False,
) -> dict[str, Path]:
    fasta_paths: dict[int, Path] = {}
    headers_by_subpool: dict[int, tuple[str, str]] = {}
    supplement_rows: list[list[object]] = []
    for subpool in (1, 2, 3):
        ref_header, ref_sequence, alt_header, alt_sequence = _headers_and_sequences(
            subpool, no_unique_first_pair=mixed_rejections
        )
        fasta_path = root / f"GSE232572_C4Sp{subpool}.fasta.gz"
        with gzip.open(fasta_path, "wt", encoding="utf-8") as handle:
            handle.write(f">{ref_header}\n{ref_sequence}\n")
            handle.write(f">{alt_header}\n{alt_sequence}\n")
            if subpool == 1:
                orphan_header = (
                    "subpool1|COSMIC|chr8:888|ORPHAN|+|reference|G|orig"
                )
                orphan_sequence = "G" * 21 + "G" * 165 + "C" * 14
                handle.write(f">{orphan_header}\n{orphan_sequence}\n")
            if mixed_rejections and subpool == 2:
                duplicate_ref_insert = list("C" * 165)
                duplicate_ref_insert[82] = "A"
                duplicate_alt_insert = duplicate_ref_insert.copy()
                duplicate_alt_insert[82] = "T"
                duplicate_ref_header = (
                    "subpool2|COSMIC_DUP|chr2:102|GENE2|+|reference|A|orig"
                )
                duplicate_alt_header = (
                    "subpool2|COSMIC_DUP|chr77:777|GENE2|+|alternate|T|orig"
                )
                duplicate_ref_sequence = (
                    "G" * 21 + "".join(duplicate_ref_insert) + "C" * 14
                )
                duplicate_alt_sequence = (
                    "G" * 21 + "".join(duplicate_alt_insert) + "C" * 14
                )
                handle.write(
                    f">{duplicate_ref_header}\n{duplicate_ref_sequence}\n"
                )
                handle.write(
                    f">{duplicate_alt_header}\n{duplicate_alt_sequence}\n"
                )
        fasta_paths[subpool] = fasta_path
        headers_by_subpool[subpool] = (ref_header, alt_header)
        supplement_rows.append(
            [
                f"chr{subpool}",
                100 + subpool,
                "A",
                "T",
                f"GENE{subpool}",
                "+",
                -math.log(2.0) if subpool == 2 else math.log(2.0),
                0.001,
                0.01,
                "COSMIC",
            ]
        )

    raw_tar = root / "GSE232572_RAW.tar"
    with tarfile.open(raw_tar, "w") as archive:
        for subpool in (1, 2, 3):
            ref_header, alt_header = headers_by_subpool[subpool]
            for molecule_code, molecule in (("D", "DNA"), ("R", "RNA")):
                for replicate in (1, 2, 3):
                    if molecule == "DNA":
                        ref_count = 10.0
                        alt_count = 10.0
                    else:
                        ref_count = 20.0
                        alt_count = 40.0
                    if subpool == 3 and molecule == "DNA" and replicate == 2:
                        ref_count = 0.0
                    rows = [f"{ref_header}\t{ref_count}"]
                    if not (
                        missing_endpoint
                        and subpool == 2
                        and molecule == "RNA"
                        and replicate == 3
                    ):
                        rows.append(f"{alt_header}\t{alt_count}")
                    text = "gene\tcount\n" + "\n".join(rows) + "\n"
                    payload = gzip.compress(text.encode("utf-8"))
                    member = tarfile.TarInfo(
                        f"GSM{subpool}{replicate}00_C4Sp{subpool}{molecule_code}{replicate}.txt.gz"
                    )
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))

    supplement = root / "41467_2024_46795_MOESM4_ESM.xlsx"
    _write_xlsx(
        supplement,
        [
            "chromosome",
            "position (hg19, 1-based)",
            "ref",
            "alt",
            "gene",
            "gene_strand",
            "lnFC",
            "pval",
            "FDR",
            "group",
        ],
        supplement_rows,
    )
    return {
        "fasta1": fasta_paths[1],
        "fasta2": fasta_paths[2],
        "fasta3": fasta_paths[3],
        "raw_tar": raw_tar,
        "supplement": supplement,
    }


def _runtime_config(
    path: Path,
    *,
    assets: dict[str, Path],
    expected_counts: dict[str, int],
    unknown_rights: bool,
    tamper_label_mapping: bool = False,
) -> Path:
    disk_document = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert disk_document["rights"]["asset_level_private_derivative_use_status"] == (
        "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED"
    )
    assert disk_document["rights"]["public_redistribution_status"] == (
        "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
    )
    assert disk_document["authority"]["geo_series_relation"] == {
        "subject": "GSE232572",
        "relation": "SUBSERIES_OF",
        "object": "GSE232573",
        "authority_url": (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232573"
        ),
        "independent_study_count_delta": 0,
    }
    assert disk_document["scope"]["independent_study_count"] == 1
    assert disk_document["pairing"]["expected_counts"] == {
        "published_universe": 11929,
        "accepted": 8068,
        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
    }
    assert (
        disk_document["matrix_contract"][
            "expected_complete_accepted_pair_count"
        ]
        == 8068
    )
    document = copy.deepcopy(disk_document)
    document["pairing"]["expected_counts"] = expected_counts
    document["matrix_contract"]["expected_complete_accepted_pair_count"] = (
        expected_counts["accepted"]
    )
    identity_entries = {
        "fasta1": document["inputs"]["fasta_by_subpool"]["1"],
        "fasta2": document["inputs"]["fasta_by_subpool"]["2"],
        "fasta3": document["inputs"]["fasta_by_subpool"]["3"],
        "raw_tar": document["inputs"]["raw_tar"],
        "supplement": document["inputs"]["published_results"],
    }
    for asset_id, entry in identity_entries.items():
        payload = assets[asset_id].read_bytes()
        entry["bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
    if unknown_rights:
        document["rights"]["asset_level_private_derivative_use_status"] = (
            "UNKNOWN_NOT_ASSERTED"
        )
    if tamper_label_mapping:
        document["published_result_contract"]["columns"][
            "published_ln_activity"
        ] = "FDR"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    *,
    mixed_rejections: bool = False,
    missing_endpoint: bool = False,
    unknown_rights: bool = False,
    tamper_label_mapping: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_generic_helper(repo_root)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    assets = _write_assets(
        inputs,
        mixed_rejections=mixed_rejections,
        missing_endpoint=missing_endpoint,
    )
    expected_counts = (
        {
            "published_universe": 3,
            "accepted": 1,
            "NO_UNIQUE_SEQUENCE_PAIR": 1,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 1,
        }
        if mixed_rejections
        else {
            "published_universe": 3,
            "accepted": 3,
            "NO_UNIQUE_SEQUENCE_PAIR": 0,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
        }
    )
    config = _runtime_config(
        tmp_path / "qualification.json",
        assets=assets,
        expected_counts=expected_counts,
        unknown_rights=unknown_rights,
        tamper_label_mapping=tamper_label_mapping,
    )
    output_dir = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo_root),
            "--config",
            str(config),
            "--fasta-subpool-1",
            str(assets["fasta1"]),
            "--fasta-subpool-2",
            str(assets["fasta2"]),
            "--fasta-subpool-3",
            str(assets["fasta3"]),
            "--raw-tar",
            str(assets["raw_tar"]),
            "--published-results",
            str(assets["supplement"]),
            "--output-dir",
            str(output_dir),
            "--recorded-at",
            RECORDED_AT,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, output_dir


def test_development_reconstruction_has_zero_credit_and_uses_published_lnfc(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run(tmp_path)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert {path.name for path in output_dir.iterdir()} == {
        "GSE232572_A1_RECOVERY_REPORT.json",
        "development_reconstruction_records.private.jsonl",
        "rejection_aggregates.private.jsonl",
    }
    report = json.loads(
        (output_dir / "GSE232572_A1_RECOVERY_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == (
        "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED"
    )
    assert report["scientific_disposition"] == (
        "DEVELOPMENT_RECONSTRUCTION_ONLY_AUDIT_PENDING_NOT_QUALIFIED"
    )
    assert report["registry_role"] == "AUDIT_ONLY"
    assert report["qualification_status"] == "AUDIT_PENDING"
    assert report["qualified"] is False
    assert report["contribution"] == {"ordinary": 0, "a1": 0, "true_a2": 0}
    assert report["development_reconstruction_record_count"] == 3
    assert report["published_universe_row_count"] == 3
    assert report["accepted_pair_count"] == 3
    assert report["rejected_published_row_count"] == 0
    assert report["rejection_reason_counts"] == {
        "NO_UNIQUE_SEQUENCE_PAIR": 0,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
    }
    assert report["accepted_pair_complete_raw_endpoint_count"] == 3
    assert report["accepted_pair_incomplete_raw_endpoint_count"] == 0
    assert report["raw_auxiliary_defined_pair_count"] == 2
    assert report["raw_auxiliary_zero_undefined_pair_count"] == 1
    rows = [
        json.loads(line)
        for line in (
            output_dir / "development_reconstruction_records.private.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    assert {row["grouping"]["gene"]: row["label"]["value"] for row in rows} == {
        "GENE1": math.log(2.0),
        "GENE2": -math.log(2.0),
        "GENE3": math.log(2.0),
    }
    assert all(
        row["label"]["source"]
        == "MOESM4_OFFICIAL_PUBLISHED_RELATIVE_ACTIVITY_LNFC"
        for row in rows
    )
    assert all("standard_error" not in row["label"] for row in rows)
    assert all(row["data_role"] == "AUDIT_ONLY" for row in rows)
    assert all(row["qualification_status"] == "AUDIT_PENDING" for row in rows)
    assert all(row["qualified"] is False for row in rows)
    assert all(
        row["claim_boundary"]
        == "DEVELOPMENT_PRIVATE_RECONSTRUCTION_NOT_CANONICAL"
        for row in rows
    )
    assert all(
        row["grouping"]["split_or_bootstrap_assignment"]
        == "NOT_CREATED_AUDIT_ONLY"
        for row in rows
    )
    assert all("split_bootstrap_group" not in row["grouping"] for row in rows)
    assert sum(
        row["raw_count_auxiliary"]["status"]
        == "ZERO_COUNT_ENDPOINT_UNDEFINED_NO_PSEUDOCOUNT"
        for row in rows
    ) == 1
    assert sum(
        row["raw_count_auxiliary"]["status"]
        == "DEFINED_DIRECTION_DISAGREES_WITH_PUBLISHED_LABEL_DIAGNOSTIC_ONLY"
        for row in rows
    ) == 1
    assert all(
        row["provenance"]["hamming_neighbor_prefilter_fields"]
        == ["gene", "strand", "orientation"]
        for row in rows
    )
    assert all(
        row["provenance"]["header_alleles_complemented_for_rc"] is False
        for row in rows
    )
    assert all(row["training_allowed"] is False for row in rows)
    rejection_aggregates = [
        json.loads(line)
        for line in (
            output_dir / "rejection_aggregates.private.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert rejection_aggregates == [
        {"reason": "NO_UNIQUE_SEQUENCE_PAIR", "count": 0},
        {"reason": "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS", "count": 0},
    ]


def test_unmapped_and_ambiguous_sheet_rows_are_aggregated(tmp_path: Path) -> None:
    completed, output_dir = _run(tmp_path, mixed_rejections=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(
        (output_dir / "GSE232572_A1_RECOVERY_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == (
        "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED"
    )
    assert report["published_universe_row_count"] == 3
    assert report["accepted_pair_count"] == 1
    assert report["rejected_published_row_count"] == 2
    assert report["rejection_reason_counts"] == {
        "NO_UNIQUE_SEQUENCE_PAIR": 1,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 1,
    }
    assert report["development_reconstruction_record_count"] == 1
    assert report["contribution"] == {"ordinary": 0, "a1": 0, "true_a2": 0}
    aggregates = [
        json.loads(line)
        for line in (
            output_dir / "rejection_aggregates.private.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert aggregates == [
        {"reason": "NO_UNIQUE_SEQUENCE_PAIR", "count": 1},
        {"reason": "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS", "count": 1},
    ]


def test_missing_accepted_raw_endpoint_writes_only_aggregate_stop(
    tmp_path: Path,
) -> None:
    completed, output_dir = _run(tmp_path, missing_endpoint=True)
    assert completed.returncode == 2, completed.stderr + completed.stdout
    assert [path.name for path in output_dir.iterdir()] == [
        "GSE232572_A1_RECOVERY_REPORT.json"
    ]
    report = json.loads(
        (output_dir / "GSE232572_A1_RECOVERY_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["accepted_pair_count"] == 3
    assert report["accepted_pair_complete_raw_endpoint_count"] == 2
    assert report["accepted_pair_incomplete_raw_endpoint_count"] == 1
    assert report["development_reconstruction_record_count"] == 0
    assert report["gates"][-1] == {
        "gate": "MATRICES",
        "status": "FAIL",
        "code": "ACCEPTED_PAIR_RAW_ENDPOINTS_INCOMPLETE",
    }


def test_unknown_rights_writes_only_aggregate_stop(tmp_path: Path) -> None:
    completed, output_dir = _run(tmp_path, unknown_rights=True)
    assert completed.returncode == 2, completed.stderr + completed.stdout
    assert [path.name for path in output_dir.iterdir()] == [
        "GSE232572_A1_RECOVERY_REPORT.json"
    ]
    report = json.loads(
        (output_dir / "GSE232572_A1_RECOVERY_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == (
        "STOP_BEFORE_DEVELOPMENT_RECONSTRUCTION_ROW_PRODUCTION"
    )
    assert report["development_reconstruction_record_count"] == 0
    assert report["gates"][-1] == {
        "gate": "RIGHTS",
        "status": "FAIL",
        "code": "ASSET_LEVEL_PRIVATE_DERIVATIVE_USE_NOT_VERIFIED",
    }


def test_label_mapping_cannot_be_changed_to_fdr(tmp_path: Path) -> None:
    completed, output_dir = _run(tmp_path, tamper_label_mapping=True)
    assert completed.returncode == 2, completed.stderr + completed.stdout
    assert [path.name for path in output_dir.iterdir()] == [
        "GSE232572_A1_RECOVERY_REPORT.json"
    ]
    report = json.loads(
        (output_dir / "GSE232572_A1_RECOVERY_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == (
        "STOP_BEFORE_DEVELOPMENT_RECONSTRUCTION_ROW_PRODUCTION"
    )
    assert report["gates"][-1] == {
        "gate": "CONFIG",
        "status": "FAIL",
        "code": "PUBLISHED_RESULT_COLUMNS_NOT_FROZEN",
    }
