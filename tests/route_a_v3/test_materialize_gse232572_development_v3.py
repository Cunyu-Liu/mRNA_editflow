from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from jsonschema import FormatChecker, validators


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    STAGING_ROOT
    / "scripts/route_a_v3/materialize_gse232572_development_v3.py"
)
CONFIG = (
    STAGING_ROOT
    / "configs/route_a_v3_gse232572_development_v3_materialization_v1.json"
)
INTEGRATED_RECOVERY_SCRIPT = (
    STAGING_ROOT / "scripts/route_a_v3/recover_gse232572_a1.py"
)
INTEGRATED_RECOVERY_CONFIG = (
    STAGING_ROOT / "configs/route_a_v3_gse232572_a1_recovery_v1.json"
)
INTEGRATED_SCHEMA = (
    STAGING_ROOT / "schemas/route_a_v3/canonical_intervention_record.schema.json"
)
WORK_ROOT = STAGING_ROOT.parent
RECOVERY_SCRIPT_SOURCE = (
    INTEGRATED_RECOVERY_SCRIPT
    if INTEGRATED_RECOVERY_SCRIPT.is_file()
    else WORK_ROOT
    / "gse232572_a1_recovery_staging/scripts/route_a_v3/recover_gse232572_a1.py"
)
RECOVERY_CONFIG_SOURCE = (
    INTEGRATED_RECOVERY_CONFIG
    if INTEGRATED_RECOVERY_CONFIG.is_file()
    else WORK_ROOT
    / "gse232572_a1_recovery_staging/configs/route_a_v3_gse232572_a1_recovery_v1.json"
)
SCHEMA_SOURCE = (
    INTEGRATED_SCHEMA
    if INTEGRATED_SCHEMA.is_file()
    else WORK_ROOT / "gse232572_schema_map/canonical_intervention_record.schema.json"
)
RECORDED_AT = "2026-08-12T23:00:00+08:00"
RECOVERY_RECORDED_AT = "2026-08-12T21:57:45+08:00"
BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)


def _load_materializer():
    spec = importlib.util.spec_from_file_location(
        "test_gse232572_development_v3_materializer", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAT = _load_materializer()


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _normalized_unbound_config(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    binding = normalized["implementation_binding"]
    for scalar in BINDING_SCALARS:
        binding[scalar] = "UNKNOWN_NOT_ASSERTED"
    return normalized


def _validate_disk_binding_and_normalize(config: dict) -> dict:
    binding = config["implementation_binding"]
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        assert all(
            binding[scalar] == "UNKNOWN_NOT_ASSERTED"
            for scalar in BINDING_SCALARS
        )
    else:
        assert binding["status"] == "BOUND"
        assert len(binding["implementation_commit"]) == 40
        assert set(binding["implementation_commit"]) <= set("0123456789abcdef")
        for scalar in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            assert len(binding[scalar]) == 64
            assert set(binding[scalar]) <= set("0123456789abcdef")
    return _normalized_unbound_config(config)


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


def _write_generic_helper(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def _headers_and_sequences(subpool: int) -> tuple[str, str, str, str]:
    gene = f"GENE{subpool}"
    chromosome_position = f"chr{subpool}:{100 + subpool}"
    reference_insert = "A" * 165
    alternate_bases = list(reference_insert)
    alternate_bases[82] = "T"
    alternate_insert = "".join(alternate_bases)
    reference_header = (
        f"subpool{subpool}|COSMIC|{chromosome_position}|{gene}|+|reference|A|orig"
    )
    alternate_header = (
        f"subpool{subpool}|COSMIC|{chromosome_position}|{gene}|+|alternate|T|orig"
    )
    reference_sequence = "G" * 21 + reference_insert + "C" * 14
    alternate_sequence = "G" * 21 + alternate_insert + "C" * 14
    return reference_header, reference_sequence, alternate_header, alternate_sequence


def _write_assets(root: Path) -> dict[str, Path]:
    root.mkdir()
    fasta_paths: dict[int, Path] = {}
    headers_by_subpool: dict[int, tuple[str, str]] = {}
    supplement_rows: list[list[object]] = []
    for subpool in (1, 2, 3):
        reference_header, reference_sequence, alternate_header, alternate_sequence = (
            _headers_and_sequences(subpool)
        )
        fasta_path = root / f"GSE232572_C4Sp{subpool}.fasta.gz"
        with gzip.open(fasta_path, "wt", encoding="utf-8") as handle:
            handle.write(f">{reference_header}\n{reference_sequence}\n")
            handle.write(f">{alternate_header}\n{alternate_sequence}\n")
        fasta_paths[subpool] = fasta_path
        headers_by_subpool[subpool] = (reference_header, alternate_header)
        supplement_rows.append(
            [
                f"chr{subpool}",
                100 + subpool,
                "A",
                "T",
                f"GENE{subpool}",
                "+",
                math.log(2.0) if subpool != 2 else -math.log(2.0),
                0.001,
                0.01,
                "COSMIC",
            ]
        )

    raw_tar = root / "GSE232572_RAW.tar"
    with tarfile.open(raw_tar, "w") as archive:
        for subpool in (1, 2, 3):
            reference_header, alternate_header = headers_by_subpool[subpool]
            for molecule_code, molecule in (("D", "DNA"), ("R", "RNA")):
                for replicate in (1, 2, 3):
                    reference_count = 10.0 if molecule == "DNA" else 20.0
                    alternate_count = 10.0 if molecule == "DNA" else 40.0
                    text = (
                        "gene\tcount\n"
                        f"{reference_header}\t{reference_count}\n"
                        f"{alternate_header}\t{alternate_count}\n"
                    )
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


def _synthetic_report() -> dict[str, object]:
    gates = json.loads(CONFIG.read_text(encoding="utf-8"))[
        "public_recovery_report"
    ]["expected"]["gates"]
    return {
        "status": "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED",
        "recorded_at": RECOVERY_RECORDED_AT,
        "published_universe_row_count": 3,
        "accepted_pair_count": 3,
        "rejected_published_row_count": 0,
        "rejection_reason_counts": {
            "NO_UNIQUE_SEQUENCE_PAIR": 0,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
        },
        "accepted_pair_complete_raw_endpoint_count": 3,
        "accepted_pair_incomplete_raw_endpoint_count": 0,
        "development_reconstruction_record_count": 3,
        "raw_auxiliary_defined_pair_count": 3,
        "raw_auxiliary_zero_undefined_pair_count": 0,
        "qualified": False,
        "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
        "gates": gates,
    }


def _bind_fake_repository(tmp_path: Path, monkeypatch) -> dict[str, object]:
    assets = _write_assets(tmp_path / "assets")
    recovery_report_path = tmp_path / "public" / "GSE232572_A1_RECOVERY_REPORT.json"
    _write_json(recovery_report_path, _synthetic_report())

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "routea-v3-test")
    _git(repo, "config", "user.name", "Route A test")
    _git(repo, "config", "user.email", "route-a-test@example.invalid")

    recovery_config = json.loads(RECOVERY_CONFIG_SOURCE.read_text(encoding="utf-8"))
    identity_entries = {
        "fasta1": recovery_config["inputs"]["fasta_by_subpool"]["1"],
        "fasta2": recovery_config["inputs"]["fasta_by_subpool"]["2"],
        "fasta3": recovery_config["inputs"]["fasta_by_subpool"]["3"],
        "raw_tar": recovery_config["inputs"]["raw_tar"],
        "supplement": recovery_config["inputs"]["published_results"],
    }
    for asset_id, entry in identity_entries.items():
        path = assets[asset_id]
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = _sha256(path)
    recovery_config["pairing"]["expected_counts"] = {
        "published_universe": 3,
        "accepted": 3,
        "NO_UNIQUE_SEQUENCE_PAIR": 0,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
    }
    recovery_config["matrix_contract"]["expected_complete_accepted_pair_count"] = 3

    recovery_config_path = repo / MAT.EXPECTED_AUTHORITY_ROLES["RECOVERY_CONFIG"]
    recovery_script_path = repo / MAT.EXPECTED_AUTHORITY_ROLES["RECOVERY_SCRIPT"]
    helper_path = repo / MAT.EXPECTED_AUTHORITY_ROLES["GENERIC_FASTA_HELPER"]
    schema_path = repo / MAT.EXPECTED_AUTHORITY_ROLES["CANONICAL_V3_SCHEMA"]
    _write_json(recovery_config_path, recovery_config)
    recovery_script_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RECOVERY_SCRIPT_SOURCE, recovery_script_path)
    _write_generic_helper(helper_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCHEMA_SOURCE, schema_path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "synthetic authority base")
    base_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(MAT, "BASE_COMMIT", base_commit)
    for name, value in {
        "EXPECTED_PUBLISHED": 3,
        "EXPECTED_ACCEPTED": 3,
        "EXPECTED_NO_UNIQUE": 0,
        "EXPECTED_AMBIGUOUS": 0,
        "EXPECTED_COMPLETE_ENDPOINTS": 3,
        "EXPECTED_INCOMPLETE_ENDPOINTS": 0,
        "EXPECTED_AUXILIARY_DEFINED": 3,
        "EXPECTED_AUXILIARY_ZERO_UNDEFINED": 0,
    }.items():
        monkeypatch.setattr(MAT, name, value)

    config = _normalized_unbound_config(
        json.loads(CONFIG.read_text(encoding="utf-8"))
    )
    config["repository_authority"].update(
        {
            "production_repo_root": str(repo.resolve()),
            "branch": "routea-v3-test",
            "base_commit": base_commit,
        }
    )
    for item in config["repository_authority"]["frozen_authority_blobs"]:
        path = item["path"]
        item["git_blob_oid"] = _git(repo, "rev-parse", f"{base_commit}:{path}")
        item["sha256"] = hashlib.sha256(
            subprocess.run(
                ["git", "-C", str(repo), "show", f"{base_commit}:{path}"],
                stdout=subprocess.PIPE,
                check=True,
            ).stdout
        ).hexdigest()
    config["inputs"] = copy.deepcopy(recovery_config["inputs"])
    expected_report = config["public_recovery_report"]["expected"]
    expected_report.update(_synthetic_report())
    config["public_recovery_report"].update(
        {
            "absolute_path": str(recovery_report_path.resolve()),
            "bytes": recovery_report_path.stat().st_size,
            "sha256": _sha256(recovery_report_path),
        }
    )
    contract = config["materialization_contract"]
    contract.update(
        {
            "required_published_universe_row_count": 3,
            "required_development_record_count": 3,
            "required_rejection_reason_counts": {
                "NO_UNIQUE_SEQUENCE_PAIR": 0,
                "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
            },
            "required_complete_raw_endpoint_pair_count": 3,
            "required_incomplete_raw_endpoint_pair_count": 0,
            "required_raw_auxiliary_defined_pair_count": 3,
            "required_raw_auxiliary_zero_undefined_pair_count": 0,
        }
    )

    config_path = repo / MAT.CONFIG_RELATIVE_PATH
    script_path = repo / MAT.SCRIPT_RELATIVE_PATH
    test_path = repo / MAT.TEST_RELATIVE_PATH
    _write_json(config_path, config)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(SCRIPT.read_bytes() + b"\n# synthetic I1 predecessor\n")
    test_path.write_bytes(Path(__file__).read_bytes() + b"\n# synthetic I1 predecessor\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "frozen exact3 implementation I1")
    production_authority_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        MAT, "PRODUCTION_AUTHORITY_COMMIT", production_authority_commit
    )

    shutil.copyfile(SCRIPT, script_path)
    shutil.copyfile(Path(__file__), test_path)
    _git(repo, "add", MAT.SCRIPT_RELATIVE_PATH, MAT.TEST_RELATIVE_PATH)
    _git(repo, "commit", "-q", "-m", "dynamic exact2 implementation I2")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    bound_config = copy.deepcopy(config)
    bound_config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": _sha256(script_path),
            "implementation_test_sha256": _sha256(test_path),
        }
    )
    _write_json(config_path, bound_config)
    _git(repo, "add", MAT.CONFIG_RELATIVE_PATH)
    _git(repo, "commit", "-q", "-m", "config-only binding B2")
    assert not _git(repo, "status", "--porcelain")
    return {
        "repo": repo,
        "config": config_path,
        "assets": assets,
        "public_report": recovery_report_path,
        "schema": schema_path,
        "production_authority_commit": production_authority_commit,
        "implementation_commit": implementation_commit,
    }


def _run_bound(tmp_path: Path, monkeypatch) -> tuple[int, dict, Path, dict]:
    bound = _bind_fake_repository(tmp_path, monkeypatch)
    assets = bound["assets"]
    output = tmp_path / "output"
    code, report = MAT.materialize(
        repo_root=bound["repo"],
        config_path=bound["config"],
        fasta_paths={
            1: assets["fasta1"],
            2: assets["fasta2"],
            3: assets["fasta3"],
        },
        raw_tar=assets["raw_tar"],
        published_results=assets["supplement"],
        public_recovery_report=bound["public_report"],
        output_dir=output,
        recorded_at=RECORDED_AT,
    )
    return code, report, output, bound


def test_disk_config_freezes_official_counts_no_credit_and_valid_i_or_b(
    tmp_path: Path,
) -> None:
    disk_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = _validate_disk_binding_and_normalize(disk_config)
    assert config["repository_authority"]["base_commit"] == (
        "aa396dbdeac083c9f88df62877ff7cbcb7e0d318"
    )
    assert all(
        config["implementation_binding"][scalar] == "UNKNOWN_NOT_ASSERTED"
        for scalar in BINDING_SCALARS
    )
    contract = config["materialization_contract"]
    assert contract["required_published_universe_row_count"] == 11929
    assert contract["required_development_record_count"] == 8068
    assert contract["required_rejection_reason_counts"] == {
        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
    }
    assert contract["required_complete_raw_endpoint_pair_count"] == 8068
    assert contract["required_incomplete_raw_endpoint_pair_count"] == 0
    assert contract["canonical_qualification_allowed"] is False
    assert contract["training_allowed"] is False
    assert config["public_recovery_report"]["private_row_artifacts_consumed"] is False

    temporary_bound = copy.deepcopy(config)
    temporary_bound["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    temporary_bound_path = tmp_path / "bound-config-node.json"
    _write_json(temporary_bound_path, temporary_bound)
    assert _validate_disk_binding_and_normalize(
        json.loads(temporary_bound_path.read_text(encoding="utf-8"))
    ) == config


def test_schema_valid_development_rows_preserve_mapping_provenance_and_no_credit(
    tmp_path: Path, monkeypatch
) -> None:
    code, report, output, bound = _run_bound(tmp_path, monkeypatch)
    assert code == 0
    assert {path.name for path in output.iterdir()} == {
        "development_v3_records.private.jsonl",
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json",
    }
    assert report["status"] == "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED"
    assert report["schema_valid_development_record_count"] == 3
    assert report["canonical_record_count"] == 0
    assert report["qualified"] is False
    assert report["contribution"] == {"ordinary": 0, "a1": 0, "true_a2": 0}
    assert report["training_allowed"] is False
    assert report["model_selection_allowed"] is False
    assert report["next_phase_allowed"] is False
    assert report["source_public_recovery_report"][
        "private_row_artifacts_consumed"
    ] is False
    assert report["license_boundary"]["row_license_status"] == "UNKNOWN_BLOCKED"
    assert report["license_boundary"]["redistribution_allowed"] is False

    private_path = output / "development_v3_records.private.jsonl"
    assert report["private_output"]["bytes"] == private_path.stat().st_size
    assert report["private_output"]["sha256"] == _sha256(private_path)
    records = [json.loads(line) for line in private_path.read_text().splitlines()]
    assert len(records) == 3
    schema = json.loads(bound["schema"].read_text(encoding="utf-8"))
    validator_class = validators.validator_for(schema)
    validator_class.check_schema(schema)
    assert validator_class.__name__ == "Draft202012Validator"
    validator = validator_class(schema, format_checker=FormatChecker())
    for record in records:
        validator.validate(record)
        assert record["data_role"] == "ORDINARY_DEVELOPMENT"
        assert record["evidence_status"] == "BLOCKED_PENDING_PUBLIC_EVIDENCE"
        assert record["claim_status"] == "NOT_ESTABLISHED"
        assert record["eligibility"]["status"] == "DEVELOPMENT_ONLY"
        assert record["exposure"] == {
            "stratum": "DEVELOPMENT_ONLY",
            "label_exposed": True,
            "sequence_exposed": True,
            "audit_id": "UNKNOWN_NOT_ASSERTED",
        }
        assert record["split"] == {
            "split_id": "GSE232572|DEVELOPMENT_ONLY|NOT_LEAKAGE_AUDITED",
            "partition": "DEVELOPMENT",
            "leakage_audit_status": "NOT_RUN",
        }
        assert record["study"]["independent_study_group_id"] == "GSE232573"
        assert record["context"]["cell_type"] == "HeLa"
        assert record["standard_error"] is None
        assert record["replicate"]["replicate_count"] == 3
        assert record["license"]["status"] == "UNKNOWN_BLOCKED"
        assert record["license"]["redistribution_allowed"] is False
        assert record["license"]["verified_at"] == RECOVERY_RECORDED_AT
        assert record["raw_measurement"]["value"] == record["delta"]
        assert record["raw_measurement"]["scale"] == "ln"
        assert "Sheet 5!row=" in record["provenance"]["raw_record_locator"]
        assert "reference_fasta_header=" in record["provenance"]["raw_record_locator"]
        assert "alternate_fasta_header=" in record["provenance"]["raw_record_locator"]
        assert record["paper_faithful_transform"]["implementation_sha256"] == (
            json.loads(bound["config"].read_text(encoding="utf-8"))[
                "implementation_binding"
            ]["implementation_script_sha256"]
        )


def test_public_recovery_report_drift_stops_before_private_rows(
    tmp_path: Path, monkeypatch
) -> None:
    bound = _bind_fake_repository(tmp_path, monkeypatch)
    report_path = bound["public_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["accepted_pair_count"] = 2
    _write_json(report_path, report)
    assets = bound["assets"]
    output = tmp_path / "output"
    code, stopped = MAT.materialize(
        repo_root=bound["repo"],
        config_path=bound["config"],
        fasta_paths={1: assets["fasta1"], 2: assets["fasta2"], 3: assets["fasta3"]},
        raw_tar=assets["raw_tar"],
        published_results=assets["supplement"],
        public_recovery_report=report_path,
        output_dir=output,
        recorded_at=RECORDED_AT,
    )
    assert code == 2
    assert stopped["status"] == "STOP_BEFORE_DEVELOPMENT_V3_ROW_PRODUCTION"
    assert {path.name for path in output.iterdir()} == {
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
    }
    assert stopped["gates"][-1]["gate"] == "PUBLIC_RECOVERY_REPORT"


def test_official_asset_drift_stops_before_private_rows(
    tmp_path: Path, monkeypatch
) -> None:
    bound = _bind_fake_repository(tmp_path, monkeypatch)
    assets = bound["assets"]
    with assets["fasta1"].open("ab") as handle:
        handle.write(b"drift")
    output = tmp_path / "output"
    code, stopped = MAT.materialize(
        repo_root=bound["repo"],
        config_path=bound["config"],
        fasta_paths={1: assets["fasta1"], 2: assets["fasta2"], 3: assets["fasta3"]},
        raw_tar=assets["raw_tar"],
        published_results=assets["supplement"],
        public_recovery_report=bound["public_report"],
        output_dir=output,
        recorded_at=RECORDED_AT,
    )
    assert code == 2
    assert {path.name for path in output.iterdir()} == {
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
    }
    assert stopped["gates"][-1] == {
        "gate": "INPUTS",
        "status": "FAIL",
        "code": "FASTA1_BYTE_COUNT_MISMATCH",
    }


def test_actual_schema_rejects_extra_property_before_private_rows(
    tmp_path: Path, monkeypatch
) -> None:
    bound = _bind_fake_repository(tmp_path, monkeypatch)
    original = MAT._build_record

    def with_extra_property(*args, **kwargs):
        record = original(*args, **kwargs)
        record["unexpected_property"] = True
        return record

    monkeypatch.setattr(MAT, "_build_record", with_extra_property)
    assets = bound["assets"]
    output = tmp_path / "output"
    code, stopped = MAT.materialize(
        repo_root=bound["repo"],
        config_path=bound["config"],
        fasta_paths={1: assets["fasta1"], 2: assets["fasta2"], 3: assets["fasta3"]},
        raw_tar=assets["raw_tar"],
        published_results=assets["supplement"],
        public_recovery_report=bound["public_report"],
        output_dir=output,
        recorded_at=RECORDED_AT,
    )
    assert code == 2
    assert {path.name for path in output.iterdir()} == {
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
    }
    assert stopped["gates"][-1] == {
        "gate": "SCHEMA",
        "status": "FAIL",
        "code": "DEVELOPMENT_RECORD_SCHEMA_VALIDATION_FAILED",
    }
