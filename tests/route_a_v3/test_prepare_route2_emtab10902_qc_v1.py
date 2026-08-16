from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from openpyxl import Workbook


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/prepare_route2_emtab10902_qc_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_route2_emtab10902_qc_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Supp_Table_2a"
    sheet.append(["caption"])
    sheet.append(["units"])
    sheet.append([
        "Source gene id",
        "Source gene name",
        "Source tile id",
        "size",
        "Mutation type",
        "Mutation position",
        "Sequence",
    ])
    source = "A" * 85
    candidate = "C" + source[1:]
    sheet.append(["GENE", "Gene", "1", 85, "WT", "WT", source])
    sheet.append(["GENE", "Gene", "1", 85, "sgl", "1-C", candidate])
    sheet.append(["GENE", "Gene", "1", 85, "mut", "1", candidate])
    workbook.save(path)


def test_outcome_blind_library_and_read_membership_use_read_count_column(tmp_path: Path) -> None:
    module = _module()
    module.EXPECTED_DESIGN_ROWS = 3
    module.EXPECTED_UNIQUE_SEQUENCES = 2
    module.PUBLISHER_REPORTED_PASSING_DESIGNS = 2
    module.SAMPLE_IDS = ("S1", "S2", "S3")
    workbook = tmp_path / "publisher.xlsx"
    _small_workbook(workbook)
    prepared = tmp_path / "prepared"
    summary = module.build_library(workbook, prepared)
    assert summary["evaluation_outcome_sheet_read"] is False
    assert summary["unique_sequence_count"] == 2
    assert summary["duplicate_sequence_group_count"] == 1

    sequence_ids = [
        json.loads(line)["sequence_id"]
        for line in (prepared / "design_row_to_sequence_id.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    unique_ids = sorted(set(sequence_ids))
    duplicated_id = Counter(sequence_ids).most_common(1)[0][0]
    count_specs = []
    for sample_index, sample_id in enumerate(module.SAMPLE_IDS):
        path = tmp_path / f"{sample_id}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for sequence_id in unique_ids:
                read_count = 25 if sequence_id == duplicated_id else (25 if sample_index == 0 else 0)
                handle.write(f"{sequence_id}\t999\t{read_count}\textra\n")
        count_specs.append(f"{sample_id}={path}")
    closed = tmp_path / "closed"
    qc_summary = module.close_qc(prepared / "design_row_to_sequence_id.jsonl", count_specs, closed)
    assert qc_summary["publisher_reported_count_reproduced"] is True
    assert qc_summary["passed_design_row_count"] == 2
    assert qc_summary["passed_unique_sequence_count"] == 1


def test_qc_closure_rejects_missing_primary_sample(tmp_path: Path) -> None:
    module = _module()
    module.EXPECTED_DESIGN_ROWS = 0
    module.EXPECTED_UNIQUE_SEQUENCES = 0
    module.SAMPLE_IDS = ("S1", "S2")
    mapping = tmp_path / "mapping.jsonl"
    mapping.write_text("", encoding="utf-8")
    with pytest.raises(module.QcPreparationError, match="sample set differs"):
        module.close_qc(mapping, [], tmp_path / "closed")
