from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from openpyxl import Workbook


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse186455_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse186455_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse186455_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_tar(path: Path, rows: list[tuple[str, str]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(["", "bc", "sequence", "seqName", "bcCount"])
    for index, (name, sequence) in enumerate(rows, start=1):
        writer.writerow([index, f"bc{index}", sequence, name, 10])
    compressed = gzip.compress(buffer.getvalue().encode("utf-8"))
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("sample.tab.gz")
        info.size = len(compressed)
        archive.addfile(info, io.BytesIO(compressed))


def test_sequence_loader_recovers_sub_geometry_and_rejects_conflicts(tmp_path: Path) -> None:
    module = _module()
    source = "A" * 120
    candidate = source[:59] + "C" + source[60:]
    archive = tmp_path / "input.tar"
    _write_tar(archive, [("GENE_ssc_ref", source), ("GENE_ssc_alt", candidate), ("GENE_ssc_shuf", source[::-1])])
    groups, stats = module._load_sequence_groups(archive)
    assert groups["GENE_ssc"] == {"ref": source, "alt": candidate, "shuf": source[::-1]}
    assert stats["tar_member_count"] == 1
    assert stats["unique_element_count"] == 3
    assert stats["complete_ref_alt_pair_count"] == 1
    assert stats["action_SUB"] == 1


def test_published_effect_reader_treats_not_tested_as_missing(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    workbook = Workbook()
    workbook.remove(workbook.active)
    header = ["Element", "logFC", "pval", "fdr", "bonferroni"]
    for sheet_name in config["input"]["expected_workbook_sheets"]:
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(header)
        if sheet_name in config["published_effect_sheets"].values():
            sheet.append(["pair1", "0.25", "0.1", "0.2", "1"])
            sheet.append(["pair2", "Not Tested", "Not Tested", "Not Tested", "Not Tested"])
    path = tmp_path / "effects.xlsx"
    workbook.save(path)
    effects, stats = module._read_published_effects(path, config)
    assert effects["N2A_TRANSCRIPT_ABUNDANCE"]["pair1"]["logFC"] == 0.25
    assert "pair2" not in effects["VGLUT_TRANSCRIPT_ABUNDANCE"]
    assert stats["finite_effect_N2A_TRANSCRIPT_ABUNDANCE"] == 1
    assert stats["not_tested_VGLUT_TRANSCRIPT_ABUNDANCE"] == 1


def test_eligibility_keeps_only_published_sub_and_author_bad_alt_is_excluded() -> None:
    module = _module()
    config = _config()
    bad_id = config["author_bad_alt_pair_ids"][0]
    source = "A" * 120
    candidate = source[:59] + "C" + source[60:]
    groups = {
        "GOOD_ssc": {"ref": source, "alt": candidate},
        bad_id: {"ref": source, "alt": candidate},
        **{pair_id: {"ref": source, "alt": source + "C"} for pair_id in config["author_bad_alt_pair_ids"][1:]},
    }
    value = {"logFC": 0.5, "pval": 0.1, "fdr": 0.2, "bonferroni": 1.0}
    effects = {
        "N2A_TRANSCRIPT_ABUNDANCE": {"GOOD_ssc": value, bad_id: value},
        "VGLUT_TRANSCRIPT_ABUNDANCE": {},
    }
    units, stats = module._eligible_units(config, groups, effects)
    assert [(unit["pair_id"], unit["biological_context_id"]) for unit in units] == [("GOOD_ssc", "N2A_TRANSCRIPT_ABUNDANCE")]
    assert stats["reject_author_bad_alt_sub_pair"] == 1
    assert stats["reject_length_change_pair"] == 12
    assert stats["reject_not_tested_VGLUT_TRANSCRIPT_ABUNDANCE"] == 1


def test_canonical_preserves_formal_lmm_delta_without_fabricating_se() -> None:
    module = _module()
    config = _config()
    source = "A" * 120
    candidate = source[:59] + "C" + source[60:]
    unit = {
        "pair_id": "GENE_ssc",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "position_zero_based": 59,
        "biological_context_id": "N2A_TRANSCRIPT_ABUNDANCE",
        "logFC": -0.35,
        "pval": 0.1,
        "fdr": 0.2,
        "bonferroni": 1.0,
    }
    record = module._canonical_records(config, [unit])[0]
    assert record["direction_normalized_delta"] == -0.35
    assert record["source_endpoint_value"] is None and record["candidate_endpoint_value"] is None
    assert record["biological_standard_error"] is None
    assert record["standard_error_status"] == "PUBLISHED_LMM_EFFECT_SE_NOT_REPORTED"
    assert record["edit_operations"] == [{"type": "SUB", "position_zero_based": 59, "ref": "A", "alt": "C"}]
    assert record["confirmatory_evaluation_eligible"] is False


def test_config_keeps_development_no_credit_and_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["development_policy"]["training_eligible"] is True
    assert config["development_policy"]["confirmatory_evaluation_eligible"] is False
    assert config["development_policy"]["exact_sequence_pair_group_binding_required"] is True
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(config, tmp_path / "missing.tar", tmp_path / "missing.xlsx", output)
    assert marker.read_text(encoding="utf-8") == "keep"
