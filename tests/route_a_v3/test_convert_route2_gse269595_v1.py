from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from openpyxl import Workbook


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse269595_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse269595_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse269595_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_keeps_exposed_library_in_development_without_credit() -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["study"]["pool_assignment"] == "DEVELOPMENT"
    assert config["development_policy"]["library_selection_exposure"] == "APARENT_AND_MEASURED_RESPONSE_GUIDED"
    assert config["development_policy"]["unseen_evaluation_eligible"] is False
    assert config["development_policy"]["publisher_exact_censor_universe_status"] == "NOT_CLAIMED_DEVELOPMENT_RULE_EXPLICIT"
    assert config["input"]["expected_processed_subaim_label_mismatch_row_count"] == 47880
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())


def test_endpoint_formula_distinguishes_finite_infinite_and_undefined() -> None:
    module = _module()
    assert module._endpoint_formula_matches(20, 8, 12, str(math.log2(12 / 8)))
    assert module._endpoint_formula_matches(20, 0, 20, "Inf")
    assert module._endpoint_formula_matches(20, 20, 0, "-Inf")
    assert module._endpoint_formula_matches(0, 0, 0, "NA")
    assert not module._endpoint_formula_matches(20, 8, 12, "0")


def test_library_loader_removes_barcode_and_recovers_dense_sub_family(tmp_path: Path) -> None:
    module = _module()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "MPRA oligo library"
    sheet.append(module.LIBRARY_HEADER)
    source = "A" * 164
    designs = [
        ("wt", source),
        ("candidate_1", "C" + source[1:]),
        ("candidate_2", source[:10] + "C" + source[11:]),
        ("candidate_3", source[:20] + "C" + source[21:]),
    ]
    for index, (experiment, sequence) in enumerate(designs, start=1):
        barcode = ("ACGT" * 5)[index:] + ("ACGT" * 5)[:index]
        sheet.append(["GENE", "PAS", "aim", "subaim", experiment, 1, barcode + sequence])
    path = tmp_path / "library.xlsx"
    workbook.save(path)
    library, stats = module._load_library(path, "MPRA oligo library")
    assert stats["library_member_count"] == 4
    assert stats["eligible_dense_family_count"] == 1
    assert stats["eligible_candidate_design_count"] == 3
    assert library["family_sources"][("GENE", "PAS")] == {source}
    assert library["edit_histogram"] == {"1": 3}


def test_eligibility_builds_paired_delta_and_listwise_contexts() -> None:
    module = _module()
    config = _config()
    family = ("GENE", "PAS")
    source = "A" * 164
    candidates = {
        family + ("candidate_1",): "C" + source[1:],
        family + ("candidate_2",): source[:10] + "CC" + source[12:],
        family + ("candidate_3",): source[:20] + "CCC" + source[23:],
    }
    source_design = family + ("wt",)
    library = {
        "design_sequences": {source_design: source, **candidates},
        "design_metadata": {design: ("aim", "subaim", 1) for design in [source_design, *candidates]},
        "family_candidates": {family: set(candidates)},
        "family_sources": {family: {source}},
        "eligible_families": {family},
    }
    pooled = defaultdict(lambda: (0, 0, 0))
    for perturbation in config["study"]["biological_context_perturbations"]:
        for distal in config["study"]["distal_reporter_contexts"]:
            for replicate in config["endpoint_policy"]["biological_replicates"]:
                pooled[source_design + (perturbation, distal, replicate)] = (200, 100, 100)
                for design in candidates:
                    pooled[design + (perturbation, distal, replicate)] = (200, 80, 120) if replicate == "rep1" else (200, 70, 130)
    units, stats = module._eligible_units(config, library, pooled)
    assert len(units) == 45
    assert stats["candidate_context_universe_count"] == 45
    assert stats["canonical_record_count"] == 45
    assert stats["source_family_with_record_count"] == 1
    assert units[0]["source_endpoint_value"] == 0.0
    assert units[0]["delta"] > 0.0
    assert units[0]["standard_error"] > 0.0


def test_canonical_materializes_each_sequence_difference_as_ordered_sub() -> None:
    module = _module()
    config = _config()
    source = "A" * 164
    candidate = "C" + source[1:10] + "G" + source[11:]
    unit = {
        "family": ("GENE", "PAS"),
        "candidate_design": ("GENE", "PAS", "candidate"),
        "source_sequence": source,
        "candidate_sequence": candidate,
        "changes": [0, 10],
        "perturbation": "NT",
        "distal_context": "bGH",
        "source_endpoint_value": 0.0,
        "candidate_endpoint_value": 0.5,
        "delta": 0.5,
        "standard_error": 0.1,
        "type": "aim",
        "subtype": "subaim",
    }
    record = module._canonical_records(config, [unit])[0]
    assert record["edit_operations"] == [
        {"type": "SUB", "position_zero_based": 0, "ref": "A", "alt": "C"},
        {"type": "SUB", "position_zero_based": 10, "ref": "A", "alt": "G"},
    ]
    assert record["multi_step_sub_trajectory"] is True
    assert record["same_position_repeated_edit_required"] is False
    assert record["biological_standard_error"] == 0.1
    assert record["unseen_evaluation_eligible"] is False


def test_execute_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(config, tmp_path / "missing.xlsx", tmp_path / "missing.gz", output)
    assert marker.read_text(encoding="utf-8") == "keep"
