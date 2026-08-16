from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_emtab10902_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_emtab10902_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_config() -> dict:
    return {
        "study": {
            "study_unit_id": "E-MTAB-10902",
            "independent_study_group_id": "E-MTAB-10902",
            "publication_doi": "10.1038/s41593-022-01243-x",
            "pool_assignment": "EVALUATION",
            "qualification_class": "EVALUATION_RESERVED",
            "study_role": "TRUE_A2_EXPLORATORY_ZERO_SHOT",
            "region": "3UTR",
            "biological_context_id": "E-MTAB-10902|primary_mouse_cortical_neuron|WT|secondary_Nzip",
            "assay_id": "E-MTAB-10902_NZIP_NEURITE_SOMA",
            "assay_type": "MPRNA_NZIP_NEURITE_SOMA",
            "endpoint_id": "E-MTAB-10902|mean_log2ratio_neurite_soma_WT",
            "endpoint_direction": "HIGHER_IS_BETTER",
        },
        "input": {
            "expected_design_row_count": 2,
            "expected_publisher_qc_passed_design_count": 2,
        },
        "evaluation_policy": {
            "outcome_access_stage": "CONVERSION_ONLY_UNTIL_PREDICTOR_GENERATOR_AND_BASELINES_FROZEN",
            "training_eligible": False,
            "model_selection_eligible": False,
            "hpo_eligible": False,
            "threshold_selection_eligible": False,
            "zero_shot_result_recorded": False,
        },
        "credit_policy": {"measured_candidate": True, "generated_candidate": False},
    }


def test_canonical_record_replays_all_substitutions_and_keeps_evaluation_closed() -> None:
    module = _module()
    source = "A" * 85
    candidate = "C" + source[1:10] + "G" + source[11:]
    unit = {
        "family": ("GENE", "Gene", "7"),
        "source_design_row_number": 1,
        "source_sequence_id": "NZSEQ00001",
        "source_sequence": source,
        "source_endpoint_value": -0.25,
        "candidate_design_row_numbers": [2, 3],
        "candidate_sequence_id": "NZSEQ00002",
        "candidate_sequence": candidate,
        "candidate_endpoint_value": 0.5,
        "mutation_labels": ["kmer10|1 - 10", "mut|10"],
        "changes": [0, 10],
    }
    record = module._canonical_records(_minimal_config(), [unit])[0]
    assert record["edit_operations"] == [
        {"type": "SUB", "position_zero_based": 0, "ref": "A", "alt": "C"},
        {"type": "SUB", "position_zero_based": 10, "ref": "A", "alt": "G"},
    ]
    assert record["direction_normalized_delta"] == 0.75
    assert record["candidate_metadata"]["identical_design_row_count"] == 2
    assert record["biological_standard_error"] is None
    assert record["training_eligible"] is False
    assert record["zero_shot_result_recorded"] is False


def test_qc_loader_requires_outcome_blind_exact_publisher_count(tmp_path: Path) -> None:
    module = _module()
    config = _minimal_config()
    summary = tmp_path / "summary.json"
    membership = tmp_path / "membership.jsonl"
    summary.write_text(json.dumps({
        "status": "PUBLISHER_READ_QC_REPRODUCED",
        "evaluation_outcome_sheet_read": False,
        "passed_design_row_count": 2,
    }), encoding="utf-8")
    membership.write_text(
        json.dumps({"design_row_number": 1, "sequence_id": "NZSEQ00001", "passes_publisher_read_qc": True}) + "\n"
        + json.dumps({"design_row_number": 2, "sequence_id": "NZSEQ00002", "passes_publisher_read_qc": True}) + "\n",
        encoding="utf-8",
    )
    loaded = module._load_qc(summary, membership, config)
    assert sorted(loaded) == [1, 2]
    bad = json.loads(summary.read_text(encoding="utf-8"))
    bad["evaluation_outcome_sheet_read"] = True
    summary.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(module.ConversionError, match="used Evaluation outcomes"):
        module._load_qc(summary, membership, config)
