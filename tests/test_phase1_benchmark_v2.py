import json
import sys

import pytest

sys.path.insert(0, "scripts")

from data.nmi_benchmark_v2 import (
    FinalTestAccessError,
    REQUIRED_SOURCE_MATCHED_FIELDS,
    iter_role_records,
    load_manifest,
)


def _make_root(tmp_path):
    root = tmp_path / "nmi"
    (root / "manifests").mkdir(parents=True)
    (root / "indices").mkdir()
    record = {
        "record_id": "r1",
        "task_kind": "local_delta",
        "confidence": "measured",
        "measured_delta": 1.0,
        **{field: None for field in REQUIRED_SOURCE_MATCHED_FIELDS},
    }
    record.update({"source_id": "s1", "candidate_id": "c1", "edit_list": [], "edit_count": 0})
    (root / "records.jsonl").write_text(json.dumps(record) + "\n")
    for role, final in (("val", False), ("test_id", True)):
        (root / "indices" / f"{role}.txt").write_text("r1\n")
        (root / "manifests" / f"{role}.json").write_text(json.dumps({
            "role": role,
            "final_test": final,
            "records_path": "records.jsonl",
            "index_path": f"indices/{role}.txt",
        }))
    return root


def test_final_role_is_fail_closed_and_task_filter_works(tmp_path):
    root = _make_root(tmp_path)
    with pytest.raises(FinalTestAccessError):
        load_manifest(root / "manifests/test_id.json")
    records = list(iter_role_records(root / "manifests/val.json", task_kind="local_delta"))
    assert [r["record_id"] for r in records] == ["r1"]


def test_required_source_matched_contract_is_explicit():
    assert REQUIRED_SOURCE_MATCHED_FIELDS == (
        "source_id", "candidate_id", "source_sequence", "candidate_sequence",
        "edit_list", "edit_count", "measured_source", "measured_candidate",
        "measured_delta", "cargo", "cell_context", "assay", "batch", "replicate",
    )


def test_budget_one_does_not_use_multi_edit_candidate():
    from evaluate_benchmark_v2 import budget_metrics

    records = [
        {"source_id": "s", "edit_count": 1, "measured_delta": 0.2},
        {"source_id": "s", "edit_count": 2, "measured_delta": 0.9},
    ]
    result = budget_metrics(records, [0.1, 1.0], budget=1)
    assert result["final_delta_mean"] == 0.2
    assert result["reference_is_not_full_legal_dp_or_beam"] is True


def test_abstention_auc_detects_high_error_when_uncertainty_is_informative():
    from evaluate_benchmark_v2 import metrics

    result = metrics([0.0, 0.9, 0.1, 0.8], [0.1, 0.9, 0.2, 0.8], [0.0, 0.0, 0.1, 0.1])
    assert result["abstention_auroc"] is not None
    assert result["abstention_auroc"] >= 0.75
