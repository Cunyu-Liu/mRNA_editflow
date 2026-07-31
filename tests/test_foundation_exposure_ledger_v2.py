from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data_registry/foundation_exposure_ledger_v2.yaml"
REQUIRED = {
    "model",
    "checkpoint",
    "checkpoint_sha256",
    "license",
    "pretraining_corpus_known",
    "pretraining_corpus_version",
    "sequence_overlap_status",
    "downstream_label_overlap_status",
    "published_task_head_used",
    "allowed_claim",
}


def test_c0_d0_foundation_ledger_is_fail_closed():
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    assert ledger["contract_id"] == "mrna_editflow_single_active_contract"
    assert ledger["selection_status"] == "NO_FOUNDATION_SELECTED"
    assert ledger["weights_downloaded_in_c0_d0"] is False
    assert set(ledger["required_fields"]) == REQUIRED
    assert len(ledger["candidates"]) >= 4
    for record in ledger["candidates"]:
        assert REQUIRED <= set(record)
        assert record["checkpoint"] is None
        assert record["checkpoint_sha256"] is None
        assert record["sequence_overlap_status"] == "UNKNOWN"
        assert record["downstream_label_overlap_status"] == "UNKNOWN"
        assert record["published_task_head_used"] is False
        assert "NONE_BEFORE_FM0_AUDIT" in record["allowed_claim"]


def test_active_contract_points_to_the_ledger():
    contract = yaml.safe_load(
        (ROOT / "configs/utr_editflow_execution_policy.yaml").read_text(encoding="utf-8")
    )
    assert contract["foundation_strategy"]["reuse_first"] is True
    assert contract["foundation_strategy"]["exposure_ledger_required"] is True
    assert contract["foundation_strategy"]["exposure_ledger"] == (
        "data_registry/foundation_exposure_ledger_v2.yaml"
    )
