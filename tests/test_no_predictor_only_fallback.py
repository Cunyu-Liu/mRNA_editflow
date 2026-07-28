from pathlib import Path

import yaml

from scripts.contracts.audit_active_contracts import audit


ROOT = Path(__file__).resolve().parents[1]


def test_predictor_only_fallback_is_fail_closed():
    contract = yaml.safe_load(
        (ROOT / "configs/utr_editflow_contract_v2.yaml").read_text(encoding="utf-8")
    )
    assert contract["method"]["predictor_role"] == "support_only"
    assert contract["method"]["predictor_only_fallback_allowed"] is False
    assert audit(ROOT)["counters"]["active_predictor_only_fallback"] == 0
