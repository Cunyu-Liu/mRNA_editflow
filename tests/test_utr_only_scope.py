from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_current_scope_is_exactly_two_utr_regions():
    contract = yaml.safe_load(
        (ROOT / "configs/utr_editflow_execution_policy.yaml").read_text(encoding="utf-8")
    )
    scope = contract["current_scope"]
    assert scope["regions"] == ["five_utr", "three_utr"]
    assert scope["cds"] == "forbidden"
    assert scope["full_length"] == "forbidden"
    assert scope["new_wetlab"] == "forbidden"
    assert contract["future_scope"]["user_approval_required"] is True
