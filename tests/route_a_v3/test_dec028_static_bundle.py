from __future__ import annotations
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def get_module():
    spec = importlib.util.spec_from_file_location("dec028_validator", ROOT / "scripts/route_a_v3/validate_dec028_static_bundle.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
def test_dec028_static_bundle_is_locked():
    assert get_module().validate(ROOT) == []
def test_dec028_protocol_cannot_train():
    data = json.loads((ROOT / "configs/route_a_v3_dec028_single_study_protocol_v1.json").read_text())
    assert data["authority"]["data_row_access_allowed"] is False
    assert data["authority"]["cuda_probe_allowed"] is False
    assert data["authority"]["training_allowed"] is False
    assert data["authority"]["g1_launched"] is False
