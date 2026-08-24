from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test.py"
)
PROTOCOL = (
    REPO_ROOT
    / "configs/route_a_v3_route2_xeditcritic_v4_frozen_test_protocol_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("atomic_xeditcritic_v4_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_three_seed_gate() -> dict[str, object]:
    return {
        "status": "XEDITCRITIC_V4_THREE_SEED_PASS",
        "required_seeds": [20260908, 20260909, 20260910],
        "development_test_authorized": True,
        "atomic_development_test_only": True,
        "additional_seed_authorized": False,
    }


def test_v4_atomic_authorization_requires_exact_three_seed_pass() -> None:
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = _passing_three_seed_gate()
    assert module.require_atomic_test_authorization_v4(protocol, gate) == (
        20260908,
        20260909,
        20260910,
    )

    gate["status"] = "XEDITCRITIC_V4_THREE_SEED_NO_GO"
    with pytest.raises(module.AtomicFrozenTestV4Error, match="does not authorize"):
        module.require_atomic_test_authorization_v4(protocol, gate)

    gate = _passing_three_seed_gate()
    gate["required_seeds"] = [20260908, 20260909, 20260910, 20260911]
    with pytest.raises(module.AtomicFrozenTestV4Error, match="does not authorize"):
        module.require_atomic_test_authorization_v4(protocol, gate)


def _prediction_rows(seed_offset: float) -> list[dict[str, object]]:
    return [
        {
            "record_id": f"record-{index}",
            "source_group_id": f"source-{index}",
            "task_id": "task-a",
            "target": float(index),
            "scaled_target": float(index),
            "prediction": float(index) + seed_offset,
            "scaled_prediction": float(index) + seed_offset,
        }
        for index in range(3)
    ]


def test_v4_atomic_ensemble_aligns_three_seed_predictions() -> None:
    module = _module()
    rows, metrics = module._ensemble_prediction_rows(
        {
            20260908: _prediction_rows(0.0),
            20260909: _prediction_rows(0.3),
            20260910: _prediction_rows(0.6),
        }
    )
    assert len(rows) == 3
    assert rows[1]["prediction"] == pytest.approx(1.3)
    assert rows[1]["per_seed_predictions"] == {
        "20260908": 1.0,
        "20260909": 1.3,
        "20260910": 1.6,
    }
    assert rows[1]["ensemble_sd"] > 0
    assert metrics["task_macro_spearman"] == pytest.approx(1.0)


def test_v4_atomic_ensemble_rejects_record_or_target_mismatch() -> None:
    module = _module()
    per_seed = {
        20260908: _prediction_rows(0.0),
        20260909: _prediction_rows(0.1),
        20260910: _prediction_rows(0.2),
    }
    per_seed[20260910] = per_seed[20260910][:-1]
    with pytest.raises(module.AtomicFrozenTestV4Error, match="records differ"):
        module._ensemble_prediction_rows(per_seed)

    per_seed[20260910] = _prediction_rows(0.2)
    per_seed[20260910][1]["target"] = 99.0
    with pytest.raises(module.AtomicFrozenTestV4Error, match="field differs"):
        module._ensemble_prediction_rows(per_seed)


def test_v4_atomic_runner_has_no_persistent_test_projection_or_cache_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_authorized_test_rows_v3" in source
    assert "assemble_frozen_bottom_encoder_chunk_cache_v4" in source
    assert "torch.save" not in source
    assert '"general_test_projection_persisted": False' in source
    assert '"test_bottom_six_cache_persisted": False' in source
    assert "authorization_consumed.json" in source
    assert "posttest_authorization_receipt.json" in source
    assert '"development_test_metrics_in_receipt": False' in source


def test_v4_atomic_test_terminal_files_are_atomic_and_exact_one(
    tmp_path: Path,
) -> None:
    module = _module()
    result_path = tmp_path / "atomic_frozen_test.json"
    payload = {"status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL"}
    module._write_atomic_once(result_path, payload)
    assert json.loads(result_path.read_text(encoding="utf-8")) == payload
    assert not result_path.with_suffix(result_path.suffix + ".partial").exists()

    partial = tmp_path / "failure.json.partial"
    partial.write_text("interrupted", encoding="utf-8")
    with pytest.raises(module.AtomicFrozenTestV4Error, match="partial artifact"):
        module._write_atomic_once(tmp_path / "failure.json", payload)

    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if not (output_directory / "atomic_frozen_test.json").exists()' in source
    assert "_write_atomic_once(output_directory / \"atomic_frozen_test.json\", result)" in source
