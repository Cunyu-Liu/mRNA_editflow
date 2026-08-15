from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/route_a_v3/record_dec028_ss1_a2_g0_static_validation.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("record_dec028_ss1_a2_g0_static_validation_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_record_writes_only_aggregate_static_audit_files(tmp_path: Path, capsys) -> None:
    runner = _load()
    run_root = tmp_path / "DEC028_SS1_A2_G0_STATIC_VALIDATION_TEST"
    assert runner.main(["--repo-root", str(ROOT), "--run-root", str(run_root)]) == 0
    acceptance_from_stdout = json.loads(capsys.readouterr().out)
    acceptance = json.loads((run_root / "ACCEPTANCE.json").read_text(encoding="utf-8"))
    payload = json.loads((run_root / "A2_G0_STATIC_VALIDATION.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))

    assert acceptance_from_stdout == acceptance
    assert acceptance["status"] == "PASS_STATIC_SYNTHETIC_G0_ONLY"
    assert acceptance["next_phase"] == "INDEPENDENT_IMPLEMENTATION_REVIEW_REQUIRED"
    assert acceptance["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert acceptance["ss1_complete"] is False
    assert payload["project_rows_accessed"] is False
    assert payload["private_rows_accessed"] is False
    assert payload["sealed_rows_accessed"] is False
    assert payload["cuda_probed"] is False
    assert payload["model_or_optimizer_constructed"] is False
    assert payload["parameter_updates"] == 0
    assert payload["result"]["component_disjoint"] is True
    assert payload["result"]["candidate_minus_source_valid"] is True
    assert manifest["outputs"] == [
        "A2_G0_STATIC_VALIDATION.json",
        "RUN_MANIFEST.json",
        "ACCEPTANCE.json",
        "RUN_LOG.md",
    ]
    assert sorted(path.name for path in run_root.iterdir()) == sorted(manifest["outputs"])


def test_record_refuses_to_overwrite_an_existing_run_directory(tmp_path: Path) -> None:
    runner = _load()
    run_root = tmp_path / "existing"
    runner.record(ROOT, run_root)
    with pytest.raises(FileExistsError):
        runner.record(ROOT, run_root)
