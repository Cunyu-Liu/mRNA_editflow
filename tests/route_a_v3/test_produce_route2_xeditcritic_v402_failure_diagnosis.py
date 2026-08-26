from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.produce_route2_xeditcritic_v402_failure_diagnosis import (
    RUN_IDS,
    consumption_marker_path,
    run,
)


def _failure(run_id: str) -> dict[str, object]:
    return {
        "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
        "run_id": run_id,
        "exception_type": "RuntimeError",
        "exception_message": "reachable technical failure",
    }


def _screen_root(tmp_path: Path) -> Path:
    root = tmp_path / "screen"
    for run_id in RUN_IDS:
        directory = root / run_id
        directory.mkdir(parents=True)
        (directory / "failure.json").write_text(
            json.dumps(_failure(run_id)), encoding="utf-8"
        )
    return root


def test_failure_diagnosis_consumes_exact_eight_failure_only_terminals(
    tmp_path: Path,
) -> None:
    root = _screen_root(tmp_path)
    output = tmp_path / "diagnosis.json"
    result = run(screen_root=root, output=output)
    assert result["status"] == (
        "XEDITCRITIC_V402_FAILURE_DIAGNOSIS_READ_ONCE_COMPLETE"
    )
    assert result["terminal_failure_payloads_read_count"] == 8
    assert result["terminal_summary_artifacts_present"] == 0
    assert result["valid_validation_performance_summary_present"] is False
    assert set(result["failure_payloads"]) == set(RUN_IDS)
    marker = json.loads(
        consumption_marker_path(output).read_text(encoding="utf-8")
    )
    assert marker["terminal_payload_content_included"] is False
    assert marker["automatic_retry_if_diagnosis_absent"] is False
    with pytest.raises(Exception, match="already exists"):
        run(screen_root=root, output=output)


def test_failure_diagnosis_does_not_consume_incomplete_package(
    tmp_path: Path,
) -> None:
    root = _screen_root(tmp_path)
    (root / RUN_IDS[-1] / "failure.json").unlink()
    output = tmp_path / "diagnosis.json"
    with pytest.raises(Exception, match="failure-only terminal"):
        run(screen_root=root, output=output)
    assert not output.exists()
    assert not consumption_marker_path(output).exists()


def test_failure_diagnosis_rejects_summary_plus_failure(
    tmp_path: Path,
) -> None:
    root = _screen_root(tmp_path)
    (root / RUN_IDS[0] / "run_summary.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "diagnosis.json"
    with pytest.raises(Exception, match="failure-only terminal"):
        run(screen_root=root, output=output)
    assert not consumption_marker_path(output).exists()
