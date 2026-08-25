from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.adjudicate_route2_xeditcritic_v3_c3_v4_reference import (
    _consumption_marker_path,
    adjudicate_c3_v4_reference,
    run,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
RUN_IDS = (
    "c3",
    "c3_source_only",
    "c3_edit_metadata_only",
    "c3_no_candidate_sequence",
    "c3_candidate_bundle_permutation",
)


def _summary(run_id: str, rho: float = 0.31) -> dict[str, object]:
    return {
        "status": "TERMINAL_SCREEN_ARM_COMPLETE",
        "run_id": run_id,
        "control_mode": "NONE",
        "candidate_bundle_permutation": run_id == "c3_candidate_bundle_permutation",
        "final_validation": {"task_macro_spearman": rho},
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _payloads() -> dict[str, tuple[str, dict[str, object]]]:
    return {run_id: ("SUMMARY", _summary(run_id)) for run_id in RUN_IDS}


def test_read_once_decision_uses_terminal_c3_and_never_authorizes_downstream() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = adjudicate_c3_v4_reference(config, _payloads(), None)
    assert result["status"] == "C3_V4_REFERENCE_READ_ONCE_COMPLETE"
    assert result["terminal_summaries_read_count"] == 5
    assert result["c3_reference_task_macro_spearman"] == 0.31
    assert result["predeclared_fallback_used"] is False
    assert result["c3_confirmation_authorized"] is False
    assert result["c3_development_test_authorized"] is False


def test_c3_technical_failure_uses_only_predeclared_valid_fallback() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payloads = _payloads()
    payloads["c3"] = (
        "FAILURE",
        {
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        },
    )
    fallback = _summary("c2", rho=0.1042656112)
    result = adjudicate_c3_v4_reference(config, payloads, fallback)
    assert result["c3_full_technical_failure"] is True
    assert result["predeclared_fallback_used"] is True
    assert result["c3_reference_task_macro_spearman"] == pytest.approx(0.1042656112)
    with pytest.raises(Exception, match="fallback is absent"):
        adjudicate_c3_v4_reference(config, payloads, None)


def test_read_once_decision_rejects_incomplete_or_protected_package() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    incomplete = _payloads()
    incomplete.pop("c3_source_only")
    with pytest.raises(Exception, match="exact five"):
        adjudicate_c3_v4_reference(config, incomplete, None)
    protected = copy.deepcopy(_payloads())
    protected["c3_edit_metadata_only"][1]["development_test_outcomes_accessed"] = True
    with pytest.raises(Exception, match="protected outcome"):
        adjudicate_c3_v4_reference(config, protected, None)


def test_runtime_resolves_all_five_terminal_paths_before_reading(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = tmp_path / "screen"
    for run_id in RUN_IDS:
        directory = root / run_id
        directory.mkdir(parents=True)
        (directory / "run_summary.json").write_text(
            json.dumps(_summary(run_id)), encoding="utf-8"
        )
    config["c3_reference"]["preferred_terminal_summary"] = str(
        root / "c3" / "run_summary.json"
    )
    config["c3_reference"]["predeclared_fallback_terminal_summary"] = str(
        tmp_path / "c2" / "run_summary.json"
    )
    config["c3_read_once_reference_adjudication"] = str(
        root / "c3_v4_reference_read_once.json"
    )
    result = run(config)
    assert result["terminal_summaries_read_count"] == 5
    output = Path(config["c3_read_once_reference_adjudication"])
    assert output.exists()
    marker = _consumption_marker_path(output)
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload["status"] == (
        "C3_V4_REFERENCE_TERMINAL_CONSUMPTION_STARTED"
    )
    assert marker_payload["terminal_payload_content_included"] is False
    assert marker_payload["automatic_retry_if_reference_absent"] is False
    assert set(marker_payload["terminal_artifacts"]) == set(RUN_IDS)
    with pytest.raises(Exception, match="already exists"):
        run(config)

    missing_config = copy.deepcopy(config)
    missing_config["c3_read_once_reference_adjudication"] = str(
        root / "second_reference.json"
    )
    (root / "c3_source_only" / "run_summary.json").unlink()
    with pytest.raises(Exception, match="exactly one terminal"):
        run(missing_config)
    assert not Path(missing_config["c3_read_once_reference_adjudication"]).exists()
    assert not _consumption_marker_path(
        Path(missing_config["c3_read_once_reference_adjudication"])
    ).exists()


def test_runtime_fails_closed_after_consumption_started_without_reference(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = tmp_path / "screen"
    for run_id in RUN_IDS:
        directory = root / run_id
        directory.mkdir(parents=True)
        (directory / "run_summary.json").write_text(
            json.dumps(_summary(run_id)), encoding="utf-8"
        )
    config["c3_reference"]["preferred_terminal_summary"] = str(
        root / "c3" / "run_summary.json"
    )
    output = root / "c3_v4_reference_read_once.json"
    config["c3_read_once_reference_adjudication"] = str(output)
    marker = _consumption_marker_path(output)
    marker.write_text("consumption began\n", encoding="utf-8")

    with pytest.raises(Exception, match="automatic reread is forbidden"):
        run(config)
    assert not output.exists()
