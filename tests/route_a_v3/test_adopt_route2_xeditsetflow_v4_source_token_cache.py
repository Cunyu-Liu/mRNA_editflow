from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.route2_source_token_cache_v3 import assemble_source_token_cache_v3
from scripts.route_a_v3 import adopt_route2_xeditsetflow_v4_source_token_cache
from scripts.route_a_v3.authorize_route2_xedit_v4_screen_stages import (
    build_cache_launch_authorization_v4,
)


ROOT = Path(__file__).resolve().parents[2]
HEAD = "a" * 40


def _c3() -> dict[str, object]:
    return {
        "status": "C3_V4_REFERENCE_READ_ONCE_COMPLETE",
        "terminal_summaries_read_count": 5,
        "c3_terminal_artifacts_retained": True,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _a100() -> dict[str, object]:
    return {
        "repository_sync": {
            "head_after": HEAD,
            "old_launch_jobs_active_before_sync": False,
            "remote_worktree_clean_after": True,
            "shared_history_rewritten": False,
        },
        "a100_current_head_verification": {
            "verified_git_head": HEAD,
            "critic_focused_total_passed": 1,
            "critic_focused_failed": 0,
            "setflow_focused_passed": 1,
            "setflow_focused_failed": 0,
            "exact_v332_passed": 96,
            "exact_v332_failed": 0,
        },
        "protected_data": {
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        },
    }


def _authorization() -> dict[str, object]:
    return build_cache_launch_authorization_v4(
        "setflow", _c3(), _a100(), current_git_head=HEAD
    )


def _legacy_summary() -> dict[str, object]:
    return {
        "schema_version": "route_a_v3_route2_setflow_source_token_cache_summary.v3",
        "status": "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE",
        "projection_record_count": 1,
        "eligible_record_count": 1,
        "unique_source_count": 1,
        "unique_source_token_count": 2,
        "maximum_source_length": 2,
        "embedding_width": 3,
        "model_id": "model-revision",
        "raw_sequence_payload_written": 0,
        "outcome_value_access_count": 0,
        "development_test_record_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_record_count": 0,
        "evaluation_outcomes_accessed": False,
        "output_path": "",
    }


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "legacy_cache_path": str(tmp_path / "source_token_cache_v1.pt"),
        "legacy_summary_path": str(tmp_path / "source_token_cache_v1.summary.json"),
        "adoption_receipt_path": str(tmp_path / "adoption.json"),
        "expected_model_id": "model-revision",
        "expected_projection_record_count": 1,
        "expected_eligible_record_count": 1,
        "expected_unique_source_count": 1,
        "expected_token_count": 2,
        "expected_maximum_source_length": 2,
        "expected_embedding_width": 3,
        "legacy_artifact_policy": "READ_ONLY_NO_REBUILD_NO_OVERWRITE",
        "encoder_forward_count": 0,
        "parameter_update_count": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _write_legacy_artifacts(config: dict[str, object]) -> tuple[bytes, str]:
    cache_path = Path(str(config["legacy_cache_path"]))
    summary_path = Path(str(config["legacy_summary_path"]))
    payload = assemble_source_token_cache_v3(
        [{"canonical_record_id": "record-1", "source_sequence": "AC"}],
        sequence_to_index={"AC": 0},
        encoded_tokens={0: torch.zeros((2, 3), dtype=torch.float32)},
        model_id="model-revision",
        pretrained_parameter_count=1,
        attention_backend="TEST",
    )
    torch.save(payload, cache_path)
    summary = _legacy_summary()
    summary["output_path"] = str(cache_path)
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    return cache_path.read_bytes(), summary_text


def test_adoption_validates_actual_payload_and_preserves_terminal_v3_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    cache_before, summary_before = _write_legacy_artifacts(config)
    monkeypatch.setattr(
        adopt_route2_xeditsetflow_v4_source_token_cache,
        "_git_head",
        lambda: HEAD,
    )
    result = adopt_route2_xeditsetflow_v4_source_token_cache.adopt(
        config, _authorization()
    )
    assert result["status"] == "XEDITSETFLOW_V4_SOURCE_CACHE_ADOPTED_READ_ONLY"
    assert result["source_token_cache_identity"] == {
        "model_id": "model-revision",
        "record_count": 1,
        "unique_source_count": 1,
        "token_count": 2,
        "maximum_source_length": 2,
        "embedding_width": 3,
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "chunk_policy": "ONE_COMPLETE_CHUNK_MAXIMUM_1000_NUCLEOTIDES",
    }
    assert Path(str(config["legacy_cache_path"])).read_bytes() == cache_before
    assert Path(str(config["legacy_summary_path"])).read_text() == summary_before
    assert result["legacy_payload_modified"] is False
    assert result["legacy_summary_modified"] is False
    with pytest.raises(Exception, match="receipt already exists"):
        adopt_route2_xeditsetflow_v4_source_token_cache.adopt(
            config, _authorization()
        )


def test_adoption_rejects_protected_or_drifted_legacy_summary_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_legacy_artifacts(config)
    summary_path = Path(str(config["legacy_summary_path"]))
    summary = _legacy_summary()
    summary["output_path"] = str(config["legacy_cache_path"])
    summary["development_test_outcomes_accessed"] = True
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        adopt_route2_xeditsetflow_v4_source_token_cache,
        "_git_head",
        lambda: HEAD,
    )
    called = False

    def forbidden_loader(path: Path) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError(path)

    monkeypatch.setattr(
        adopt_route2_xeditsetflow_v4_source_token_cache,
        "load_source_token_cache_v3",
        forbidden_loader,
    )
    with pytest.raises(Exception, match="not outcome isolated"):
        adopt_route2_xeditsetflow_v4_source_token_cache.adopt(
            config, _authorization()
        )
    assert called is False


def test_adoption_requires_authorization_before_opening_terminal_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adopt_route2_xeditsetflow_v4_source_token_cache,
        "_git_head",
        lambda: HEAD,
    )
    with pytest.raises(Exception, match="launch authorization is absent"):
        adopt_route2_xeditsetflow_v4_source_token_cache.adopt({}, {})


def test_frozen_screen_config_points_to_read_only_adoption_receipt() -> None:
    screen = json.loads(
        (ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json").read_text()
    )
    adoption = json.loads(
        (ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_source_cache_adoption_v1.json").read_text()
    )
    assert screen["source_token_cache_adoption_receipt_path"] == adoption[
        "adoption_receipt_path"
    ]
    assert adoption["legacy_artifact_policy"] == "READ_ONLY_NO_REBUILD_NO_OVERWRITE"
    assert adoption["encoder_forward_count"] == 0
    assert adoption["parameter_update_count"] == 0
