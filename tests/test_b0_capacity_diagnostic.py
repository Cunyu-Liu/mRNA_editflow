from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.data.diagnose_b0_path_capacity as capacity_diagnostic
from scripts.data.diagnose_b0_path_capacity import (
    CapacityDiagnosticError,
)
from scripts.data.diagnose_b0_path_capacity import (
    FROZEN_B0_LIMITS,
)
from scripts.data.diagnose_b0_path_capacity import (
    _apply_diagnostic_scope,
)
from scripts.data.diagnose_b0_path_capacity import (
    _claim_boundary,
)
from scripts.data.diagnose_b0_path_capacity import (
    _create_exclusive_run_root,
)
from scripts.data.diagnose_b0_path_capacity import (
    _load_structural_selection,
)
from scripts.data.diagnose_b0_path_capacity import (
    _validate_config,
)
from scripts.data.diagnose_b0_path_capacity import (
    _validate_d1_snapshot_trust,
)
from scripts.data.diagnose_b0_path_capacity import (
    _write_replay_script,
)


WITNESS_ID = "GSE217518:record:025e56d3b64660abb559dcbd"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_sha(values: list[str]) -> str:
    return hashlib.sha256(
        (("\n".join(sorted(values)) + "\n") if values else "").encode("utf-8")
    ).hexdigest()


def _record(
    record_id: str,
    source: str,
    candidate: str,
    *,
    dataset_id: str = "GSE217518",
    canonical_split: str = "unassigned_B0",
) -> dict:
    edit_distance = abs(len(source) - len(candidate))
    if not edit_distance:
        edit_distance = sum(a != b for a, b in zip(source, candidate))
    return {
        "record_id": record_id,
        "dataset_id": dataset_id,
        "region": "five_utr",
        "source_id": f"{record_id}:source",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_distance": edit_distance,
        "edit_count": edit_distance,
        "edit_script": (
            [{"op": "DEL", "pos": 0, "ref": "A"}]
            if len(source) > len(candidate)
            else [{"op": "SUB", "pos": 0, "ref": "A", "alt": "C"}]
        ),
        "pair_type": "true_wt_mutant",
        "canonical_split": canonical_split,
        "paper_split": "unassigned_D1",
        "quality_flags": [],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _record("single", "AA", "CA"),
        _record(WITNESS_ID, "AAAA", "AA"),
        _record("multi", "AC", "CA"),
        _record(
            "retrospective",
            "AAAA",
            "AA",
            dataset_id="GSE246381",
        ),
    ]
    store = tmp_path / "candidates.jsonl"
    store.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="",
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "artifact_type": "d1_canonical_snapshot",
                "schema_version": "d1_canonical_snapshot.v1",
                "global_stores": {
                    "record_ids_sha256": _ids_sha([row["record_id"] for row in rows]),
                    "canonical_label_store": {
                        "path": str(tmp_path / "labels.jsonl"),
                        "bytes": 123,
                        "sha256": "a" * 64,
                        "records": len(rows),
                        "record_ids_sha256": _ids_sha(
                            [row["record_id"] for row in rows]
                        ),
                    },
                    "sealed_label_free_candidate_store": {
                        "path": str(store),
                        "bytes": store.stat().st_size,
                        "sha256": _sha(store),
                        "records": len(rows),
                        "record_ids_sha256": _ids_sha(
                            [row["record_id"] for row in rows]
                        ),
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return store, snapshot


def _config() -> dict:
    return {
        "schema_version": "b0_capacity_config.v1",
        "selection": {
            "regions": ["five_utr", "three_utr"],
            "minimum_edit_distance": 2,
        },
        "frozen_b0_limits": dict(FROZEN_B0_LIMITS),
        "diagnostic_safety_limits": {
            "max_dag_cells_per_record": 1_000_000,
            "max_reachable_states_per_record": 150_000,
            "max_neighbor_expansions_per_record": 10_000_000,
            "max_state_dp_cells_per_record": 100_000_000,
            "max_spill_bytes_per_record": 20_000_000_000,
            "chunk_size": 50_000,
            "max_open_chunks": 128,
            "minimum_free_bytes": 100_000_000_000,
            "max_rss_bytes": 64_000_000_000,
            "max_wall_seconds": 86_400,
            "max_spill_bytes": 200_000_000_000,
            "heartbeat_seconds": 300,
        },
    }


def test_structural_selection_is_label_free_accounted_and_witness_first(
    tmp_path: Path,
) -> None:
    store, snapshot = _fixture(tmp_path)
    selection = _load_structural_selection(
        candidate_store=store,
        d1_snapshot=snapshot,
        minimum_edit_distance=2,
        witness_record_id=WITNESS_ID,
    )

    assert selection.source_record_count == 4
    assert selection.split_eligible_record_count == 3
    assert selection.selected_record_count == 2
    assert selection.excluded_record_count == 2
    assert [row["record_id"] for row in selection.records] == [
        WITNESS_ID,
        "multi",
    ]
    assert selection.records[0]["_canonical_jsonl_line"] == 2
    assert selection.label_fields_read == ()
    assert selection.canonical_label_store_opened is False
    assert selection.record_ids_sha256 == _ids_sha(
        ["single", WITNESS_ID, "multi", "retrospective"]
    )
    assert selection.selected_record_ids_sha256 == _ids_sha([WITNESS_ID, "multi"])


def test_witness_scope_contains_only_witness_endpoints_and_cannot_be_global_census(
    tmp_path: Path,
) -> None:
    store, snapshot = _fixture(tmp_path)
    full = _load_structural_selection(
        candidate_store=store,
        d1_snapshot=snapshot,
        minimum_edit_distance=2,
        witness_record_id=WITNESS_ID,
    )
    witness = _apply_diagnostic_scope(full, scope="witness")

    assert [row["record_id"] for row in witness.records] == [WITNESS_ID]
    assert witness.eligible_endpoints == ("AA", "AAAA")
    assert set(witness.eligible_endpoints) < set(full.eligible_endpoints)
    assert witness.selected_record_count == 1
    assert witness.excluded_record_count == witness.source_record_count - 1
    assert witness.exclusion_reason_counts["scope_witness_only_unscheduled"] == 1


def test_structural_store_hash_record_count_and_id_binding_fail_closed(
    tmp_path: Path,
) -> None:
    store, snapshot = _fixture(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["global_stores"]["sealed_label_free_candidate_store"]["sha256"] = "0" * 64
    snapshot.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(CapacityDiagnosticError, match="structural store"):
        _load_structural_selection(
            candidate_store=store,
            d1_snapshot=snapshot,
            minimum_edit_distance=2,
            witness_record_id=WITNESS_ID,
        )


def test_unexpected_top_level_label_field_fails_closed(tmp_path: Path) -> None:
    store, snapshot = _fixture(tmp_path)
    rows = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    rows[0]["final_label"] = 1.0
    store.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    metadata = payload["global_stores"]["sealed_label_free_candidate_store"]
    metadata["bytes"] = store.stat().st_size
    metadata["sha256"] = _sha(store)
    snapshot.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(CapacityDiagnosticError, match="label"):
        _load_structural_selection(
            candidate_store=store,
            d1_snapshot=snapshot,
            minimum_edit_distance=2,
            witness_record_id=WITNESS_ID,
        )


def test_nested_label_path_inside_allowed_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    store, snapshot = _fixture(tmp_path)
    rows = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    rows[0]["endpoint_provenance"] = {
        "source": "fixture",
        "nested": {"delta_raw": 0.25},
    }
    store.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    metadata = payload["global_stores"]["sealed_label_free_candidate_store"]
    metadata["bytes"] = store.stat().st_size
    metadata["sha256"] = _sha(store)
    snapshot.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(CapacityDiagnosticError, match="recursively detected"):
        _load_structural_selection(
            candidate_store=store,
            d1_snapshot=snapshot,
            minimum_edit_distance=2,
            witness_record_id=WITNESS_ID,
        )


def test_d1_snapshot_trust_calls_exact_validator_and_records_opaque_hash_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    snapshot = project_root / "data/d1/manifests/snapshot.json"
    acceptance = project_root / "acceptance.json"
    git_dir = tmp_path / "gitdir"
    snapshot.parent.mkdir(parents=True)
    git_dir.mkdir()
    snapshot.write_text('{"snapshot":true}\n', encoding="utf-8", newline="")
    acceptance.write_text("{}\n", encoding="utf-8", newline="")

    def fake_git(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if "ls-files" in command:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=snapshot.read_bytes(),
            stderr=b"",
        )

    validator_calls: list[tuple[Path, Path]] = []

    def fake_validate_snapshot(
        snapshot_path: Path,
        *,
        repo_root: Path,
    ) -> list[str]:
        assert os.environ["GIT_DIR"] == str(git_dir.resolve())
        assert os.environ["GIT_WORK_TREE"] == str(project_root.resolve())
        validator_calls.append((snapshot_path, repo_root))
        return []

    monkeypatch.setattr(capacity_diagnostic.subprocess, "run", fake_git)
    monkeypatch.setattr(
        capacity_diagnostic,
        "validate_snapshot",
        fake_validate_snapshot,
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "validate_phase_acceptance",
        lambda *_args, **_kwargs: [],
    )
    evidence = _validate_d1_snapshot_trust(
        snapshot_path=snapshot,
        acceptance_path=acceptance,
        project_root=project_root,
        git_dir=git_dir,
    )

    assert validator_calls == [(snapshot, project_root)]
    assert evidence["status"] == "PASS"
    assert evidence["canonical_label_store_access"] == {
        "mode": "OPAQUE_SHA256_INTEGRITY_VALIDATION_ONLY",
        "opened_for_integrity_hash": True,
        "jsonl_parsed": False,
        "label_values_accessed": False,
        "used_for_selection": False,
        "used_for_capacity": False,
    }


def test_d1_snapshot_exact_validator_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    snapshot = project_root / "snapshot.json"
    acceptance = project_root / "acceptance.json"
    git_dir = tmp_path / "gitdir"
    project_root.mkdir()
    git_dir.mkdir()
    snapshot.write_text('{"snapshot":true}\n', encoding="utf-8", newline="")
    acceptance.write_text("{}\n", encoding="utf-8", newline="")

    def fake_git(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if "ls-files" in command:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=snapshot.read_bytes(),
            stderr=b"",
        )

    monkeypatch.setattr(capacity_diagnostic.subprocess, "run", fake_git)
    monkeypatch.setattr(
        capacity_diagnostic,
        "validate_snapshot",
        lambda *_args, **_kwargs: ["snapshot_differs_from_exact_live_recomputation"],
    )
    with pytest.raises(CapacityDiagnosticError, match="exact live recomputation"):
        _validate_d1_snapshot_trust(
            snapshot_path=snapshot,
            acceptance_path=acceptance,
            project_root=project_root,
            git_dir=git_dir,
        )


def test_replay_captures_cwd_and_replaces_equals_form_output_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    launch_cwd = tmp_path / "project"
    run_root.mkdir()
    launch_cwd.mkdir()
    _write_replay_script(
        run_root=run_root,
        argv=(
            "/runtime/python",
            "/project/diagnose.py",
            "--output-root=/old/run",
        ),
        launch_cwd=launch_cwd,
    )
    replay = (run_root / "replay.sh").read_text(encoding="utf-8")
    assert f"cd {launch_cwd}\n" in replay
    assert '--output-root="${1}"' in replay
    assert "/old/run" not in replay
    assert replay.count("${1}") == 1


def test_replay_rejects_missing_or_duplicate_output_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    launch_cwd = tmp_path / "project"
    run_root.mkdir()
    launch_cwd.mkdir()
    with pytest.raises(CapacityDiagnosticError, match="exactly one"):
        _write_replay_script(
            run_root=run_root,
            argv=("/runtime/python", "/project/diagnose.py"),
            launch_cwd=launch_cwd,
        )
    with pytest.raises(CapacityDiagnosticError, match="exactly one"):
        _write_replay_script(
            run_root=run_root,
            argv=(
                "/runtime/python",
                "/project/diagnose.py",
                "--output-root=/one",
                "--output-root",
                "/two",
            ),
            launch_cwd=launch_cwd,
        )


def test_config_separates_frozen_gates_from_diagnostic_safety_limits() -> None:
    resolved = _validate_config(_config())
    assert resolved["frozen_b0_limits"] == FROZEN_B0_LIMITS
    assert (
        resolved["diagnostic_safety_limits"]["max_reachable_states_per_record"]
        == 150_000
    )

    changed = _config()
    changed["frozen_b0_limits"]["max_reachable_states"] = 50_001
    with pytest.raises(CapacityDiagnosticError, match="frozen"):
        _validate_config(changed)


def test_claim_boundary_cannot_authorize_b0_or_budget_change() -> None:
    assert _claim_boundary() == {
        "formal_b0_attempt_started": False,
        "b0_gate_values_changed": False,
        "b0_phase_acceptance_claimed": False,
        "scientific_result_claimed": False,
        "budget_change_authorized": False,
        "approximation_emitted": False,
        "allowed_claim": "CAPACITY_DIAGNOSTIC_ONLY",
    }


def test_run_root_creation_is_exclusive(tmp_path: Path) -> None:
    fresh = tmp_path / "B0_capacity_20260729T150000Z_deadbee"
    _create_exclusive_run_root(fresh)
    assert fresh.is_dir()
    with pytest.raises(CapacityDiagnosticError, match="already exists"):
        _create_exclusive_run_root(fresh)


def test_dag_capacity_stop_emits_typed_lower_bound_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("dag-stop", "AAAA", "AA")
    record["_canonical_jsonl_line"] = 1
    run_root = tmp_path / "run"
    run_root.mkdir()

    def raise_dag_stop(*_args: object, **_kwargs: object) -> None:
        raise capacity_diagnostic.PathStateError(
            "minimum-alignment DAG exceeds the configured cell budget"
        )

    monkeypatch.setattr(
        capacity_diagnostic,
        "minimum_alignment_statistics",
        raise_dag_stop,
    )
    watchdog = SimpleNamespace(prototype_callback=lambda _progress: None)
    row, state_file = capacity_diagnostic._run_record(
        record=record,
        ordinal=0,
        run_root=run_root,
        safety=_config()["diagnostic_safety_limits"],
        watchdog=watchdog,
    )

    assert state_file is None
    assert row["outcome"] == "LOWER_BOUND_STOPPED"
    assert row["capacity"]["exact"] is None
    assert row["alignment_statistics"] == {
        "minimum_edit_count": 2,
        "minimum_alignment_count": None,
        "evaluated_dag_cell_count": 1_000_001,
        "counts_exact": False,
    }
    assert row["stop"]["dimension"] == "max_dag_cells"
    assert row["stop"]["observed_lower_bound"] == 1_000_001


@pytest.mark.parametrize(
    ("message", "dimension", "frozen_limit", "observed_lower_bound"),
    (
        (
            "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
            "closure exceeded 150000 reachable states; "
            "no approximation was emitted",
            "max_reachable_states",
            50_000,
            150_001,
        ),
        (
            "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
            "closure exceeded 10000000 evaluated primitive actions; "
            "no approximation was emitted",
            "max_neighbor_expansions",
            5_000_000,
            10_000_001,
        ),
        (
            "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
            "closure exceeded 100000000 state DP cells; "
            "no approximation was emitted",
            "max_state_dp_cells",
            50_000_000,
            100_000_001,
        ),
    ),
)
def test_streaming_capacity_stop_emits_typed_lower_bound_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    dimension: str,
    frozen_limit: int,
    observed_lower_bound: int,
) -> None:
    record = _record("streaming-stop", "AC", "CA")
    record["_canonical_jsonl_line"] = 1
    run_root = tmp_path / "run"
    run_root.mkdir()

    def raise_streaming_stop(*_args: object, **_kwargs: object) -> None:
        raise capacity_diagnostic.StreamingPathStateError(message)

    monkeypatch.setattr(
        capacity_diagnostic,
        "minimum_alignment_state_summary",
        raise_streaming_stop,
    )
    safety = _config()["diagnostic_safety_limits"]
    watchdog = SimpleNamespace(prototype_callback=lambda _progress: None)
    row, state_file = capacity_diagnostic._run_record(
        record=record,
        ordinal=0,
        run_root=run_root,
        safety=safety,
        watchdog=watchdog,
    )

    assert state_file is None
    assert row["outcome"] == "LOWER_BOUND_STOPPED"
    assert row["capacity"]["exact"] is None
    assert row["capacity"]["lower_bound"] == {
        "reachable_node_count": 1,
        "reachable_transition_count": 0,
        "minimum_state_path_count": 1,
        "evaluated_primitive_action_count": 0,
        "evaluated_state_dp_cell_count": 0,
    }
    assert row["state_universe_artifact"] is None
    assert row["alignment_statistics"]["counts_exact"] is True
    assert row["alignment_statistics"]["minimum_edit_count"] == 2
    assert row["evidence_semantics"] == {
        "counts_exact": False,
        "state_set_complete": False,
        "no_approximation_emitted": True,
        "usable_for_b0_acceptance": False,
    }
    assert row["stop"] == {
        "stop_rule": "STOP_RULE_B0_PATH_STATE_COMPLEXITY",
        "dimension": dimension,
        "frozen_limit": frozen_limit,
        "observed_lower_bound": observed_lower_bound,
        "message": message,
    }
    assert row["frozen_gate_assessment"] == {
        "would_pass_frozen_b0_limits": False,
        "exceeded_limits": [dimension],
    }


def test_non_streaming_non_dag_path_error_remains_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("raw-path-error", "AC", "CA")
    record["_canonical_jsonl_line"] = 1
    run_root = tmp_path / "run"
    run_root.mkdir()

    def raise_raw_path_error(*_args: object, **_kwargs: object) -> None:
        raise capacity_diagnostic.PathStateError(
            "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
            "closure exceeded 100000000 state DP cells; "
            "no approximation was emitted"
        )

    monkeypatch.setattr(
        capacity_diagnostic,
        "minimum_alignment_state_summary",
        raise_raw_path_error,
    )
    watchdog = SimpleNamespace(prototype_callback=lambda _progress: None)
    with pytest.raises(
        CapacityDiagnosticError,
        match="path-state oracle failure was not the typed DAG capacity stop",
    ):
        capacity_diagnostic._run_record(
            record=record,
            ordinal=0,
            run_root=run_root,
            safety=_config()["diagnostic_safety_limits"],
            watchdog=watchdog,
        )


def test_file_descriptor_capture_records_os_level_streams_and_restores_fds(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    capture = capacity_diagnostic.FileDescriptorCapture(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    capture.start()
    os.write(1, b"captured-stdout\n")
    os.write(2, b"captured-stderr\n")
    capture.stop()

    assert stdout_path.read_bytes() == b"captured-stdout\n"
    assert stderr_path.read_bytes() == b"captured-stderr\n"
    assert capture.saved_stdout_fd is None
    assert capture.saved_stderr_fd is None


def test_small_merge_tail_forces_resource_callback_before_publish(
    tmp_path: Path,
) -> None:
    output = tmp_path / "merged.tsv"
    callbacks: list[dict] = []

    def stop_on_final(progress: dict) -> None:
        callbacks.append(progress)
        if progress.get("final_callback"):
            raise capacity_diagnostic.SafeCapacityPause("small-tail-stop")

    with pytest.raises(
        capacity_diagnostic.SafeCapacityPause,
        match="small-tail-stop",
    ):
        capacity_diagnostic._merge_unique_iterators(
            [iter(("A", "C"))],
            output,
            progress_callback=stop_on_final,
        )

    assert callbacks == [
        {
            "phase": "merge_unique_sequences",
            "processed_input_sequences": 2,
            "unique_output_sequences": 2,
            "final_callback": True,
        }
    ]
    assert not output.exists()
    assert (
        output.with_name("merged.tsv.partial").read_text(encoding="utf-8") == "A\nC\n"
    )


def test_exact_record_verifier_replays_all_deterministic_evidence_in_scratch(
    tmp_path: Path,
) -> None:
    record = _record("exact-replay", "A", "C")
    record["_canonical_jsonl_line"] = 1
    safety = _config()["diagnostic_safety_limits"]
    safety["minimum_free_bytes"] = 1
    safety["max_rss_bytes"] = 10**15
    result = capacity_diagnostic._recompute_exact_record_evidence(
        record=record,
        ordinal=0,
        scratch_parent=tmp_path,
        safety=safety,
        verification_started=capacity_diagnostic.time.monotonic(),
    )

    expected_state_bytes = b"A\nC\n"
    assert result["alignment_statistics"] == {
        "minimum_edit_count": 1,
        "minimum_alignment_count": 1,
        "evaluated_dag_cell_count": 4,
        "counts_exact": True,
    }
    assert result["exact"]["reachable_node_count"] == 2
    assert result["exact"]["reachable_transition_count"] == 1
    assert result["exact"]["minimum_state_path_count"] == 1
    assert (
        result["exact"]["reachable_states_sha256"]
        == hashlib.sha256(expected_state_bytes).hexdigest()
    )
    assert result["state_universe"] == {
        "bytes": len(expected_state_bytes),
        "sha256": hashlib.sha256(expected_state_bytes).hexdigest(),
    }
    assert result["frozen_gate_assessment"] == {
        "would_pass_frozen_b0_limits": True,
        "exceeded_limits": [],
    }
    assert list(tmp_path.iterdir()) == []
