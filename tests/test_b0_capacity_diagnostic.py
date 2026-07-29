from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.data.diagnose_b0_path_capacity import (
    CapacityDiagnosticError,
)
from scripts.data.diagnose_b0_path_capacity import (
    FROZEN_B0_LIMITS,
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
