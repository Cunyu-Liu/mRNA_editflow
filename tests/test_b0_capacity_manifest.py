from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

import scripts.data.diagnose_b0_path_capacity as capacity_diagnostic
from scripts.data.diagnose_b0_path_capacity import CapacityDiagnosticError
from scripts.data.diagnose_b0_path_capacity import _artifact_ref
from scripts.data.diagnose_b0_path_capacity import _canonical_sha256
from scripts.data.diagnose_b0_path_capacity import _checksum_index
from scripts.data.diagnose_b0_path_capacity import _code_files
from scripts.data.diagnose_b0_path_capacity import _record_ids_sha256
from scripts.data.diagnose_b0_path_capacity import _render_replay_script
from scripts.data.diagnose_b0_path_capacity import _validate_bundle
from scripts.data.diagnose_b0_path_capacity import _validate_verified_marker
from scripts.data.diagnose_b0_path_capacity import _verified_marker_payload
from scripts.data.diagnose_b0_path_capacity import RESOURCE_LIMIT_SCOPE
from scripts.data.diagnose_b0_path_capacity import STREAM_CAPTURE_SCOPE


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/b0_capacity_diagnostic.schema.json"
GOAL_SHA256 = "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
HEX64 = "a" * 64
WITNESS_ID = "GSE217518:record:025e56d3b64660abb559dcbd"
BOUNDED_ID = "GSE217518:record:bounded"
TINY_WITNESS_STATES = ("A",)
_ORIGINAL_VALIDATE_EXTERNAL_PARENT_AUTHORIZATION = (
    capacity_diagnostic._validate_external_parent_authorization
)


def _sequence_file_bytes(states: tuple[str, ...]) -> bytes:
    return "".join(f"{state}\n" for state in states).encode("utf-8")


def _states_sha256(states: tuple[str, ...]) -> str:
    return hashlib.sha256(_sequence_file_bytes(states)).hexdigest()


def _record_workspace_path(ordinal: int, record_id: str) -> str:
    record_digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:12]
    return f"record_workspaces/{ordinal:06d}-{record_digest}/state_universe.tsv"


def _frozen_witness_counts() -> dict:
    """The immutable schema declaration; never used as the test state file."""

    return {
        "reachable_node_count": 95_217,
        "reachable_transition_count": 751_771,
        "minimum_state_path_count": 3_934_510_691_993,
        "evaluated_primitive_action_count": 1_205_477,
        "evaluated_state_dp_cell_count": 0,
        "reachable_states_sha256": (
            "900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332"
        ),
    }


def _tiny_exact_counts(
    states: tuple[str, ...] = TINY_WITNESS_STATES,
) -> dict:
    return {
        "reachable_node_count": len(states),
        "reachable_transition_count": max(len(states) - 1, 0),
        "minimum_state_path_count": 1,
        "evaluated_primitive_action_count": max(len(states) - 1, 0),
        "evaluated_state_dp_cell_count": 0,
        "reachable_states_sha256": _states_sha256(states),
    }


def _structural_record(*, exact: bool) -> dict:
    if exact:
        return {
            "record_id": WITNESS_ID,
            "dataset_id": "GSE217518",
            "region": "five_utr",
            "source_sequence": capacity_diagnostic.WITNESS_SOURCE,
            "candidate_sequence": capacity_diagnostic.WITNESS_CANDIDATE,
            "edit_distance": 15,
            "_canonical_jsonl_line": (capacity_diagnostic.WITNESS_CANONICAL_JSONL_LINE),
        }
    return {
        "record_id": BOUNDED_ID,
        "dataset_id": "GSE217518",
        "region": "five_utr",
        "source_sequence": "AC",
        "candidate_sequence": "CA",
        "edit_distance": 2,
        "_canonical_jsonl_line": (capacity_diagnostic.WITNESS_CANONICAL_JSONL_LINE + 1),
    }


def _trusted_structural_selection(candidate_store: Path):
    records = (
        _structural_record(exact=True),
        _structural_record(exact=False),
    )
    selected_ids = [str(record["record_id"]) for record in records]
    return capacity_diagnostic.StructuralSelection(
        records=records,
        eligible_endpoints=TINY_WITNESS_STATES,
        source_record_count=44_151,
        split_eligible_record_count=2,
        selected_record_count=2,
        excluded_record_count=44_149,
        record_ids_sha256="e" * 64,
        selected_record_ids_sha256=_record_ids_sha256(selected_ids),
        structural_store_sha256=hashlib.sha256(
            candidate_store.read_bytes()
        ).hexdigest(),
        exclusion_reason_counts={"fixture_unselected": 44_149},
    )


@pytest.fixture(autouse=True)
def _small_live_recomputation_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise live validation without materialising the 95k-state witness."""

    monkeypatch.setattr(
        capacity_diagnostic,
        "WITNESS_EXPECTED",
        _tiny_exact_counts(),
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "_load_structural_selection",
        lambda *, candidate_store, **_kwargs: _trusted_structural_selection(
            candidate_store
        ),
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "_validate_d1_snapshot_trust",
        lambda **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        capacity_diagnostic.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10**15),
    )

    def replay_fixture_exact_record(
        *,
        record: dict,
        **_kwargs: object,
    ) -> dict:
        is_witness = record["record_id"] == WITNESS_ID
        states = TINY_WITNESS_STATES if is_witness else ("C",)
        exact = _tiny_exact_counts(states)
        alignment = {
            "minimum_edit_count": record["edit_distance"],
            "minimum_alignment_count": 2_340 if is_witness else 2,
            "evaluated_dag_cell_count": 1_950 if is_witness else 9,
            "counts_exact": True,
        }
        return {
            "alignment_statistics": alignment,
            "exact": exact,
            "frozen_gate_assessment": (
                capacity_diagnostic._frozen_gate_assessment(alignment, exact)
            ),
            "state_universe": {
                "bytes": len(_sequence_file_bytes(states)),
                "sha256": _states_sha256(states),
            },
        }

    monkeypatch.setattr(
        capacity_diagnostic,
        "_recompute_exact_record_evidence",
        replay_fixture_exact_record,
    )

    def validate_fixture_parent_paths(
        *,
        parent_id: str,
        parent_authorization: dict,
        **_kwargs: object,
    ) -> None:
        parent_root = capacity_diagnostic.ALLOWED_OUTPUT_PARENT / parent_id
        expected = {
            "diagnostic_manifest": parent_root / "diagnostic_manifest.json",
            "verified_marker": parent_root / "VERIFIED",
            "bundle_seal": parent_root / "bundle_seal.json",
            "terminal_lock": parent_root / "terminal.lock",
            "process_result": parent_root / "provenance/process_result.json",
        }
        if parent_authorization.get("diagnostic_id") != parent_id or any(
            parent_authorization.get(field, {}).get("path") != str(path)
            for field, path in expected.items()
        ):
            raise CapacityDiagnosticError(
                "census bundle does not freeze the exact parent authorization"
            )

    monkeypatch.setattr(
        capacity_diagnostic,
        "_validate_external_parent_authorization",
        validate_fixture_parent_paths,
    )


def _reference(path: str, *, sha256: str = HEX64, size: int = 1) -> dict:
    return {
        "path": path,
        "bytes": size,
        "sha256": sha256,
    }


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator() -> jsonschema.Draft202012Validator:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )


def _row_validator() -> jsonschema.Draft202012Validator:
    schema = _schema()
    row_schema = {
        "$schema": schema["$schema"],
        "$ref": "#/$defs/recordResult",
        "$defs": schema["$defs"],
    }
    return jsonschema.Draft202012Validator(row_schema)


def _exact_counts() -> dict:
    return _tiny_exact_counts()


def _lower_bounds() -> dict:
    return {
        "reachable_node_count": 50_001,
        "reachable_transition_count": 1,
        "minimum_state_path_count": 1,
        "evaluated_primitive_action_count": 1,
        "evaluated_state_dp_cell_count": 0,
    }


def _record_result(*, exact: bool) -> dict:
    record = _structural_record(exact=exact)
    ordinal = 0 if exact else 1
    record_id = str(record["record_id"])
    source = str(record["source_sequence"])
    candidate = str(record["candidate_sequence"])
    return {
        "schema_version": "b0_capacity_record.v1",
        "ordinal": ordinal,
        "record_id": record_id,
        "dataset_id": record["dataset_id"],
        "region": record["region"],
        "canonical_jsonl_line": record["_canonical_jsonl_line"],
        "input_record_structural_sha256": (
            capacity_diagnostic.record_structural_sha256(record)
        ),
        "source_sequence_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "candidate_sequence_sha256": hashlib.sha256(
            candidate.encode("utf-8")
        ).hexdigest(),
        "source_length": len(source),
        "candidate_length": len(candidate),
        "alignment_statistics": {
            "minimum_edit_count": record["edit_distance"],
            "minimum_alignment_count": 2_340 if exact else 2,
            "evaluated_dag_cell_count": 1_950 if exact else 9,
            "counts_exact": True,
        },
        "outcome": "EXACT_COMPLETED" if exact else "LOWER_BOUND_STOPPED",
        "state_universe_artifact": (
            {
                "path": _record_workspace_path(ordinal, record_id),
                "bytes": len(_sequence_file_bytes(TINY_WITNESS_STATES)),
                "sha256": _states_sha256(TINY_WITNESS_STATES),
            }
            if exact
            else None
        ),
        "capacity": {
            "exact": _exact_counts() if exact else None,
            "lower_bound": None if exact else _lower_bounds(),
        },
        "frozen_gate_assessment": {
            "would_pass_frozen_b0_limits": exact,
            "exceeded_limits": [] if exact else ["max_reachable_states"],
        },
        "evidence_semantics": {
            "counts_exact": exact,
            "state_set_complete": exact,
            "no_approximation_emitted": True,
            "usable_for_b0_acceptance": False,
        },
        "stop": (
            None
            if exact
            else {
                "stop_rule": "STOP_RULE_B0_PATH_STATE_COMPLEXITY",
                "dimension": "max_reachable_states",
                "frozen_limit": 50_000,
                "observed_lower_bound": 50_001,
                "message": "exact diagnostic stopped without approximation",
            }
        ),
        "elapsed_seconds": 1.0,
        "peak_rss_bytes": 1,
        "spill_bytes": 0,
    }


def _manifest() -> dict:
    return {
        "artifact_type": "b0_capacity_diagnostic",
        "schema_version": "b0_capacity_diagnostic.v1",
        "diagnostic_id": "B0_capacity_20260729T150000Z_2e58254",
        "parent_diagnostic_id": "B0_capacity_20260729T140000Z_2e58254",
        "parent_authorization": {
            "diagnostic_id": "B0_capacity_20260729T140000Z_2e58254",
            "diagnostic_manifest": _reference(
                "/mnt/cunyuliu/mrna_editflow_b0_capacity/"
                "B0_capacity_20260729T140000Z_2e58254/"
                "diagnostic_manifest.json",
                sha256="1" * 64,
            ),
            "verified_marker": _reference(
                "/mnt/cunyuliu/mrna_editflow_b0_capacity/"
                "B0_capacity_20260729T140000Z_2e58254/VERIFIED",
                sha256="2" * 64,
            ),
            "bundle_seal": _reference(
                "/mnt/cunyuliu/mrna_editflow_b0_capacity/"
                "B0_capacity_20260729T140000Z_2e58254/bundle_seal.json",
                sha256="3" * 64,
            ),
            "terminal_lock": _reference(
                "/mnt/cunyuliu/mrna_editflow_b0_capacity/"
                "B0_capacity_20260729T140000Z_2e58254/terminal.lock",
                sha256="4" * 64,
            ),
            "process_result": _reference(
                "/mnt/cunyuliu/mrna_editflow_b0_capacity/"
                "B0_capacity_20260729T140000Z_2e58254/"
                "provenance/process_result.json",
                sha256="5" * 64,
            ),
        },
        "state": "COMPLETED",
        "terminal": True,
        "started_at_utc": "2026-07-29T15:00:00Z",
        "ended_at_utc": "2026-07-29T16:00:00Z",
        "workload_class": "NON_NEURAL_DATA_BENCHMARK",
        "purpose": "B0_PATH_CAPACITY_DIAGNOSTIC_ONLY",
        "goal_contract": {
            "id": "utr_editflow_goal_v2",
            "sha256": GOAL_SHA256,
            "repository_snapshot": "docs/contracts/mrna_latest_build_contract_v2.md",
        },
        "claim_boundary": {
            "formal_b0_attempt_started": False,
            "b0_gate_values_changed": False,
            "b0_phase_acceptance_claimed": False,
            "scientific_result_claimed": False,
            "budget_change_authorized": False,
            "approximation_emitted": False,
            "allowed_claim": "CAPACITY_DIAGNOSTIC_ONLY",
        },
        "selection": {
            "source_store_role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
            "selection_algorithm": (
                "split_graph.select_split_eligible_records+edit_distance_gte_2"
            ),
            "diagnostic_scope": "census",
            "state_universe_scope": "FULL_CENSUS",
            "regions": ["five_utr", "three_utr"],
            "record_order": (
                "FROZEN_WITNESS_FIRST_THEN_CANONICAL_JSONL_LINE_ASCENDING"
            ),
            "source_record_count": 44_151,
            "selected_record_count": 2,
            "excluded_record_count": 44_149,
            "selected_record_ids_sha256": _record_ids_sha256([WITNESS_ID, BOUNDED_ID]),
            "label_fields_read": [],
            "canonical_label_store_opened_by_selection": False,
            "selection_manifest": _reference(
                "provenance/selection_manifest.json",
                sha256="f" * 64,
            ),
        },
        "witnesses": [
            {
                "witness_kind": "FROZEN_FIRST_PATH_COMPLEXITY_WITNESS",
                "record_id": "GSE217518:record:025e56d3b64660abb559dcbd",
                "canonical_jsonl_line": 39_913,
                "source_length": 129,
                "candidate_length": 114,
                "expected": _frozen_witness_counts(),
                "observed_record_ordinal": 0,
                "parity_passed": True,
            }
        ],
        "provenance": {
            "git": {
                "project_root": "/mnt/cunyuliu/worktree",
                "git_dir": "/mnt/cunyuliu/gitdir",
                "head": "2" * 40,
                "clean": True,
                "dirty_state_sha256": "3" * 64,
            },
            "exact_argv": [
                "/mnt/runtime/bin/python",
                "/mnt/cunyuliu/worktree/scripts/data/diagnose_b0_path_capacity.py",
                "--config",
                "configs/b0_capacity_v1.json",
            ],
            "exact_argv_sha256": "4" * 64,
            "launch_cwd": "/mnt/cunyuliu/worktree",
            "resolved_config": _reference(
                "resolved_config.json",
                sha256="5" * 64,
            ),
            "runtime_manifest": _reference(
                "provenance/runtime_manifest.json",
                sha256="6" * 64,
            ),
            "python_launcher": _reference(
                "provenance/python_launcher.bin",
                sha256="7" * 64,
            ),
            "d1_snapshot": _reference(
                "data/d1/manifests/d1_canonical_snapshot.json",
                sha256="0" * 64,
            ),
            "d1_snapshot_validation": _reference(
                "provenance/d1_snapshot_validation.json",
                sha256="1" * 64,
            ),
            "d1_acceptance": _reference(
                "provenance/d1_acceptance.json",
                sha256="8" * 64,
            ),
            "d1_build_manifest": _reference(
                "provenance/d1_build_manifest.json",
                sha256="9" * 64,
            ),
            "canonical_store_metadata_only": {
                **_reference(
                    "provenance/canonical_store.binding.json",
                    sha256="a" * 64,
                ),
                "record_count": 44_151,
                "record_ids_sha256": "b" * 64,
                "content_sha256": "c" * 64,
                "selection_opened": False,
                "capacity_algorithm_opened": False,
                "opaque_integrity_hash_opened": True,
                "jsonl_parsed": False,
                "label_values_accessed": False,
            },
            "structural_store": {
                **_reference(
                    "provenance/structural_store.binding.json",
                    sha256="d" * 64,
                ),
                "record_count": 44_151,
                "record_ids_sha256": "e" * 64,
                "structural_content_sha256": "f" * 64,
                "role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
            },
            "ambiguity_report": _reference(
                "provenance/edit_script_ambiguity_report.json",
                sha256="1" * 64,
            ),
            "process_result": _reference(
                "provenance/process_result.json",
                sha256="2" * 64,
            ),
            "code_files": [
                {
                    **_reference(
                        "data/utr_benchmark_v2/path_states.py",
                        sha256="2" * 64,
                    ),
                    "role": "PATH_STATE_ORACLE",
                },
                {
                    **_reference(
                        "data/utr_benchmark_v2/near_neighbors.py",
                        sha256="3" * 64,
                    ),
                    "role": "NEAR_NEIGHBOR_ORACLE",
                },
                {
                    **_reference(
                        "data/utr_benchmark_v2/split_graph.py",
                        sha256="4" * 64,
                    ),
                    "role": "SELECTION_ORACLE",
                },
                {
                    **_reference(
                        "data/utr_benchmark_v2/symbolic_path_states.py",
                        sha256="5" * 64,
                    ),
                    "role": "STREAMING_PROTOTYPE",
                },
                {
                    **_reference(
                        "scripts/data/diagnose_b0_path_capacity.py",
                        sha256="6" * 64,
                    ),
                    "role": "DIAGNOSTIC_ENTRYPOINT",
                },
                {
                    **_reference(
                        "schemas/b0_capacity_diagnostic.schema.json",
                        sha256="7" * 64,
                    ),
                    "role": "DIAGNOSTIC_SCHEMA",
                },
            ],
        },
        "algorithm_contract": {
            "path_state_algorithm": "all_shortest_dynamic_edit_state_closure_v3",
            "state_closure_scope": (
                "all_shortest_primitive_dynamic_edit_execution_orders_"
                "sequence_identity"
            ),
            "state_path_count_scope": (
                "minimum_primitive_edit_state_paths_coordinate_equivalent_"
                "transitions_collapsed"
            ),
            "primitive_action_evaluation_scope": (
                "distinct_dynamic_geodesic_actions_before_sequence_identity_collapse"
            ),
            "near_neighbor_algorithm": (
                "six_block_pigeonhole_all_substring_candidates_exact_banded_"
                "levenshtein_v1"
            ),
            "near_neighbor_edit_distance_threshold": 5,
            "frozen_b0_limits": {
                "max_dag_cells": 1_000_000,
                "max_reachable_states": 50_000,
                "max_neighbor_expansions": 5_000_000,
                "max_state_dp_cells": 50_000_000,
                "max_sequences": 100_000,
                "max_block_postings": 600_000,
                "max_substring_probes": 50_000_000,
                "max_candidate_pairs": 1_000_000,
                "max_exact_dp_cells": 100_000_000,
            },
            "diagnostic_safety_limits": {
                "minimum_free_bytes": 1,
                "max_rss_bytes": 1,
                "max_wall_seconds": 1,
                "max_spill_bytes": 1,
                "max_dag_cells_per_record": 1,
                "max_reachable_states_per_record": 50_001,
                "max_neighbor_expansions_per_record": 1,
                "max_state_dp_cells_per_record": 1,
                "max_spill_bytes_per_record": 1,
                "chunk_size": 1,
                "max_open_chunks": 2,
                "heartbeat_seconds": 300,
            },
        },
        "accounting": {
            "scheduled_record_count": 2,
            "terminal_record_count": 2,
            "exact_completed_count": 1,
            "lower_bound_count": 1,
            "failed_record_count": 0,
            "outcome_counts": {
                "EXACT_COMPLETED": 1,
                "LOWER_BOUND_STOPPED": 1,
            },
            "all_scheduled_records_accounted": True,
            "accounting_reconciled": True,
            "record_results": _reference(
                "records/results.jsonl",
                sha256="6" * 64,
            ),
            "record_shard_index": _reference(
                "records/shard_index.json",
                sha256="7" * 64,
            ),
            "capacity_summary": _reference(
                "capacity_summary.json",
                sha256="0" * 64,
            ),
        },
        "result_semantics": {
            "exact_capacity_complete": False,
            "lower_bounds_present": True,
            "no_approximation_emitted": True,
            "usable_for_budget_decision": False,
            "usable_for_b0_acceptance": False,
        },
        "stop_reason": None,
        "completion_seal": {
            "sealed_at_utc": "2026-07-29T16:00:00Z",
            "terminal_marker": "DONE",
            "terminal_marker_ref": _reference("DONE", sha256="8" * 64, size=0),
            "artifact_checksum_index_path": "artifact_checksums.json",
            "bundle_seal_path": "bundle_seal.json",
            "terminal_lock_path": "terminal.lock",
            "record_shard_index": _reference(
                "records/shard_index.json",
                sha256="7" * 64,
            ),
        },
        "failure": None,
    }


def _errors(payload: dict) -> list[jsonschema.ValidationError]:
    return sorted(_validator().iter_errors(payload), key=lambda error: list(error.path))


def _row_errors(payload: dict) -> list[jsonschema.ValidationError]:
    return sorted(
        _row_validator().iter_errors(payload),
        key=lambda error: list(error.path),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def _seal_test_bundle(
    run_root: Path,
    manifest: dict,
    *,
    update_marker_ref: bool = True,
) -> None:
    marker_path = run_root / "DONE"
    marker_path.write_text("", encoding="utf-8", newline="")
    if update_marker_ref:
        manifest["completion_seal"]["terminal_marker_ref"] = _artifact_ref(
            marker_path,
            root=run_root,
        )
    _write_json(run_root / "diagnostic_manifest.json", manifest)
    for name in (
        "artifact_checksums.json",
        "bundle_seal.json",
        "terminal.lock",
        "VERIFIED",
    ):
        path = run_root / name
        if path.exists():
            path.unlink()
    preseal_paths = [path for path in run_root.rglob("*") if path.is_file()]
    _write_json(
        run_root / "artifact_checksums.json",
        _checksum_index(run_root, preseal_paths),
    )
    _write_json(
        run_root / "bundle_seal.json",
        {
            "schema_version": "b0_capacity_bundle_seal.v1",
            "diagnostic_id": manifest["diagnostic_id"],
            "state": "COMPLETED",
            "sealed_at_utc": manifest["ended_at_utc"],
            "diagnostic_manifest": _artifact_ref(
                run_root / "diagnostic_manifest.json",
                root=run_root,
            ),
            "artifact_checksum_index": _artifact_ref(
                run_root / "artifact_checksums.json",
                root=run_root,
            ),
            "terminal_marker": _artifact_ref(marker_path, root=run_root),
            "status": _artifact_ref(run_root / "status.json", root=run_root),
            "process_result": _artifact_ref(
                run_root / "provenance/process_result.json",
                root=run_root,
            ),
        },
    )
    _write_json(
        run_root / "terminal.lock",
        {
            "schema_version": "b0_capacity_terminal_lock.v1",
            "diagnostic_id": manifest["diagnostic_id"],
            "state": "COMPLETED",
            "sealed": True,
            "bundle_seal": _artifact_ref(
                run_root / "bundle_seal.json",
                root=run_root,
            ),
        },
    )


def _bundle_fixture(tmp_path: Path) -> tuple[Path, dict]:
    run_root = tmp_path / "B0_capacity_20260729T150000Z_2e58254"
    for relative in ("logs", "provenance", "records", "global"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    external_root = tmp_path / "external"
    external_root.mkdir()
    git_dir = external_root / "gitdir"
    git_dir.mkdir()
    d1_snapshot = external_root / "d1_snapshot.json"
    d1_acceptance = external_root / "d1_acceptance.json"
    d1_build_manifest = external_root / "d1_build_manifest.json"
    ambiguity_report = external_root / "ambiguity_report.json"
    canonical_store = external_root / "canonical.jsonl"
    candidate_store = external_root / "candidates.jsonl"
    d1_snapshot.write_text("{}\n", encoding="utf-8", newline="")
    d1_acceptance.write_text("{}\n", encoding="utf-8", newline="")
    d1_build_manifest.write_text("{}\n", encoding="utf-8", newline="")
    ambiguity_report.write_text("{}\n", encoding="utf-8", newline="")
    canonical_store.write_text('{"label":1}\n', encoding="utf-8", newline="")
    candidate_store.write_text('{"fixture":true}\n', encoding="utf-8", newline="")
    config_path = ROOT / "configs/b0_capacity_diagnostic_v1.json"
    contract_path = ROOT / "docs/contracts/mrna_latest_build_contract_v2.md"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    launcher = Path(sys.executable).resolve(strict=True)
    entrypoint = (ROOT / "scripts/data/diagnose_b0_path_capacity.py").resolve(
        strict=True
    )

    (run_root / "logs/stdout.log").write_text("", encoding="utf-8")
    (run_root / "logs/stderr.log").write_text("", encoding="utf-8")
    (run_root / "logs/events.jsonl").write_text("{}\n", encoding="utf-8")
    (run_root / "logs/system_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(run_root / "resolved_config.json", config)
    launcher_binding = {
        "path": str(launcher),
        "bytes": launcher.stat().st_size,
        "sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
    }
    _write_json(
        run_root / "provenance/runtime_manifest.json",
        {
            "executable": launcher_binding["path"],
            "executable_bytes": launcher_binding["bytes"],
            "executable_sha256": launcher_binding["sha256"],
            "resource_limit_scope": RESOURCE_LIMIT_SCOPE,
        },
    )
    _write_json(
        run_root / "provenance/python_launcher.json",
        launcher_binding,
    )
    _write_json(
        run_root / "provenance/d1_snapshot_validation.json",
        {"status": "PASS"},
    )
    _write_json(
        run_root / "provenance/input_manifest.json",
        {
            "contract": _artifact_ref(contract_path),
            "d1_snapshot": _artifact_ref(d1_snapshot),
            "d1_snapshot_validation": _artifact_ref(
                run_root / "provenance/d1_snapshot_validation.json",
                root=run_root,
            ),
            "d1_acceptance": _artifact_ref(d1_acceptance),
            "d1_build_manifest": _artifact_ref(d1_build_manifest),
            "ambiguity_report": _artifact_ref(ambiguity_report),
            "structural_store": {
                **_artifact_ref(candidate_store),
                "records": 44_151,
                "record_ids_sha256": "e" * 64,
            },
            "canonical_store_metadata_only": {
                **_artifact_ref(canonical_store),
                "records": 44_151,
                "record_ids_sha256": "b" * 64,
                "selection_opened": False,
                "capacity_algorithm_opened": False,
                "opaque_integrity_hash_opened": True,
                "jsonl_parsed": False,
                "label_values_accessed": False,
            },
            "label_fields_read": [],
        },
    )
    fixture_code_files = _code_files(ROOT)
    _write_json(
        run_root / "provenance/code_manifest.json",
        {
            "code_commit": "2" * 40,
            "files": fixture_code_files,
        },
    )

    rows = [_record_result(exact=True), _record_result(exact=False)]
    exact_state_path = run_root / rows[0]["state_universe_artifact"]["path"]
    exact_state_path.parent.mkdir(parents=True)
    exact_state_path.write_bytes(_sequence_file_bytes(TINY_WITNESS_STATES))
    records_path = run_root / "records/results.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    observed_ids = [str(row["record_id"]) for row in rows]
    _write_json(
        run_root / "records/shard_index.json",
        {
            "schema_version": "b0_capacity_shard_index.v1",
            "row_count": len(rows),
            "record_ids_sha256": _record_ids_sha256(observed_ids),
            "shards": [_artifact_ref(records_path, root=run_root)],
        },
    )
    trusted_selection = _trusted_structural_selection(candidate_store)
    selection_manifest = capacity_diagnostic._selection_manifest_payload(
        trusted_selection,
        scope="census",
    )
    _write_json(
        run_root / "provenance/selection_manifest.json",
        selection_manifest,
    )
    (run_root / "global/state_universe.tsv").write_text(
        "A\n",
        encoding="utf-8",
        newline="",
    )
    (run_root / "global/eligible_endpoints.tsv").write_text(
        "A\n",
        encoding="utf-8",
        newline="",
    )
    global_path = run_root / "global/state_universe.tsv"
    global_sha256 = hashlib.sha256(global_path.read_bytes()).hexdigest()
    _write_json(
        run_root / "capacity_summary.json",
        capacity_diagnostic._diagnostic_summary(
            run_root=run_root,
            diagnostic_scope="census",
            selection=trusted_selection,
            rows=rows,
            global_state_count=1,
            global_state_digest=global_sha256,
            global_state_exact=False,
            global_state_role=(
                "LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS"
            ),
            global_state_path=global_path,
        ),
    )
    _write_json(
        run_root / "status.json",
        {
            "state": "COMPLETED",
            "terminal": True,
            "formal_b0_attempt_started": False,
        },
    )
    _write_json(
        run_root / "provenance/process_result.json",
        {
            "schema_version": "b0_capacity_process_result.v1",
            "terminal_state": "COMPLETED",
            "return_code": 0,
            "stdout": _artifact_ref(run_root / "logs/stdout.log", root=run_root),
            "stderr": _artifact_ref(run_root / "logs/stderr.log", root=run_root),
            "capture_mode": "OS_FD_DUP2",
            "capture_complete": True,
            "capture_scope": STREAM_CAPTURE_SCOPE,
            "return_code_semantics": (
                "COMMITTED_AFTER_BUNDLE_VALIDATION_AND_VERIFIED_MARKER"
            ),
        },
    )
    parent_diagnostic_id = "B0_capacity_20260729T140000Z_2e58254"
    exact_argv = [
        str(launcher),
        str(entrypoint),
        "--config",
        str(config_path),
        "--d1-snapshot",
        str(d1_snapshot),
        "--d1-acceptance",
        str(d1_acceptance),
        "--contract",
        str(contract_path),
        "--candidate-store",
        str(candidate_store),
        "--output-root",
        str(run_root),
        "--project-root",
        str(ROOT),
        "--git-dir",
        str(git_dir),
        "--expected-code-commit",
        "2" * 40,
        "--scope",
        "census",
        "--parent-diagnostic-id",
        parent_diagnostic_id,
    ]
    _write_json(
        run_root / "command.json",
        {
            "exact_argv": exact_argv,
            "exact_argv_sha256": _canonical_sha256(exact_argv),
            "launch_cwd": str(ROOT),
            "output_root_replay_policy": (
                "replace_with_a_fresh_nonexistent_output_root"
            ),
        },
    )
    replay_path = run_root / "replay.sh"
    replay_path.write_text(
        _render_replay_script(exact_argv, ROOT),
        encoding="utf-8",
        newline="",
    )
    replay_path.chmod(0o750)

    manifest = _manifest()
    manifest["parent_diagnostic_id"] = parent_diagnostic_id
    manifest["provenance"]["git"] = {
        "project_root": str(ROOT),
        "git_dir": str(git_dir),
        "head": "2" * 40,
        "clean": True,
        "dirty_state_sha256": "3" * 64,
    }
    manifest["provenance"]["exact_argv"] = exact_argv
    manifest["provenance"]["exact_argv_sha256"] = _canonical_sha256(exact_argv)
    manifest["provenance"]["launch_cwd"] = str(ROOT)
    manifest["provenance"]["resolved_config"] = _artifact_ref(
        run_root / "resolved_config.json",
        root=run_root,
    )
    manifest["provenance"]["runtime_manifest"] = _artifact_ref(
        run_root / "provenance/runtime_manifest.json",
        root=run_root,
    )
    manifest["provenance"]["python_launcher"] = _artifact_ref(
        run_root / "provenance/python_launcher.json",
        root=run_root,
    )
    manifest["provenance"]["d1_snapshot_validation"] = _artifact_ref(
        run_root / "provenance/d1_snapshot_validation.json",
        root=run_root,
    )
    manifest["provenance"]["d1_snapshot"] = _artifact_ref(d1_snapshot)
    manifest["provenance"]["d1_acceptance"] = _artifact_ref(d1_acceptance)
    manifest["provenance"]["d1_build_manifest"] = _artifact_ref(d1_build_manifest)
    manifest["provenance"]["ambiguity_report"] = _artifact_ref(ambiguity_report)
    canonical_ref = _artifact_ref(canonical_store)
    manifest["provenance"]["canonical_store_metadata_only"] = {
        **canonical_ref,
        "record_count": 44_151,
        "record_ids_sha256": "b" * 64,
        "content_sha256": canonical_ref["sha256"],
        "selection_opened": False,
        "capacity_algorithm_opened": False,
        "opaque_integrity_hash_opened": True,
        "jsonl_parsed": False,
        "label_values_accessed": False,
    }
    structural_ref = _artifact_ref(candidate_store)
    manifest["provenance"]["structural_store"] = {
        **structural_ref,
        "record_count": 44_151,
        "record_ids_sha256": "e" * 64,
        "structural_content_sha256": structural_ref["sha256"],
        "role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
    }
    manifest["provenance"]["code_files"] = fixture_code_files
    manifest["provenance"]["process_result"] = _artifact_ref(
        run_root / "provenance/process_result.json",
        root=run_root,
    )
    manifest["selection"]["selection_manifest"] = _artifact_ref(
        run_root / "provenance/selection_manifest.json",
        root=run_root,
    )
    manifest["accounting"]["record_results"] = _artifact_ref(
        records_path,
        root=run_root,
    )
    manifest["accounting"]["record_shard_index"] = _artifact_ref(
        run_root / "records/shard_index.json",
        root=run_root,
    )
    manifest["accounting"]["capacity_summary"] = _artifact_ref(
        run_root / "capacity_summary.json",
        root=run_root,
    )
    manifest["completion_seal"]["record_shard_index"] = manifest["accounting"][
        "record_shard_index"
    ]
    _seal_test_bundle(run_root, manifest)
    return run_root, manifest


def _rewrite_rows_and_bindings(
    run_root: Path,
    manifest: dict,
    rows: list[dict],
) -> None:
    records_path = run_root / "records/results.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    observed_ids = [str(row["record_id"]) for row in rows]
    shard_path = run_root / "records/shard_index.json"
    _write_json(
        shard_path,
        {
            "schema_version": "b0_capacity_shard_index.v1",
            "row_count": len(rows),
            "record_ids_sha256": _record_ids_sha256(observed_ids),
            "shards": [_artifact_ref(records_path, root=run_root)],
        },
    )
    manifest["accounting"]["record_results"] = _artifact_ref(
        records_path,
        root=run_root,
    )
    manifest["accounting"]["record_shard_index"] = _artifact_ref(
        shard_path,
        root=run_root,
    )
    manifest["completion_seal"]["record_shard_index"] = manifest["accounting"][
        "record_shard_index"
    ]


def _rewrite_global_summary(
    run_root: Path,
    manifest: dict,
    summary: dict,
    states: tuple[str, ...],
) -> None:
    global_path = run_root / "global/state_universe.tsv"
    global_path.write_bytes(_sequence_file_bytes(states))
    summary["global_unique_state_count"] = len(states)
    summary["global_state_universe_sha256"] = _states_sha256(states)
    summary["global_state_universe_artifact"] = _artifact_ref(
        global_path,
        root=run_root,
    )
    summary_path = run_root / "capacity_summary.json"
    _write_json(summary_path, summary)
    manifest["accounting"]["capacity_summary"] = _artifact_ref(
        summary_path,
        root=run_root,
    )


def _promote_second_record_to_exact(
    run_root: Path,
    manifest: dict,
) -> list[dict]:
    rows = [
        dict(row)
        for row in capacity_diagnostic._read_jsonl_objects(
            run_root / "records/results.jsonl"
        )
    ]
    states = ("C",)
    state_path = run_root / _record_workspace_path(1, rows[1]["record_id"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(_sequence_file_bytes(states))
    rows[1].update(
        {
            "outcome": "EXACT_COMPLETED",
            "state_universe_artifact": _artifact_ref(
                state_path,
                root=run_root,
            ),
            "capacity": {
                "exact": _tiny_exact_counts(states),
                "lower_bound": None,
            },
            "frozen_gate_assessment": {
                "would_pass_frozen_b0_limits": True,
                "exceeded_limits": [],
            },
            "evidence_semantics": {
                "counts_exact": True,
                "state_set_complete": True,
                "no_approximation_emitted": True,
                "usable_for_b0_acceptance": False,
            },
            "stop": None,
        }
    )
    _rewrite_rows_and_bindings(run_root, manifest, rows)
    manifest["accounting"].update(
        {
            "exact_completed_count": 2,
            "lower_bound_count": 0,
            "outcome_counts": {
                "EXACT_COMPLETED": 2,
                "LOWER_BOUND_STOPPED": 0,
            },
        }
    )
    manifest["result_semantics"].update(
        {
            "exact_capacity_complete": True,
            "lower_bounds_present": False,
            "usable_for_budget_decision": True,
        }
    )
    global_states = ("A", "C")
    global_path = run_root / "global/state_universe.tsv"
    global_path.write_bytes(_sequence_file_bytes(global_states))
    selection = _trusted_structural_selection(
        Path(manifest["provenance"]["structural_store"]["path"])
    )
    summary = capacity_diagnostic._diagnostic_summary(
        run_root=run_root,
        diagnostic_scope="census",
        selection=selection,
        rows=rows,
        global_state_count=len(global_states),
        global_state_digest=_states_sha256(global_states),
        global_state_exact=True,
        global_state_role="EXACT",
        global_state_path=global_path,
    )
    summary_path = run_root / "capacity_summary.json"
    _write_json(summary_path, summary)
    manifest["accounting"]["capacity_summary"] = _artifact_ref(
        summary_path,
        root=run_root,
    )
    return rows


def test_schema_is_strict_draft_2020_12_and_valid_manifest_passes() -> None:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert _errors(_manifest()) == []


def test_detached_bundle_seal_and_cross_accounting_validate(tmp_path: Path) -> None:
    run_root, _manifest_payload = _bundle_fixture(tmp_path)
    _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_resealed_selection_and_rows_must_match_live_d1_recomputation(
    tmp_path: Path,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    records_path = run_root / "records/results.jsonl"
    rows = [dict(row) for row in capacity_diagnostic._read_jsonl_objects(records_path)]
    forged = {
        **_structural_record(exact=False),
        "record_id": "GSE217518:record:forged-selection",
    }
    rows[1]["record_id"] = forged["record_id"]
    rows[1]["canonical_jsonl_line"] = forged["_canonical_jsonl_line"]
    rows[1]["input_record_structural_sha256"] = (
        capacity_diagnostic.record_structural_sha256(forged)
    )
    rows[1]["source_sequence_sha256"] = hashlib.sha256(
        forged["source_sequence"].encode("utf-8")
    ).hexdigest()
    rows[1]["candidate_sequence_sha256"] = hashlib.sha256(
        forged["candidate_sequence"].encode("utf-8")
    ).hexdigest()
    selection_path = run_root / "provenance/selection_manifest.json"
    sealed_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sealed_selection["selected_records"][1] = {
        "canonical_jsonl_line": forged["_canonical_jsonl_line"],
        "record_id": forged["record_id"],
        "structural_sha256": rows[1]["input_record_structural_sha256"],
    }
    forged_ids = [str(row["record_id"]) for row in rows]
    forged_ids_sha256 = _record_ids_sha256(forged_ids)
    sealed_selection["selected_record_ids_sha256"] = forged_ids_sha256
    _write_json(selection_path, sealed_selection)
    manifest["selection"]["selected_record_ids_sha256"] = forged_ids_sha256
    manifest["selection"]["selection_manifest"] = _artifact_ref(
        selection_path,
        root=run_root,
    )
    _rewrite_rows_and_bindings(run_root, manifest, rows)
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError, match="selection"):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


@pytest.mark.parametrize(
    "attack",
    [
        "missing",
        "non_deterministic_path",
        "count_digest_mismatch",
    ],
)
def test_exact_row_state_universe_is_required_deterministic_and_recomputed(
    tmp_path: Path,
    attack: str,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    records_path = run_root / "records/results.jsonl"
    rows = [dict(row) for row in capacity_diagnostic._read_jsonl_objects(records_path)]
    canonical_path = run_root / rows[0]["state_universe_artifact"]["path"]
    if attack == "missing":
        canonical_path.unlink()
    elif attack == "non_deterministic_path":
        forged_path = run_root / "record_workspaces/forged/state_universe.tsv"
        forged_path.parent.mkdir(parents=True)
        forged_path.write_bytes(_sequence_file_bytes(TINY_WITNESS_STATES))
        rows[0]["state_universe_artifact"] = _artifact_ref(
            forged_path,
            root=run_root,
        )
    else:
        mismatched_states = ("A", "AA")
        canonical_path.write_bytes(_sequence_file_bytes(mismatched_states))
        rows[0]["state_universe_artifact"] = _artifact_ref(
            canonical_path,
            root=run_root,
        )
        summary_path = run_root / "capacity_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _rewrite_global_summary(
            run_root,
            manifest,
            summary,
            mismatched_states,
        )
    _rewrite_rows_and_bindings(run_root, manifest, rows)
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_full_exact_global_census_is_bounded_union_of_endpoints_and_exact_states(
    tmp_path: Path,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    records_path = run_root / "records/results.jsonl"
    rows = [dict(row) for row in capacity_diagnostic._read_jsonl_objects(records_path)]
    omitted_states = ("C",)
    second_state_path = run_root / _record_workspace_path(1, rows[1]["record_id"])
    second_state_path.parent.mkdir(parents=True)
    second_state_path.write_bytes(_sequence_file_bytes(omitted_states))
    rows[1]["outcome"] = "EXACT_COMPLETED"
    rows[1]["state_universe_artifact"] = _artifact_ref(
        second_state_path,
        root=run_root,
    )
    rows[1]["capacity"] = {
        "exact": _tiny_exact_counts(omitted_states),
        "lower_bound": None,
    }
    rows[1]["frozen_gate_assessment"] = {
        "would_pass_frozen_b0_limits": True,
        "exceeded_limits": [],
    }
    rows[1]["evidence_semantics"] = {
        "counts_exact": True,
        "state_set_complete": True,
        "no_approximation_emitted": True,
        "usable_for_b0_acceptance": False,
    }
    rows[1]["stop"] = None
    _rewrite_rows_and_bindings(run_root, manifest, rows)

    manifest["accounting"].update(
        {
            "exact_completed_count": 2,
            "lower_bound_count": 0,
            "outcome_counts": {
                "EXACT_COMPLETED": 2,
                "LOWER_BOUND_STOPPED": 0,
            },
        }
    )
    manifest["result_semantics"].update(
        {
            "exact_capacity_complete": True,
            "lower_bounds_present": False,
            "usable_for_budget_decision": True,
        }
    )
    summary_path = run_root / "capacity_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "exact_completed_record_count": 2,
            "lower_bound_record_count": 0,
            "global_state_universe_exact": True,
            "full_census_state_universe_exact": True,
            "global_state_universe_role": "EXACT",
        }
    )
    # The forged bundle omits "C", even though the second exact record binds it.
    _rewrite_global_summary(
        run_root,
        manifest,
        summary,
        TINY_WITNESS_STATES,
    )
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


@pytest.mark.parametrize(
    "attack",
    [
        "transition_count",
        "path_count",
        "primitive_action_count",
        "state_dp_count",
        "alignment_count",
        "dag_count",
        "gate_assessment",
    ],
)
def test_non_witness_exact_counters_and_gate_require_endpoint_replay(
    tmp_path: Path,
    attack: str,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    rows = _promote_second_record_to_exact(run_root, manifest)
    exact = rows[1]["capacity"]["exact"]
    if attack == "transition_count":
        exact["reachable_transition_count"] = 999
    elif attack == "path_count":
        exact["minimum_state_path_count"] = 999
    elif attack == "primitive_action_count":
        exact["evaluated_primitive_action_count"] = 888
    elif attack == "state_dp_count":
        exact["evaluated_state_dp_cell_count"] = 777
    elif attack == "alignment_count":
        rows[1]["alignment_statistics"]["minimum_alignment_count"] = 3
    elif attack == "dag_count":
        rows[1]["alignment_statistics"]["evaluated_dag_cell_count"] = 10
    else:
        rows[1]["frozen_gate_assessment"] = {
            "would_pass_frozen_b0_limits": False,
            "exceeded_limits": ["max_dag_cells"],
        }
    _rewrite_rows_and_bindings(run_root, manifest, rows)
    global_path = run_root / "global/state_universe.tsv"
    selection = _trusted_structural_selection(
        Path(manifest["provenance"]["structural_store"]["path"])
    )
    summary = capacity_diagnostic._diagnostic_summary(
        run_root=run_root,
        diagnostic_scope="census",
        selection=selection,
        rows=rows,
        global_state_count=2,
        global_state_digest=_states_sha256(("A", "C")),
        global_state_exact=True,
        global_state_role="EXACT",
        global_state_path=global_path,
    )
    summary_path = run_root / "capacity_summary.json"
    _write_json(summary_path, summary)
    manifest["accounting"]["capacity_summary"] = _artifact_ref(
        summary_path,
        root=run_root,
    )
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(
        CapacityDiagnosticError,
        match="independent endpoint replay",
    ):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("formal_b0_attempt_started", True),
        ("b0_gate_values_changed", True),
        ("budget_change_authorized", True),
        ("scientific_result_claimed", True),
        ("edit_class_counts", {"counterfeit": 2}),
        ("sum_exact_per_record_reachable_nodes", 999_999),
        ("maximum_exact_record_reachable_nodes", 999_999),
        ("records_above_frozen_state_gate", ["counterfeit"]),
        ("usable_for_b0_acceptance", True),
        ("allowed_interpretation", "B0_ACCEPTANCE"),
    ],
)
def test_capacity_summary_requires_full_independent_recomputation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    summary_path = run_root / "capacity_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    _write_json(summary_path, summary)
    manifest["accounting"]["capacity_summary"] = _artifact_ref(
        summary_path,
        root=run_root,
    )
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError, match="summary differs"):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_partial_evidence_is_included_in_seal_and_validates(
    tmp_path: Path,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    partial_path = run_root / "record_workspaces/000001/layer.tsv.partial"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_text("A\t1\n", encoding="utf-8", newline="")
    _seal_test_bundle(run_root, manifest)

    _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)
    checksum_index = json.loads(
        (run_root / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    assert "record_workspaces/000001/layer.tsv.partial" in {
        item["path"] for item in checksum_index["artifacts"]
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "wrong"),
        ("terminal_state", "SAFE_PAUSED"),
        ("return_code", 3),
        ("capture_mode", "PLACEHOLDER"),
        ("capture_complete", False),
        ("capture_scope", "incomplete"),
        ("return_code_semantics", "observed_without_wrapper"),
    ],
)
def test_process_result_contract_is_independently_verified(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    process_path = run_root / "provenance/process_result.json"
    process = json.loads(process_path.read_text(encoding="utf-8"))
    process[field] = value
    _write_json(process_path, process)
    manifest["provenance"]["process_result"] = _artifact_ref(
        process_path,
        root=run_root,
    )
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError, match="process result"):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_nonempty_captured_warning_is_retained_and_hash_bound(
    tmp_path: Path,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    stdout_path = run_root / "logs/stdout.log"
    stdout_path.write_text(
        "unexpected warning retained as evidence\n",
        encoding="utf-8",
        newline="",
    )
    process_path = run_root / "provenance/process_result.json"
    process = json.loads(process_path.read_text(encoding="utf-8"))
    process["stdout"] = _artifact_ref(stdout_path, root=run_root)
    _write_json(process_path, process)
    manifest["provenance"]["process_result"] = _artifact_ref(
        process_path,
        root=run_root,
    )
    _seal_test_bundle(run_root, manifest)

    _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)
    assert process["stdout"]["bytes"] > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_unique_state_count", 999_999_999),
        ("global_state_universe_sha256", "0" * 64),
        (
            "global_state_universe_artifact",
            {
                "path": "global/eligible_endpoints.tsv",
                "bytes": 2,
                "sha256": hashlib.sha256(b"A\n").hexdigest(),
            },
        ),
    ],
)
def test_global_universe_summary_cannot_be_counterfeited_and_resealed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    summary_path = run_root / "capacity_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    _write_json(summary_path, summary)
    manifest["accounting"]["capacity_summary"] = _artifact_ref(
        summary_path,
        root=run_root,
    )
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError, match=r"global[- ]universe"):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_replay_script_cannot_be_replaced_by_an_arbitrary_command(
    tmp_path: Path,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    replay_path = run_root / "replay.sh"
    replay_path.write_text(
        "#!/bin/sh\nexec /bin/true\n",
        encoding="utf-8",
        newline="",
    )
    replay_path.chmod(0o750)
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError, match="exact rendering"):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_census_parent_authorization_paths_are_frozen_in_child_manifest(
    tmp_path: Path,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    manifest["parent_authorization"]["verified_marker"]["path"] = (
        "/mnt/cunyuliu/mrna_editflow_b0_capacity/"
        "B0_capacity_20260729T140000Z_2e58254/COUNTERFEIT"
    )
    _seal_test_bundle(run_root, manifest)

    with pytest.raises(CapacityDiagnosticError, match="parent authorization"):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


def test_census_parent_authorization_requires_live_bundle_hashes_and_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_id = "B0_capacity_20260729T140000Z_2e58254"
    monkeypatch.setattr(capacity_diagnostic, "ALLOWED_OUTPUT_PARENT", tmp_path)
    parent_root = tmp_path / parent_id
    (parent_root / "provenance").mkdir(parents=True)
    shared_git = {
        "project_root": str(ROOT),
        "git_dir": str(tmp_path / "gitdir"),
        "head": "2" * 40,
        "clean": True,
        "dirty_state_sha256": "3" * 64,
    }
    shared_snapshot = _reference(str(tmp_path / "snapshot.json"))
    shared_acceptance = _reference(str(tmp_path / "acceptance.json"))
    shared_structural = {
        **_reference(str(tmp_path / "candidates.jsonl")),
        "record_count": 2,
        "record_ids_sha256": "4" * 64,
        "structural_content_sha256": "5" * 64,
        "role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
    }
    goal_contract = {
        "id": "utr_editflow_goal_v2",
        "sha256": GOAL_SHA256,
        "repository_snapshot": "docs/contracts/mrna_latest_build_contract_v2.md",
    }
    algorithm_contract = {"frozen": True}
    parent_manifest = {
        "diagnostic_id": parent_id,
        "state": "COMPLETED",
        "parent_diagnostic_id": None,
        "selection": {"diagnostic_scope": "witness"},
        "goal_contract": goal_contract,
        "algorithm_contract": algorithm_contract,
        "provenance": {
            "git": shared_git,
            "d1_snapshot": shared_snapshot,
            "d1_acceptance": shared_acceptance,
            "structural_store": shared_structural,
        },
    }
    _write_json(parent_root / "diagnostic_manifest.json", parent_manifest)
    resolved_config = json.loads(
        (ROOT / "configs/b0_capacity_diagnostic_v1.json").read_text(encoding="utf-8")
    )
    _write_json(parent_root / "resolved_config.json", resolved_config)
    for path in (
        parent_root / "VERIFIED",
        parent_root / "bundle_seal.json",
        parent_root / "terminal.lock",
        parent_root / "provenance/process_result.json",
    ):
        path.write_text("{}\n", encoding="utf-8", newline="")
    expected_paths = {
        "diagnostic_manifest": parent_root / "diagnostic_manifest.json",
        "verified_marker": parent_root / "VERIFIED",
        "bundle_seal": parent_root / "bundle_seal.json",
        "terminal_lock": parent_root / "terminal.lock",
        "process_result": parent_root / "provenance/process_result.json",
    }
    authorization = {
        "diagnostic_id": parent_id,
        **{field: _artifact_ref(path) for field, path in expected_paths.items()},
    }
    child_manifest = {
        "goal_contract": goal_contract,
        "algorithm_contract": algorithm_contract,
        "provenance": {
            "git": shared_git,
            "d1_snapshot": shared_snapshot,
            "d1_acceptance": shared_acceptance,
            "structural_store": shared_structural,
        },
    }
    recursive_calls: list[Path] = []
    verified_calls: list[Path] = []
    monkeypatch.setattr(
        capacity_diagnostic,
        "_validate_bundle",
        lambda *, run_root, schema_path: recursive_calls.append(run_root),
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "_validate_verified_marker",
        lambda run_root: verified_calls.append(run_root),
    )

    _ORIGINAL_VALIDATE_EXTERNAL_PARENT_AUTHORIZATION(
        parent_id=parent_id,
        parent_authorization=authorization,
        child_manifest=child_manifest,
        resolved_config=resolved_config,
        schema_path=SCHEMA_PATH,
    )
    assert recursive_calls == [parent_root]
    assert verified_calls == [parent_root]

    forged = copy.deepcopy(authorization)
    forged["verified_marker"]["sha256"] = "f" * 64
    with pytest.raises(CapacityDiagnosticError, match="bytes or SHA-256"):
        _ORIGINAL_VALIDATE_EXTERNAL_PARENT_AUTHORIZATION(
            parent_id=parent_id,
            parent_authorization=forged,
            child_manifest=child_manifest,
            resolved_config=resolved_config,
            schema_path=SCHEMA_PATH,
        )

    (parent_root / "VERIFIED").unlink()
    with pytest.raises(CapacityDiagnosticError, match="bytes or SHA-256"):
        _ORIGINAL_VALIDATE_EXTERNAL_PARENT_AUTHORIZATION(
            parent_id=parent_id,
            parent_authorization=authorization,
            child_manifest=child_manifest,
            resolved_config=resolved_config,
            schema_path=SCHEMA_PATH,
        )


def test_verified_marker_is_post_seal_and_binds_committed_return_code(
    tmp_path: Path,
) -> None:
    run_root, _manifest_payload = _bundle_fixture(tmp_path)
    metric = {
        "captured_at_utc": "2026-07-29T16:00:01Z",
        "elapsed_seconds": 3600.0,
        "free_bytes": 1,
        "peak_rss_bytes": 1,
        "run_bytes": 1,
        "total_bytes": 2,
        "used_bytes": 1,
    }
    marker_path = run_root / "VERIFIED"
    _write_json(
        marker_path,
        _verified_marker_payload(
            run_root=run_root,
            state="COMPLETED",
            verified_at_utc="2026-07-29T16:00:01Z",
            postseal_system_metric=metric,
        ),
    )

    _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)
    _validate_verified_marker(run_root)

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["committed_return_code"] = 3
    _write_json(marker_path, marker)
    with pytest.raises(CapacityDiagnosticError, match="VERIFIED"):
        _validate_verified_marker(run_root)


def test_parent_witness_rejects_constant_row_with_counterfeit_state_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_id = "B0_capacity_20260729T140000Z_2e58254"
    parent_root = tmp_path / parent_id
    (parent_root / "provenance").mkdir(parents=True)
    (parent_root / "records").mkdir()
    candidate_store = tmp_path / "candidates.jsonl"
    d1_snapshot = tmp_path / "d1_snapshot.json"
    candidate_store.write_text("{}\n", encoding="utf-8", newline="")
    d1_snapshot.write_text("{}\n", encoding="utf-8", newline="")
    config = json.loads(
        (ROOT / "configs/b0_capacity_diagnostic_v1.json").read_text(encoding="utf-8")
    )
    witness_record = {
        "record_id": capacity_diagnostic.WITNESS_RECORD_ID,
        "_canonical_jsonl_line": (capacity_diagnostic.WITNESS_CANONICAL_JSONL_LINE),
        "source_sequence": capacity_diagnostic.WITNESS_SOURCE,
        "candidate_sequence": capacity_diagnostic.WITNESS_CANDIDATE,
    }
    row = {
        "ordinal": 0,
        "record_id": capacity_diagnostic.WITNESS_RECORD_ID,
        "canonical_jsonl_line": (capacity_diagnostic.WITNESS_CANONICAL_JSONL_LINE),
        "source_length": len(capacity_diagnostic.WITNESS_SOURCE),
        "candidate_length": len(capacity_diagnostic.WITNESS_CANDIDATE),
        "input_record_structural_sha256": "a" * 64,
        "source_sequence_sha256": hashlib.sha256(
            capacity_diagnostic.WITNESS_SOURCE.encode("utf-8")
        ).hexdigest(),
        "candidate_sequence_sha256": hashlib.sha256(
            capacity_diagnostic.WITNESS_CANDIDATE.encode("utf-8")
        ).hexdigest(),
        "outcome": "EXACT_COMPLETED",
        "capacity": {
            "exact": capacity_diagnostic.WITNESS_EXPECTED,
            "lower_bound": None,
        },
        "alignment_statistics": {"counts_exact": True},
        "evidence_semantics": {
            "counts_exact": True,
            "state_set_complete": True,
        },
    }
    _write_json(
        parent_root / "diagnostic_manifest.json",
        {
            "diagnostic_id": parent_id,
            "parent_diagnostic_id": None,
            "state": "COMPLETED",
            "selection": {
                "diagnostic_scope": "witness",
                "selected_record_count": 1,
            },
            "accounting": {
                "scheduled_record_count": 1,
                "terminal_record_count": 1,
                "exact_completed_count": 1,
            },
            "result_semantics": {
                "exact_capacity_complete": False,
                "usable_for_budget_decision": False,
            },
            "provenance": {
                "git": {"head": "2" * 40},
                "d1_snapshot": _artifact_ref(d1_snapshot),
                "structural_store": _artifact_ref(candidate_store),
            },
            "algorithm_contract": {
                "frozen_b0_limits": capacity_diagnostic.FROZEN_B0_LIMITS,
                "diagnostic_safety_limits": config["diagnostic_safety_limits"],
            },
        },
    )
    _write_json(
        parent_root / "provenance/selection_manifest.json",
        {
            "scope": "witness",
            "selected_record_count": 1,
            "selected_records": [{"record_id": capacity_diagnostic.WITNESS_RECORD_ID}],
        },
    )
    _write_json(
        parent_root / "capacity_summary.json",
        {
            "diagnostic_scope": "witness",
            "state_universe_scope": "FROZEN_WITNESS_SUBSET",
            "global_state_universe_exact": True,
            "global_unique_state_count": 1,
            "global_state_universe_sha256": hashlib.sha256(
                b"COUNTERFEIT\n"
            ).hexdigest(),
            "full_census_state_universe_exact": False,
        },
    )
    _write_json(parent_root / "resolved_config.json", config)
    (parent_root / "records/results.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )

    monkeypatch.setattr(
        capacity_diagnostic,
        "ALLOWED_OUTPUT_PARENT",
        tmp_path,
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "_validate_bundle",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "_validate_verified_marker",
        lambda _root: None,
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "_load_structural_selection",
        lambda **_kwargs: type(
            "Selection",
            (),
            {"records": (witness_record,)},
        )(),
    )
    monkeypatch.setattr(
        capacity_diagnostic,
        "record_structural_sha256",
        lambda _record: "a" * 64,
    )

    with pytest.raises(
        CapacityDiagnosticError,
        match="not the exact completed witness",
    ):
        capacity_diagnostic._validate_parent_witness(
            parent_diagnostic_id=parent_id,
            project_root=ROOT,
            expected_code_commit="2" * 40,
            d1_snapshot=d1_snapshot,
            candidate_store=candidate_store,
            resolved_config=config,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "selection_count",
        "outcome_count",
        "seal_marker_path",
        "empty_terminal_lock",
    ],
)
def test_bundle_verifier_rejects_schema_valid_cross_field_and_seal_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_root, manifest = _bundle_fixture(tmp_path)
    if mutation == "selection_count":
        manifest["selection"]["selected_record_count"] = 3
        manifest["selection"]["excluded_record_count"] = 44_148
        _seal_test_bundle(run_root, manifest)
    elif mutation == "outcome_count":
        manifest["accounting"]["outcome_counts"]["EXACT_COMPLETED"] = 2
        _seal_test_bundle(run_root, manifest)
    elif mutation == "seal_marker_path":
        manifest["completion_seal"]["terminal_marker_ref"]["path"] = "WRONG"
        _seal_test_bundle(run_root, manifest, update_marker_ref=False)
    else:
        (run_root / "terminal.lock").write_text("", encoding="utf-8", newline="")
    with pytest.raises(CapacityDiagnosticError):
        _validate_bundle(run_root=run_root, schema_path=SCHEMA_PATH)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claim_boundary", "formal_b0_attempt_started"), True),
        (("claim_boundary", "b0_gate_values_changed"), True),
        (("claim_boundary", "b0_phase_acceptance_claimed"), True),
        (("claim_boundary", "scientific_result_claimed"), True),
        (("claim_boundary", "budget_change_authorized"), True),
        (("claim_boundary", "approximation_emitted"), True),
    ],
)
def test_claim_boundary_is_fail_closed(path: tuple[str, str], value: bool) -> None:
    manifest = _manifest()
    manifest[path[0]][path[1]] = value
    assert _errors(manifest)


def test_provenance_hashes_and_nested_objects_are_strict() -> None:
    manifest = _manifest()
    manifest["provenance"]["structural_store"]["structural_content_sha256"] = "bad"
    assert _errors(manifest)

    manifest = _manifest()
    manifest["provenance"]["unhashed_note"] = "not allowed"
    assert _errors(manifest)

    manifest = _manifest()
    manifest["provenance"]["code_files"] = []
    assert _errors(manifest)


def test_scope_and_opaque_label_integrity_access_are_explicit() -> None:
    manifest = _manifest()
    manifest["selection"]["diagnostic_scope"] = "witness"
    assert _errors(manifest)

    manifest = _manifest()
    manifest["provenance"]["canonical_store_metadata_only"][
        "opaque_integrity_hash_opened"
    ] = False
    assert _errors(manifest)

    manifest = _manifest()
    manifest["provenance"]["canonical_store_metadata_only"][
        "label_values_accessed"
    ] = True
    assert _errors(manifest)


def test_exact_and_lower_bound_rows_are_mutually_exclusive() -> None:
    exact = _record_result(exact=True)
    lower = _record_result(exact=False)
    assert _row_errors(exact) == []
    assert _row_errors(lower) == []

    mixed = copy.deepcopy(exact)
    mixed["capacity"]["lower_bound"] = _lower_bounds()
    assert _row_errors(mixed)

    false_exact = copy.deepcopy(lower)
    false_exact["capacity"]["exact"] = _exact_counts()
    assert _row_errors(false_exact)

    overstated = copy.deepcopy(lower)
    overstated["evidence_semantics"]["counts_exact"] = True
    assert _row_errors(overstated)

    dag_lower = copy.deepcopy(lower)
    dag_lower["alignment_statistics"]["minimum_alignment_count"] = None
    dag_lower["alignment_statistics"]["counts_exact"] = False
    dag_lower["stop"]["dimension"] = "max_dag_cells"
    dag_lower["stop"]["frozen_limit"] = 1_000_000
    dag_lower["stop"]["observed_lower_bound"] = 1_000_001
    assert _row_errors(dag_lower) == []

    contradictory_alignment = copy.deepcopy(dag_lower)
    contradictory_alignment["alignment_statistics"]["minimum_alignment_count"] = 2_340
    assert _row_errors(contradictory_alignment)


def test_completed_manifest_requires_reconciled_accounting_and_seal() -> None:
    manifest = _manifest()
    manifest["accounting"]["all_scheduled_records_accounted"] = False
    assert _errors(manifest)

    manifest = _manifest()
    manifest["accounting"]["accounting_reconciled"] = False
    assert _errors(manifest)

    manifest = _manifest()
    manifest["accounting"]["failed_record_count"] = 1
    assert _errors(manifest)

    manifest = _manifest()
    manifest["completion_seal"] = None
    assert _errors(manifest)


def test_safe_pause_and_failure_semantics_are_distinct_from_completion() -> None:
    paused = _manifest()
    paused["state"] = "SAFE_PAUSED"
    paused["stop_reason"] = "DIAGNOSTIC_DISK_SAFETY_LIMIT"
    paused["completion_seal"] = None
    paused["accounting"]["all_scheduled_records_accounted"] = False
    paused["accounting"]["accounting_reconciled"] = False
    paused["failure"] = {
        "failed_at_utc": "2026-07-29T16:00:00Z",
        "failure_class": "SAFE_RESOURCE_PAUSE",
        "reason": "DIAGNOSTIC_DISK_SAFETY_LIMIT",
        "evidence": _reference("failure/failure.json", sha256="1" * 64),
        "partial_outputs_preserved": True,
        "resume_allowed": True,
    }
    assert _errors(paused) == []

    failed = copy.deepcopy(paused)
    failed["state"] = "FAILED_WITH_EVIDENCE"
    failed["failure"]["failure_class"] = "INTEGRITY_FAILURE"
    failed["failure"]["resume_allowed"] = False
    assert _errors(failed) == []

    failed["completion_seal"] = _manifest()["completion_seal"]
    assert _errors(failed)

    completed_with_failure = _manifest()
    completed_with_failure["failure"] = paused["failure"]
    assert _errors(completed_with_failure)


def test_selection_and_frozen_witness_cannot_claim_label_access_or_false_parity() -> (
    None
):
    manifest = _manifest()
    manifest["selection"]["canonical_label_store_opened_by_selection"] = True
    assert _errors(manifest)

    manifest = _manifest()
    manifest["selection"]["label_fields_read"] = ["delta_raw"]
    assert _errors(manifest)

    manifest = _manifest()
    manifest["witnesses"][0]["parity_passed"] = False
    assert _errors(manifest)

    manifest = _manifest()
    manifest["witnesses"][0]["expected"]["reachable_node_count"] = 50_000
    assert _errors(manifest)


def test_frozen_limits_cannot_be_raised_inside_diagnostic_manifest() -> None:
    manifest = _manifest()
    manifest["algorithm_contract"]["frozen_b0_limits"]["max_reachable_states"] = 100_000
    assert _errors(manifest)
