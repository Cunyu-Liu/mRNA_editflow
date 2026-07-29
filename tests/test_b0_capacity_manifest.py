from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/b0_capacity_diagnostic.schema.json"
GOAL_SHA256 = "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
HEX64 = "a" * 64


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


def _lower_bounds() -> dict:
    return {
        "reachable_node_count": 50_001,
        "reachable_transition_count": 1,
        "minimum_state_path_count": 1,
        "evaluated_primitive_action_count": 1,
        "evaluated_state_dp_cell_count": 0,
    }


def _record_result(*, exact: bool) -> dict:
    return {
        "schema_version": "b0_capacity_record.v1",
        "ordinal": 0 if exact else 1,
        "record_id": (
            "GSE217518:record:025e56d3b64660abb559dcbd"
            if exact
            else "GSE217518:record:bounded"
        ),
        "dataset_id": "GSE217518",
        "region": "five_utr",
        "canonical_jsonl_line": 39_913 if exact else 39_914,
        "input_record_structural_sha256": "b" * 64,
        "source_sequence_sha256": "c" * 64,
        "candidate_sequence_sha256": "d" * 64,
        "source_length": 129,
        "candidate_length": 114,
        "alignment_statistics": {
            "minimum_edit_count": 15,
            "minimum_alignment_count": 2_340,
            "evaluated_dag_cell_count": 1_950,
            "counts_exact": True,
        },
        "outcome": "EXACT_COMPLETED" if exact else "LOWER_BOUND_STOPPED",
        "capacity": {
            "exact": _exact_counts() if exact else None,
            "lower_bound": None if exact else _lower_bounds(),
        },
        "frozen_gate_assessment": {
            "would_pass_frozen_b0_limits": False,
            "exceeded_limits": ["max_reachable_states"],
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
        "parent_diagnostic_id": None,
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
            "regions": ["five_utr", "three_utr"],
            "record_order": (
                "FROZEN_WITNESS_FIRST_THEN_CANONICAL_JSONL_LINE_ASCENDING"
            ),
            "source_record_count": 44_151,
            "selected_record_count": 2,
            "excluded_record_count": 44_149,
            "selected_record_ids_sha256": "e" * 64,
            "label_fields_read": [],
            "canonical_label_store_opened": False,
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
                "expected": _exact_counts(),
                "observed_record_ordinal": 0,
                "parity_passed": True,
            }
        ],
        "provenance": {
            "git": {
                "project_root": "/mnt/cunyuliu/worktree",
                "head": "2" * 40,
                "clean": True,
                "dirty_state_sha256": "3" * 64,
            },
            "exact_argv": [
                "/mnt/runtime/bin/python",
                "scripts/data/diagnose_b0_path_capacity.py",
                "--config",
                "configs/b0_capacity_v1.json",
            ],
            "exact_argv_sha256": "4" * 64,
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
                "opened": False,
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
            "artifact_checksum_index": _reference(
                "artifact_checksums.json",
                sha256="9" * 64,
            ),
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


def test_schema_is_strict_draft_2020_12_and_valid_manifest_passes() -> None:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert _errors(_manifest()) == []


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
    manifest["selection"]["canonical_label_store_opened"] = True
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
