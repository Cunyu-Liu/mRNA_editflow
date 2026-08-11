from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse200304_dec019_upstream_pass_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse200304_dec019_upstream_pass_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("gse200304_evt042_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


EXPECTED_SOURCE_OUTPUT_NAMES = [
    "PMC10540565_EUROPE_PMC_FULLTEXT.xml",
    "GSE200302_family.soft.gz",
    "GSE200302_log2_cpm_counts_all_samples.txt.gz",
    "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json",
    "SHA256SUMS",
    "PUBLICATION_COMMIT.json",
    "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json",
    "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json",
    "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json",
    "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json",
    "SHA256SUMS",
    "PUBLICATION_COMMIT.json",
    "ADJUDICATION_REPORT.json",
    "INPUT_EVIDENCE_AUDIT.json",
    "SHA256SUMS",
    "PUBLICATION_COMMIT.json",
]


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)


def normalize_i_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    result = copy.deepcopy(config if config is not None else read_config())
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        result["implementation_binding"][key] = runtime_sync.UNKNOWN
    return result


def bind_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    result = normalize_i_config(config)
    binding = result["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": "b" * 40,
            "implementation_script_sha256": runtime_sync.sha256(b"synthetic EVT042 runtime script\n"),
            "implementation_test_sha256": runtime_sync.sha256(b"synthetic EVT042 runtime test\n"),
        }
    )
    refresh_core(result)
    return result


def update_member_identities(
    config: dict[str, Any], source_key: str, payloads: Mapping[str, bytes]
) -> None:
    for item in config["registered_evidence"][source_key]["members"]:
        payload = payloads[item["name"]]
        item["bytes"] = len(payload)
        item["sha256"] = runtime_sync.sha256(payload)


def synthetic_sources(config: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    registered = config["registered_evidence"]

    upstream_spec = registered[runtime_sync.UPSTREAM_AUTHORITY_KEY]
    upstream: dict[str, bytes] = {
        "PMC10540565_EUROPE_PMC_FULLTEXT.xml": b"<article>synthetic authority</article>\n",
        "GSE200302_family.soft.gz": b"synthetic-gzip-soft-body\n",
        "GSE200302_log2_cpm_counts_all_samples.txt.gz": b"synthetic-gzip-aggregate-matrix-body-never-decode-rows\n",
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json": runtime_sync.json_bytes(
            {
                "schema_version": "1.0.0",
                "record_type": "GSE200304_UPSTREAM_SOURCE_AUTHORITY_VIABILITY_AUDIT_V1",
                "protocol_id": upstream_spec["protocol_id"],
                "contract_id": config["contract_id"],
                "phase_id": "A1",
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE",
                "mode": "AUDIT_ONLY_NO_GATE_CHANGE",
                "endpoint_crosswalk": {
                    "status_if_all_source_checks_pass": "READY_FOR_PASS_RECORD_NOT_YET_BOUND",
                    "consumer_gate_pass": False,
                },
                "replicate_branch": {
                    "status_if_all_source_checks_pass": "READY_FOR_REPLICATE_BRANCH_PASS_RECORD_NOT_YET_BOUND",
                    "consumer_gate_pass": False,
                },
                "private_only_rights": {
                    "status_if_all_source_checks_pass": "READY_FOR_PRIVATE_CANONICAL_ONLY_PASS_RECORD_NOT_YET_BOUND",
                    "consumer_gate_pass": False,
                },
                "biological_group_authority": {
                    "status": "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT"
                },
                "processed_matrix_authority": {
                    "matrix_key_set_equals_s3_key_set": True,
                    "matrix_covers_every_finite_totalpoly_key": True,
                },
            }
        ),
    }
    upstream_content_names = tuple(upstream)
    upstream["SHA256SUMS"] = "".join(
        f"{runtime_sync.sha256(upstream[name])}  {name}\n"
        for name in sorted(upstream_content_names)
    ).encode("ascii")
    upstream["PUBLICATION_COMMIT.json"] = runtime_sync.json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": upstream_spec["terminal_record_type"],
            "protocol_id": upstream_spec["protocol_id"],
            "contract_id": config["contract_id"],
            "dataset_id": "GSE200304",
            "bundle_id": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1",
            "preterminal_member_names": sorted((*upstream_content_names, "SHA256SUMS")),
            "preterminal_member_count": 5,
            "exact_final_member_count": 6,
            "sha256sums_sha256": runtime_sync.sha256(upstream["SHA256SUMS"]),
            "final_output_target_sha256": upstream_spec["final_output_target_sha256"],
            "publication_mode": upstream_spec["publication_mode"],
            "committed": True,
            "terminal_marker_written_last": True,
            "no_overwrite": True,
            "partial_default": upstream_spec["partial_default"],
        }
    )
    update_member_identities(config, runtime_sync.UPSTREAM_AUTHORITY_KEY, upstream)

    pass_spec = registered[runtime_sync.UPSTREAM_PASS_PACK_KEY]
    gate_names = [
        "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json",
        "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json",
        "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json",
    ]
    pass_pack: dict[str, bytes] = {
        name: runtime_sync.json_bytes({"status": "PASS", "aggregate_only": True})
        for name in gate_names
    }
    pass_audit_name = "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json"
    pass_pack[pass_audit_name] = runtime_sync.json_bytes(
        {
            "record_type": "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT_V1",
            "protocol_id": pass_spec["protocol_id"],
            "contract_id": config["contract_id"],
            "dataset_id": "GSE200304",
            "decision_id": "V3-DEC-019",
            "status": "PASS_EXACT_THREE_CONSUMER_ACCEPTED_GATES_NO_ADJUDICATION",
            "upstream_exact6_verified": True,
            "decoded_raw_source_count": 0,
            "pass_gate_ids": [
                "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
                "LICENSE_RIGHTS",
                "ROW_REPLICATE_OR_VALID_SE",
            ],
            "consumer_validate_gate_record_pass_count": 3,
            "consumer_slot_gate_pass_exact_true_count": 3,
            "ordinary_study_contribution_delta": 0,
            "a1_study_contribution_delta": 0,
            "true_a2_study_contribution_delta": 0,
            "canonical_record_count_delta": 0,
            "aggregate_only": True,
        }
    )
    pass_content_names = tuple(sorted((*gate_names, pass_audit_name)))
    pass_pack["SHA256SUMS"] = "".join(
        f"{runtime_sync.sha256(pass_pack[name])}  {name}\n" for name in pass_content_names
    ).encode("ascii")
    pass_spec["payload_set_sha256"] = runtime_sync._pass_pack_payload_set_sha256(pass_pack)
    pass_pack["PUBLICATION_COMMIT.json"] = runtime_sync.json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": pass_spec["terminal_record_type"],
            "protocol_id": pass_spec["protocol_id"],
            "contract_id": config["contract_id"],
            "dataset_id": "GSE200304",
            "decision_id": "V3-DEC-019",
            "publication_mode": pass_spec["publication_mode"],
            "preterminal_member_names": sorted((*pass_content_names, "SHA256SUMS")),
            "preterminal_member_count": 5,
            "exact_final_member_count": 6,
            "gate_record_names": sorted(gate_names),
            "descriptor_binding_scope": "THREE_GATE_JSON_FILES_ONLY",
            "sha256sums_sha256": runtime_sync.sha256(pass_pack["SHA256SUMS"]),
            "payload_set_sha256": pass_spec["payload_set_sha256"],
            "final_output_target_sha256": pass_spec["final_output_target_sha256"],
            "committed": True,
            "commit_marker_written_last": True,
            "overwrite_allowed": False,
        }
    )
    update_member_identities(config, runtime_sync.UPSTREAM_PASS_PACK_KEY, pass_pack)

    adjudication_spec = registered[runtime_sync.ADJUDICATION_KEY]
    adjudication: dict[str, bytes] = {
        "ADJUDICATION_REPORT.json": runtime_sync.json_bytes(
            {
                "status": adjudication_spec["scientific_status"],
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "aggregate_only": True,
                "blockers": adjudication_spec["unresolved_blockers"],
                "config_core_sha256": adjudication_spec["science_core_sha256"],
                "evidence_descriptor_set_sha256": adjudication_spec[
                    "evidence_descriptor_set_sha256"
                ],
            }
        ),
        "INPUT_EVIDENCE_AUDIT.json": runtime_sync.json_bytes(
            {
                "mode": "ALL_HASH_BOUND_AGGREGATES_VERIFIED",
                "all_inputs_aggregate_only": True,
                "row_level_payload_read_count": 0,
                "sequence_read_count": 0,
                "opened_input_count": 8,
                "evidence_descriptor_set_sha256": adjudication_spec[
                    "evidence_descriptor_set_sha256"
                ],
                "slots": [
                    {
                        "slot_id": "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE",
                        "gate_status": "PASS",
                    },
                    {
                        "slot_id": "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
                        "gate_status": "PASS",
                    },
                    {"slot_id": "BIOLOGICAL_GROUP_AUTHORITY", "gate_status": "BLOCKED"},
                    {"slot_id": "ROW_REPLICATE_OR_VALID_SE", "gate_status": "PASS"},
                    {
                        "slot_id": "CHECKPOINT_SPECIFIC_EXPOSURE",
                        "gate_status": runtime_sync.UNKNOWN,
                    },
                    {"slot_id": "LICENSE_RIGHTS", "gate_status": "PASS"},
                    {"slot_id": "OUTCOME_BLIND_SPLIT_LEAKAGE", "gate_status": "NOT_RUN"},
                    {"slot_id": "PREFROZEN_POWER_PRECISION", "gate_status": "NOT_RUN"},
                ],
            }
        ),
    }
    adjudication["SHA256SUMS"] = "".join(
        f"{runtime_sync.sha256(adjudication[name])}  {name}\n"
        for name in sorted(adjudication)
    ).encode("ascii")
    adjudication["PUBLICATION_COMMIT.json"] = runtime_sync.json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": adjudication_spec["terminal_record_type"],
            "contract_id": config["contract_id"],
            "decision_id": "V3-DEC-019",
            "dataset_id": "GSE200304",
            "output_id": adjudication_spec["output_id"],
            "scientific_status": adjudication_spec["scientific_status"],
            "publication_mode": adjudication_spec["publication_mode"],
            "sha256sums_sha256": runtime_sync.sha256(adjudication["SHA256SUMS"]),
            "bundle_member_names_excluding_commit_marker": [
                "ADJUDICATION_REPORT.json",
                "INPUT_EVIDENCE_AUDIT.json",
                "SHA256SUMS",
            ],
            "bundle_file_count_excluding_commit_marker": 3,
            "final_output_target_sha256": adjudication_spec["final_output_target_sha256"],
            "committed": True,
            "commit_marker_written_last": True,
            "aggregate_acceptance_requires_exact_marker": True,
        }
    )
    update_member_identities(config, runtime_sync.ADJUDICATION_KEY, adjudication)
    refresh_core(config)
    return {
        runtime_sync.UPSTREAM_AUTHORITY_KEY: upstream,
        runtime_sync.UPSTREAM_PASS_PACK_KEY: pass_pack,
        runtime_sync.ADJUDICATION_KEY: adjudication,
    }


def predecessor_payloads() -> dict[str, bytes]:
    status = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "training_started": False,
        "next_phase_authorized": False,
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "updated_at": "2026-08-11T21:40:09+08:00",
        "historical_field": "PRESERVED",
    }
    manifest = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        "outputs": [
            {
                "artifact_type": f"PREDECESSOR_{index:03d}",
                "absolute_path": f"/predecessor/{index:03d}",
                "sha256": f"{index:064x}",
                "status": "COMPLETE",
            }
            for index in range(143)
        ],
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-10T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 41)
    ]
    events.append(
        {
            "event_id": "A1-EVT-041",
            "at": "2026-08-11T21:40:09+08:00",
            "event": "GSE200304_UPSTREAM_AUTHORITY_PASS_GATES_LEDGER_REGISTERED_PENDING_RUNTIME_SYNC",
            "training_started": False,
            "training_allowed": False,
            "next_phase_authorized": False,
        }
    )
    return {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(runtime_sync.compact_json_line(item) for item in events),
    }


def make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, dict[str, bytes]],
    dict[str, Any],
    Path,
    Path,
]:
    config = bind_config()
    run_root = tmp_path / "run"
    allowed = tmp_path / "allowed"
    prepared = allowed / "evt042-job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    for source_key in runtime_sync.SOURCE_KEYS:
        config["registered_evidence"][source_key]["absolute_directory"] = str(
            tmp_path / f"source-{source_key}"
        )
    sources = synthetic_sources(config)
    predecessor = predecessor_payloads()
    for name, payload in predecessor.items():
        spec = config["runtime"]["predecessor_mutables"][name]
        spec["bytes"] = len(payload)
        spec["sha256"] = runtime_sync.sha256(payload)
    tail_line = predecessor["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail_event"].update(
        {
            "bytes": len(tail_line),
            "sha256": runtime_sync.sha256(tail_line),
            "at": "2026-08-11T21:40:09+08:00",
        }
    )
    refresh_core(config)
    run_root.mkdir()
    allowed.mkdir()
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    authority = {
        "status": "PASS_SYNTHETIC_D3_LEDGER_RUNTIME_I_CONFIG_ONLY_B",
        "binding_commit": "c" * 40,
        "head_commit": "c" * 40,
        "origin_branch_head_commit": "c" * 40,
        "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config)),
        "base_commit": config["repository_authority"]["base_commit"],
        "implementation_commit": config["implementation_binding"]["implementation_commit"],
        "predecessor_ledger_commit": config["repository_authority"]["predecessor_ledger"][
            "commit"
        ],
        "upstream_authority_binding_commit": config["repository_authority"][
            "upstream_authority_producer_lifecycle"
        ]["binding_commit"],
        "upstream_pass_gate_binding_commit": config["repository_authority"][
            "upstream_pass_gate_producer_lifecycle"
        ]["binding_commit"],
        "adjudicator_descriptor_commit": config["repository_authority"]["adjudicator_lifecycle"][
            "descriptor_commit"
        ],
    }
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    return config, predecessor, sources, authority, run_root, prepared


def prepare_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, dict[str, bytes]],
    dict[str, Any],
    Path,
    Path,
]:
    context = make_context(tmp_path, monkeypatch)
    config, _predecessor, sources, authority, run_root, prepared = context
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T12:00:00+08:00",
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        source_payload_overrides=sources,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["event_id"] == "A1-EVT-042"
    assert result["manifest_output_transition"] == "143_TO_163"
    assert result["runtime_artifact_count"] == 7
    return context


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in root.iterdir()
        if item.is_file() and not item.is_symlink()
    }


def test_static_config_closes_final_ledger_types_core_and_i_to_b() -> None:
    disk_config = read_config()
    runtime_sync.validate_static_config(disk_config)
    disk_binding = disk_config["implementation_binding"]
    assert disk_binding["compiled_core_sha256"] == runtime_sync.compiled_core_sha256(
        disk_config
    )
    if runtime_sync._binding_values_are_unknown(disk_binding):
        with pytest.raises(runtime_sync.BindingError, match="implementation is not BOUND"):
            runtime_sync.validate_bound_config(disk_config)
    else:
        runtime_sync.validate_bound_config(disk_config)
        assert disk_binding["status"] == "BOUND"
        assert runtime_sync.HEX40.fullmatch(disk_binding["implementation_commit"])
        assert disk_binding["implementation_script_sha256"] == runtime_sync.sha256(
            SCRIPT_PATH.read_bytes()
        )
        assert disk_binding["implementation_test_sha256"] == runtime_sync.sha256(
            Path(__file__).read_bytes()
        )

    ledger = disk_config["repository_authority"]["predecessor_ledger"]
    assert ledger["commit"] == "ef2666e7a3e224f2043e7c647e10a4b8cadf01e8"
    assert ledger["expected_parent"] == "8084a1e2b68eaf84bd4befb2f232759d7540b97c"
    assert [item["sha256"] for item in ledger["frozen_blobs"]] == [
        "b1aceb1cf3d7dc2de4b77270045949659de64f70d7bc677e084b67c176a8beb1",
        "9ed5e415d96bbfe2c0fc6161fe4caa691b12ddae8fcc74c2f7dde0293123af8b",
        "f087235353a53574e63b969c1e06c110572c07244063e551bb1e98dd7b612028",
        "b35f0e3e22eebb19a03f0a8bff42b658a6561aa54e5f56c8231ea0ef2d6a9920",
    ]
    assert ledger["integration_id"] == (
        "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_UPSTREAM_PASS_GATE_PACK_V1"
    )
    assert ledger["registered_lineage_ids"] == {
        "upstream_authority": "gse200304_upstream_authority_viability_v1",
        "upstream_pass_gate_pack": "gse200304_dec019_upstream_pass_gate_pack_v1",
        "updated_blocked_adjudication": (
            "gse200304_dec019_reported_endpoint_a1_adjudication_v3_upstream_pass_gate_pack_v1"
        ),
    }
    assert [len(disk_config["registered_evidence"][key]["members"]) for key in runtime_sync.SOURCE_KEYS] == [
        6,
        6,
        4,
    ]
    assert disk_config["runtime"]["allowed_mutable_states"] == [
        ["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "OLD_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "NEW_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "NEW_EXACT", "NEW_EXACT"],
    ]

    i_config = normalize_i_config(disk_config)
    runtime_sync.validate_static_config(i_config)
    with pytest.raises(runtime_sync.BindingError, match="implementation is not BOUND"):
        runtime_sync.validate_bound_config(i_config)
    bound = bind_config(i_config)
    runtime_sync.validate_bound_config(bound)
    changed_binding_paths = [
        f"implementation_binding.{key}"
        for key in i_config["implementation_binding"]
        if i_config["implementation_binding"][key] != bound["implementation_binding"][key]
    ]
    assert changed_binding_paths == i_config["implementation_binding"][
        "unknown_to_bound_scalar_paths"
    ]
    assert runtime_sync.expected_unknown_i_config(bound) == i_config
    assert runtime_sync.compiled_core_projection(bound) == runtime_sync.compiled_core_projection(
        i_config
    ) == runtime_sync.compiled_core_projection(disk_config)

    partial = copy.deepcopy(i_config)
    partial["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(runtime_sync.BindingError, match="partially known"):
        runtime_sync.validate_static_config(partial)

    wrong_type = bind_config(i_config)
    wrong_type["successor_invariants"]["training_allowed"] = 0
    refresh_core(wrong_type)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_static_config(wrong_type)

    wrong_artifact_type = bind_config(i_config)
    wrong_artifact_type["registered_evidence"][runtime_sync.UPSTREAM_AUTHORITY_KEY]["members"][0][
        "artifact_type"
    ] = False
    refresh_core(wrong_artifact_type)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_static_config(wrong_artifact_type)


@pytest.mark.parametrize(
    "mode",
    [
        "commit",
        "integration_id",
        "manifest_status",
        "lineage_upstream_authority",
        "lineage_upstream_pass_gate_pack",
        "lineage_updated_blocked_adjudication",
        "blob_0",
        "blob_1",
        "blob_2",
        "blob_3",
    ],
)
def test_static_config_rejects_recomputed_core_ledger_authority_mutation(mode: str) -> None:
    config = read_config()
    authority = config["repository_authority"]
    ledger = authority["predecessor_ledger"]
    if mode == "commit":
        mutated_commit = "d" * 40
        authority["base_commit"] = mutated_commit
        authority["current_pre_runtime_sync_head"] = mutated_commit
        ledger["commit"] = mutated_commit
    elif mode == "integration_id":
        ledger["integration_id"] = "MUTATED_CONCRETE_INTEGRATION_ID"
    elif mode == "manifest_status":
        ledger["manifest_status"] = "MUTATED_CONCRETE_MANIFEST_STATUS"
    elif mode.startswith("lineage_"):
        ledger["registered_lineage_ids"][mode.removeprefix("lineage_")] = (
            "mutated_concrete_lineage_v1"
        )
    else:
        ledger["frozen_blobs"][int(mode.removeprefix("blob_"))]["sha256"] = "0" * 64
    refresh_core(config)
    with pytest.raises(runtime_sync.RuntimeSyncError, match="frozen predecessor ledger authority"):
        runtime_sync.validate_static_config(config)


def test_unknown_binding_stops_before_repository_runtime_or_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("external-access")
        raise AssertionError("repository, runtime, or source was accessed")

    monkeypatch.setattr(runtime_sync, "audit_repo_authority", forbidden)
    monkeypatch.setattr(runtime_sync, "open_directory", forbidden)
    monkeypatch.setattr(runtime_sync, "validate_registered_bundles", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="implementation is not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-12T12:00:00+08:00",
            production=False,
            config_override=read_config(),
            repo_root=tmp_path / "repo",
            run_root_override=tmp_path / "run",
        )
    assert accessed == []


def test_exact20_output_delta_uses_ledger_order_and_registers_exact16_in_place() -> None:
    config = bind_config()
    delta = runtime_sync.expected_output_delta(config, "f" * 64)
    assert len(delta) == 20
    assert [Path(item["absolute_path"]).name for item in delta[:16]] == EXPECTED_SOURCE_OUTPUT_NAMES
    assert [Path(item["absolute_path"]).name for item in delta[16:]] == [
        config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in runtime_sync.MUTABLE_NAMES
    ] + [config["runtime"]["sync_name"]]
    expected_types = []
    for source_key, member_name in runtime_sync.SOURCE_MEMBER_OUTPUT_ORDER:
        members = {
            item["name"]: item
            for item in config["registered_evidence"][source_key]["members"]
        }
        expected_types.append(members[member_name]["artifact_type"])
    assert [item["artifact_type"] for item in delta[:16]] == expected_types
    assert len({item["absolute_path"] for item in delta}) == 20
    assert all(
        Path(item["absolute_path"]).parent
        == Path(config["registered_evidence"][source_key]["absolute_directory"])
        for item, (source_key, _member_name) in zip(delta[:16], runtime_sync.SOURCE_MEMBER_OUTPUT_ORDER)
    )
    assert not any(
        item["absolute_path"] in {
            frozen["path"]
            for frozen in config["repository_authority"]["predecessor_ledger"]["frozen_blobs"]
        }
        for item in delta
    )


@pytest.mark.parametrize(
    "mode",
    [
        "positive",
        "dirty",
        "i2_parent",
        "i2_paths",
        "i2_blob",
        "i_transition",
        "ancestor",
    ],
)
def test_repo_audit_proves_d3_ledger_i1_i2_config_only_b2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = bind_config()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo)
    authority = config["repository_authority"]
    authority["production_repo_root"] = str(repo)
    refresh_core(config)
    runtime_sync.validate_bound_config(config)

    binding = config["implementation_binding"]
    ledger = authority["predecessor_ledger"]
    upstream = authority["upstream_authority_producer_lifecycle"]
    pass_pack = authority["upstream_pass_gate_producer_lifecycle"]
    adjudicator = authority["adjudicator_lifecycle"]
    implementation = binding["implementation_commit"]
    historical_i1 = runtime_sync.HISTORICAL_RUNTIME_I1_COMMIT
    head = "c" * 40
    config_payload = runtime_sync.json_bytes(config)
    i_config = runtime_sync.expected_unknown_i_config(config)
    if mode == "i_transition":
        i_config["successor_invariants"]["training_allowed"] = 0
    i_payload = runtime_sync.json_bytes(i_config)
    runtime_script = b"synthetic EVT042 runtime script\n"
    runtime_test = b"synthetic EVT042 runtime test\n"
    historical_i1_script = b"frozen historical EVT042 runtime I1 script\n"
    historical_i1_test = b"frozen historical EVT042 runtime I1 test\n"

    blobs: dict[tuple[str, str], bytes] = {
        (head, runtime_sync.CONFIG_REPO_PATH): config_payload,
        (implementation, runtime_sync.CONFIG_REPO_PATH): i_payload,
        (historical_i1, runtime_sync.CONFIG_REPO_PATH): i_payload,
        (head, runtime_sync.SCRIPT_REPO_PATH): runtime_script,
        (implementation, runtime_sync.SCRIPT_REPO_PATH): runtime_script,
        (historical_i1, runtime_sync.SCRIPT_REPO_PATH): historical_i1_script,
        (head, runtime_sync.TEST_REPO_PATH): runtime_test,
        (implementation, runtime_sync.TEST_REPO_PATH): runtime_test,
        (historical_i1, runtime_sync.TEST_REPO_PATH): historical_i1_test,
    }
    digest_overrides: dict[bytes, str] = {
        i_payload: runtime_sync.HISTORICAL_RUNTIME_I1_BLOBS[runtime_sync.CONFIG_REPO_PATH],
        runtime_script: binding["implementation_script_sha256"],
        runtime_test: binding["implementation_test_sha256"],
        historical_i1_script: runtime_sync.HISTORICAL_RUNTIME_I1_BLOBS[
            runtime_sync.SCRIPT_REPO_PATH
        ],
        historical_i1_test: runtime_sync.HISTORICAL_RUNTIME_I1_BLOBS[
            runtime_sync.TEST_REPO_PATH
        ],
    }
    worktree_payloads: dict[str, bytes] = {
        runtime_sync.SCRIPT_REPO_PATH: runtime_script,
        runtime_sync.TEST_REPO_PATH: runtime_test,
    }

    for item in ledger["frozen_blobs"]:
        payload = f"ledger:{item['path']}\n".encode()
        digest_overrides[payload] = item["sha256"]
        worktree_payloads[item["path"]] = payload
        for commit in (ledger["commit"], historical_i1, implementation, head):
            blobs[(commit, item["path"])] = payload

    for lifecycle, blob_key, label in (
        (upstream, "bound_blobs", "upstream"),
        (pass_pack, "bound_blobs", "pass"),
        (adjudicator, "descriptor_blobs", "adjudicator"),
    ):
        current = lifecycle.get("binding_commit", lifecycle.get("descriptor_commit"))
        assert isinstance(current, str)
        expected = lifecycle[blob_key]
        for path_key, digest_key in (
            ("config_path", "config_sha256"),
            ("script_path", "script_sha256"),
            ("test_path", "test_sha256"),
        ):
            payload = f"{label}:{path_key}\n".encode()
            if path_key == "config_path" and "config_bytes" in expected:
                payload = payload.ljust(expected["config_bytes"], b"x")[: expected["config_bytes"]]
            digest_overrides[payload] = expected[digest_key]
            blobs[(current, lifecycle[path_key])] = payload
            blobs[(head, lifecycle[path_key])] = payload

    real_sha256 = runtime_sync.sha256
    monkeypatch.setattr(
        runtime_sync,
        "sha256",
        lambda payload: digest_overrides.get(payload, real_sha256(payload)),
    )
    parent_map = {
        head: implementation,
        implementation: historical_i1,
        historical_i1: ledger["commit"],
        ledger["commit"]: adjudicator["descriptor_commit"],
        adjudicator["descriptor_commit"]: pass_pack["binding_commit"],
    }
    changed_paths = {
        ledger["commit"]: ledger["commit_exact_changed_paths"],
        historical_i1: authority["implementation_commit_exact_changed_paths"],
        implementation: runtime_sync.RUNTIME_I2_EXACT_CHANGED_PATHS,
        head: authority["binding_commit_exact_changed_paths"],
        upstream["binding_commit"]: [upstream["config_path"]],
        pass_pack["binding_commit"]: [pass_pack["config_path"]],
        adjudicator["descriptor_commit"]: [adjudicator["config_path"]],
    }

    def fake_git(
        _repo: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)
    ) -> bytes:
        del allowed_returncodes
        branch = authority["branch"]
        if args == ("rev-parse", "HEAD") or args == ("rev-parse", "@{upstream}"):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--verify", f"refs/remotes/origin/{branch}"):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{branch}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{branch}\n".encode()
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b"M dirty\n" if mode == "dirty" else b""
        if len(args) == 2 and args[0] == "rev-parse" and args[1].endswith("^"):
            child = args[1][:-1]
            parent = parent_map[child]
            if mode == "i2_parent" and child == implementation:
                parent = "d" * 40
            return f"{parent}\n".encode()
        if len(args) == 2 and args[0] == "rev-parse":
            return f"{args[1]}\n".encode()
        if args[:2] == ("merge-base", "--is-ancestor"):
            if mode == "ancestor":
                raise runtime_sync.AuthorityError("synthetic non-ancestor")
            assert args[2:] == (upstream["binding_commit"], pass_pack["binding_commit"])
            return b""
        raise AssertionError(args)

    def fake_blob(_repo: Path, commit: str, path: str) -> bytes:
        payload = blobs[(commit, path)]
        if mode == "i2_blob" and commit == implementation and path == runtime_sync.SCRIPT_REPO_PATH:
            return payload + b"drift"
        return payload

    def fake_paths(_repo: Path, commit: str) -> list[str]:
        result = list(changed_paths[commit])
        if mode == "i2_paths" and commit == implementation:
            result.append("unexpected")
        return sorted(result)

    def fake_read(path: Path) -> bytes:
        return worktree_payloads[str(path.relative_to(repo))]

    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "_git_blob", fake_blob)
    monkeypatch.setattr(runtime_sync, "_paths_changed_by_commit", fake_paths)
    monkeypatch.setattr(runtime_sync, "read_regular_path", fake_read)
    if mode == "positive":
        result = runtime_sync.audit_repo_authority(repo, config, config_payload)
        assert result["status"] == (
            "PASS_STRICT_LINEAR_DAG_D3_LEDGER_RUNTIME_I1_I2_CONFIG_ONLY_B2"
        )
        assert result["predecessor_ledger_commit"] == ledger["commit"]
        assert result["historical_runtime_i1_commit"] == historical_i1
        assert result["runtime_i2_commit"] == implementation
        assert result["ledger_blob_check_count"] == 4
        assert result["historical_runtime_i1_blob_check_count"] == 3
        assert result["runtime_i2_changed_path_count"] == 2
        assert result["producer_and_adjudicator_blob_check_count"] == 18
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_repo_authority(repo, config, config_payload)


def test_exact16_terminal_closure_hashes_compressed_matrix_without_decoding_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_config()
    sources = synthetic_sources(config)
    real_load_json = runtime_sync.load_json
    decoded_labels: list[str] = []

    def recording_load_json(payload: bytes, *, label: str) -> dict[str, Any]:
        decoded_labels.append(label)
        return real_load_json(payload, label=label)

    monkeypatch.setattr(runtime_sync, "load_json", recording_load_json)
    selected = runtime_sync.validate_registered_bundles(config, payload_overrides=sources)
    assert [selected[key]["member_count"] for key in runtime_sync.SOURCE_KEYS] == [6, 6, 4]
    assert not any(
        token in label
        for label in decoded_labels
        for token in ("log2_cpm", "family.soft", "FULLTEXT.xml")
    )
    matrix = "GSE200302_log2_cpm_counts_all_samples.txt.gz"
    sources[runtime_sync.UPSTREAM_AUTHORITY_KEY][matrix] += b"drift"
    with pytest.raises(runtime_sync.PublicationError, match="bytes or SHA-256 drift"):
        runtime_sync.validate_registered_bundles(config, payload_overrides=sources)


def test_prepare_exact7_preserves_delta_order_one_way_hash_and_blocked_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, sources, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    artifacts = tree_bytes(prepared)
    assert set(artifacts) == set(runtime_sync.MUTABLE_NAMES) | set(
        runtime_sync.immutable_names(config)
    )
    assert len(artifacts) == 7
    assert not (set(EXPECTED_SOURCE_OUTPUT_NAMES) & set(artifacts))
    sync_name = config["runtime"]["sync_name"]
    sync_digest = runtime_sync.sha256(artifacts[sync_name])
    sync = runtime_sync.load_json(artifacts[sync_name], label="sync")
    manifest = runtime_sync.load_json(artifacts["RUN_MANIFEST.json"], label="manifest")
    old_manifest = runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="old manifest")
    assert len(manifest["outputs"]) == 163
    assert manifest["outputs"][:143] == old_manifest["outputs"]
    assert manifest["outputs"][143:] == runtime_sync.expected_output_delta(config, sync_digest)
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][143:159]] == (
        EXPECTED_SOURCE_OUTPUT_NAMES
    )
    assert [sync["registered_evidence"][key]["member_count"] for key in runtime_sync.SOURCE_KEYS] == [
        6,
        6,
        4,
    ]
    assert all(
        sync["registered_evidence"][key]["bodies_embedded"] is False
        for key in runtime_sync.SOURCE_KEYS
    )
    assert sync["adjudication_transition"]["input_status_counts"] == {
        "PASS": 4,
        "BLOCKED": 1,
        runtime_sync.UNKNOWN: 1,
        "NOT_RUN": 2,
    }
    assert sync["adjudication_transition"]["unresolved_blockers"] == list(
        runtime_sync.UNRESOLVED_BLOCKERS
    )
    access = sync["access_and_materialization_boundary"]
    assert access["upstream_public_aggregate_artifact_body_opened_for_hash_validation"] is True
    assert access["upstream_public_aggregate_rows_decoded_or_parsed"] is False
    assert access["row_level_payload_read_count"] == access["sequence_read_count"] == 0
    assert access["canonical_read_count"] == access["canonical_write_count"] == 0
    assert access["training_run_count"] == access["model_selection_run_count"] == 0
    assert all(
        runtime_sync.sha256(artifacts[name]).encode("ascii") not in artifacts[sync_name]
        for name in runtime_sync.MUTABLE_NAMES
    )
    assert all(
        sync_digest.encode("ascii") in artifacts[name] for name in runtime_sync.MUTABLE_NAMES
    )
    events = runtime_sync.load_json_lines(artifacts["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 42
    event = events[-1]
    assert event["event_id"] == "A1-EVT-042"
    assert event["manifest_output_count_before"] == 143
    assert event["manifest_output_count_after"] == 163
    assert event["registered_in_place_artifact_count"] == 16
    assert event["adjudication_unresolved_blockers"] == list(runtime_sync.UNRESOLVED_BLOCKERS)
    assert event["qualification_changed"] is False
    assert event["training_allowed"] is False
    assert event["model_selection_allowed"] is False
    assert event["next_phase_authorized"] is False
    validated = runtime_sync.validate_target_only(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        source_payload_overrides=sources,
    )
    assert validated["status"] == "VALIDATED_NOT_PUBLISHED"


def test_sources_are_rechecked_after_immutables_before_any_mutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, sources, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    real_validate = runtime_sync.validate_registered_bundles
    calls = 0

    def fail_second(
        config_value: dict[str, Any],
        *,
        payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise runtime_sync.PublicationError("synthetic source drift before mutables")
        return real_validate(config_value, payload_overrides=payload_overrides)

    monkeypatch.setattr(runtime_sync, "validate_registered_bundles", fail_second)
    result = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        source_payload_overrides=sources,
    )
    assert result["status"] == "PARTIAL_STATE_REQUIRES_IDEMPOTENT_RETRY"
    assert len(result["committed_members"]) == 4
    source_result = result["results"]["GSE200304_DEC019_REGISTERED_EVIDENCE"]
    assert source_result["state"] == "BEFORE_MUTABLES_VALIDATION_FAILED"
    assert source_result["last_validation_phase"] == "BEFORE_MUTABLES"
    assert all((run_root / name).read_bytes() == predecessor[name] for name in runtime_sync.MUTABLE_NAMES)


def test_final_source_validation_fault_reports_complete_event_commit_not_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    real_validate = runtime_sync.validate_registered_bundles
    calls = 0

    def fail_third(
        config_value: dict[str, Any],
        *,
        payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise runtime_sync.PublicationError("synthetic FINAL source validation fault")
        return real_validate(config_value, payload_overrides=payload_overrides)

    monkeypatch.setattr(runtime_sync, "validate_registered_bundles", fail_third)
    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
        "source_payload_overrides": sources,
    }
    result = runtime_sync.publish_prepared(**kwargs)
    complete_commit_order = [*runtime_sync.immutable_names(config), *runtime_sync.MUTABLE_NAMES]
    assert calls == 3
    assert result["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert result["event_committed"] is True
    assert result["committed_members"] == complete_commit_order
    assert result["preexisting_partial_state"] is False
    source_result = result["results"]["GSE200304_DEC019_REGISTERED_EVIDENCE"]
    assert source_result["state"] == "FINAL_VALIDATION_FAILED"
    assert source_result["last_validation_phase"] == "FINAL"
    assert source_result["accepted"] is False
    prepared_payloads = tree_bytes(prepared)
    assert all((run_root / name).read_bytes() == prepared_payloads[name] for name in complete_commit_order)
    events = runtime_sync.load_json_lines(
        (run_root / "EVENT_LOG.jsonl").read_bytes(), label="FINAL-warning events"
    )
    assert len(events) == 42 and events[-1]["event_id"] == "A1-EVT-042"
    manifest = runtime_sync.load_json(
        (run_root / "RUN_MANIFEST.json").read_bytes(), label="FINAL-warning manifest"
    )
    assert len(manifest["outputs"]) == 163

    monkeypatch.setattr(runtime_sync, "validate_registered_bundles", real_validate)
    retried = runtime_sync.publish_prepared(**kwargs)
    assert retried["status"] == "PUBLISHED_VERIFIED"
    assert retried["committed_members"] == []


@pytest.mark.parametrize("prefix_length", [0, 1, 2, 3])
def test_all_four_mutable_recovery_prefixes_publish_status_manifest_event_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix_length: int,
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    artifacts = tree_bytes(prepared)
    for name in runtime_sync.MUTABLE_NAMES[:prefix_length]:
        (run_root / name).write_bytes(artifacts[name])
    result = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        source_payload_overrides=sources,
    )
    assert result["status"] == "PUBLISHED_VERIFIED"
    assert list(result["mutable_preflight"])[0:3] == list(runtime_sync.MUTABLE_NAMES)
    expected_states = ["NEW_EXACT"] * prefix_length + ["OLD_EXACT"] * (3 - prefix_length)
    assert [result["mutable_preflight"][name] for name in runtime_sync.MUTABLE_NAMES] == expected_states
    assert [
        name for name in result["committed_members"] if name in runtime_sync.MUTABLE_NAMES
    ] == list(runtime_sync.MUTABLE_NAMES[prefix_length:])
    assert all((run_root / name).read_bytes() == artifacts[name] for name in runtime_sync.MUTABLE_NAMES)
    assert runtime_sync.load_json_lines(
        (run_root / "EVENT_LOG.jsonl").read_bytes(), label="published events"
    )[-1]["event_id"] == "A1-EVT-042"


def test_event_postcommit_warning_reports_committed_evt042_and_retry_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    calls = 0

    def fail_event_fsync(point: str) -> None:
        nonlocal calls
        if point == "mutable_post_replace_directory_fsync":
            calls += 1
            if calls == 3:
                raise OSError("synthetic EVENT post-commit fsync warning")

    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
        "source_payload_overrides": sources,
    }
    warned = runtime_sync.publish_prepared(**kwargs, fault_injector=fail_event_fsync)
    assert warned["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert warned["warning_member"] == "EVENT_LOG.jsonl"
    assert warned["committed_members"][-1] == "EVENT_LOG.jsonl"
    events = runtime_sync.load_json_lines(
        (run_root / "EVENT_LOG.jsonl").read_bytes(), label="warning-state events"
    )
    assert len(events) == 42 and events[-1]["event_id"] == "A1-EVT-042"
    retried = runtime_sync.publish_prepared(**kwargs)
    assert retried["status"] == "PUBLISHED_VERIFIED"
    assert retried["committed_members"] == []


def test_stale_evt042_temp_and_differing_immutable_fail_before_mutables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, sources, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
        "source_payload_overrides": sources,
    }
    stale = run_root / ".evt042.123.0123456789abcdef.STATUS.json.tmp"
    stale.write_bytes(b"stale")
    with pytest.raises(runtime_sync.PublicationError, match="stale EVT-042"):
        runtime_sync.publish_prepared(**kwargs)
    assert all((run_root / name).read_bytes() == predecessor[name] for name in runtime_sync.MUTABLE_NAMES)
    stale.unlink()
    immutable = runtime_sync.immutable_names(config)[0]
    (run_root / immutable).write_bytes(b"foreign")
    with pytest.raises(runtime_sync.PublicationError, match="existing immutable artifact differs"):
        runtime_sync.publish_prepared(**kwargs)
    assert all((run_root / name).read_bytes() == predecessor[name] for name in runtime_sync.MUTABLE_NAMES)
