from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse200304_dec019_blocked_adjudication_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse200304_dec019_blocked_adjudication_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("gse200304_evt041_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)


def bind_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(config if config is not None else read_config())
    authority = value["repository_authority"]
    ledger = authority["predecessor_ledger"]
    ledger_commit = "a" * 40
    authority["base_commit"] = ledger_commit
    authority["current_pre_runtime_sync_head"] = ledger_commit
    ledger["status"] = "BOUND"
    ledger["commit"] = ledger_commit
    for index, item in enumerate(ledger["frozen_blobs"], start=1):
        item["sha256"] = f"{index:064x}"
    binding = value["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "b" * 40
    binding["implementation_script_sha256"] = runtime_sync.sha256(b"synthetic runtime script\n")
    binding["implementation_test_sha256"] = runtime_sync.sha256(b"synthetic runtime test\n")
    refresh_core(value)
    return value


def unknown_implementation(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    for key in ("status", "implementation_commit", "implementation_script_sha256", "implementation_test_sha256"):
        value["implementation_binding"][key] = runtime_sync.UNKNOWN
    return value


def synthetic_sources(config: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    registered = config["registered_evidence"]
    negative_spec = registered[runtime_sync.NEGATIVE_PACK_KEY]
    negative: dict[str, bytes] = {}
    for item in negative_spec["members"]:
        if item["name"] != "PUBLICATION_COMMIT.json":
            negative[item["name"]] = runtime_sync.json_bytes(
                {"status": item["terminal_status"], "aggregate_only": True}
            )
    gate_names = sorted(negative)
    negative["PUBLICATION_COMMIT.json"] = runtime_sync.json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": negative_spec["terminal_record_type"],
            "contract_id": config["contract_id"],
            "protocol_id": negative_spec["protocol_id"],
            "dataset_id": "GSE200304",
            "decision_id": "V3-DEC-019",
            "publication_mode": negative_spec["publication_mode"],
            "gate_record_count": 7,
            "gate_record_names": gate_names,
            "gate_payload_set_sha256": negative_spec["gate_payload_set_sha256"],
            "final_output_target_sha256": negative_spec["final_output_target_sha256"],
            "committed": True,
            "commit_marker_written_last": True,
        }
    )
    for item in negative_spec["members"]:
        payload = negative[item["name"]]
        item["bytes"] = len(payload)
        item["sha256"] = runtime_sync.sha256(payload)

    adjudication_spec = registered[runtime_sync.ADJUDICATION_KEY]
    report = runtime_sync.json_bytes(
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
        }
    )
    audit = runtime_sync.json_bytes(
        {
            "mode": "ALL_HASH_BOUND_AGGREGATES_VERIFIED",
            "all_inputs_aggregate_only": True,
            "row_level_payload_read_count": 0,
            "sequence_read_count": 0,
            "opened_input_count": 8,
        }
    )
    adjudication: dict[str, bytes] = {
        "ADJUDICATION_REPORT.json": report,
        "INPUT_EVIDENCE_AUDIT.json": audit,
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
            "publication_mode": "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1",
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
    for item in adjudication_spec["members"]:
        payload = adjudication[item["name"]]
        item["bytes"] = len(payload)
        item["sha256"] = runtime_sync.sha256(payload)
    refresh_core(config)
    return {
        runtime_sync.NEGATIVE_PACK_KEY: negative,
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
        "updated_at": "2026-08-11T11:00:00+08:00",
        "legacy": "PRESERVED",
    }
    outputs = [
        {
            "artifact_type": f"OLD_{index:03d}",
            "absolute_path": f"/old/{index:03d}",
            "sha256": f"{index:064x}",
            "status": "COMPLETE",
        }
        for index in range(127)
    ]
    manifest = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        "outputs": outputs,
    }
    events = [
        {"event_id": f"A1-EVT-{index:03d}", "at": "2026-08-10T00:00:00+08:00", "event": "OLD"}
        for index in range(1, 40)
    ]
    events.append(
        {
            "event_id": "A1-EVT-040",
            "at": "2026-08-11T11:30:00+08:00",
            "event": "DEC019_AUTHORITY_AND_GSE114002_PUBLIC_GAP_LINEAGE_SYNCED_GATE_UNCHANGED",
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
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, dict[str, bytes]], dict[str, Any], Path, Path]:
    config = bind_config()
    run_root = tmp_path / "run"
    allowed = tmp_path / "allowed"
    prepared = allowed / "job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    config["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["absolute_directory"] = str(run_root / "negative")
    config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["absolute_directory"] = str(tmp_path / "adjudication")
    sources = synthetic_sources(config)
    predecessor = predecessor_payloads()
    for name, payload in predecessor.items():
        config["runtime"]["predecessor_mutables"][name]["bytes"] = len(payload)
        config["runtime"]["predecessor_mutables"][name]["sha256"] = runtime_sync.sha256(payload)
    tail = predecessor["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail_event"]["bytes"] = len(tail)
    config["runtime"]["predecessor_tail_event"]["sha256"] = runtime_sync.sha256(tail)
    refresh_core(config)
    run_root.mkdir()
    allowed.mkdir()
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    authority = {
        "status": "PASS_SYNTHETIC_HISTORICAL_LIFECYCLES_AND_RUNTIME_I_B",
        "binding_commit": "c" * 40,
        "head_commit": "c" * 40,
        "origin_branch_head_commit": "c" * 40,
        "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config)),
        "base_commit": config["repository_authority"]["base_commit"],
        "implementation_commit": config["implementation_binding"]["implementation_commit"],
        "predecessor_ledger_commit": config["repository_authority"]["predecessor_ledger"]["commit"],
        "negative_nfs_binding_commit": config["repository_authority"]["negative_producer_lifecycle"]["nfs_binding_commit"],
        "adjudicator_descriptor_commit": config["repository_authority"]["adjudicator_lifecycle"]["descriptor_commit"],
    }
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    return config, predecessor, sources, authority, run_root, prepared


def prepare_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, dict[str, bytes]], dict[str, Any], Path, Path]:
    context = make_context(tmp_path, monkeypatch)
    config, _predecessor, sources, authority, run_root, prepared = context
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-11T12:00:00+08:00",
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        source_payload_overrides=sources,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "127_TO_143"
    assert result["runtime_artifact_count"] == 7
    return context


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in root.iterdir() if item.is_file() and not item.is_symlink()}


def test_static_i_form_freezes_delta16_exact8_exact4_and_only_expected_placeholders() -> None:
    config = read_config()
    runtime_sync.validate_static_config(config)
    assert config["implementation_binding"]["status"] == runtime_sync.UNKNOWN
    authority = config["repository_authority"]
    ledger = authority["predecessor_ledger"]
    assert ledger["status"] == "BOUND"
    assert authority["base_commit"] == authority["current_pre_runtime_sync_head"] == ledger["commit"] == "f465dd03ae792b98c0604b1d225cd2df37d28f9e"
    assert [item["sha256"] for item in ledger["frozen_blobs"]] == [
        "552f705445a36df99a5ae071c85625c729ad2f69f0a375bb7b3118c3b400e16c",
        "9548ccfc7150b8b6381a58f19bd6ed39638b70869d776efdb36a3f725d7bffed",
        "cb7f278e1daed071ea691c7c1817e800c94a46982915348b8f678656ea99b528",
        "76a78ec060060feff243366ab93bd6a721096d1847b197f2ff4da6518d3625f5",
    ]
    assert (config["runtime"]["predecessor_manifest_output_count"], config["runtime"]["successor_manifest_output_count"], config["runtime"]["output_delta_count"]) == (127, 143, 16)
    assert len(config["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["members"]) == 8
    assert len(config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["members"]) == 4
    blocked = config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]
    assert blocked["input_count"] == 8
    assert len(blocked["unresolved_blockers"]) == 7
    assert [blocked[key] for key in ("ordinary_study_contribution", "a1_study_contribution", "true_a2_study_contribution", "canonical_record_count")] == [0, 0, 0, 0]
    assert all(blocked[key] is False for key in ("qualified", "canonical_materialization_allowed", "training_allowed", "model_selection_allowed", "next_phase_authorized"))
    with pytest.raises(runtime_sync.BindingError, match="implementation is not BOUND"):
        runtime_sync.validate_bound_config(config)
    bound = bind_config(config)
    runtime_sync.validate_bound_config(bound)
    assert runtime_sync.expected_unknown_i_config(bound) == unknown_implementation(bound)
    assert runtime_sync.compiled_core_projection(bound) == runtime_sync.compiled_core_projection(unknown_implementation(bound))


@pytest.mark.parametrize(
    "mode",
    ["delta15", "negative_hash", "adjudication_member", "artifact_type", "member_order", "source_order", "blocker_count", "training_integer", "outer_truth", "partial_ledger"],
)
def test_static_config_rejects_p0_p1_map_or_truth_drift(mode: str) -> None:
    config = bind_config()
    if mode == "delta15":
        config["runtime"]["output_delta_count"] = 15
    elif mode == "negative_hash":
        config["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["members"][0]["sha256"] = "0" * 64
    elif mode == "adjudication_member":
        config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["members"].pop()
    elif mode == "artifact_type":
        config["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["members"][0]["artifact_type"] = False
    elif mode == "member_order":
        members = config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["members"]
        members[0], members[1] = members[1], members[0]
    elif mode == "source_order":
        registered = config["registered_evidence"]
        config["registered_evidence"] = {
            runtime_sync.ADJUDICATION_KEY: registered[runtime_sync.ADJUDICATION_KEY],
            runtime_sync.NEGATIVE_PACK_KEY: registered[runtime_sync.NEGATIVE_PACK_KEY],
        }
    elif mode == "blocker_count":
        config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["unresolved_blockers"].pop()
    elif mode == "training_integer":
        config["successor_invariants"]["training_allowed"] = 0
    elif mode == "outer_truth":
        config["runtime_authority"]["historical_outer_runtime_authority"]["code_commit"] = "0" * 40
    else:
        config["repository_authority"]["predecessor_ledger"]["frozen_blobs"][0]["sha256"] = runtime_sync.UNKNOWN
    refresh_core(config)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_static_config(config)


def test_manifest_source_output_order_is_explicit_and_insertion_order_independent() -> None:
    config = bind_config()
    expected = runtime_sync.expected_output_delta(config, "f" * 64)
    assert [Path(item["absolute_path"]).name for item in expected[:12]] == [
        member_name for _source_key, member_name in runtime_sync.SOURCE_MEMBER_OUTPUT_ORDER
    ]

    reordered = copy.deepcopy(config)
    registered = reordered["registered_evidence"]
    reordered["registered_evidence"] = {
        runtime_sync.ADJUDICATION_KEY: registered[runtime_sync.ADJUDICATION_KEY],
        runtime_sync.NEGATIVE_PACK_KEY: registered[runtime_sync.NEGATIVE_PACK_KEY],
    }
    for source in reordered["registered_evidence"].values():
        source["members"].reverse()
    assert runtime_sync.expected_output_delta(reordered, "f" * 64) == expected


def test_unknown_bindings_stop_before_repository_runtime_or_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("accessed")
        raise AssertionError("external source accessed")

    monkeypatch.setattr(runtime_sync, "audit_repo_authority", forbidden)
    monkeypatch.setattr(runtime_sync, "open_directory", forbidden)
    monkeypatch.setattr(runtime_sync, "validate_registered_bundles", forbidden)
    with pytest.raises(runtime_sync.BindingError):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-11T12:00:00+08:00",
            production=False,
            config_override=read_config(),
            repo_root=tmp_path / "repo",
            run_root_override=tmp_path / "run",
        )
    assert accessed == []


@pytest.mark.parametrize(
    "mode",
    ["positive", "dirty", "parent_drift", "path_drift", "negative_drift", "adjudicator_drift", "ledger_drift", "current_drift", "lineage_drift", "i_type_drift"],
)
def test_repo_audit_proves_historical_negative_adjudicator_d2_ledger_and_runtime_i_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = bind_config()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo)
    authority = config["repository_authority"]
    authority["production_repo_root"] = str(repo)
    refresh_core(config)
    binding = config["implementation_binding"]
    ledger = authority["predecessor_ledger"]
    negative = authority["negative_producer_lifecycle"]
    adjudicator = authority["adjudicator_lifecycle"]
    base = authority["base_commit"]
    implementation = binding["implementation_commit"]
    head = "c" * 40
    config_payload = runtime_sync.json_bytes(config)
    i_config = runtime_sync.expected_unknown_i_config(config)
    if mode == "i_type_drift":
        i_config["successor_invariants"]["training_allowed"] = 0
    i_payload = runtime_sync.json_bytes(i_config)
    runtime_script = b"synthetic runtime script\n"
    runtime_test = b"synthetic runtime test\n"

    blob_payloads: dict[tuple[str, str], bytes] = {}
    digest_overrides: dict[bytes, str] = {
        runtime_script: binding["implementation_script_sha256"],
        runtime_test: binding["implementation_test_sha256"],
    }

    def add_three(commit: str, spec: Mapping[str, Any], paths: Mapping[str, str], label: str) -> None:
        for path_key, digest_key in (("config_path", "config_sha256"), ("script_path", "script_sha256"), ("test_path", "test_sha256")):
            payload = f"{label}:{path_key}\n".encode()
            if path_key == "config_path" and "config_bytes" in spec:
                payload = payload.ljust(spec["config_bytes"], b"x")[: spec["config_bytes"]]
            blob_payloads[(commit, paths[path_key])] = payload
            digest_overrides[payload] = spec[digest_key]

    add_three(negative["initial_implementation_commit"], negative["initial_implementation_blobs"], negative, "negative-initial-i")
    add_three(negative["initial_binding_commit"], negative["initial_binding_blobs"], negative, "negative-initial-b")
    add_three(negative["nfs_implementation_commit"], negative["nfs_implementation_blobs"], negative, "negative-nfs-i")
    for commit, label in ((negative["nfs_binding_commit"], "negative-nfs-b"), (negative["current_descriptor_commit"], "negative-d2"), (head, "negative-head")):
        add_three(commit, negative["nfs_binding_and_current_blobs"], negative, label)
    add_three(adjudicator["implementation_commit"], adjudicator["implementation_blobs"], adjudicator, "adjudicator-i")
    add_three(adjudicator["binding_commit"], adjudicator["binding_blobs"], adjudicator, "adjudicator-b")
    for commit, label in ((adjudicator["descriptor_commit"], "adjudicator-d2"), (head, "adjudicator-head")):
        add_three(commit, adjudicator["descriptor_blobs"], adjudicator, label)
    ledger_payloads: dict[str, bytes] = {}
    for item in ledger["frozen_blobs"]:
        payload = f"ledger:{item['path']}\n".encode()
        ledger_payloads[item["path"]] = payload
        digest_overrides[payload] = item["sha256"]
        for commit in (base, implementation, head):
            blob_payloads[(commit, item["path"])] = payload
    real_sha256 = runtime_sync.sha256
    monkeypatch.setattr(runtime_sync, "sha256", lambda payload: digest_overrides.get(payload, real_sha256(payload)))

    parent_map = {
        head: implementation,
        implementation: base,
        base: ledger["expected_parent"],
        negative["initial_binding_commit"]: negative["initial_implementation_commit"],
        negative["nfs_implementation_commit"]: negative["initial_binding_commit"],
        negative["nfs_binding_commit"]: negative["nfs_implementation_commit"],
        adjudicator["binding_commit"]: adjudicator["implementation_commit"],
        adjudicator["descriptor_commit"]: adjudicator["binding_commit"],
    }
    changed_paths = {
        base: ledger["commit_exact_changed_paths"],
        implementation: authority["implementation_commit_exact_changed_paths"],
        head: authority["binding_commit_exact_changed_paths"],
        negative["initial_implementation_commit"]: negative["implementation_commit_exact_changed_paths"],
        negative["initial_binding_commit"]: negative["binding_commit_exact_changed_paths"],
        negative["nfs_implementation_commit"]: negative["implementation_commit_exact_changed_paths"],
        negative["nfs_binding_commit"]: negative["binding_commit_exact_changed_paths"],
        adjudicator["implementation_commit"]: adjudicator["implementation_commit_exact_changed_paths"],
        adjudicator["binding_commit"]: adjudicator["binding_and_descriptor_commit_exact_changed_paths"],
        adjudicator["descriptor_commit"]: adjudicator["binding_and_descriptor_commit_exact_changed_paths"],
    }

    def fake_git(_repo: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
        del allowed_returncodes
        branch = authority["branch"]
        if args == ("rev-parse", "HEAD") or args == ("rev-parse", "@{upstream}") or args == ("rev-parse", "--verify", f"refs/remotes/origin/{branch}"):
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
            if mode == "parent_drift" and child == negative["nfs_binding_commit"]:
                parent = "f" * 40
            return f"{parent}\n".encode()
        if len(args) == 2 and args[0] == "rev-parse":
            return f"{args[1]}\n".encode()
        if args[:2] == ("merge-base", "--is-ancestor"):
            if mode == "lineage_drift" and args[2] == negative["nfs_binding_commit"]:
                raise runtime_sync.AuthorityError("synthetic ancestry failure")
            return b""
        raise AssertionError(args)

    def fake_blob(_repo: Path, commit: str, path: str) -> bytes:
        if path == runtime_sync.CONFIG_REPO_PATH:
            return config_payload if commit == head else i_payload
        if path == runtime_sync.SCRIPT_REPO_PATH:
            return runtime_script
        if path == runtime_sync.TEST_REPO_PATH:
            return runtime_test
        payload = blob_payloads[(commit, path)]
        if mode == "negative_drift" and commit == negative["nfs_implementation_commit"] and path == negative["config_path"]:
            return payload + b"drift"
        if mode == "adjudicator_drift" and commit == adjudicator["descriptor_commit"] and path == adjudicator["config_path"]:
            return payload + b"drift"
        if mode == "ledger_drift" and commit == base and path == ledger["frozen_blobs"][0]["path"]:
            return payload + b"drift"
        if mode == "current_drift" and commit == head and path == adjudicator["config_path"]:
            return payload + b"drift"
        return payload

    def fake_changed(_repo: Path, commit: str) -> list[str]:
        result = list(changed_paths[commit])
        if mode == "path_drift" and commit == negative["nfs_implementation_commit"]:
            result.append("unexpected")
        return sorted(result)

    def fake_read(path: Path) -> bytes:
        relative = str(path.relative_to(repo))
        if relative == runtime_sync.SCRIPT_REPO_PATH:
            return runtime_script
        if relative == runtime_sync.TEST_REPO_PATH:
            return runtime_test
        if relative in ledger_payloads:
            return ledger_payloads[relative]
        if relative in (negative["config_path"], negative["script_path"], negative["test_path"]):
            return blob_payloads[(head, relative)]
        if relative in (adjudicator["config_path"], adjudicator["script_path"], adjudicator["test_path"]):
            return blob_payloads[(head, relative)]
        raise AssertionError(relative)

    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "_git_blob", fake_blob)
    monkeypatch.setattr(runtime_sync, "_paths_changed_by_commit", fake_changed)
    monkeypatch.setattr(runtime_sync, "read_regular_path", fake_read)
    if mode == "positive":
        result = runtime_sync.audit_repo_authority(repo, config, config_payload)
        assert result["negative_nfs_binding_commit"] == negative["nfs_binding_commit"]
        assert result["adjudicator_descriptor_commit"] == adjudicator["descriptor_commit"]
        assert result["predecessor_ledger_commit"] == base
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_repo_authority(repo, config, config_payload)


@pytest.mark.parametrize(
    "mode",
    ["positive", "missing", "extra", "negative_hash", "negative_status", "negative_marker", "adjudication_hash", "adjudication_sums"],
)
def test_exact8_exact4_terminal_bundle_validation(mode: str) -> None:
    config = bind_config()
    sources = synthetic_sources(config)
    if mode == "missing":
        sources[runtime_sync.NEGATIVE_PACK_KEY].pop(next(iter(sources[runtime_sync.NEGATIVE_PACK_KEY])))
    elif mode == "extra":
        sources[runtime_sync.ADJUDICATION_KEY]["EXTRA"] = b"extra"
    elif mode == "negative_hash":
        name = next(name for name in sources[runtime_sync.NEGATIVE_PACK_KEY] if name != "PUBLICATION_COMMIT.json")
        sources[runtime_sync.NEGATIVE_PACK_KEY][name] += b"drift"
    elif mode == "negative_status":
        item = next(item for item in config["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["members"] if item["name"] != "PUBLICATION_COMMIT.json")
        payload = runtime_sync.json_bytes({"status": "PASS", "aggregate_only": True})
        sources[runtime_sync.NEGATIVE_PACK_KEY][item["name"]] = payload
        item["bytes"], item["sha256"] = len(payload), runtime_sync.sha256(payload)
    elif mode == "negative_marker":
        item = next(item for item in config["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["members"] if item["name"] == "PUBLICATION_COMMIT.json")
        marker = runtime_sync.load_json(sources[runtime_sync.NEGATIVE_PACK_KEY][item["name"]], label="marker")
        marker["committed"] = False
        payload = runtime_sync.json_bytes(marker)
        sources[runtime_sync.NEGATIVE_PACK_KEY][item["name"]] = payload
        item["bytes"], item["sha256"] = len(payload), runtime_sync.sha256(payload)
    elif mode == "adjudication_hash":
        sources[runtime_sync.ADJUDICATION_KEY]["ADJUDICATION_REPORT.json"] += b"drift"
    elif mode == "adjudication_sums":
        item = next(item for item in config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["members"] if item["name"] == "SHA256SUMS")
        payload = sources[runtime_sync.ADJUDICATION_KEY]["SHA256SUMS"] + b"0"
        sources[runtime_sync.ADJUDICATION_KEY]["SHA256SUMS"] = payload
        item["bytes"], item["sha256"] = len(payload), runtime_sync.sha256(payload)
    if mode == "positive":
        selected = runtime_sync.validate_registered_bundles(config, payload_overrides=sources)
        assert selected[runtime_sync.NEGATIVE_PACK_KEY]["member_count"] == 8
        assert selected[runtime_sync.ADJUDICATION_KEY]["member_count"] == 4
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.validate_registered_bundles(config, payload_overrides=sources)


@pytest.mark.parametrize("mode", ["hardlink", "symlink_directory"])
def test_registered_bundle_link_boundary_fails_closed(tmp_path: Path, mode: str) -> None:
    config = bind_config()
    sources = synthetic_sources(config)
    roots: dict[str, Path] = {}
    for key in runtime_sync.SOURCE_KEYS:
        root = tmp_path / key
        root.mkdir()
        for name, payload in sources[key].items():
            (root / name).write_bytes(payload)
        roots[key] = root
        config["registered_evidence"][key]["absolute_directory"] = str(root)
    if mode == "hardlink":
        member = next(iter(sources[runtime_sync.NEGATIVE_PACK_KEY]))
        os.link(roots[runtime_sync.NEGATIVE_PACK_KEY] / member, tmp_path / "external-link")
    else:
        link = tmp_path / "adjudication-link"
        link.symlink_to(roots[runtime_sync.ADJUDICATION_KEY], target_is_directory=True)
        config["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["absolute_directory"] = str(link)
    with pytest.raises((runtime_sync.RuntimeSyncError, OSError)):
        runtime_sync.validate_registered_bundles(config)


def test_prepare_exact7_registers_exact8_plus_exact4_and_appends_delta16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    artifacts = tree_bytes(prepared)
    assert set(artifacts) == set(runtime_sync.MUTABLE_NAMES) | set(runtime_sync.immutable_names(config))
    assert len(artifacts) == 7
    manifest = runtime_sync.load_json(artifacts["RUN_MANIFEST.json"], label="manifest")
    old_manifest = runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="old manifest")
    assert len(manifest["outputs"]) == 143
    assert manifest["outputs"][:127] == old_manifest["outputs"]
    sync_digest = runtime_sync.sha256(artifacts[config["runtime"]["sync_name"]])
    assert manifest["outputs"][127:] == runtime_sync.expected_output_delta(config, sync_digest)
    assert len(manifest["outputs"][127:139]) == 12
    sync = runtime_sync.load_json(artifacts[config["runtime"]["sync_name"]], label="sync")
    assert sync["registered_evidence"][runtime_sync.NEGATIVE_PACK_KEY]["member_count"] == 8
    assert sync["registered_evidence"][runtime_sync.ADJUDICATION_KEY]["member_count"] == 4
    assert all(item["bodies_embedded"] is False for item in sync["registered_evidence"].values())
    assert sync["a1_gate_snapshot"]["training_allowed"] is False
    assert all(runtime_sync.sha256(artifacts[name]).encode() not in artifacts[config["runtime"]["sync_name"]] for name in runtime_sync.MUTABLE_NAMES)
    events = runtime_sync.load_json_lines(artifacts["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 41
    assert events[-1]["event_id"] == "A1-EVT-041"
    assert events[-1]["manifest_output_count_after"] == 143
    assert events[-1]["adjudication_input_count"] == 8
    assert events[-1]["adjudication_unresolved_blocker_count"] == 7
    validated = runtime_sync.validate_target_only(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        source_payload_overrides=sources,
    )
    assert validated["status"] == "VALIDATED_NOT_PUBLISHED"


def test_one_way_sync_hash_policy_rejects_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, _sources, _authority, _run_root, prepared = prepare_context(tmp_path, monkeypatch)
    artifacts = tree_bytes(prepared)
    sync_name = config["runtime"]["sync_name"]
    sync = runtime_sync.load_json(artifacts[sync_name], label="sync")
    sync["hash_linkage"]["sync_record_references_successor_hashes"] = True
    artifacts[sync_name] = runtime_sync.json_bytes(sync)
    with pytest.raises(runtime_sync.RuntimeSyncError, match="one-way sync hash linkage"):
        runtime_sync.validate_successors(
            config,
            artifacts,
            predecessor,
            runtime_sync.load_json(predecessor["STATUS.json"], label="status"),
            runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="manifest"),
            runtime_sync.sha256(artifacts[sync_name]),
        )


def test_source_terminal_closure_is_rechecked_after_immutables_before_mutables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    real_validate = runtime_sync.validate_registered_bundles
    calls = 0

    def drift_on_second(config_value: dict[str, Any], *, payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise runtime_sync.PublicationError("synthetic source drift after immutable publication")
        return real_validate(config_value, payload_overrides=payload_overrides)

    monkeypatch.setattr(runtime_sync, "validate_registered_bundles", drift_on_second)
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
    assert result["preexisting_partial_state"] is False
    source_result = result["results"]["GSE200304_DEC019_REGISTERED_EVIDENCE"]
    assert source_result["accepted"] is False
    assert source_result["state"] == "BEFORE_MUTABLES_VALIDATION_FAILED"
    assert source_result["last_validation_phase"] == "BEFORE_MUTABLES"
    for name in runtime_sync.MUTABLE_NAMES:
        assert (run_root / name).read_bytes() == predecessor[name]


def test_first_publish_and_idempotent_retry_preserve_source_bundles_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
        "source_payload_overrides": sources,
    }
    first = runtime_sync.publish_prepared(**kwargs)
    assert first["status"] == "PUBLISHED_VERIFIED"
    assert first["manifest_output_transition"] == "127_TO_143"
    assert first["committed_members"][-1] == "EVENT_LOG.jsonl"
    assert len(runtime_sync.immutable_names(config)) == 4
    for name in runtime_sync.immutable_names(config):
        assert (run_root / name).stat().st_nlink == 1
    second = runtime_sync.publish_prepared(**kwargs)
    assert second["status"] == "PUBLISHED_VERIFIED"
    assert second["committed_members"] == []


@pytest.mark.parametrize("prefix_length", [0, 1, 2, 3])
def test_four_allowed_mutable_recovery_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix_length: int
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
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
    assert all((run_root / name).read_bytes() == artifacts[name] for name in runtime_sync.MUTABLE_NAMES)


def test_event_is_last_and_postcommit_warning_recovers_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    calls = 0

    def fail_event_fsync(point: str) -> None:
        nonlocal calls
        if point == "mutable_post_replace_directory_fsync":
            calls += 1
            if calls == 3:
                raise OSError("synthetic event post-commit fsync failure")

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
    assert runtime_sync.load_json_lines((run_root / "EVENT_LOG.jsonl").read_bytes(), label="events")[-1]["event_id"] == "A1-EVT-041"
    assert runtime_sync.publish_prepared(**kwargs)["status"] == "PUBLISHED_VERIFIED"


def test_immutable_temp_unlink_failure_reports_committed_manual_truth_with_evt041_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    failed = False

    def fail_once(point: str) -> None:
        nonlocal failed
        if point == "immutable_post_link_unlink" and not failed:
            failed = True
            raise OSError("synthetic unlink failure")

    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
        "source_payload_overrides": sources,
    }
    result = runtime_sync.publish_prepared(
        **kwargs,
        fault_injector=fail_once,
    )
    assert result["status"] == "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION"
    assert len(result["committed_members"]) == 1
    stale = list(run_root.glob(".evt041.*.tmp"))
    assert stale

    retried = runtime_sync.publish_prepared(**kwargs)
    assert retried["status"] == "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION"
    assert retried["preexisting_partial_state"] is True
    assert retried["warning_member"] in retried["committed_members"]
    assert retried["stale_temporary_members"] == [stale[0].name]


def test_stale_evt041_temp_and_differing_immutable_never_touch_mutables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, sources, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    stale = run_root / ".evt041.123.0123456789abcdef.STATUS.json.tmp"
    stale.write_bytes(b"stale")
    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
        "source_payload_overrides": sources,
    }
    with pytest.raises(runtime_sync.PublicationError, match="stale EVT-041"):
        runtime_sync.publish_prepared(**kwargs)
    stale.unlink()
    immutable = runtime_sync.immutable_names(config)[0]
    (run_root / immutable).write_bytes(b"foreign")
    with pytest.raises(runtime_sync.PublicationError, match="existing immutable artifact differs"):
        runtime_sync.publish_prepared(**kwargs)
    assert all((run_root / name).read_bytes() == predecessor[name] for name in runtime_sync.MUTABLE_NAMES)


def test_recorded_at_and_prepared_path_boundary_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, sources, authority, run_root, _prepared = make_context(tmp_path, monkeypatch)
    with pytest.raises(runtime_sync.RuntimeSyncError, match="EVT-041 window"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "allowed" / "late",
            recorded_at="2026-08-13T12:00:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
            source_payload_overrides=sources,
        )
    with pytest.raises(runtime_sync.PublicationError, match="below"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "outside",
            recorded_at="2026-08-11T12:00:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
            source_payload_overrides=sources,
        )
