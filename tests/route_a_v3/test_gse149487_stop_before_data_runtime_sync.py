from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    STAGING_ROOT / "configs/route_a_v3_gse149487_stop_before_data_runtime_sync_v1.json"
)
SCRIPT_PATH = (
    STAGING_ROOT / "scripts/route_a_v3/gse149487_stop_before_data_runtime_sync.py"
)
SPEC = importlib.util.spec_from_file_location("gse149487_evt036_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def replace_pending(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("PENDING_"):
        return "a" * 64
    if isinstance(value, list):
        return [replace_pending(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_pending(item) for key, item in value.items()}
    return value


def source_artifact(config: dict[str, Any]) -> bytes:
    truth = config["source_truth"]
    authority = config["repository_authority"]
    value = {
        "schema_version": truth["schema_version"],
        "protocol_id": truth["protocol_id"],
        "dataset_id": truth["dataset_id"],
        "recorded_at_utc": truth["recorded_at_utc"],
        "outcome": truth["outcome"],
        "ready_for_study_qualification": truth["ready_for_study_qualification"],
        "blockers": truth["blockers"],
        "counters": truth["zero_counters"],
        "gate_truth": truth["gate_truth"],
        "authority_audit": {
            "status": truth["authority_status"],
            "accepted_a0_base_commit": authority["accepted_a0_base_commit"],
            "active_authority_commit": authority["active_authority_commit"],
            "active_amendment_decision_ids": authority["active_amendment_decision_ids"],
            "implementation_commit": "d10a42a564ecac2af048b39c05cbc863ebdacd02",
            "binding_commit": config["implementation_binding"]["base_commit"],
        },
        "environment_audit": {
            "status": truth["environment_status"],
            "claim_absent": True,
            "failure_absent": True,
            "output_absent": True,
        },
        "inventory_audit": {
            "status": truth["inventory_status"],
            **truth["inventory_counts"],
            "payload_open_count": 0,
            "manifest_open_count": 0,
            "payload_hash_count": 0,
            "scientific_processing_count": 0,
            "hash_reverification": "NOT_RUN_STOP_BEFORE_DATA",
        },
        "external_evidence_audit": truth["external_evidence_truth"],
        "historical_r4_closure": {
            "reference_only_not_reopened": True,
            "rerun_is_qualification_path": False,
        },
    }
    return runtime_sync.json_bytes(value)


def predecessor_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    invariants = config["successor_invariants"]
    status = {
        **invariants,
        "updated_at": "2026-08-11T00:08:49+08:00",
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "legacy_field_preserved": "EXACT",
    }
    manifest = {
        "run_status": invariants["run_status"],
        "evidence_status": invariants["evidence_status"],
        "claim_status": invariants["claim_status"],
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        "v3_contract_sha256": "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
        "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
        "dataset_hashes": [{"dataset_id": "SYNTHETIC_EXISTING", "manifest_sha256": "f" * 64}],
        "outputs": [
            {
                "artifact_type": f"SYNTHETIC_{index:03d}",
                "absolute_path": f"/synthetic/existing/{index:03d}.json",
                "sha256": f"{index:064x}",
                "status": "COMPLETE",
            }
            for index in range(88)
        ],
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-10T00:00:00+08:00",
            "event": "SYNTHETIC_PREDECESSOR",
        }
        for index in range(1, 35)
    ]
    events.append(
        {
            "event_id": "A1-EVT-035",
            "at": "2026-08-11T00:08:49+08:00",
            "event": "SYNTHETIC_EXACT_EVT_035",
        }
    )
    return {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(runtime_sync.compact_json_line(event) for event in events),
    }


def bound_test_context() -> tuple[dict[str, Any], dict[str, bytes], bytes, dict[str, Any]]:
    config = replace_pending(read_config())
    config["implementation_binding"]["status"] = "BOUND"
    config["implementation_binding"]["implementation_commit"] = "1" * 40
    predecessor = predecessor_payloads(config)
    for name, payload in predecessor.items():
        config["runtime"]["predecessor_mutables"][name]["bytes"] = len(payload)
        config["runtime"]["predecessor_mutables"][name]["sha256"] = runtime_sync.sha256(payload)
    source = source_artifact(config)
    config["runtime"]["source_artifact"]["bytes"] = len(source)
    config["runtime"]["source_artifact"]["sha256"] = runtime_sync.sha256(source)
    runtime_sync.validate_bound_config(config)
    authority = {
        "status": "PASS_EXACT_AEECF0F_TO_I_TO_CONFIG_ONLY_B",
        "binding_commit": "2" * 40,
        "head_commit": "2" * 40,
        "origin_branch_head_commit": "2" * 40,
        "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config)),
    }
    return config, predecessor, source, authority


def bind_allowed_prepared_root(
    config: dict[str, Any], authority: dict[str, Any], allowed_root: Path
) -> Path:
    allowed_root.mkdir()
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    authority["config_sha256"] = runtime_sync.sha256(runtime_sync.json_bytes(config))
    return allowed_root / "job"


def seed_run_root(root: Path, config: dict[str, Any], predecessor: dict[str, bytes], source: bytes) -> None:
    root.mkdir()
    for name, payload in predecessor.items():
        (root / name).write_bytes(payload)
    (root / config["runtime"]["source_artifact"]["name"]).write_bytes(source)


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in root.iterdir()
        if item.is_file() and not item.is_symlink()
    }


def prepare_synthetic(tmp_path: Path) -> tuple[
    dict[str, Any], dict[str, bytes], bytes, dict[str, Any], Path, Path
]:
    config, predecessor, source, authority = bound_test_context()
    run_root = tmp_path / "run"
    prepared = bind_allowed_prepared_root(config, authority, tmp_path / "allowed")
    seed_run_root(run_root, config, predecessor, source)
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-11T01:00:00+08:00",
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    return config, predecessor, source, authority, run_root, prepared


def test_static_i_config_has_only_two_unresolved_binding_scalars() -> None:
    config = read_config()
    assert config["implementation_binding"]["status"] == "UNKNOWN_NOT_ASSERTED"
    assert config["implementation_binding"]["implementation_commit"] == "UNKNOWN_NOT_ASSERTED"
    pending = set()

    def walk(value: Any) -> None:
        if isinstance(value, str) and value.startswith("PENDING_"):
            pending.add(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(config)
    assert pending == set()
    assert {
        item["path"]: item["sha256"]
        for item in config["repository_authority"]["implementation_ledger_files"]
    } == {
        "docs/execution/route_a_v3_a1_interim.yaml": "e441ee0321b4947edff7d24a4c6fae67aece926be64d334c4f45a5de4d2c98d5",
        "docs/execution/route_a_v3_registry_manifest.json": "0695f755f11e1ff6fb27b5bd706ee51914a5b74a7647d74737142537e871484b",
        "scripts/route_a_v3/validate_a0_bundle.py": "dfe10a694d398af4e6b30d2f0dc93c7b71511739a6c377ca8bc8dea8a4b2bf28",
        "tests/route_a_v3/test_a0_integrity_guards.py": "4702887a304936d5060f09612960356b089931205f9f3dbd2d075eddcdad9eaa",
    }


def test_unknown_binding_fails_before_run_root_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = read_config()
    opened = 0

    def forbidden_open(_path: Path) -> int:
        nonlocal opened
        opened += 1
        raise AssertionError("run root opened before binding failure")

    monkeypatch.setattr(runtime_sync, "open_directory", forbidden_open)
    with pytest.raises(runtime_sync.BindingError, match="not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-11T01:00:00+08:00",
            production=False,
            config_override=config,
            authority_override={},
            run_root_override=tmp_path / "run",
        )
    assert opened == 0
    assert not (tmp_path / "prepared").exists()


def test_pending_ledger_fails_before_run_root_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = read_config()
    config["implementation_binding"]["status"] = "BOUND"
    config["implementation_binding"]["implementation_commit"] = "1" * 40
    config["repository_authority"]["implementation_ledger_files"][0][
        "sha256"
    ] = "PENDING_A1_INTERIM_SHA256"
    opened = 0

    def forbidden_open(_path: Path) -> int:
        nonlocal opened
        opened += 1
        raise AssertionError("run root opened before placeholder failure")

    monkeypatch.setattr(runtime_sync, "open_directory", forbidden_open)
    with pytest.raises(runtime_sync.BindingError, match="lowercase hex"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-11T01:00:00+08:00",
            production=False,
            config_override=config,
            authority_override={},
            run_root_override=tmp_path / "run",
        )
    assert opened == 0


def test_unknown_i_to_bound_b_changes_only_two_scalars() -> None:
    config, _predecessor, _source, _authority = bound_test_context()
    unknown = runtime_sync.expected_unknown_i_config(config)
    assert unknown["implementation_binding"]["status"] == "UNKNOWN_NOT_ASSERTED"
    assert unknown["implementation_binding"]["implementation_commit"] == "UNKNOWN_NOT_ASSERTED"
    rebound = copy.deepcopy(unknown)
    rebound["implementation_binding"]["status"] = "BOUND"
    rebound["implementation_binding"]["implementation_commit"] = "1" * 40
    assert rebound == config


def test_repo_audit_accepts_only_exact_base_i_config_only_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, _source, _authority = bound_test_context()
    implementation = config["implementation_binding"]["implementation_commit"]
    binding_commit = "2" * 40
    base = config["implementation_binding"]["base_commit"]
    blob_payloads: dict[str, bytes] = {}

    implementation_paths = [
        config["implementation_binding"]["implementation_script_path"],
        config["implementation_binding"]["implementation_test_path"],
        *[item["path"] for item in config["repository_authority"]["implementation_ledger_files"]],
    ]
    for index, relative in enumerate(implementation_paths):
        payload = f"implementation blob {index} {relative}\n".encode()
        blob_payloads[relative] = payload
        if relative == config["implementation_binding"]["implementation_script_path"]:
            config["implementation_binding"]["implementation_script_sha256"] = runtime_sync.sha256(
                payload
            )
        elif relative == config["implementation_binding"]["implementation_test_path"]:
            config["implementation_binding"]["implementation_test_sha256"] = runtime_sync.sha256(
                payload
            )
        else:
            next(
                item
                for item in config["repository_authority"]["implementation_ledger_files"]
                if item["path"] == relative
            )["sha256"] = runtime_sync.sha256(payload)
    for index, item in enumerate(config["repository_authority"]["fixed_authority_files"]):
        payload = f"fixed authority blob {index} {item['path']}\n".encode()
        blob_payloads[item["path"]] = payload
        item["sha256"] = runtime_sync.sha256(payload)
    runtime_sync.validate_bound_config(config)
    config_payload = runtime_sync.json_bytes(config)
    unknown_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    branch = config["repository_authority"]["branch"]
    bad_binding_paths = False

    def fake_git(
        observed_root: Path, *arguments: str, allowed_returncodes: tuple[int, ...] = (0,)
    ) -> bytes:
        assert observed_root == repo_root
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{branch}\n".encode()
        if arguments == ("rev-parse", "HEAD"):
            return f"{binding_commit}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{branch}\n".encode()
        if arguments == ("rev-parse", "@{upstream}"):
            return f"{binding_commit}\n".encode()
        if arguments == ("rev-parse", "--verify", f"refs/remotes/origin/{branch}"):
            return f"{binding_commit}\n".encode()
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if arguments == ("rev-parse", f"{binding_commit}^"):
            return f"{implementation}\n".encode()
        if arguments == ("rev-parse", f"{implementation}^"):
            return f"{base}\n".encode()
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return b""
        if arguments == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            implementation,
        ):
            return ("\n".join(config["repository_authority"]["implementation_commit_exact_changed_paths"]) + "\n").encode()
        if arguments == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            binding_commit,
        ):
            paths = [runtime_sync.CONFIG_REPO_PATH]
            if bad_binding_paths:
                paths.append(runtime_sync.SCRIPT_REPO_PATH)
            return ("\n".join(paths) + "\n").encode()
        if arguments[0] == "show":
            commit, relative = arguments[1].split(":", 1)
            if relative == runtime_sync.CONFIG_REPO_PATH:
                return config_payload if commit == binding_commit else unknown_payload
            return blob_payloads[relative]
        raise AssertionError(f"unexpected fake git call: {arguments!r}")

    def fake_read(path: Path) -> bytes:
        return blob_payloads[str(path.relative_to(repo_root))]

    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo_root)
    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "read_regular_path", fake_read)
    audit = runtime_sync.audit_repo_authority(repo_root, config, config_payload)
    assert audit["binding_commit"] == binding_commit
    assert audit["implementation_commit"] == implementation
    assert audit["binding_commit_is_config_only"] is True

    bad_binding_paths = True
    with pytest.raises(runtime_sync.RuntimeSyncError, match="B exact changed paths"):
        runtime_sync.audit_repo_authority(repo_root, config, config_payload)


def test_prepare_builds_exact_seven_file_evt036_closure(tmp_path: Path) -> None:
    config, predecessor, _source, authority, _run_root, prepared = prepare_synthetic(tmp_path)
    names = {item.name for item in prepared.iterdir()}
    assert names == set(runtime_sync.MUTABLE_NAMES) | set(runtime_sync.immutable_names(config))
    observed_prepared, payloads = runtime_sync.read_prepared_directory(
        prepared, config, production=False
    )
    assert observed_prepared == prepared
    snapshots = runtime_sync.snapshot_names(config)
    for mutable, snapshot in snapshots.items():
        assert payloads[snapshot] == predecessor[mutable]
    sync_name = config["runtime"]["sync_name"]
    sync_digest = runtime_sync.sha256(payloads[sync_name])
    sync = runtime_sync.load_json(payloads[sync_name], label=sync_name)
    assert sync["runtime_sync_publisher_authority"]["binding_commit"] == authority["binding_commit"]
    assert sync["scientific_blockers"]["count"] == 11
    assert sync["runtime_sync_scope"]["data_payload_open_count"] == 0
    for mutable in runtime_sync.MUTABLE_NAMES:
        assert runtime_sync.sha256(payloads[mutable]).encode() not in payloads[sync_name]
        assert sync_digest.encode() in payloads[mutable]
    manifest = runtime_sync.load_json(payloads["RUN_MANIFEST.json"], label="manifest")
    assert len(manifest["outputs"]) == 93
    assert manifest["outputs"][:88] == json.loads(predecessor["RUN_MANIFEST.json"])["outputs"]
    events = runtime_sync.load_json_lines(payloads["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 36 and events[-1]["event_id"] == "A1-EVT-036"
    status = runtime_sync.load_json(payloads["STATUS.json"], label="status")
    for key, value in config["successor_invariants"].items():
        assert status[key] == value


def test_source_drift_fails_before_prepared_output(tmp_path: Path) -> None:
    config, predecessor, source, authority = bound_test_context()
    run_root = tmp_path / "run"
    prepared = bind_allowed_prepared_root(config, authority, tmp_path / "allowed")
    seed_run_root(run_root, config, predecessor, source + b" ")
    with pytest.raises(runtime_sync.PublicationError, match="source aggregate artifact"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-11T01:00:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
    assert not prepared.exists()


def test_predecessor_drift_fails_before_prepared_output(tmp_path: Path) -> None:
    config, predecessor, source, authority = bound_test_context()
    run_root = tmp_path / "run"
    prepared = bind_allowed_prepared_root(config, authority, tmp_path / "allowed")
    seed_run_root(run_root, config, predecessor, source)
    (run_root / "RUN_MANIFEST.json").write_bytes(predecessor["RUN_MANIFEST.json"] + b" ")
    with pytest.raises(runtime_sync.PublicationError, match="exact predecessor mutable drift"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-11T01:00:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
    assert not prepared.exists()


def test_publish_is_no_overwrite_and_idempotent(tmp_path: Path) -> None:
    config, _predecessor, _source, authority, run_root, prepared = prepare_synthetic(tmp_path)
    first = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert first["status"] == "PUBLISHED_VERIFIED"
    assert first["source_artifact_state"] == "EXISTING_SOURCE_EXACT_REUSED"
    for name in runtime_sync.immutable_names(config):
        assert first["results"][name]["state"] == "CREATED_EXCLUSIVE"
    for name in runtime_sync.MUTABLE_NAMES:
        assert first["results"][name]["state"] == "REPLACED_OLD_EXACT"

    after_first = tree_bytes(run_root)
    second = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert second["status"] == "PUBLISHED_VERIFIED"
    assert tree_bytes(run_root) == after_first
    for name in runtime_sync.immutable_names(config):
        assert second["results"][name]["state"] == "EXISTING_EXACT_REUSED"
    for name in runtime_sync.MUTABLE_NAMES:
        assert second["results"][name]["state"] == "EXISTING_NEW_EXACT_REUSED"
    events = runtime_sync.load_json_lines((run_root / "EVENT_LOG.jsonl").read_bytes(), label="events")
    assert len(events) == 36 and events[-1]["event_id"] == "A1-EVT-036"


def test_immutable_collision_fails_before_any_publish_write(tmp_path: Path) -> None:
    config, _predecessor, _source, authority, run_root, prepared = prepare_synthetic(tmp_path)
    sync_name = config["runtime"]["sync_name"]
    (run_root / sync_name).write_bytes(b"DIFFERING")
    before = tree_bytes(run_root)
    with pytest.raises(runtime_sync.PublicationError, match="existing immutable artifact differs"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
    assert tree_bytes(run_root) == before


def test_invalid_mutable_order_fails_before_any_publish_write(tmp_path: Path) -> None:
    config, _predecessor, _source, authority, run_root, prepared = prepare_synthetic(tmp_path)
    (run_root / "EVENT_LOG.jsonl").write_bytes((prepared / "EVENT_LOG.jsonl").read_bytes())
    before = tree_bytes(run_root)
    with pytest.raises(runtime_sync.PublicationError, match="publication-order state"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
    assert tree_bytes(run_root) == before


def test_allowed_partial_mutable_state_recovers_in_order(tmp_path: Path) -> None:
    config, _predecessor, _source, authority, run_root, prepared = prepare_synthetic(tmp_path)
    (run_root / "STATUS.json").write_bytes((prepared / "STATUS.json").read_bytes())
    result = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert result["status"] == "PUBLISHED_VERIFIED"
    assert result["mutable_preflight"] == {
        "STATUS.json": "NEW_EXACT",
        "RUN_MANIFEST.json": "OLD_EXACT",
        "EVENT_LOG.jsonl": "OLD_EXACT",
    }
    assert result["results"]["STATUS.json"]["state"] == "EXISTING_NEW_EXACT_REUSED"


@pytest.mark.parametrize(
    "fault_point",
    [
        "immutable_post_link_unlink",
        "immutable_post_link_directory_fsync",
        "immutable_post_link_close",
    ],
)
def test_immutable_post_commit_failures_preserve_committed_truth(
    tmp_path: Path, fault_point: str
) -> None:
    root = tmp_path / fault_point
    root.mkdir()
    directory_fd = runtime_sync.open_directory(root)

    def inject(point: str) -> None:
        if point == fault_point:
            raise OSError(f"injected {point}")

    try:
        outcome = runtime_sync.publish_immutable_at(
            directory_fd, "FINAL.json", b"exact\n", fault_injector=inject
        )
    finally:
        os.close(directory_fd)
    assert outcome["state"] == "CREATED_EXCLUSIVE"
    assert outcome["committed_by_this_call"] is True
    assert outcome["accepted"] is False
    assert outcome["warnings"][0]["point"] == fault_point
    assert (root / "FINAL.json").read_bytes() == b"exact\n"


def test_mutable_post_commit_failure_preserves_committed_truth(tmp_path: Path) -> None:
    root = tmp_path / "mutable"
    root.mkdir()
    (root / "STATUS.json").write_bytes(b"old\n")
    directory_fd = runtime_sync.open_directory(root)

    def inject(point: str) -> None:
        if point == "mutable_post_replace_directory_fsync":
            raise OSError("injected mutable fsync")

    try:
        outcome = runtime_sync.replace_mutable_at(
            directory_fd,
            "STATUS.json",
            b"old\n",
            b"new\n",
            fault_injector=inject,
        )
    finally:
        os.close(directory_fd)
    assert outcome["state"] == "REPLACED_OLD_EXACT"
    assert outcome["committed_by_this_call"] is True
    assert outcome["accepted"] is False
    assert (root / "STATUS.json").read_bytes() == b"new\n"


def test_publish_surfaces_committed_warning_then_idempotently_recovers(tmp_path: Path) -> None:
    config, _predecessor, _source, authority, run_root, prepared = prepare_synthetic(tmp_path)
    injected = False

    def inject(point: str) -> None:
        nonlocal injected
        if point == "immutable_post_link_directory_fsync" and not injected:
            injected = True
            raise OSError("injected post-link directory fsync")

    warning = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=inject,
    )
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    warning_member = warning["warning_member"]
    assert (run_root / warning_member).read_bytes() == (prepared / warning_member).read_bytes()
    assert (run_root / "STATUS.json").read_bytes() != (prepared / "STATUS.json").read_bytes()

    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    assert (run_root / "EVENT_LOG.jsonl").read_bytes() == (prepared / "EVENT_LOG.jsonl").read_bytes()


def test_lock_exit_fault_after_commits_returns_warning_and_idempotently_recovers(
    tmp_path: Path,
) -> None:
    config, _predecessor, _source, authority, run_root, prepared = prepare_synthetic(tmp_path)
    injected = False

    def inject(point: str) -> None:
        nonlocal injected
        if point == "lock_exit_unlock" and not injected:
            injected = True
            raise OSError("injected lock-exit unlock failure")

    warning = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=inject,
    )
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert warning["committed_members"]
    assert warning["lock_cleanup_warnings"][0]["point"] == "lock_exit_unlock"
    assert (run_root / "EVENT_LOG.jsonl").read_bytes() == (prepared / "EVENT_LOG.jsonl").read_bytes()

    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    for name in runtime_sync.immutable_names(config):
        assert recovered["results"][name]["state"] == "EXISTING_EXACT_REUSED"
    for name in runtime_sync.MUTABLE_NAMES:
        assert recovered["results"][name]["state"] == "EXISTING_NEW_EXACT_REUSED"


def test_prepared_path_rejects_dotdot_empty_and_intermediate_symlink_escape(
    tmp_path: Path,
) -> None:
    config, predecessor, source, authority = bound_test_context()
    allowed = tmp_path / "allowed"
    escape = tmp_path / "escape"
    allowed.mkdir()
    escape.mkdir()
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    authority["config_sha256"] = runtime_sync.sha256(runtime_sync.json_bytes(config))

    with pytest.raises(runtime_sync.PublicationError, match="dot-dot"):
        runtime_sync._validate_prepared_path(
            f"{allowed}/../escape/job", config, production=False
        )
    with pytest.raises(runtime_sync.PublicationError, match="empty"):
        runtime_sync._validate_prepared_path(f"{allowed}//job", config, production=False)

    alias = allowed / "alias"
    alias.symlink_to(escape, target_is_directory=True)
    run_root = tmp_path / "run"
    seed_run_root(run_root, config, predecessor, source)
    with pytest.raises(runtime_sync.PublicationError, match="nofollow directory"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=f"{allowed}/alias/job",
            recorded_at="2026-08-11T01:00:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
    assert not (escape / "job").exists()
    assert list(escape.iterdir()) == []
