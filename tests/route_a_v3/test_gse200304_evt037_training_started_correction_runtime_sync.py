from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse200304_evt037_training_started_correction_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse200304_evt037_training_started_correction_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("evt038_correction_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)


def unknown_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(config if config is not None else read_config())
    binding = value["implementation_binding"]
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        binding[key] = "UNKNOWN_NOT_ASSERTED"
    return value


def bind_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    binding = value["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    refresh_core(value)
    return value


def evt037_sync(config: dict[str, Any]) -> bytes:
    return runtime_sync.json_bytes(
        {
            "record_type": "ROUTE_A_V3_A1_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC",
            "scientific_blockers": {"count": 8, "exact": config["unresolved_blockers"]},
            "a1_gate_snapshot": copy.deepcopy(config["successor_invariants"]),
            "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
        }
    )


def predecessor_payloads(config: dict[str, Any], sync_payload: bytes) -> dict[str, bytes]:
    status = {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "claim_status": "NOT_ESTABLISHED",
        "training_started": False, "next_phase_authorized": False,
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "updated_at": "2026-08-11T05:30:00+08:00", "legacy": "PRESERVED",
    }
    sync_path = str(Path(config["runtime"]["run_root"]) / config["runtime"]["evt037_sync"]["name"])
    outputs = [
        {"artifact_type": f"OLD_{index:03d}", "absolute_path": f"/old/{index:03d}", "sha256": f"{index:064x}", "status": "COMPLETE"}
        for index in range(101)
    ]
    outputs.append({"artifact_type": "A1_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1", "absolute_path": sync_path, "sha256": runtime_sync.sha256(sync_payload), "status": "COMPLETE"})
    manifest = {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED", "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        "outputs": outputs,
    }
    events = [
        {"event_id": f"A1-EVT-{index:03d}", "at": "2026-08-10T00:00:00+08:00", "event": "OLD"}
        for index in range(1, 37)
    ]
    events.append(
        {
            "event_id": "A1-EVT-037", "at": "2026-08-11T05:30:00+08:00",
            "event": "GSE200304_PUBLISHED_ENDPOINT_COMMITTED_ACCEPTED_SYNCED_GATE_UNCHANGED",
            "training_authorized": False, "next_phase_authorized": False,
        }
    )
    return {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(runtime_sync.compact_json_line(item) for item in events),
    }


def make_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any], Path, Path]:
    config = bind_config(unknown_config())
    run_root, allowed = tmp_path / "run", tmp_path / "allowed"
    prepared = allowed / "job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    sync_payload = evt037_sync(config)
    config["runtime"]["evt037_sync"]["bytes"] = len(sync_payload)
    config["runtime"]["evt037_sync"]["sha256"] = runtime_sync.sha256(sync_payload)
    predecessor = predecessor_payloads(config, sync_payload)
    for name, payload in predecessor.items():
        config["runtime"]["predecessor_mutables"][name]["bytes"] = len(payload)
        config["runtime"]["predecessor_mutables"][name]["sha256"] = runtime_sync.sha256(payload)
    tail = predecessor["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["evt037_tail_event"]["bytes"] = len(tail)
    config["runtime"]["evt037_tail_event"]["sha256"] = runtime_sync.sha256(tail)
    refresh_core(config)
    run_root.mkdir(); allowed.mkdir()
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    (run_root / config["runtime"]["evt037_sync"]["name"]).write_bytes(sync_payload)
    authority = {"status": "PASS_EXACT_EVT037_B_TO_CORRECTION_I_TO_CONFIG_ONLY_B", "binding_commit": "4" * 40, "head_commit": "4" * 40, "origin_branch_head_commit": "4" * 40, "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config))}
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    return config, predecessor, authority, run_root, prepared


def prepare_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any], Path, Path]:
    context = make_context(tmp_path, monkeypatch)
    config, _predecessor, authority, run_root, prepared = context
    result = runtime_sync.prepare_runtime_sync(prepared_directory=prepared, recorded_at="2026-08-11T06:00:00+08:00", production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    return context


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in root.iterdir() if item.is_file() and not item.is_symlink()}


def test_static_config_freezes_published_evt037_predecessor() -> None:
    config = read_config()
    assert config["protocol_id"].endswith("_RUNTIME_SYNC_V1")
    assert runtime_sync.CONFIG_REPO_PATH.endswith("_runtime_sync_v1.json")
    assert runtime_sync.SCRIPT_REPO_PATH.endswith("_runtime_sync.py")
    assert runtime_sync.TEST_REPO_PATH.endswith("_runtime_sync.py")
    assert config["implementation_binding"]["status"] in {"UNKNOWN_NOT_ASSERTED", "BOUND"}
    assert config["runtime"]["predecessor_manifest_output_count"] == 102
    assert config["runtime"]["successor_manifest_output_count"] == 106
    assert config["runtime"]["predecessor_mutables"]["EVENT_LOG.jsonl"] == {
        "bytes": 42299,
        "sha256": "70941937b50b1f8e4bbc9b67196ffcd34328ea434756b891be01b561d5dffdaa",
        "snapshot_name": "EVENT_LOG_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_V1.jsonl",
    }
    assert config["runtime"]["evt037_tail_event"] == {
        "bytes": 1757, "sha256": "9a1c4b32ac88ac3ca7c4f9f36b0722fcf10835b86b1a7614e6f736aebd8bda16",
        "training_started_key_present": False,
    }
    unknown = unknown_config(config)
    bound = config if config["implementation_binding"]["status"] == "BOUND" else bind_config(unknown)
    runtime_sync.validate_bound_config(bound)
    assert runtime_sync.expected_unknown_i_config(bound) == unknown
    assert runtime_sync.compiled_core_projection(bound) == runtime_sync.compiled_core_projection(unknown)


@pytest.mark.parametrize(
    "drift",
    [
        "root_extra",
        "privacy_true",
        "run_root_redirect",
        "sync_name",
        "training_started_integer",
        "producer_config_path",
        "publication_policy_extra",
    ],
)
def test_bound_config_is_closed_type_strict_and_frozen(drift: str) -> None:
    config = bind_config(unknown_config())
    if drift == "root_extra":
        config["extra"] = False
    elif drift == "privacy_true":
        config["privacy_boundary"]["raw_reads_or_alignments_opened"] = True
    elif drift == "run_root_redirect":
        config["runtime"]["run_root"] = "/mnt/cunyuliu/redirected"
    elif drift == "sync_name":
        config["runtime"]["sync_name"] = "drift.json"
    elif drift == "training_started_integer":
        config["successor_invariants"]["training_started"] = 0
    elif drift == "producer_config_path":
        config["repository_authority"]["evt037_producer_binding"]["config_path"] = "configs/drift.json"
    else:
        config["publication_policy"]["extra"] = False
    refresh_core(config)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_bound_config(config)


def test_unknown_binding_stops_before_runtime_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = False
    def forbidden(_path: Path) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("runtime accessed")
    monkeypatch.setattr(runtime_sync, "open_directory", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="not BOUND"):
        runtime_sync.prepare_runtime_sync(prepared_directory=tmp_path / "prepared", recorded_at="2026-08-11T06:00:00+08:00", production=False, config_override=unknown_config(), authority_override={}, run_root_override=tmp_path / "run")
    assert opened is False


@pytest.mark.parametrize("mode", ["positive", "dirty", "non_config_binding", "script_drift"])
def test_repo_binding_audit_exact_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = bind_config(unknown_config())
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo)
    authority, binding = config["repository_authority"], config["implementation_binding"]
    implementation, base, head = binding["implementation_commit"], authority["base_commit"], "4" * 40
    blobs: dict[str, bytes] = {}
    for path, field in ((runtime_sync.SCRIPT_REPO_PATH, "implementation_script_sha256"), (runtime_sync.TEST_REPO_PATH, "implementation_test_sha256")):
        blobs[path] = f"correction {path}\n".encode()
        binding[field] = runtime_sync.sha256(blobs[path])
    producer = authority["evt037_producer_binding"]
    for path_field, digest_field in (("config_path", "config_sha256"), ("script_path", "script_sha256"), ("test_path", "test_sha256")):
        blobs[producer[path_field]] = f"EVT037 {path_field}\n".encode()
        producer[digest_field] = runtime_sync.sha256(blobs[producer[path_field]])
    refresh_core(config)
    config_payload = runtime_sync.json_bytes(config)
    explicit_unknown = unknown_config(config)
    assert runtime_sync.compiled_core_projection(explicit_unknown) == runtime_sync.compiled_core_projection(config)
    assert runtime_sync.expected_unknown_i_config(config) == explicit_unknown
    i_payload = runtime_sync.json_bytes(explicit_unknown)

    def fake_git(_repo: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
        if args == ("rev-parse", "HEAD"): return f"{head}\n".encode()
        if args == ("rev-parse", "--verify", f"refs/remotes/origin/{authority['branch']}"): return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "HEAD"): return f"{authority['branch']}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "@{upstream}"): return f"origin/{authority['branch']}\n".encode()
        if args == ("rev-parse", "@{upstream}"): return f"{head}\n".encode()
        if args == ("rev-parse", f"{head}^"): return f"{implementation}\n".encode()
        if args == ("rev-parse", f"{implementation}^"): return f"{base}\n".encode()
        if args == ("status", "--porcelain=v1", "--untracked-files=all"): return b"M dirty\n" if mode == "dirty" else b""
        if args[:2] == ("merge-base", "--is-ancestor"): return b""
        if args[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            paths = authority["implementation_commit_exact_changed_paths"] if args[4] == implementation else authority["binding_commit_exact_changed_paths"]
            if mode == "non_config_binding" and args[4] == head: paths = paths + [runtime_sync.SCRIPT_REPO_PATH]
            return ("\n".join(paths) + "\n").encode()
        if args[0] == "show":
            commit, path = args[1].split(":", 1)
            if path == runtime_sync.CONFIG_REPO_PATH: return config_payload if commit == head else i_payload
            payload = blobs[path]
            if mode == "script_drift" and path == runtime_sync.SCRIPT_REPO_PATH and commit == implementation: return payload + b"drift"
            return payload
        raise AssertionError(args)

    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "read_regular_path", lambda path: blobs[str(path.relative_to(repo))])
    if mode == "positive":
        assert runtime_sync.audit_repo_authority(repo, config, config_payload)["binding_commit"] == head
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_repo_authority(repo, config, config_payload)


def test_prepare_appends_exact_correction_and_preserves_evt037(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, predecessor, _authority, _run_root, prepared = prepare_context(tmp_path, monkeypatch)
    payloads = {item.name: item.read_bytes() for item in prepared.iterdir()}
    manifest = runtime_sync.load_json(payloads["RUN_MANIFEST.json"], label="manifest")
    old_manifest = runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="old manifest")
    assert len(manifest["outputs"]) == 106 and manifest["outputs"][:102] == old_manifest["outputs"]
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][102:]] == [
        "STATUS_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_V1.json",
        "RUN_MANIFEST_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_V1.json",
        "EVENT_LOG_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_V1.jsonl",
        "A1_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_V1.json",
    ]
    assert [item["artifact_type"] for item in manifest["outputs"][102:]] == [
        "A1_STATUS_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_SNAPSHOT",
        "A1_RUN_MANIFEST_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_SNAPSHOT",
        "A1_EVENT_LOG_PRE_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_SNAPSHOT",
        "A1_GSE200304_EVT037_TRAINING_STARTED_CORRECTION_RUNTIME_SYNC_V1",
    ]
    sync = runtime_sync.load_json(payloads[config["runtime"]["sync_name"]], label="sync")
    assert sync["correction_type"] == "APPEND_ONLY_SEMANTIC_COMPLETION_NO_TRUTH_CHANGE"
    assert sync["access_and_materialization_boundary"]["published_endpoint_artifact_body_opened"] is False
    assert sync["access_and_materialization_boundary"]["published_endpoint_artifact_registration_count"] == 0
    events = runtime_sync.load_json_lines(payloads["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 38 and events[:-1] == runtime_sync.load_json_lines(predecessor["EVENT_LOG.jsonl"], label="old events")
    assert "training_started" not in events[-2]
    assert events[-1]["training_started"] is False
    assert events[-1]["training_allowed"] is False
    assert events[-1]["correction_type"] == "APPEND_ONLY_SEMANTIC_COMPLETION_NO_TRUTH_CHANGE"
    assert events[-1]["published_endpoint_artifact_body_opened"] is False
    assert events[-1]["published_endpoint_artifact_registration_count"] == 0
    for projection in (config["successor_invariants"], config["privacy_boundary"]):
        for key, value in projection.items():
            assert type(events[-1][key]) is type(value)
            assert events[-1][key] == value
    assert events[-1]["unresolved_blockers"] == config["unresolved_blockers"]
    assert payloads["EVENT_LOG.jsonl"].startswith(predecessor["EVENT_LOG.jsonl"])


@pytest.mark.parametrize(
    "drift",
    [
        "missing_gate",
        "missing_privacy",
        "true_gate",
        "true_privacy",
        "nonzero_privacy",
        "bool_as_integer",
        "integer_as_bool",
        "extra_key",
    ],
)
def test_evt038_closed_event_rejects_missing_true_nonzero_and_type_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    config, predecessor, _authority, _run_root, prepared = prepare_context(tmp_path, monkeypatch)
    artifacts = {item.name: item.read_bytes() for item in prepared.iterdir()}
    events = runtime_sync.load_json_lines(artifacts["EVENT_LOG.jsonl"], label="events")
    if drift == "missing_gate": events[-1].pop("run_status")
    elif drift == "missing_privacy": events[-1].pop("row_identifier_payload_included")
    elif drift == "true_gate": events[-1]["training_allowed"] = True
    elif drift == "true_privacy": events[-1]["gene_payload_included"] = True
    elif drift == "nonzero_privacy": events[-1]["raw_replay_run_count"] = 1
    elif drift == "bool_as_integer": events[-1]["raw_fastq_body_read_count"] = False
    elif drift == "integer_as_bool": events[-1]["training_started"] = 0
    else: events[-1]["unexpected"] = False
    artifacts["EVENT_LOG.jsonl"] = predecessor["EVENT_LOG.jsonl"] + runtime_sync.compact_json_line(events[-1])
    with pytest.raises(runtime_sync.RuntimeSyncError, match="EVT-038 closed event"):
        runtime_sync.validate_successors(config, artifacts, predecessor, runtime_sync.load_json(predecessor["STATUS.json"], label="status"), runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="manifest"), runtime_sync.sha256(artifacts[config["runtime"]["sync_name"]]))


def test_evt037_sync_true_and_tail_present_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, predecessor, _authority, run_root, _prepared = make_context(tmp_path, monkeypatch)
    sync_name = config["runtime"]["evt037_sync"]["name"]
    sync = runtime_sync.load_json((run_root / sync_name).read_bytes(), label="sync")
    sync["a1_gate_snapshot"]["training_started"] = True
    payload = runtime_sync.json_bytes(sync)
    (run_root / sync_name).write_bytes(payload)
    config["runtime"]["evt037_sync"]["bytes"], config["runtime"]["evt037_sync"]["sha256"] = len(payload), runtime_sync.sha256(payload)
    fd = runtime_sync.open_directory(run_root)
    try:
        with pytest.raises(runtime_sync.RuntimeSyncError, match="training_started"):
            runtime_sync.validate_evt037_source(fd, config, predecessor["EVENT_LOG.jsonl"])
    finally:
        runtime_sync.os.close(fd)

    events = runtime_sync.load_json_lines(predecessor["EVENT_LOG.jsonl"], label="events")
    events[-1]["training_started"] = False
    changed = b"".join(runtime_sync.compact_json_line(item) for item in events)
    predecessor["EVENT_LOG.jsonl"] = changed
    config["runtime"]["predecessor_mutables"]["EVENT_LOG.jsonl"]["bytes"] = len(changed)
    config["runtime"]["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"] = runtime_sync.sha256(changed)
    with pytest.raises(runtime_sync.PublicationError, match="omission"):
        runtime_sync._validate_predecessor_objects(runtime_sync.load_json(predecessor["STATUS.json"], label="status"), runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="manifest"), events, config)


def test_first_publish_idempotent_and_all_recovery_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    first = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert first["status"] == "PUBLISHED_VERIFIED"
    after = tree_bytes(run_root)
    second = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert second["status"] == "PUBLISHED_VERIFIED" and tree_bytes(run_root) == after


@pytest.mark.parametrize("new_count", [0, 1, 2, 3])
def test_four_allowed_mutable_recovery_prefixes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_count: int) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    for name in runtime_sync.MUTABLE_NAMES[:new_count]:
        (run_root / name).write_bytes((prepared / name).read_bytes())
    result = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert result["status"] == "PUBLISHED_VERIFIED"
    assert [result["mutable_preflight"][name] for name in runtime_sync.MUTABLE_NAMES] == ["NEW_EXACT"] * new_count + ["OLD_EXACT"] * (3 - new_count)


def test_event_directory_fsync_warning_retry_reconfirms_all_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    attempts = 0
    def inject(point: str) -> None:
        nonlocal attempts
        if point == "mutable_post_replace_directory_fsync":
            attempts += 1
            if attempts == 3: raise OSError("EVENT fsync fault")
    warning = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root, fault_injector=inject)
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY" and warning["warning_member"] == "EVENT_LOG.jsonl"
    real_fsync, calls = runtime_sync.os.fsync, []
    def track(fd: int) -> None:
        calls.append(fd); real_fsync(fd)
    monkeypatch.setattr(runtime_sync.os, "fsync", track)
    recovered = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    assert len(calls) == len(runtime_sync.immutable_names(config)) + len(runtime_sync.MUTABLE_NAMES)


def test_immutable_temp_unlink_failure_requires_manual_adjudication_and_retry_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(tmp_path, monkeypatch)
    injected = False
    def inject(point: str) -> None:
        nonlocal injected
        if point == "immutable_post_link_unlink" and not injected:
            injected = True
            raise OSError("unlink fault")
    result = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root, fault_injector=inject)
    assert result["status"] == "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION"
    stale = [item.name for item in run_root.iterdir() if item.name.startswith(".evt038.")]
    assert len(stale) == 1
    with pytest.raises(runtime_sync.PublicationError, match="stale EVT-038 publisher temporary"):
        runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
