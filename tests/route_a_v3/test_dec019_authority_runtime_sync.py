from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec019_authority_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec019_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec019_evt040_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )


def unknown_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(config if config is not None else read_config())
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        value["implementation_binding"][key] = "UNKNOWN_NOT_ASSERTED"
    return value


def bind_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(config if config is not None else unknown_config())
    binding = value["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    refresh_core(value)
    return value


def synthetic_public_gap(config: dict[str, Any]) -> bytes:
    gap = config["legacy_gse114002_public_gap"]
    payload = runtime_sync.json_bytes(
        {
            "schema_version": "synthetic.public.gap.v1",
            "record_id": gap["audit_record_id"],
            "status": gap["audit_status"],
            "lineage": {
                "historical_failed_attempt_lineage_id": gap[
                    "historical_failed_attempt_lineage_id"
                ],
                "current_mechanical_closure_lineage_id": gap[
                    "current_mechanical_closure_lineage_id"
                ],
                "predecessor_runtime_event_id": "A1-EVT-039",
                "runtime_sync_status": "PENDING_NO_EVT_040",
            },
        }
    )
    gap["audit_bytes"] = len(payload)
    gap["audit_sha256"] = runtime_sync.sha256(payload)
    return payload


def predecessor_payloads() -> dict[str, bytes]:
    status = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "training_started": False,
        "next_phase_authorized": False,
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "updated_at": "2026-08-11T09:30:00+08:00",
        "legacy": "PRESERVED",
    }
    outputs = [
        {
            "artifact_type": f"OLD_{index:03d}",
            "absolute_path": f"/old/{index:03d}",
            "sha256": f"{index:064x}",
            "status": "COMPLETE",
        }
        for index in range(122)
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
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-10T00:00:00+08:00",
            "event": "OLD",
        }
        for index in range(1, 39)
    ]
    events.append(
        {
            "event_id": "A1-EVT-039",
            "at": "2026-08-11T10:00:00+08:00",
            "event": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_ATTEMPT_LINEAGE_SYNCED_GATE_UNCHANGED",
            "training_started": False,
            "training_allowed": False,
            "next_phase_authorized": False,
        }
    )
    return {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(
            runtime_sync.compact_json_line(item) for item in events
        ),
    }


def make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    bytes,
    dict[str, Any],
    Path,
    Path,
]:
    config = bind_config()
    run_root = tmp_path / "run"
    allowed = tmp_path / "allowed"
    prepared = allowed / "job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    public_gap = synthetic_public_gap(config)
    predecessor = predecessor_payloads()
    for name, payload in predecessor.items():
        config["runtime"]["predecessor_mutables"][name]["bytes"] = len(payload)
        config["runtime"]["predecessor_mutables"][name]["sha256"] = (
            runtime_sync.sha256(payload)
        )
    tail_line = predecessor["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail_event"]["bytes"] = len(tail_line)
    config["runtime"]["predecessor_tail_event"]["sha256"] = runtime_sync.sha256(
        tail_line
    )
    refresh_core(config)
    run_root.mkdir()
    allowed.mkdir()
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    authority = {
        "status": "PASS_EXACT_SYNTHETIC_DEC019_I_TO_CONFIG_ONLY_B",
        "binding_commit": "4" * 40,
        "head_commit": "4" * 40,
        "origin_branch_head_commit": "4" * 40,
        "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config)),
        "dec019_authority_blob_count": 13,
    }
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    return config, predecessor, public_gap, authority, run_root, prepared


def prepare_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    bytes,
    dict[str, Any],
    Path,
    Path,
]:
    context = make_context(tmp_path, monkeypatch)
    config, _predecessor, public_gap, authority, run_root, prepared = context
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-11T10:30:00+08:00",
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        public_gap_payload_override=public_gap,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "122_TO_127"
    assert result["runtime_artifact_count"] == 8
    return context


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in root.iterdir()
        if item.is_file() and not item.is_symlink()
    }


def locate_authority_root(config: dict[str, Any]) -> Path:
    contract = config["dec019_authority"]["authority_files"][0]
    candidates = [STAGING_ROOT]
    candidates.extend(sorted(STAGING_ROOT.parent.glob("dec019_validation_overlay.*"), reverse=True))
    for candidate in candidates:
        path = candidate / contract["path"]
        if path.is_file() and runtime_sync.sha256(path.read_bytes()) == contract["sha256"]:
            return candidate
    raise AssertionError("exact DEC019 authority root is unavailable")


def test_static_config_supports_i_or_b_and_freezes_evt040_authority() -> None:
    config = read_config()
    binding = config["implementation_binding"]
    assert binding["status"] in {"UNKNOWN_NOT_ASSERTED", "BOUND"}
    assert binding["compiled_core_sha256"] == runtime_sync.compiled_core_sha256(config)
    runtime = config["runtime"]
    assert (
        runtime["predecessor_event_count"],
        runtime["successor_event_count"],
        runtime["predecessor_manifest_output_count"],
        runtime["successor_manifest_output_count"],
        runtime["output_delta_count"],
    ) == (39, 40, 122, 127, 5)
    assert runtime["predecessor_mutables"] == {
        "STATUS.json": {
            "bytes": 21042,
            "sha256": "a94fedccd0b19801b2d82dff55063bf0b03740ecb351628bc26e3e1c440d8376",
            "snapshot_name": "STATUS_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json",
        },
        "RUN_MANIFEST.json": {
            "bytes": 50117,
            "sha256": "4669f52f2ac1946f91f6121a80a310025c8f6825162bffb0f19ee472e5394d86",
            "snapshot_name": "RUN_MANIFEST_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json",
        },
        "EVENT_LOG.jsonl": {
            "bytes": 51402,
            "sha256": "6e82f992616fbd503b1ea79c5fd200910fbbe79361fdabf5714e11e93c4241cb",
            "snapshot_name": "EVENT_LOG_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_V1.jsonl",
        },
    }
    assert runtime["predecessor_tail_event"] == {
        "event_id": "A1-EVT-039",
        "bytes": 6104,
        "sha256": "a878458c70d2a6b9dd08d3448f0ccb1c89372831238e198b9025ba94b0e32994",
        "training_started_key_present": True,
        "training_started": False,
    }
    gap = config["legacy_gse114002_public_gap"]
    assert gap["audit_sha256"] == runtime_sync.PUBLIC_GAP_AUDIT_SHA256
    assert gap["source_role"] == "REPOSITORY_AGGREGATE_AUDIT_AUTHORITY_BLOB_PRE_DEC019"
    assert gap["runtime_snapshot_name"] == runtime_sync.PUBLIC_GAP_AUDIT_RUNTIME_NAME
    assert gap["runtime_snapshot_role"] == runtime_sync.PUBLIC_GAP_AUDIT_RUNTIME_ROLE
    assert gap["source_runtime_sync_status"] == "PENDING_NO_EVT_040"
    assert gap["evt040_registration_status"] == (
        "REGISTERED_HASH_BOUND_NO_SOURCE_REWRITE"
    )
    assert config["runtime_authority"] == {
        "historical_outer_runtime_authority": {
            "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
            "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        },
        "current_contract_authority": {
            "implementation_commit": "d54de63605a2df51e91262c99218684a80cb6515",
            "binding_commit": "78827501c7efcef28550b04876c98206d94d4808",
            "scope": "DEC019_AUTHORITY_AND_SUCCESSOR_ADJUDICATOR_BINDING",
            "active_amendment_decision_ids": [
                "V3-DEC-017",
                "V3-DEC-018",
                "V3-DEC-019",
            ],
        },
    }
    assert len(config["dec019_authority"]["authority_files"]) == 12
    assert config["dec019_authority"]["approval_itself_qualifies_any_study"] is False
    assert config["successor_invariants"]["qualified_independent_ordinary_studies"] == 0
    assert config["successor_invariants"]["qualified_a1_studies"] == 0
    assert config["successor_invariants"]["qualified_a2_dense_studies"] == 0
    assert config["successor_invariants"]["canonical_intervention_record_count"] == 0
    assert config["successor_invariants"]["training_allowed"] is False
    assert config["access_and_materialization_boundary"]["new_runtime_output_count"] == 5
    unknown = unknown_config(config)
    bound = config if binding["status"] == "BOUND" else bind_config(unknown)
    runtime_sync.validate_bound_config(bound)
    assert runtime_sync.expected_unknown_i_config(bound) == unknown
    assert runtime_sync.compiled_core_projection(bound) == (
        runtime_sync.compiled_core_projection(unknown)
    )


def test_frozen_test_remains_valid_after_config_only_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound_path = tmp_path / CONFIG_PATH.name
    bound_path.write_bytes(runtime_sync.json_bytes(bind_config(read_config())))
    monkeypatch.setattr(sys.modules[__name__], "CONFIG_PATH", bound_path)
    test_static_config_supports_i_or_b_and_freezes_evt040_authority()


def test_frozen_authority_and_public_gap_source_hashes_revalidate_exactly(
    tmp_path: Path,
) -> None:
    config = bind_config()
    source_root = locate_authority_root(config)
    for record in config["dec019_authority"]["authority_files"]:
        payload = (source_root / record["path"]).read_bytes()
        assert runtime_sync.sha256(payload) == record["sha256"]
    gap = config["legacy_gse114002_public_gap"]
    source_payload = (source_root / gap["audit_path"]).read_bytes()
    assert len(source_payload) == gap["audit_bytes"]
    assert runtime_sync.sha256(source_payload) == gap["audit_sha256"]
    repo = tmp_path / "repo"
    target = repo / gap["audit_path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(source_payload)
    assert runtime_sync.read_exact_public_gap_source(repo, config) == source_payload
    target.write_bytes(source_payload + b"drift")
    with pytest.raises(runtime_sync.AuthorityError, match="source identity drift"):
        runtime_sync.read_exact_public_gap_source(repo, config)


@pytest.mark.parametrize(
    "drift",
    [
        "root_extra",
        "delta_four",
        "successor_126",
        "short_audit_name",
        "approval_true",
        "training_integer",
        "outer_authority_drift",
        "authority_hash_drift",
        "snapshot_role_drift",
    ],
)
def test_bound_config_is_closed_typed_and_rejects_old_delta_four(drift: str) -> None:
    config = bind_config()
    if drift == "root_extra":
        config["extra"] = False
    elif drift == "delta_four":
        config["runtime"]["output_delta_count"] = 4
    elif drift == "successor_126":
        config["runtime"]["successor_manifest_output_count"] = 126
    elif drift == "short_audit_name":
        config["legacy_gse114002_public_gap"]["runtime_snapshot_name"] = (
            "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1.json"
        )
    elif drift == "approval_true":
        config["dec019_authority"]["approval_itself_qualifies_any_study"] = True
    elif drift == "training_integer":
        config["successor_invariants"]["training_started"] = 0
    elif drift == "outer_authority_drift":
        config["runtime_authority"]["historical_outer_runtime_authority"][
            "code_commit"
        ] = "0" * 40
    elif drift == "authority_hash_drift":
        config["dec019_authority"]["authority_files"][0]["sha256"] = "0" * 64
    else:
        config["legacy_gse114002_public_gap"]["runtime_snapshot_role"] = "DRIFT"
    refresh_core(config)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_bound_config(config)


def test_unknown_binding_stops_before_repository_runtime_or_audit_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("accessed")
        raise AssertionError("repository, runtime, or audit source accessed")

    monkeypatch.setattr(runtime_sync, "audit_repo_authority", forbidden)
    monkeypatch.setattr(runtime_sync, "open_directory", forbidden)
    monkeypatch.setattr(runtime_sync, "read_exact_public_gap_source", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-11T10:30:00+08:00",
            production=False,
            config_override=unknown_config(),
            repo_root=tmp_path / "repo",
            run_root_override=tmp_path / "run",
        )
    assert accessed == []


@pytest.mark.parametrize(
    "mode",
    [
        "positive",
        "dirty",
        "non_config_binding",
        "script_drift",
        "authority_drift",
        "base_parent_drift",
        "introduced_parent_exists",
        "introduced_parent_git_fault",
    ],
)
def test_repo_binding_audit_requires_exact_i_b_and_thirteen_authority_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = bind_config()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo)
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    implementation = binding["implementation_commit"]
    base = authority["base_commit"]
    parent = authority["base_commit_expected_parent"]
    introduced = config["legacy_gse114002_public_gap"]["introduced_commit"]
    head = "4" * 40
    script_blob = b"synthetic runtime sync script\n"
    test_blob = b"synthetic runtime sync tests\n"
    binding["implementation_script_sha256"] = runtime_sync.sha256(script_blob)
    binding["implementation_test_sha256"] = runtime_sync.sha256(test_blob)
    refresh_core(config)
    config_payload = runtime_sync.json_bytes(config)
    i_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))

    authority_blobs: dict[str, bytes] = {
        record["path"]: f"authority:{record['role']}:{record['path']}".encode()
        for record in config["dec019_authority"]["authority_files"]
    }
    gap_blob = b"G" * config["legacy_gse114002_public_gap"]["audit_bytes"]
    authority_blobs[runtime_sync.PUBLIC_GAP_AUDIT_REPO_PATH] = gap_blob
    digest_overrides = {
        authority_blobs[record["path"]]: record["sha256"]
        for record in config["dec019_authority"]["authority_files"]
    }
    digest_overrides[gap_blob] = config["legacy_gse114002_public_gap"]["audit_sha256"]
    real_sha256 = runtime_sync.sha256

    def synthetic_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    monkeypatch.setattr(runtime_sync, "sha256", synthetic_sha256)

    def fake_git(
        _repo: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)
    ) -> bytes:
        del allowed_returncodes
        if args == ("rev-parse", "HEAD"):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--verify", f"refs/remotes/origin/{authority['branch']}"):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{authority['branch']}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{authority['branch']}\n".encode()
        if args == ("rev-parse", "@{upstream}"):
            return f"{head}\n".encode()
        if len(args) == 2 and args[0] == "rev-parse" and args[1].endswith("^"):
            commit = args[1][:-1]
            mapping = {head: implementation, implementation: base, base: parent}
            value = mapping[commit]
            if mode == "base_parent_drift" and commit == base:
                value = "9" * 40
            return f"{value}\n".encode()
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b"M dirty\n" if mode == "dirty" else b""
        if args[:2] == ("merge-base", "--is-ancestor"):
            return b""
        if args == (
            "ls-tree",
            "-z",
            "--full-tree",
            f"{introduced}^",
            "--",
            runtime_sync.PUBLIC_GAP_AUDIT_REPO_PATH,
        ):
            if mode == "introduced_parent_git_fault":
                raise runtime_sync.AuthorityError("synthetic ls-tree git fault")
            if mode == "introduced_parent_exists":
                return (
                    "100644 blob "
                    + "a" * 40
                    + "\t"
                    + runtime_sync.PUBLIC_GAP_AUDIT_REPO_PATH
                    + "\0"
                ).encode()
            return b""
        if args[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            paths = (
                authority["implementation_commit_exact_changed_paths"]
                if args[4] == implementation
                else authority["binding_commit_exact_changed_paths"]
            )
            if mode == "non_config_binding" and args[4] == head:
                paths = paths + [runtime_sync.SCRIPT_REPO_PATH]
            return ("\n".join(paths) + "\n").encode()
        if args[0] == "show":
            commit, path = args[1].split(":", 1)
            if path == runtime_sync.CONFIG_REPO_PATH:
                return config_payload if commit == head else i_payload
            if path == runtime_sync.SCRIPT_REPO_PATH:
                if mode == "script_drift" and commit == implementation:
                    return script_blob + b"drift"
                return script_blob
            if path == runtime_sync.TEST_REPO_PATH:
                return test_blob
            blob = authority_blobs[path]
            if mode == "authority_drift" and commit == implementation and path == (
                config["dec019_authority"]["authority_files"][0]["path"]
            ):
                return blob + b"drift"
            return blob
        raise AssertionError(args)

    def fake_worktree_read(path: Path) -> bytes:
        relative = str(path.relative_to(repo))
        if relative == runtime_sync.SCRIPT_REPO_PATH:
            return script_blob
        if relative == runtime_sync.TEST_REPO_PATH:
            return test_blob
        return authority_blobs[relative]

    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "read_regular_path", fake_worktree_read)
    if mode == "positive":
        result = runtime_sync.audit_repo_authority(repo, config, config_payload)
        assert result["binding_commit"] == head
        assert result["base_commit_parent"] == parent
        assert result["dec019_authority_blob_count"] == 13
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_repo_authority(repo, config, config_payload)


def test_prepare_closes_exact_eight_members_and_appends_exact_five_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    artifacts = tree_bytes(prepared)
    assert len(artifacts) == 8
    assert set(artifacts) == set(runtime_sync.MUTABLE_NAMES) | set(
        runtime_sync.immutable_names(config)
    )
    gap_name = config["legacy_gse114002_public_gap"]["runtime_snapshot_name"]
    assert artifacts[gap_name] == public_gap
    assert artifacts[config["runtime"]["predecessor_mutables"]["STATUS.json"]["snapshot_name"]] == predecessor["STATUS.json"]
    sync = runtime_sync.load_json(
        artifacts[config["runtime"]["sync_name"]], label="sync"
    )
    assert sync["legacy_gse114002_public_gap"]["source_runtime_sync_status"] == (
        "PENDING_NO_EVT_040"
    )
    assert sync["legacy_gse114002_public_gap"]["live_runtime_sync_status"] == (
        "SYNCED_EVT_040"
    )
    assert sync["legacy_gse114002_public_gap"]["source_authority_file_rewritten"] is False
    assert sync["runtime_authority"] == config["runtime_authority"]
    assert sync["a1_gate_snapshot"]["qualified_independent_ordinary_studies"] == 0
    assert sync["a1_gate_snapshot"]["qualified_a1_studies"] == 0
    assert sync["a1_gate_snapshot"]["qualified_a2_dense_studies"] == 0
    assert sync["a1_gate_snapshot"]["canonical_intervention_record_count"] == 0
    assert sync["a1_gate_snapshot"]["training_allowed"] is False
    manifest = runtime_sync.load_json(artifacts["RUN_MANIFEST.json"], label="manifest")
    assert len(manifest["outputs"]) == 127
    assert manifest["outputs"][:122] == runtime_sync.load_json(
        predecessor["RUN_MANIFEST.json"], label="predecessor manifest"
    )["outputs"]
    sync_digest = runtime_sync.sha256(artifacts[config["runtime"]["sync_name"]])
    assert manifest["outputs"][122:] == runtime_sync.expected_output_delta(
        config, sync_digest
    )
    status = runtime_sync.load_json(artifacts["STATUS.json"], label="status")
    assert status["dec019_authority_runtime_sync_status"] == "SYNCED_EVT_040"
    assert status["gse114002_public_authority_gap_runtime_sync_status"] == (
        "SYNCED_EVT_040"
    )
    assert status["code_commit"] == "28cd2f132d022fea6ac43e1f89d6673d02a9c97d"
    events = runtime_sync.load_json_lines(artifacts["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 40
    assert events[-1]["event_id"] == "A1-EVT-040"
    assert events[-1]["public_gap_source_runtime_sync_status"] == "PENDING_NO_EVT_040"
    assert events[-1]["public_gap_runtime_sync_status"] == "SYNCED_EVT_040"
    assert events[-1]["dec019_authority_runtime_sync_status"] == "SYNCED_EVT_040"
    assert events[-1]["approval_itself_qualifies_any_study"] is False
    validated = runtime_sync.validate_target_only(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert validated["status"] == "VALIDATED_NOT_PUBLISHED"


@pytest.mark.parametrize("mode", ["missing_audit", "extra_member", "tampered_audit"])
def test_prepared_directory_rejects_delta_four_seven_member_and_audit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config, _predecessor, _public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    gap_name = config["legacy_gse114002_public_gap"]["runtime_snapshot_name"]
    if mode == "missing_audit":
        (prepared / gap_name).unlink()
    elif mode == "extra_member":
        (prepared / "EXTRA").write_bytes(b"extra")
    else:
        (prepared / gap_name).write_bytes(b"drift")
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_target_only(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )


def test_first_publish_and_idempotent_retry_preserve_exact_audit_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    kwargs = {
        "prepared_directory": prepared,
        "production": False,
        "config_override": config,
        "authority_override": authority,
        "run_root_override": run_root,
    }
    first = runtime_sync.publish_prepared(**kwargs)
    assert first["status"] == "PUBLISHED_VERIFIED"
    assert first["manifest_output_transition"] == "122_TO_127"
    assert first["committed_members"][-1] == "EVENT_LOG.jsonl"
    gap_name = config["legacy_gse114002_public_gap"]["runtime_snapshot_name"]
    assert (run_root / gap_name).read_bytes() == public_gap
    for name in runtime_sync.immutable_names(config):
        assert (run_root / name).stat().st_nlink == 1
    second = runtime_sync.publish_prepared(**kwargs)
    assert second["status"] == "PUBLISHED_VERIFIED"
    assert second["committed_members"] == []
    assert (run_root / gap_name).read_bytes() == public_gap


@pytest.mark.parametrize("prefix_length", [0, 1, 2, 3])
def test_four_allowed_mutable_recovery_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix_length: int
) -> None:
    config, _predecessor, _public_gap, authority, run_root, prepared = prepare_context(
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
    )
    assert result["status"] == "PUBLISHED_VERIFIED"
    for name in runtime_sync.MUTABLE_NAMES:
        assert (run_root / name).read_bytes() == artifacts[name]


def test_event_is_last_and_postcommit_warning_recovers_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, _public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
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
    }
    warned = runtime_sync.publish_prepared(**kwargs, fault_injector=fail_event_fsync)
    assert warned["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert warned["warning_member"] == "EVENT_LOG.jsonl"
    events = runtime_sync.load_json_lines(
        (run_root / "EVENT_LOG.jsonl").read_bytes(), label="events"
    )
    assert events[-1]["event_id"] == "A1-EVT-040"
    retried = runtime_sync.publish_prepared(**kwargs)
    assert retried["status"] == "PUBLISHED_VERIFIED"


def test_immutable_temp_unlink_failure_reports_committed_manual_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, _public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    failed = False

    def fail_once(point: str) -> None:
        nonlocal failed
        if point == "immutable_post_link_unlink" and not failed:
            failed = True
            raise OSError("synthetic unlink failure")

    result = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=fail_once,
    )
    assert result["status"] == "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION"
    assert len(result["committed_members"]) == 1
    committed = result["committed_members"][0]
    assert (run_root / committed).read_bytes() == (prepared / committed).read_bytes()
    assert list(run_root.glob(".evt040.*.tmp"))


@pytest.mark.parametrize("immutable_index", range(5))
def test_existing_exact_immutable_with_external_hardlink_fails_before_mutables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    immutable_index: int,
) -> None:
    config, predecessor, _public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    immutable_name = runtime_sync.immutable_names(config)[immutable_index]
    target = run_root / immutable_name
    target.write_bytes((prepared / immutable_name).read_bytes())
    external = tmp_path / f"external-hardlink-{immutable_index}"
    os.link(target, external)
    assert target.stat().st_nlink == 2

    with pytest.raises(
        runtime_sync.PublicationError,
        match="member does not have exactly one hard link",
    ):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )

    assert target.stat().st_nlink == 2
    for name in runtime_sync.MUTABLE_NAMES:
        assert (run_root / name).read_bytes() == predecessor[name]


def test_immutable_fileexists_race_rejects_exact_multi_link_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    payload = b"exact immutable payload\n"
    final_name = "IMMUTABLE.json"
    external_name = "EXTERNAL_HARDLINK"
    real_link = os.link

    def synthesize_multilink_race(
        _source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        del src_dir_fd, follow_symlinks
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=dst_dir_fd,
        )
        try:
            runtime_sync.write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        real_link(
            target,
            external_name,
            src_dir_fd=dst_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=False,
        )
        raise FileExistsError("synthetic FileExists race")

    directory_fd = os.open(runtime_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    monkeypatch.setattr(runtime_sync.os, "link", synthesize_multilink_race)
    try:
        with pytest.raises(
            runtime_sync.PublicationError,
            match="member does not have exactly one hard link",
        ):
            runtime_sync.publish_immutable_at(directory_fd, final_name, payload)
    finally:
        os.close(directory_fd)
    assert (runtime_root / final_name).stat().st_nlink == 2
    assert (runtime_root / external_name).read_bytes() == payload
    assert not list(runtime_root.glob(".evt040.*.tmp"))


def test_differing_existing_public_gap_snapshot_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, _public_gap, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    gap_name = config["legacy_gse114002_public_gap"]["runtime_snapshot_name"]
    (run_root / gap_name).write_bytes(b"foreign")
    with pytest.raises(runtime_sync.PublicationError, match="existing immutable artifact differs"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
    assert (run_root / gap_name).read_bytes() == b"foreign"
    for name in runtime_sync.MUTABLE_NAMES:
        assert (run_root / name).read_bytes() == predecessor[name]


def test_recorded_at_and_path_boundary_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, public_gap, authority, run_root, _prepared = make_context(
        tmp_path, monkeypatch
    )
    with pytest.raises(runtime_sync.RuntimeSyncError, match="EVT-040 window"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "allowed" / "late",
            recorded_at="2026-08-13T10:30:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
            public_gap_payload_override=public_gap,
        )
    with pytest.raises(runtime_sync.PublicationError, match="below"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "outside",
            recorded_at="2026-08-11T10:30:00+08:00",
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
            public_gap_payload_override=public_gap,
        )
