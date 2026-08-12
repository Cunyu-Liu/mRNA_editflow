from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse200304_dec019_group_split_power_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse200304_dec019_group_split_power_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("evt044_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_disk_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def bound_config() -> dict[str, Any]:
    config = read_disk_config()
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    ledger = config["repository_authority"]["predecessor_ledger"]
    ledger["status"] = "BOUND"
    ledger["commit"] = "4" * 40
    for index, blob in enumerate(ledger["frozen_blobs"]):
        blob["sha256"] = str(5 + index) * 64
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    return config


def predecessor_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    status = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "training_started": False,
        "historical_status_field": "PRESERVED",
    }
    manifest = {
        "run_status": "IN_PROGRESS",
        "claim_status": "NOT_ESTABLISHED",
        "active_authority_commit": "a" * 40,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}",
                "artifact_type": f"EXISTING_{index:03d}",
                "bytes": index,
                "sha256": f"{index:064x}",
            }
            for index in range(163)
        ],
        "historical_manifest_field": "PRESERVED",
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-12T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 43)
    ]
    events.append(
        {
            "event_id": "A1-EVT-043",
            "at": "2026-08-12T03:00:00+08:00",
            "event": "EVT043",
            "decision_id": "V3-DEC-019",
        }
    )
    payloads = {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(runtime_sync.compact_json_line(event) for event in events),
    }
    for name, payload in payloads.items():
        config["runtime"]["predecessor_mutables"][name].update(
            {"bytes": len(payload), "sha256": runtime_sync.sha256(payload)}
        )
    tail = events[-1]
    config["runtime"]["predecessor_tail"].update(
        {"event_id": tail["event_id"], "decision_id": tail["decision_id"]}
    )
    return payloads


def make_context(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = bound_config()
    run_root = tmp_path / "run"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt044-job"
    run_root.mkdir()
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    predecessor = predecessor_payloads(config)
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {name: (run_root / name).read_bytes() for name in runtime_sync.MUTABLE_NAMES}


def install_fake_repository_authority(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    mode: str = "valid",
) -> bytes:
    """Bind a minimal L -> I -> B repository and install a direct fake Git."""

    ledger_commit = "4" * 40
    implementation_commit = "1" * 40
    binding_commit = "b" * 40
    script_payload = b"bound runtime publisher\n"
    test_payload = b"bound focused test\n"
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": runtime_sync.sha256(script_payload),
            "implementation_test_sha256": runtime_sync.sha256(test_payload),
        }
    )
    ledger = config["repository_authority"]["predecessor_ledger"]
    ledger["status"] = "BOUND"
    ledger["commit"] = ledger_commit
    ledger_payloads: dict[str, bytes] = {}
    for index, item in enumerate(ledger["frozen_blobs"]):
        payload = f"ledger member {index}\n".encode()
        ledger_payloads[item["path"]] = payload
        item["sha256"] = runtime_sync.sha256(payload)
    binding["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    config_payload = runtime_sync.json_bytes(config)
    i_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))
    changed_paths = {
        ledger_commit: [item["path"] for item in ledger["frozen_blobs"]],
        implementation_commit: list(
            config["repository_authority"]["implementation_exact_changed_paths"]
        ),
        binding_commit: list(config["repository_authority"]["binding_exact_changed_paths"]),
    }
    if mode in {"L_paths", "I_paths", "B_paths"}:
        commit = {
            "L_paths": ledger_commit,
            "I_paths": implementation_commit,
            "B_paths": binding_commit,
        }[mode]
        changed_paths[commit].append("unexpected/path")

    blobs = {
        **{(ledger_commit, path): payload for path, payload in ledger_payloads.items()},
        (implementation_commit, runtime_sync.CONFIG_REPO_PATH): i_payload,
        (binding_commit, runtime_sync.CONFIG_REPO_PATH): config_payload,
        (implementation_commit, runtime_sync.SCRIPT_REPO_PATH): script_payload,
        (binding_commit, runtime_sync.SCRIPT_REPO_PATH): script_payload,
        (implementation_commit, runtime_sync.TEST_REPO_PATH): test_payload,
        (binding_commit, runtime_sync.TEST_REPO_PATH): test_payload,
    }
    if mode == "script_hash":
        blobs[(implementation_commit, runtime_sync.SCRIPT_REPO_PATH)] += b"drift"

    def fake_git(_repo: Path, *arguments: str) -> bytes:
        branch = config["repository_authority"]["branch"]
        if arguments in (("rev-parse", "HEAD"), ("rev-parse", "@{upstream}")):
            return f"{binding_commit}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{branch}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{branch}\n".encode()
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if arguments == ("rev-parse", f"{binding_commit}^"):
            return f"{implementation_commit}\n".encode()
        if arguments == ("rev-parse", f"{implementation_commit}^"):
            return f"{ledger_commit}\n".encode()
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return ("\n".join(changed_paths[arguments[4]]) + "\n").encode()
        if arguments[0] == "show":
            commit, path = arguments[1].split(":", 1)
            return blobs[(commit, path)]
        raise AssertionError(arguments)

    worktree_payloads = {
        runtime_sync.CONFIG_REPO_PATH: config_payload,
        runtime_sync.SCRIPT_REPO_PATH: script_payload,
        runtime_sync.TEST_REPO_PATH: test_payload,
    }
    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(
        runtime_sync,
        "_read_repo_file",
        lambda _repo, path: worktree_payloads[path],
    )
    return config_payload


def test_disk_i_unbound_implementation_stops_before_runtime_or_source_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The checked-in I config has a bound ledger but no implementation binding.
    # It must stop before opening any runtime or registered-artifact path.
    config = read_disk_config()
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("runtime-or-source")
        raise AssertionError("runtime/source I/O occurred before binding")

    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="runtime-sync implementation is not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-12T04:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=tmp_path / "run",
        )
    assert accessed == []


def test_normal_transaction_registers_exact14_adds_exact4_and_preserves_scientific_gate(
    tmp_path: Path,
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T04:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["prepared_member_count"] == 7
    assert {item.name for item in prepared.iterdir()} == {
        *runtime_sync.MUTABLE_NAMES,
        *config["runtime"]["immutable_publish_order"],
    }
    published = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    status, manifest, events = runtime_sync._parse_runtime(read_runtime(run_root))
    assert len(events) == 44
    assert events[:-1] == runtime_sync.load_events(
        predecessor["EVENT_LOG.jsonl"], label="predecessor"
    )
    assert events[-1]["event_id"] == "A1-EVT-044"
    assert events[-1]["scientific_state_changed"] is True
    assert events[-1]["evidence_gate_statuses_changed_since_evt043"] is True
    assert events[-1]["overall_qualification_gate_changed"] is False
    assert len(manifest["outputs"]) == 181
    assert manifest["outputs"][:163] == runtime_sync._parse_runtime(predecessor)[1]["outputs"]
    assert manifest["outputs"][163:177] == config["registered_artifacts"]
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][-4:]] == list(
        config["runtime"]["immutable_publish_order"]
    )
    assert status["unresolved_blockers"] == ["CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"]
    assert status["qualified"] is False
    assert status["training_started"] is False
    assert status["model_selection_allowed"] is False
    sync = runtime_sync.load_json(
        (run_root / config["runtime"]["sync_name"]).read_bytes(), label="sync"
    )
    assert sync["scientific_state_changed"] is True
    assert sync["evidence_gate_statuses_changed_since_evt043"] is True
    assert sync["overall_qualification_gate_changed"] is False
    assert runtime_sync.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )["status"] == "PUBLISHED_VERIFIED"


def test_predecessor_drift_stops_before_prepared_or_runtime_write(tmp_path: Path) -> None:
    # This detects applying EVT044 to a runtime other than the exact EVT043
    # predecessor, and proves the failed prepare has zero writes.
    config, _predecessor, run_root, prepared = make_context(tmp_path)
    status = runtime_sync.load_json((run_root / "STATUS.json").read_bytes(), label="status")
    status["drift"] = True
    (run_root / "STATUS.json").write_bytes(runtime_sync.json_bytes(status))
    before = read_runtime(run_root)
    with pytest.raises(runtime_sync.PredecessorError, match="predecessor identity drift"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-12T04:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert read_runtime(run_root) == before
    assert not prepared.exists()


def test_supported_prefix_recovery_and_idempotent_retry(tmp_path: Path) -> None:
    config, _predecessor, run_root, prepared = make_context(tmp_path)
    runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T04:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )

    def fail_before_manifest(point: str) -> None:
        if point == "before_replace:RUN_MANIFEST.json":
            raise OSError("injected prefix interruption")

    with pytest.raises(runtime_sync.PublicationError, match="not committed"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=fail_before_manifest,
        )
    partial = read_runtime(run_root)
    # STATUS is the first committed mutable; RUN_MANIFEST and EVENT_LOG remain old.
    old_status = runtime_sync.load_json(
        (prepared / "STATUS_PRE_GSE200304_DEC019_GROUP_SPLIT_POWER_RUNTIME_SYNC_V1.json").read_bytes(),
        label="snapshot status",
    )
    new_status = runtime_sync.load_json(partial["STATUS.json"], label="new status")
    assert new_status != old_status
    assert partial["EVENT_LOG.jsonl"] == (
        prepared / "EVENT_LOG_PRE_GSE200304_DEC019_GROUP_SPLIT_POWER_RUNTIME_SYNC_V1.jsonl"
    ).read_bytes()
    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    reused = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert reused["status"] == "PUBLISHED_VERIFIED"
    assert reused["reused"] is True
    assert len(runtime_sync.load_events(read_runtime(run_root)["EVENT_LOG.jsonl"], label="events")) == 44


def test_production_authority_accepts_exact_chain_without_source_or_output_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = read_disk_config()
    config_payload = install_fake_repository_authority(monkeypatch, config)
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("source-or-output")
        raise AssertionError("source/output I/O is outside repository authority audit")

    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_prepared", forbidden)
    result = runtime_sync.audit_production_repository_authority(config, config_payload)
    assert result == {
        "status": "PASS_EXACT_L_TO_I_TO_CONFIG_ONLY_B",
        "ledger_commit": "4" * 40,
        "implementation_commit": "1" * 40,
        "binding_commit": "b" * 40,
    }
    assert accessed == []


@pytest.mark.parametrize("mode", ["L_paths", "I_paths", "B_paths", "script_hash"])
def test_production_authority_drift_stops_before_source_or_output_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = read_disk_config()
    config_payload = install_fake_repository_authority(monkeypatch, config, mode=mode)
    config_path = tmp_path / "production-config.json"
    config_path.write_bytes(config_payload)
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("source-or-output")
        raise AssertionError("source/output I/O occurred before repository authority")

    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_prepared", forbidden)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-12T04:00:00+08:00",
            config_path=config_path,
            production=True,
        )
    assert accessed == []
    assert not (tmp_path / "prepared").exists()
